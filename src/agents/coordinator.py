"""
Coordinator agent implementation for orchestrating problem-solving workflows.
"""
from typing import Dict, List, Any, Optional
import json
from datetime import datetime

from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

from ..core.agent import Agent, AgentRole, AgentCapability, Message
from ..core.task_manager import TaskManager, TaskPriority, TaskStatus

class CoordinationPrompt:
    """Prompts for coordination decisions."""
    
    ANALYZE_TASK = PromptTemplate(
        input_variables=["task_description"],
        template="""
        Analyze the following task and determine the best approach to solve it:
        
        Task: {task_description}
        
        Provide:
        1. Required agent roles and their responsibilities
        2. High-level steps needed
        3. Potential challenges and mitigation strategies
        4. Success criteria
        
        Format your response as a JSON object.
        """
    )
    
    ASSIGN_TASK = PromptTemplate(
        input_variables=["task", "available_agents"],
        template="""
        Given the following task and available agents, determine the best agent assignments:
        
        Task: {task}
        Available Agents: {available_agents}
        
        For each subtask, specify:
        1. Most suitable agent(s)
        2. Rationale for selection
        3. Required capabilities
        
        Format your response as a JSON object.
        """
    )

class CoordinatorAgent(Agent):
    """Agent responsible for coordinating problem-solving workflows."""
    
    def __init__(
        self,
        agent_id: str,
        task_manager: TaskManager,
        llm_chain: Optional[LLMChain] = None
    ):
        super().__init__(
            agent_id=agent_id,
            role=AgentRole.COORDINATOR,
            capabilities=[
                AgentCapability.TASK_DECOMPOSITION,
                AgentCapability.SOLUTION_SYNTHESIS
            ],
            llm=llm_chain.llm if llm_chain else None
        )
        self.task_manager = task_manager
        self.llm_chain = llm_chain
        self.active_workflows: Dict[str, Dict[str, Any]] = {}
    
    async def _coordinate_task(self, task_content: Dict[str, Any]) -> Dict[str, Any]:
        """Coordinate task execution."""
        task_id = task_content.get("task_id")
        task = task_content.get("task", {})
        
        # Analyze task and determine approach
        analysis = await self._analyze_task(task)
        
        # Create main task
        main_task = await self.task_manager.create_task(
            name=task.get("name", "Unnamed Task"),
            description=task.get("description", ""),
            priority=TaskPriority(task.get("priority", 2))
        )
        
        # Decompose into subtasks
        subtasks = await self.task_manager.decompose_task(main_task.id)
        
        # Create workflow
        workflow_id = await self._create_workflow(
            main_task.id,
            analysis,
            subtasks
        )
        
        # Start execution
        await self._execute_workflow(workflow_id)
        
        return {
            "workflow_id": workflow_id,
            "main_task_id": main_task.id,
            "analysis": analysis
        }
    
    async def _analyze_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze task and determine solution approach."""
        if not self.llm_chain:
            return {}
        
        response = await self.llm_chain.arun(
            CoordinationPrompt.ANALYZE_TASK,
            task_description=task.get("description", "")
        )
        
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            self.logger.error("Failed to parse task analysis response")
            return {}
    
    async def _create_workflow(
        self,
        main_task_id: str,
        analysis: Dict[str, Any],
        subtasks: List[Any]
    ) -> str:
        """Create a new workflow."""
        workflow_id = f"workflow_{main_task_id}"
        
        self.active_workflows[workflow_id] = {
            "main_task_id": main_task_id,
            "analysis": analysis,
            "subtasks": {
                subtask.id: {
                    "status": TaskStatus.PENDING,
                    "assigned_agent": None,
                    "started_at": None,
                    "completed_at": None
                }
                for subtask in subtasks
            },
            "created_at": datetime.now(),
            "status": "created"
        }
        
        return workflow_id
    
    async def _execute_workflow(self, workflow_id: str) -> None:
        """Execute workflow tasks."""
        workflow = self.active_workflows.get(workflow_id)
        if not workflow:
            return
        
        workflow["status"] = "in_progress"
        
        # Get available tasks
        available_tasks = self.task_manager.get_available_tasks()
        
        for task in available_tasks:
            if task.id not in workflow["subtasks"]:
                continue
            
            # Assign task to appropriate agent
            assigned = await self._assign_task(task)
            if assigned:
                workflow["subtasks"][task.id]["assigned_agent"] = assigned
                workflow["subtasks"][task.id]["started_at"] = datetime.now()
    
    async def _assign_task(self, task: Any) -> Optional[str]:
        """Assign task to appropriate agent."""
        # This would interact with the agent manager to find and assign
        # the most suitable agent for the task
        pass
    
    async def _handle_task_completion(
        self,
        task_id: str,
        result: Dict[str, Any]
    ) -> None:
        """Handle task completion."""
        # Update task status
        await self.task_manager.update_task_status(
            task_id,
            TaskStatus.COMPLETED,
            result
        )
        
        # Update workflow
        for workflow in self.active_workflows.values():
            if task_id in workflow["subtasks"]:
                workflow["subtasks"][task_id].update({
                    "status": TaskStatus.COMPLETED,
                    "completed_at": datetime.now(),
                    "result": result
                })
                
                # Check if workflow is complete
                if all(
                    subtask["status"] == TaskStatus.COMPLETED
                    for subtask in workflow["subtasks"].values()
                ):
                    workflow["status"] = "completed"
                    
                # Execute next available tasks
                await self._execute_workflow(
                    f"workflow_{workflow['main_task_id']}"
                )
                break
    
    def get_workflow_status(self, workflow_id: str) -> Dict[str, Any]:
        """Get workflow status."""
        workflow = self.active_workflows.get(workflow_id, {})
        return {
            "status": workflow.get("status"),
            "tasks": {
                task_id: {
                    "status": task_info["status"],
                    "assigned_agent": task_info["assigned_agent"],
                    "started_at": task_info["started_at"],
                    "completed_at": task_info["completed_at"]
                }
                for task_id, task_info in workflow.get("subtasks", {}).items()
            },
            "created_at": workflow.get("created_at"),
            "analysis": workflow.get("analysis")
        }
