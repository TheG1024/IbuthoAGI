"""
Executor agent implementation for task execution and resource management.
"""
from typing import Dict, List, Any, Optional
import json
from datetime import datetime
import asyncio
from concurrent.futures import ThreadPoolExecutor

from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

from ..core.agent import Agent, AgentRole, AgentCapability
from ..core.task_manager import TaskManager, TaskStatus
from ..utils.monitoring import MonitoringSystem

class ExecutionPrompt:
    """Prompts for execution operations."""
    
    EXECUTION_PLAN = PromptTemplate(
        input_variables=["task", "resources", "constraints"],
        template="""
        Create an execution plan for the following task:
        
        Task: {task}
        Available Resources: {resources}
        Constraints: {constraints}
        
        Provide:
        1. Execution steps and commands
        2. Resource allocation
        3. Error handling strategy
        4. Success criteria
        5. Rollback plan
        
        Format your response as a JSON object.
        """
    )
    
    ERROR_RESOLUTION = PromptTemplate(
        input_variables=["error", "context", "previous_attempts"],
        template="""
        Analyze and provide resolution for the following error:
        
        Error: {error}
        Context: {context}
        Previous Attempts: {previous_attempts}
        
        Provide:
        1. Root cause analysis
        2. Resolution steps
        3. Prevention measures
        4. Verification steps
        
        Format your response as a JSON object.
        """
    )

class ExecutorAgent(Agent):
    """Agent responsible for task execution and resource management."""
    
    def __init__(
        self,
        agent_id: str,
        task_manager: TaskManager,
        monitoring_system: MonitoringSystem,
        llm_chain: Optional[LLMChain] = None,
        max_concurrent_tasks: int = 5
    ):
        super().__init__(
            agent_id=agent_id,
            role=AgentRole.EXECUTOR,
            capabilities=[
                AgentCapability.TASK_EXECUTION,
                AgentCapability.RESOURCE_MANAGEMENT
            ],
            llm=llm_chain.llm if llm_chain else None
        )
        self.task_manager = task_manager
        self.monitoring_system = monitoring_system
        self.llm_chain = llm_chain
        self.max_concurrent_tasks = max_concurrent_tasks
        self.active_executions: Dict[str, Dict[str, Any]] = {}
        self.thread_pool = ThreadPoolExecutor(max_workers=max_concurrent_tasks)
        self.resource_pool: Dict[str, Any] = {}
    
    async def execute_task(
        self,
        task_id: str,
        resources: Optional[Dict[str, Any]] = None,
        constraints: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Execute a task with given resources and constraints."""
        # Get task details
        task = await self.task_manager.get_task(task_id)
        if not task:
            return {"status": "error", "message": "Task not found"}
        
        # Create execution ID
        execution_id = f"exec_{task_id}_{datetime.now().timestamp()}"
        
        # Generate execution plan
        plan = await self._create_execution_plan(
            task,
            resources,
            constraints
        )
        
        # Initialize execution
        self.active_executions[execution_id] = {
            "task_id": task_id,
            "plan": plan,
            "status": "initializing",
            "resources": resources or {},
            "started_at": datetime.now(),
            "errors": [],
            "metrics": {},
            "results": None
        }
        
        try:
            # Allocate resources
            await self._allocate_resources(execution_id)
            
            # Execute plan
            results = await self._execute_plan(execution_id)
            
            # Update execution status
            self.active_executions[execution_id].update({
                "status": "completed",
                "completed_at": datetime.now(),
                "results": results
            })
            
            # Update task status
            await self.task_manager.update_task_status(
                task_id,
                TaskStatus.COMPLETED,
                results
            )
            
            return {
                "execution_id": execution_id,
                "status": "completed",
                "results": results
            }
            
        except Exception as e:
            error_resolution = await self._handle_error(execution_id, str(e))
            return {
                "execution_id": execution_id,
                "status": "error",
                "error": str(e),
                "resolution": error_resolution
            }
        finally:
            # Release resources
            await self._release_resources(execution_id)
    
    async def _create_execution_plan(
        self,
        task: Dict[str, Any],
        resources: Optional[Dict[str, Any]],
        constraints: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Create detailed execution plan for task."""
        if not self.llm_chain:
            return {}
            
        response = await self.llm_chain.arun(
            ExecutionPrompt.EXECUTION_PLAN,
            task=json.dumps(task),
            resources=json.dumps(resources) if resources else "",
            constraints=json.dumps(constraints) if constraints else ""
        )
        
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            self.logger.error("Failed to parse execution plan response")
            return {}
    
    async def _allocate_resources(self, execution_id: str) -> None:
        """Allocate required resources for execution."""
        execution = self.active_executions.get(execution_id)
        if not execution:
            return
            
        resources = execution.get("resources", {})
        
        # Check resource availability
        for resource_id, requirements in resources.items():
            if resource_id not in self.resource_pool:
                self.resource_pool[resource_id] = {
                    "total": requirements.get("quantity", 1),
                    "available": requirements.get("quantity", 1)
                }
            
            # Wait for resource availability
            while self.resource_pool[resource_id]["available"] < requirements.get("quantity", 1):
                await asyncio.sleep(1)
            
            # Allocate resources
            self.resource_pool[resource_id]["available"] -= requirements.get("quantity", 1)
    
    async def _release_resources(self, execution_id: str) -> None:
        """Release allocated resources."""
        execution = self.active_executions.get(execution_id)
        if not execution:
            return
            
        resources = execution.get("resources", {})
        
        # Release all allocated resources
        for resource_id, requirements in resources.items():
            if resource_id in self.resource_pool:
                self.resource_pool[resource_id]["available"] += requirements.get("quantity", 1)
    
    async def _execute_plan(self, execution_id: str) -> Dict[str, Any]:
        """Execute the plan steps."""
        execution = self.active_executions.get(execution_id)
        if not execution:
            return {}
            
        plan = execution.get("plan", {})
        results = {}
        
        # Update status
        execution["status"] = "executing"
        
        # Execute each step in the plan
        for step in plan.get("steps", []):
            step_id = step.get("id")
            
            try:
                # Start monitoring
                self.monitoring_system.start_monitoring(
                    f"{execution_id}_{step_id}",
                    step.get("metrics", [])
                )
                
                # Execute step
                step_result = await self._execute_step(step)
                
                # Store results
                results[step_id] = step_result
                
                # Collect metrics
                metrics = self.monitoring_system.get_metrics(
                    f"{execution_id}_{step_id}"
                )
                execution["metrics"][step_id] = metrics
                
            except Exception as e:
                self.logger.error(f"Error executing step {step_id}: {str(e)}")
                raise
            finally:
                # Stop monitoring
                self.monitoring_system.stop_monitoring(
                    f"{execution_id}_{step_id}"
                )
        
        return results
    
    async def _execute_step(self, step: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single step of the plan."""
        # This would contain the actual execution logic
        # For now, we'll simulate execution
        await asyncio.sleep(1)
        return {"status": "completed", "output": f"Executed {step.get('id')}"}
    
    async def _handle_error(
        self,
        execution_id: str,
        error: str
    ) -> Dict[str, Any]:
        """Handle execution error."""
        if not self.llm_chain:
            return {}
            
        execution = self.active_executions.get(execution_id)
        if not execution:
            return {}
            
        # Update execution status
        execution["status"] = "error"
        execution["errors"].append({
            "error": error,
            "timestamp": datetime.now().isoformat()
        })
        
        # Get error resolution
        response = await self.llm_chain.arun(
            ExecutionPrompt.ERROR_RESOLUTION,
            error=error,
            context=json.dumps(execution),
            previous_attempts=json.dumps(execution["errors"])
        )
        
        try:
            resolution = json.loads(response)
            return resolution
        except json.JSONDecodeError:
            self.logger.error("Failed to parse error resolution response")
            return {}
    
    def get_execution_status(self, execution_id: str) -> Dict[str, Any]:
        """Get execution status and metrics."""
        execution = self.active_executions.get(execution_id, {})
        return {
            "status": execution.get("status"),
            "started_at": execution.get("started_at"),
            "completed_at": execution.get("completed_at"),
            "metrics": execution.get("metrics", {}),
            "errors": execution.get("errors", []),
            "results": execution.get("results")
        }
    
    def get_resource_status(self) -> Dict[str, Any]:
        """Get status of resource pool."""
        return {
            resource_id: {
                "total": info["total"],
                "available": info["available"],
                "utilized": info["total"] - info["available"]
            }
            for resource_id, info in self.resource_pool.items()
        }
