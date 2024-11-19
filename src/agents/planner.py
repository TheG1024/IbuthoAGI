"""
Planner agent implementation for strategic planning and solution design.
"""
from typing import Dict, List, Any, Optional
import json
from datetime import datetime
import networkx as nx

from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

from ..core.agent import Agent, AgentRole, AgentCapability
from ..core.task_manager import TaskManager, TaskPriority, TaskStatus

class PlanningPrompt:
    """Prompts for planning operations."""
    
    ANALYZE_PROBLEM = PromptTemplate(
        input_variables=["problem", "context", "constraints"],
        template="""
        Analyze the following problem and provide a strategic solution approach:
        
        Problem: {problem}
        Context: {context}
        Constraints: {constraints}
        
        Provide:
        1. Problem breakdown and dependencies
        2. Solution components and interfaces
        3. Resource requirements
        4. Risk assessment and mitigation
        5. Success metrics
        
        Format your response as a JSON object.
        """
    )
    
    DESIGN_SOLUTION = PromptTemplate(
        input_variables=["analysis", "requirements"],
        template="""
        Design a detailed solution based on the following analysis:
        
        Analysis: {analysis}
        Requirements: {requirements}
        
        Provide:
        1. Architecture overview
        2. Component specifications
        3. Interface definitions
        4. Implementation roadmap
        5. Testing strategy
        
        Format your response as a JSON object.
        """
    )

class PlannerAgent(Agent):
    """Agent responsible for strategic planning and solution design."""
    
    def __init__(
        self,
        agent_id: str,
        task_manager: TaskManager,
        llm_chain: Optional[LLMChain] = None
    ):
        super().__init__(
            agent_id=agent_id,
            role=AgentRole.PLANNER,
            capabilities=[
                AgentCapability.TASK_DECOMPOSITION,
                AgentCapability.SOLUTION_SYNTHESIS
            ],
            llm=llm_chain.llm if llm_chain else None
        )
        self.task_manager = task_manager
        self.llm_chain = llm_chain
        self.active_plans: Dict[str, Dict[str, Any]] = {}
        self.dependency_graphs: Dict[str, nx.DiGraph] = {}
    
    async def create_plan(
        self,
        problem: str,
        context: Optional[Dict[str, Any]] = None,
        constraints: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Create a strategic plan for solving a problem."""
        # Generate plan ID
        plan_id = f"plan_{datetime.now().timestamp()}"
        
        # Analyze problem
        analysis = await self._analyze_problem(
            problem,
            context,
            constraints
        )
        
        # Initialize plan
        self.active_plans[plan_id] = {
            "problem": problem,
            "context": context,
            "constraints": constraints,
            "analysis": analysis,
            "solution": None,
            "tasks": [],
            "status": "in_progress",
            "started_at": datetime.now()
        }
        
        # Create dependency graph
        self.dependency_graphs[plan_id] = self._create_dependency_graph(analysis)
        
        # Design solution
        solution = await self._design_solution(
            plan_id,
            analysis
        )
        
        # Generate tasks
        tasks = await self._generate_tasks(plan_id, solution)
        
        return {
            "plan_id": plan_id,
            "analysis": analysis,
            "solution": solution,
            "tasks": tasks
        }
    
    async def _analyze_problem(
        self,
        problem: str,
        context: Optional[Dict[str, Any]] = None,
        constraints: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Analyze problem and determine solution approach."""
        if not self.llm_chain:
            return {}
            
        response = await self.llm_chain.arun(
            PlanningPrompt.ANALYZE_PROBLEM,
            problem=problem,
            context=json.dumps(context) if context else "",
            constraints=json.dumps(constraints) if constraints else ""
        )
        
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            self.logger.error("Failed to parse problem analysis response")
            return {}
    
    def _create_dependency_graph(self, analysis: Dict[str, Any]) -> nx.DiGraph:
        """Create dependency graph from problem analysis."""
        graph = nx.DiGraph()
        
        # Extract components and dependencies from analysis
        components = analysis.get("solution_components", [])
        dependencies = analysis.get("dependencies", [])
        
        # Add nodes for components
        for component in components:
            graph.add_node(
                component["id"],
                **component
            )
        
        # Add edges for dependencies
        for dep in dependencies:
            graph.add_edge(
                dep["from"],
                dep["to"],
                **dep.get("metadata", {})
            )
        
        return graph
    
    async def _design_solution(
        self,
        plan_id: str,
        analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Design detailed solution based on analysis."""
        if not self.llm_chain:
            return {}
            
        # Get requirements from analysis
        requirements = analysis.get("requirements", {})
        
        response = await self.llm_chain.arun(
            PlanningPrompt.DESIGN_SOLUTION,
            analysis=json.dumps(analysis),
            requirements=json.dumps(requirements)
        )
        
        try:
            solution = json.loads(response)
            # Update plan
            plan = self.active_plans.get(plan_id)
            if plan:
                plan["solution"] = solution
            return solution
        except json.JSONDecodeError:
            self.logger.error("Failed to parse solution design response")
            return {}
    
    async def _generate_tasks(
        self,
        plan_id: str,
        solution: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate tasks from solution design."""
        tasks = []
        
        # Get implementation roadmap
        roadmap = solution.get("implementation_roadmap", [])
        
        # Create tasks for each roadmap item
        for item in roadmap:
            task = await self.task_manager.create_task(
                name=item.get("name", "Unnamed Task"),
                description=item.get("description", ""),
                priority=TaskPriority(item.get("priority", 2)),
                metadata={
                    "plan_id": plan_id,
                    "component": item.get("component"),
                    "dependencies": item.get("dependencies", [])
                }
            )
            tasks.append(task)
        
        # Update plan
        plan = self.active_plans.get(plan_id)
        if plan:
            plan["tasks"] = tasks
            
        return tasks
    
    def get_critical_path(self, plan_id: str) -> List[str]:
        """Get critical path of tasks in the plan."""
        graph = self.dependency_graphs.get(plan_id)
        if not graph:
            return []
            
        try:
            # Find longest path in the graph
            critical_path = nx.dag_longest_path(graph)
            return critical_path
        except nx.NetworkXError:
            self.logger.error(f"Failed to find critical path for plan {plan_id}")
            return []
    
    def get_plan_status(self, plan_id: str) -> Dict[str, Any]:
        """Get plan status and progress."""
        plan = self.active_plans.get(plan_id, {})
        
        # Calculate completion percentage
        total_tasks = len(plan.get("tasks", []))
        completed_tasks = sum(
            1 for task in plan.get("tasks", [])
            if task.get("status") == TaskStatus.COMPLETED
        )
        completion_percentage = (
            (completed_tasks / total_tasks * 100)
            if total_tasks > 0
            else 0
        )
        
        return {
            "status": plan.get("status"),
            "started_at": plan.get("started_at"),
            "completed_at": plan.get("completed_at"),
            "total_tasks": total_tasks,
            "completed_tasks": completed_tasks,
            "completion_percentage": completion_percentage,
            "critical_path": self.get_critical_path(plan_id)
        }
    
    async def update_plan(
        self,
        plan_id: str,
        updates: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Update plan with new information or changes."""
        plan = self.active_plans.get(plan_id)
        if not plan:
            return {}
            
        # Update plan fields
        for key, value in updates.items():
            if key in plan:
                plan[key] = value
                
        # If solution is updated, regenerate tasks
        if "solution" in updates:
            new_tasks = await self._generate_tasks(
                plan_id,
                updates["solution"]
            )
            plan["tasks"] = new_tasks
            
        # Update dependency graph if needed
        if "analysis" in updates:
            self.dependency_graphs[plan_id] = self._create_dependency_graph(
                updates["analysis"]
            )
            
        return plan
