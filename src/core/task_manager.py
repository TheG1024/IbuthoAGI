"""
Task manager for handling task decomposition and workflow.
"""
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, field
from datetime import datetime
import uuid
import logging
from enum import Enum

from pydantic import BaseModel
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

class TaskStatus(str, Enum):
    """Task status enum."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    BLOCKED = "blocked"
    COMPLETED = "completed"
    FAILED = "failed"

class TaskPriority(int, Enum):
    """Task priority enum."""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

@dataclass
class TaskDependency:
    """Task dependency information."""
    task_id: str
    dependency_type: str
    optional: bool = False
    timeout_seconds: Optional[int] = None

@dataclass
class Task:
    """Task representation."""
    id: str
    name: str
    description: str
    priority: TaskPriority
    status: TaskStatus = TaskStatus.PENDING
    dependencies: List[TaskDependency] = field(default_factory=list)
    subtasks: List[str] = field(default_factory=list)
    parent_task: Optional[str] = None
    assigned_agent: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

class TaskDecompositionPrompt:
    """Prompts for task decomposition."""
    
    DECOMPOSE_TASK = PromptTemplate(
        input_variables=["task_description"],
        template="""
        Given the following task, break it down into smaller, manageable subtasks:
        
        Task: {task_description}
        
        For each subtask, provide:
        1. Name
        2. Description
        3. Dependencies (if any)
        4. Estimated complexity (low/medium/high)
        
        Format your response as a JSON object.
        """
    )
    
    IDENTIFY_DEPENDENCIES = PromptTemplate(
        input_variables=["subtasks"],
        template="""
        Given the following subtasks, identify any dependencies between them:
        
        Subtasks:
        {subtasks}
        
        For each dependency, specify:
        1. Source task
        2. Target task (dependent on source)
        3. Type of dependency
        4. Whether it's optional
        
        Format your response as a JSON object.
        """
    )

class TaskManager:
    """Manages task decomposition and workflow."""
    
    def __init__(self, llm_chain: Optional[LLMChain] = None):
        self.tasks: Dict[str, Task] = {}
        self.llm_chain = llm_chain
        self.logger = logging.getLogger("task_manager")
        self.dependency_graph: Dict[str, Set[str]] = {}
    
    async def create_task(
        self,
        name: str,
        description: str,
        priority: TaskPriority,
        parent_task: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Task:
        """Create a new task."""
        task_id = str(uuid.uuid4())
        task = Task(
            id=task_id,
            name=name,
            description=description,
            priority=priority,
            parent_task=parent_task,
            metadata=metadata or {}
        )
        
        self.tasks[task_id] = task
        self.logger.info(f"Created task: {task_id} ({name})")
        return task
    
    async def decompose_task(self, task_id: str) -> List[Task]:
        """Decompose a task into subtasks."""
        task = self.tasks.get(task_id)
        if not task:
            raise ValueError(f"Task {task_id} not found")
        
        if not self.llm_chain:
            raise RuntimeError("LLM chain not initialized")
        
        # Generate subtasks using LLM
        response = await self.llm_chain.arun(
            TaskDecompositionPrompt.DECOMPOSE_TASK,
            task_description=task.description
        )
        
        subtasks_data = self._parse_decomposition_response(response)
        created_subtasks = []
        
        for subtask_data in subtasks_data:
            subtask = await self.create_task(
                name=subtask_data["name"],
                description=subtask_data["description"],
                priority=self._calculate_subtask_priority(
                    task.priority,
                    subtask_data["complexity"]
                ),
                parent_task=task_id,
                metadata={"complexity": subtask_data["complexity"]}
            )
            created_subtasks.append(subtask)
            task.subtasks.append(subtask.id)
        
        # Identify dependencies
        await self._identify_dependencies(created_subtasks)
        
        return created_subtasks
    
    def get_task(self, task_id: str) -> Optional[Task]:
        """Get task by ID."""
        return self.tasks.get(task_id)
    
    def get_tasks_by_status(self, status: TaskStatus) -> List[Task]:
        """Get all tasks with specified status."""
        return [
            task for task in self.tasks.values()
            if task.status == status
        ]
    
    def get_available_tasks(self) -> List[Task]:
        """Get tasks that are ready to be executed."""
        available_tasks = []
        
        for task in self.tasks.values():
            if task.status != TaskStatus.PENDING:
                continue
            
            if not self._are_dependencies_met(task):
                continue
            
            available_tasks.append(task)
        
        return sorted(
            available_tasks,
            key=lambda t: t.priority.value,
            reverse=True
        )
    
    async def update_task_status(
        self,
        task_id: str,
        status: TaskStatus,
        result: Optional[Dict[str, Any]] = None
    ) -> None:
        """Update task status."""
        task = self.tasks.get(task_id)
        if not task:
            raise ValueError(f"Task {task_id} not found")
        
        old_status = task.status
        task.status = status
        
        if status == TaskStatus.IN_PROGRESS and not task.started_at:
            task.started_at = datetime.now()
        elif status in [TaskStatus.COMPLETED, TaskStatus.FAILED]:
            task.completed_at = datetime.now()
            if result:
                task.result = result
        
        self.logger.info(
            f"Updated task {task_id} status: {old_status} -> {status}"
        )
        
        # Update parent task status if needed
        if task.parent_task:
            await self._update_parent_task_status(task.parent_task)
    
    def _are_dependencies_met(self, task: Task) -> bool:
        """Check if all task dependencies are met."""
        for dep in task.dependencies:
            dep_task = self.tasks.get(dep.task_id)
            if not dep_task:
                continue
            
            if not dep.optional and dep_task.status != TaskStatus.COMPLETED:
                return False
            
            if (
                dep.timeout_seconds
                and dep_task.started_at
                and (datetime.now() - dep_task.started_at).total_seconds()
                > dep.timeout_seconds
            ):
                return False
        
        return True
    
    async def _update_parent_task_status(self, parent_id: str) -> None:
        """Update parent task status based on subtasks."""
        parent = self.tasks.get(parent_id)
        if not parent:
            return
        
        subtask_statuses = [
            self.tasks[subtask_id].status
            for subtask_id in parent.subtasks
            if subtask_id in self.tasks
        ]
        
        if all(status == TaskStatus.COMPLETED for status in subtask_statuses):
            await self.update_task_status(parent_id, TaskStatus.COMPLETED)
        elif any(status == TaskStatus.FAILED for status in subtask_statuses):
            await self.update_task_status(parent_id, TaskStatus.FAILED)
        elif any(status == TaskStatus.IN_PROGRESS for status in subtask_statuses):
            await self.update_task_status(parent_id, TaskStatus.IN_PROGRESS)
    
    def _calculate_subtask_priority(
        self,
        parent_priority: TaskPriority,
        complexity: str
    ) -> TaskPriority:
        """Calculate subtask priority based on parent and complexity."""
        priority_value = parent_priority.value
        
        if complexity == "high":
            priority_value = min(priority_value + 1, TaskPriority.CRITICAL.value)
        elif complexity == "low":
            priority_value = max(priority_value - 1, TaskPriority.LOW.value)
        
        return TaskPriority(priority_value)
    
    def _parse_decomposition_response(
        self,
        response: str
    ) -> List[Dict[str, Any]]:
        """Parse LLM decomposition response."""
        # Implement parsing logic
        # This should convert the LLM's JSON response into a list of subtask data
        pass
    
    async def _identify_dependencies(self, tasks: List[Task]) -> None:
        """Identify dependencies between tasks."""
        if not self.llm_chain:
            return
        
        # Format tasks for prompt
        tasks_str = "\n".join([
            f"- {task.name}: {task.description}"
            for task in tasks
        ])
        
        # Get dependencies from LLM
        response = await self.llm_chain.arun(
            TaskDecompositionPrompt.IDENTIFY_DEPENDENCIES,
            subtasks=tasks_str
        )
        
        # Parse and add dependencies
        dependencies = self._parse_dependency_response(response)
        self._add_dependencies(dependencies)
    
    def _parse_dependency_response(
        self,
        response: str
    ) -> List[Dict[str, Any]]:
        """Parse LLM dependency response."""
        # Implement parsing logic
        # This should convert the LLM's JSON response into a list of dependency data
        pass
    
    def _add_dependencies(
        self,
        dependencies: List[Dict[str, Any]]
    ) -> None:
        """Add dependencies to tasks."""
        for dep_data in dependencies:
            source_task = self.tasks.get(dep_data["source"])
            target_task = self.tasks.get(dep_data["target"])
            
            if not source_task or not target_task:
                continue
            
            dependency = TaskDependency(
                task_id=source_task.id,
                dependency_type=dep_data["type"],
                optional=dep_data.get("optional", False),
                timeout_seconds=dep_data.get("timeout")
            )
            
            target_task.dependencies.append(dependency)
            
            # Update dependency graph
            if target_task.id not in self.dependency_graph:
                self.dependency_graph[target_task.id] = set()
            self.dependency_graph[target_task.id].add(source_task.id)
    
    def get_task_dependencies(self, task_id: str) -> Dict[str, Any]:
        """Get detailed dependency information for a task."""
        task = self.tasks.get(task_id)
        if not task:
            raise ValueError(f"Task {task_id} not found")
        
        return {
            "direct_dependencies": [
                {
                    "task_id": dep.task_id,
                    "type": dep.dependency_type,
                    "optional": dep.optional,
                    "status": self.tasks[dep.task_id].status
                    if dep.task_id in self.tasks else None
                }
                for dep in task.dependencies
            ],
            "dependent_tasks": [
                task_id for task_id, deps in self.dependency_graph.items()
                if task_id in deps
            ]
        }
