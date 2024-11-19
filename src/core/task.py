"""
Task management module for IbuthoAGI.
"""
from enum import Enum
from typing import Optional, Dict, Any
from datetime import datetime
from dataclasses import dataclass, field

from .message import MessagePriority

class TaskStatus(str, Enum):
    """Task execution status."""
    PENDING = "pending"
    ASSIGNED = "assigned"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    VALIDATED = "validated"
    REJECTED = "rejected"

@dataclass
class Task:
    """Represents a task to be executed by agents."""
    
    id: str
    description: str
    priority: MessagePriority
    created_at: str
    status: TaskStatus = TaskStatus.PENDING
    assigned_to: Optional[str] = None
    result: Optional[Any] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    validated_by: Optional[str] = None
    completed_at: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert task to dictionary format."""
        return {
            "id": self.id,
            "description": self.description,
            "priority": self.priority.value,
            "status": self.status.value,
            "created_at": self.created_at,
            "assigned_to": self.assigned_to,
            "result": self.result,
            "error": self.error,
            "metadata": self.metadata,
            "validated_by": self.validated_by,
            "completed_at": self.completed_at
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Task':
        """Create task from dictionary data."""
        return cls(
            id=data["id"],
            description=data["description"],
            priority=MessagePriority(data["priority"]),
            status=TaskStatus(data.get("status", TaskStatus.PENDING.value)),
            created_at=data["created_at"],
            assigned_to=data.get("assigned_to"),
            result=data.get("result"),
            error=data.get("error"),
            metadata=data.get("metadata", {}),
            validated_by=data.get("validated_by"),
            completed_at=data.get("completed_at")
        )
    
    def assign(self, agent_id: str) -> None:
        """Assign task to an agent."""
        self.assigned_to = agent_id
        self.status = TaskStatus.ASSIGNED
    
    def start(self) -> None:
        """Mark task as in progress."""
        if self.status != TaskStatus.ASSIGNED:
            raise ValueError("Task must be assigned before starting")
        self.status = TaskStatus.IN_PROGRESS
    
    def complete(self, result: Any) -> None:
        """Mark task as completed with result."""
        self.result = result
        self.status = TaskStatus.COMPLETED
        self.completed_at = datetime.now().isoformat()
    
    def fail(self, error: str) -> None:
        """Mark task as failed with error."""
        self.error = error
        self.status = TaskStatus.FAILED
        self.completed_at = datetime.now().isoformat()
    
    def validate(self, validator_id: str) -> None:
        """Mark task as validated."""
        if self.status != TaskStatus.COMPLETED:
            raise ValueError("Only completed tasks can be validated")
        self.validated_by = validator_id
        self.status = TaskStatus.VALIDATED
    
    def reject(self, validator_id: str, reason: str) -> None:
        """Mark task as rejected."""
        if self.status != TaskStatus.COMPLETED:
            raise ValueError("Only completed tasks can be rejected")
        self.validated_by = validator_id
        self.status = TaskStatus.REJECTED
        self.error = reason
