"""
Message class and related types for agent communication.
"""
from typing import Dict, Optional, Any
from datetime import datetime
from enum import Enum

class MessagePriority(int, Enum):
    """Message priority levels."""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    URGENT = 4

class Message:
    """Represents a message in the agent communication system."""
    
    def __init__(
        self,
        id: str,
        sender: str,
        receiver: str,
        type: str,
        content: Dict[str, Any],
        timestamp: datetime,
        priority: MessagePriority = MessagePriority.MEDIUM,
        requires_response: bool = False,
        context: Optional[Dict[str, Any]] = None
    ):
        self.id = id
        self.sender = sender
        self.receiver = receiver
        self.type = type
        self.content = content
        self.timestamp = timestamp
        self.priority = priority
        self.requires_response = requires_response
        self.context = context or {}
    
    def dict(self) -> Dict[str, Any]:
        """Convert message to dictionary format."""
        return {
            "id": self.id,
            "sender": self.sender,
            "receiver": self.receiver,
            "type": self.type,
            "content": self.content,
            "timestamp": self.timestamp,
            "priority": self.priority,
            "requires_response": self.requires_response,
            "context": self.context
        }
