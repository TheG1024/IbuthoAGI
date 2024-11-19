"""
Message validation and transformation system for IbuthoAGI.
"""
from typing import Dict, List, Optional, Any, Callable, Type, Union
from datetime import datetime
import json
from enum import Enum
import logging
from pydantic import BaseModel, Field, validator

from .message import Message, MessagePriority

class MessageValidationError(Exception):
    """Raised when message validation fails."""
    pass

class MessageTransformationError(Exception):
    """Raised when message transformation fails."""
    pass

class ContentSchema(BaseModel):
    """Base schema for message content validation."""
    type: str = Field(..., description="Content type identifier")
    data: Dict[str, Any] = Field(..., description="Actual content data")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Optional metadata")

    @validator("type")
    def validate_type(cls, v):
        if not v.strip():
            raise ValueError("Content type cannot be empty")
        return v.lower()

class TaskContent(ContentSchema):
    """Schema for task-related message content."""
    type: str = "task"
    task_id: str = Field(..., description="Unique task identifier")
    description: str = Field(..., description="Task description")
    requirements: List[str] = Field(default_factory=list, description="Task requirements")
    priority: int = Field(default=1, ge=1, le=4, description="Task priority")
    deadline: Optional[datetime] = Field(None, description="Task deadline")

class QueryContent(ContentSchema):
    """Schema for query-related message content."""
    type: str = "query"
    query_id: str = Field(..., description="Unique query identifier")
    query: str = Field(..., description="Query string")
    context: Dict[str, Any] = Field(default_factory=dict, description="Query context")

class ResultContent(ContentSchema):
    """Schema for result-related message content."""
    type: str = "result"
    task_id: str = Field(..., description="Related task identifier")
    status: str = Field(..., description="Result status")
    data: Dict[str, Any] = Field(..., description="Result data")
    metrics: Dict[str, Any] = Field(default_factory=dict, description="Performance metrics")

class ErrorContent(ContentSchema):
    """Schema for error-related message content."""
    type: str = "error"
    error_code: str = Field(..., description="Error code")
    message: str = Field(..., description="Error message")
    details: Dict[str, Any] = Field(default_factory=dict, description="Error details")

class StatusContent(ContentSchema):
    """Schema for status-related message content."""
    type: str = "status"
    agent_id: str = Field(..., description="Agent identifier")
    status: str = Field(..., description="Current status")
    metrics: Dict[str, float] = Field(default_factory=dict, description="Status metrics")

# Map message types to their content schemas
CONTENT_SCHEMAS = {
    "task_request": TaskContent,
    "task_response": ResultContent,
    "query_request": QueryContent,
    "query_response": ResultContent,
    "error_response": ErrorContent,
    "status_response": StatusContent
}

class MessageValidator:
    """Validates message structure and content."""
    
    @staticmethod
    def validate_message(message: Message) -> None:
        """Validate a message."""
        try:
            # Validate basic message structure
            if not message.id or not message.sender or not message.type:
                raise MessageValidationError("Missing required message fields")
            
            # Validate content schema
            if message.type in CONTENT_SCHEMAS:
                schema = CONTENT_SCHEMAS[message.type]
                try:
                    schema(**message.content)
                except Exception as e:
                    raise MessageValidationError(f"Invalid content schema: {str(e)}")
            
            # Validate priority
            if not 1 <= message.priority <= 4:
                raise MessageValidationError("Invalid priority level")
            
            # Validate timestamp
            if message.timestamp > datetime.now():
                raise MessageValidationError("Future timestamp not allowed")
            
        except Exception as e:
            raise MessageValidationError(f"Message validation failed: {str(e)}")

class MessageTransformer:
    """Transforms messages between different formats and schemas."""
    
    @staticmethod
    def transform_for_external(message: Message) -> Dict[str, Any]:
        """Transform message for external systems."""
        return {
            "id": message.id,
            "sender": message.sender,
            "receiver": message.receiver,
            "type": message.type,
            "content": message.content,
            "timestamp": message.timestamp.isoformat(),
            "priority": message.priority,
            "requires_response": message.requires_response,
            "context": message.context
        }
    
    @staticmethod
    def transform_from_external(data: Dict[str, Any]) -> Message:
        """Transform external data into a message."""
        try:
            # Convert ISO timestamp to datetime
            if isinstance(data.get("timestamp"), str):
                data["timestamp"] = datetime.fromisoformat(data["timestamp"])
            
            return Message(**data)
        except Exception as e:
            raise MessageTransformationError(f"Failed to transform external data: {str(e)}")
    
    @staticmethod
    def transform_for_logging(message: Message) -> Dict[str, Any]:
        """Transform message for logging purposes."""
        return {
            "id": message.id,
            "sender": message.sender,
            "receiver": message.receiver,
            "type": message.type,
            "content_type": message.content.get("type", "unknown"),
            "timestamp": message.timestamp.isoformat(),
            "priority": message.priority
        }
    
    @staticmethod
    def transform_content(
        content: Dict[str, Any],
        source_type: str,
        target_type: str
    ) -> Dict[str, Any]:
        """Transform content between different schemas."""
        try:
            # Validate source content
            if source_type in CONTENT_SCHEMAS:
                source_schema = CONTENT_SCHEMAS[source_type]
                content = source_schema(**content).dict()
            
            # Transform to target schema
            if target_type in CONTENT_SCHEMAS:
                target_schema = CONTENT_SCHEMAS[target_type]
                return target_schema(**content).dict()
            
            return content
            
        except Exception as e:
            raise MessageTransformationError(
                f"Failed to transform content from {source_type} to {target_type}: {str(e)}"
            )

class MessageProcessor:
    """Processes messages with validation and transformation."""
    
    def __init__(self):
        self.validator = MessageValidator()
        self.transformer = MessageTransformer()
        self.logger = logging.getLogger("message_processor")
    
    async def process_outgoing(
        self,
        message: Message,
        target_system: Optional[str] = None
    ) -> Dict[str, Any]:
        """Process outgoing message."""
        try:
            # Validate message
            self.validator.validate_message(message)
            
            # Transform message
            if target_system:
                transformed = self.transformer.transform_for_external(message)
            else:
                transformed = message.dict()
            
            # Log processed message
            self.logger.debug(
                "Processed outgoing message",
                extra=self.transformer.transform_for_logging(message)
            )
            
            return transformed
            
        except Exception as e:
            self.logger.error(f"Failed to process outgoing message: {str(e)}")
            raise
    
    async def process_incoming(
        self,
        data: Dict[str, Any],
        source_system: Optional[str] = None
    ) -> Message:
        """Process incoming message."""
        try:
            # Transform incoming data
            if source_system:
                message = self.transformer.transform_from_external(data)
            else:
                message = Message(**data)
            
            # Validate message
            self.validator.validate_message(message)
            
            # Log processed message
            self.logger.debug(
                "Processed incoming message",
                extra=self.transformer.transform_for_logging(message)
            )
            
            return message
            
        except Exception as e:
            self.logger.error(f"Failed to process incoming message: {str(e)}")
            raise
