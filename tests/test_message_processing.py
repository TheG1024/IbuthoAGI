"""
Tests for message processing module.
"""
import pytest
from datetime import datetime, timedelta
import uuid

from src.core.message_processing import (
    MessageValidator,
    MessageTransformer,
    MessageProcessor,
    MessageValidationError,
    MessageTransformationError,
    TaskContent,
    QueryContent,
    ResultContent,
    ErrorContent,
    StatusContent
)
from src.core.agent import Message
from src.core.communication import MessagePriority

@pytest.fixture
def valid_message():
    return Message(
        id=str(uuid.uuid4()),
        sender="test_sender",
        receiver="test_receiver",
        type="task_request",
        content={
            "type": "task",
            "task_id": str(uuid.uuid4()),
            "description": "Test task",
            "requirements": ["req1", "req2"],
            "priority": 2,
            "data": {}
        },
        timestamp=datetime.now(),
        priority=MessagePriority.MEDIUM,
        requires_response=True,
        context={}
    )

@pytest.fixture
def invalid_message():
    return Message(
        id="",  # Invalid: empty ID
        sender="test_sender",
        receiver="test_receiver",
        type="task_request",
        content={},  # Invalid: empty content
        timestamp=datetime.now() + timedelta(days=1),  # Invalid: future timestamp
        priority=5,  # Invalid: priority out of range
        requires_response=True,
        context={}
    )

@pytest.fixture
def message_processor():
    return MessageProcessor()

class TestMessageValidator:
    def test_validate_valid_message(self, valid_message):
        validator = MessageValidator()
        validator.validate_message(valid_message)  # Should not raise
    
    def test_validate_invalid_message(self, invalid_message):
        validator = MessageValidator()
        with pytest.raises(MessageValidationError):
            validator.validate_message(invalid_message)
    
    def test_validate_task_content(self):
        content = {
            "type": "task",
            "task_id": str(uuid.uuid4()),
            "description": "Test task",
            "requirements": ["req1"],
            "priority": 2,
            "data": {}
        }
        TaskContent(**content)  # Should not raise
    
    def test_validate_invalid_task_content(self):
        content = {
            "type": "task",
            "description": "Missing task_id",
            "data": {}
        }
        with pytest.raises(Exception):
            TaskContent(**content)

class TestMessageTransformer:
    def test_transform_for_external(self, valid_message):
        transformer = MessageTransformer()
        result = transformer.transform_for_external(valid_message)
        assert isinstance(result["timestamp"], str)
        assert result["id"] == valid_message.id
    
    def test_transform_from_external(self, valid_message):
        transformer = MessageTransformer()
        external_data = transformer.transform_for_external(valid_message)
        result = transformer.transform_from_external(external_data)
        assert isinstance(result, Message)
        assert result.id == valid_message.id
    
    def test_transform_invalid_external_data(self):
        transformer = MessageTransformer()
        with pytest.raises(MessageTransformationError):
            transformer.transform_from_external({"invalid": "data"})
    
    def test_transform_for_logging(self, valid_message):
        transformer = MessageTransformer()
        result = transformer.transform_for_logging(valid_message)
        assert "content" not in result
        assert "content_type" in result
    
    def test_transform_content(self):
        transformer = MessageTransformer()
        content = {
            "type": "task",
            "task_id": str(uuid.uuid4()),
            "description": "Test task",
            "status": "completed",
            "data": {"result": "success"}
        }
        result = transformer.transform_content(content, "task_request", "task_response")
        assert result["type"] == "result"

class TestMessageProcessor:
    @pytest.mark.asyncio
    async def test_process_outgoing_valid(self, message_processor, valid_message):
        result = await message_processor.process_outgoing(valid_message)
        assert isinstance(result, dict)
        assert result["id"] == valid_message.id
    
    @pytest.mark.asyncio
    async def test_process_outgoing_invalid(self, message_processor, invalid_message):
        with pytest.raises(MessageValidationError):
            await message_processor.process_outgoing(invalid_message)
    
    @pytest.mark.asyncio
    async def test_process_incoming_valid(self, message_processor, valid_message):
        data = MessageTransformer.transform_for_external(valid_message)
        result = await message_processor.process_incoming(data)
        assert isinstance(result, Message)
        assert result.id == valid_message.id
    
    @pytest.mark.asyncio
    async def test_process_incoming_invalid(self, message_processor):
        with pytest.raises(Exception):
            await message_processor.process_incoming({"invalid": "data"})

class TestContentSchemas:
    def test_task_content(self):
        content = TaskContent(
            type="task",
            task_id=str(uuid.uuid4()),
            description="Test task",
            requirements=["req1"],
            priority=2,
            data={}
        )
        assert content.type == "task"
    
    def test_query_content(self):
        content = QueryContent(
            type="query",
            query_id=str(uuid.uuid4()),
            query="test query",
            data={},
            context={"key": "value"}
        )
        assert content.type == "query"
    
    def test_result_content(self):
        content = ResultContent(
            type="result",
            task_id=str(uuid.uuid4()),
            status="completed",
            data={"result": "success"},
            metrics={"time": 1.0}
        )
        assert content.type == "result"
    
    def test_error_content(self):
        content = ErrorContent(
            type="error",
            error_code="E001",
            message="Test error",
            data={},
            details={"stack": "trace"}
        )
        assert content.type == "error"
    
    def test_status_content(self):
        content = StatusContent(
            type="status",
            agent_id="test_agent",
            status="active",
            data={},
            metrics={"cpu": 0.5}
        )
        assert content.type == "status"
