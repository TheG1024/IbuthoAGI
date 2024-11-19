"""
Integration tests for the entire IbuthoAGI system.
Tests agent interactions, message processing, and task execution.
"""
import asyncio
import pytest
from datetime import datetime

from src.core.message import Message, MessagePriority
from src.core.agent import Agent, AgentRole
from src.core.communication import (
    CommunicationType,
    CommunicationProtocol,
    RoutingInfo,
    CommunicationManager
)
from src.core.task import Task, TaskStatus
from src.core.message_processing import MessageValidator

class TestAgent(Agent):
    """Test agent implementation for system testing."""
    
    def __init__(self, agent_id: str, role: AgentRole):
        super().__init__(agent_id=agent_id, role=role)
        self.received_messages = []
        self.processed_tasks = []
    
    async def process_message(self, message: Message) -> None:
        """Record received messages."""
        self.received_messages.append(message)
        await super().process_message(message)
    
    async def execute_task(self, task: Task) -> None:
        """Record processed tasks."""
        self.processed_tasks.append(task)
        # Simulate task execution
        await asyncio.sleep(0.1)
        task.status = TaskStatus.COMPLETED
        task.result = f"Executed by {self.agent_id}"

@pytest.fixture
def comm_manager():
    return CommunicationManager()

@pytest.fixture
def coordinator():
    return TestAgent("coordinator", AgentRole.COORDINATOR)

@pytest.fixture
def executor():
    return TestAgent("executor", AgentRole.EXECUTOR)

@pytest.fixture
def validator():
    return TestAgent("validator", AgentRole.VALIDATOR)

@pytest.fixture
async def setup_system(comm_manager, coordinator, executor, validator):
    """Setup the test system with agents and channels."""
    # Subscribe agents to appropriate channels
    await comm_manager.subscribe(coordinator.agent_id, "coordinator_channel")
    await comm_manager.subscribe(executor.agent_id, "executor_channel")
    await comm_manager.subscribe(validator.agent_id, "validator_channel")
    
    # Connect agents to communication manager
    coordinator.connect_to_manager(comm_manager)
    executor.connect_to_manager(comm_manager)
    validator.connect_to_manager(comm_manager)
    
    return {
        "manager": comm_manager,
        "coordinator": coordinator,
        "executor": executor,
        "validator": validator
    }

@pytest.mark.asyncio
async def test_task_execution_flow(setup_system):
    """Test the complete task execution flow through the system."""
    system = await setup_system
    coordinator = system["coordinator"]
    executor = system["executor"]
    validator = system["validator"]
    
    # Create a test task
    task = Task(
        id="test_task_1",
        description="Test task execution",
        priority=MessagePriority.MEDIUM,
        created_at=datetime.now().isoformat()
    )
    
    # Coordinator assigns task to executor
    assign_message = Message(
        id="msg_1",
        sender=coordinator.agent_id,
        receiver=executor.agent_id,
        content=task.to_dict(),
        priority=MessagePriority.MEDIUM,
        timestamp=datetime.now().isoformat()
    )
    
    await coordinator.send_message(assign_message)
    await asyncio.sleep(0.2)  # Wait for message processing
    
    # Verify executor received the task
    assert len(executor.received_messages) == 1
    assert executor.received_messages[0].id == assign_message.id
    
    # Verify task execution
    assert len(executor.processed_tasks) == 1
    executed_task = executor.processed_tasks[0]
    assert executed_task.id == task.id
    assert executed_task.status == TaskStatus.COMPLETED
    
    # Verify validation flow
    validation_messages = [msg for msg in validator.received_messages 
                         if msg.sender == executor.agent_id]
    assert len(validation_messages) > 0
    
    # Check system metrics
    metrics = system["manager"].get_metrics()
    assert metrics["total_messages"] > 0
    assert metrics["errors"] == 0

@pytest.mark.asyncio
async def test_broadcast_communication(setup_system):
    """Test broadcast communication between agents."""
    system = await setup_system
    coordinator = system["coordinator"]
    
    # Create broadcast message
    broadcast_msg = Message(
        id="broadcast_1",
        sender=coordinator.agent_id,
        receiver="broadcast",
        content="System-wide notification",
        priority=MessagePriority.HIGH,
        timestamp=datetime.now().isoformat()
    )
    
    # Send broadcast
    await system["manager"].broadcast_message(
        broadcast_msg,
        "system_broadcast"
    )
    await asyncio.sleep(0.2)  # Wait for message processing
    
    # Verify all agents received the broadcast
    for agent in [system["executor"], system["validator"]]:
        broadcasts = [msg for msg in agent.received_messages 
                     if msg.id == broadcast_msg.id]
        assert len(broadcasts) == 1

@pytest.mark.asyncio
async def test_message_validation(setup_system):
    """Test message content validation."""
    system = await setup_system
    coordinator = system["coordinator"]
    validator = MessageValidator()
    
    # Test valid task message
    task = Task(
        id="test_task_2",
        description="Test validation",
        priority=MessagePriority.MEDIUM,
        created_at=datetime.now().isoformat()
    )
    
    valid_msg = Message(
        id="valid_msg_1",
        sender=coordinator.agent_id,
        receiver="test_receiver",
        type="task_request",
        content={
            "type": "task",
            "task_id": task.id,
            "description": task.description,
            "priority": task.priority.value
        },
        priority=MessagePriority.MEDIUM,
        timestamp=datetime.now().isoformat()
    )
    
    # Should not raise exception
    validator.validate_message(valid_msg)
    
    # Test invalid message
    invalid_msg = Message(
        id="invalid_msg_1",
        sender=coordinator.agent_id,
        receiver="test_receiver",
        type="unknown_type",
        content={"invalid_key": "invalid_value"},
        priority=MessagePriority.MEDIUM,
        timestamp=datetime.now().isoformat()
    )
    
    # Should raise exception
    with pytest.raises(Exception):
        validator.validate_message(invalid_msg)

@pytest.mark.asyncio
async def test_error_handling(setup_system):
    """Test system error handling capabilities."""
    system = await setup_system
    coordinator = system["coordinator"]
    executor = system["executor"]
    
    # Create a message with invalid content
    invalid_msg = Message(
        id="error_msg_1",
        sender=coordinator.agent_id,
        receiver=executor.agent_id,
        type="unknown_type",
        content={"invalid": "content"},
        priority=MessagePriority.MEDIUM,
        timestamp=datetime.now().isoformat()
    )
    
    # Send invalid message
    await coordinator.send_message(invalid_msg)
    await asyncio.sleep(0.2)  # Wait for error handling
    
    # Check error metrics
    metrics = system["manager"].get_metrics()
    assert metrics["errors"] > 0
    
    # Verify error handling didn't crash the system
    # Send a valid message after error
    valid_msg = Message(
        id="valid_after_error",
        sender=coordinator.agent_id,
        receiver=executor.agent_id,
        type="task_request",
        content={
            "type": "task",
            "task_id": "recovery_task",
            "description": "Test system recovery",
            "priority": MessagePriority.MEDIUM.value
        },
        priority=MessagePriority.MEDIUM,
        timestamp=datetime.now().isoformat()
    )
    
    await coordinator.send_message(valid_msg)
    await asyncio.sleep(0.2)
    
    # Verify system still processes valid messages
    assert any(msg.id == valid_msg.id for msg in executor.received_messages)
