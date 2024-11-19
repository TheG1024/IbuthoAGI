"""
Tests for the communication module.
"""
import asyncio
import pytest
from datetime import datetime

from src.core.message import Message, MessagePriority
from src.core.communication import (
    CommunicationType,
    CommunicationProtocol,
    RoutingInfo,
    CommunicationChannel,
    CommunicationManager
)

@pytest.fixture
def test_message():
    return Message(
        id="test_message_id",
        sender="test_sender",
        receiver="test_receiver",
        content="test content",
        priority=MessagePriority.MEDIUM,
        timestamp=datetime.now().isoformat()
    )

@pytest.fixture
def test_routing_info():
    return RoutingInfo(
        source="test_sender",
        destination="test_receiver",
        route_type=CommunicationType.DIRECT,
        protocol=CommunicationProtocol.ASYNC
    )

@pytest.fixture
def communication_channel():
    return CommunicationChannel("test_channel")

@pytest.fixture
def communication_manager():
    return CommunicationManager()

# Channel Tests
@pytest.mark.asyncio
async def test_channel_subscribe(communication_channel):
    await communication_channel.subscribe("test_agent")
    assert "test_agent" in communication_channel.subscribers
    assert communication_channel.get_metrics()["subscribers"] == 1

@pytest.mark.asyncio
async def test_channel_unsubscribe(communication_channel):
    await communication_channel.subscribe("test_agent")
    await communication_channel.unsubscribe("test_agent")
    assert "test_agent" not in communication_channel.subscribers
    assert communication_channel.get_metrics()["subscribers"] == 0

@pytest.mark.asyncio
async def test_channel_publish(communication_channel, test_message, test_routing_info):
    await communication_channel.subscribe("test_receiver")
    await communication_channel.publish(test_message, test_routing_info)
    
    # Check metrics
    metrics = communication_channel.get_metrics()
    assert metrics["messages_processed"] == 1
    assert metrics["subscribers"] == 1
    assert metrics["errors"] == 0

@pytest.mark.asyncio
async def test_channel_broadcast(communication_channel, test_message):
    # Subscribe multiple agents
    await communication_channel.subscribe("agent1")
    await communication_channel.subscribe("agent2")
    await communication_channel.subscribe("agent3")
    
    # Create broadcast routing info
    broadcast_routing = RoutingInfo(
        source="test_sender",
        destination="broadcast",
        route_type=CommunicationType.BROADCAST,
        protocol=CommunicationProtocol.ASYNC
    )
    
    # Publish broadcast message
    await communication_channel.publish(test_message, broadcast_routing)
    
    # Check metrics
    metrics = communication_channel.get_metrics()
    assert metrics["messages_processed"] == 1
    assert metrics["subscribers"] == 3
    assert metrics["errors"] == 0

# Manager Tests
@pytest.mark.asyncio
async def test_manager_create_channel(communication_manager):
    channel = communication_manager.create_channel("test_channel")
    assert isinstance(channel, CommunicationChannel)
    assert "test_channel" in communication_manager.channels
    assert communication_manager.metrics["channels"] == 1

@pytest.mark.asyncio
async def test_manager_subscribe(communication_manager):
    await communication_manager.subscribe("test_agent", "test_channel")
    assert "test_channel" in communication_manager.channels
    assert "test_agent" in communication_manager.channels["test_channel"].subscribers
    assert communication_manager.routing_table["test_agent"] == "test_channel"

@pytest.mark.asyncio
async def test_manager_send_message(communication_manager, test_message, test_routing_info):
    # Setup channel and subscriber
    await communication_manager.subscribe("test_receiver", "test_channel")
    
    # Send message
    await communication_manager.send_message(test_message, test_routing_info)
    
    # Check metrics
    metrics = communication_manager.get_metrics()
    assert metrics["total_messages"] == 1
    assert metrics["active_routes"] == 1
    assert metrics["errors"] == 0

@pytest.mark.asyncio
async def test_manager_broadcast_message(communication_manager, test_message):
    # Setup channel and subscribers
    channel_id = "broadcast_channel"
    await communication_manager.subscribe("agent1", channel_id)
    await communication_manager.subscribe("agent2", channel_id)
    
    # Broadcast message
    await communication_manager.broadcast_message(test_message, channel_id)
    
    # Check metrics
    metrics = communication_manager.get_metrics()
    assert metrics["total_messages"] == 1
    assert metrics["total_subscribers"] == 2
    assert metrics["errors"] == 0

@pytest.mark.asyncio
async def test_manager_routing(communication_manager):
    # Test direct routing
    channel_id = communication_manager._get_channel_for_destination("test_agent")
    assert channel_id is None  # No route yet
    
    # Add route
    await communication_manager.subscribe("test_agent", "test_channel")
    channel_id = communication_manager._get_channel_for_destination("test_agent")
    assert channel_id == "test_channel"
    
    # Test role-based routing
    channel_id = communication_manager._get_channel_for_destination("role:executor")
    assert channel_id == "role:executor"

@pytest.mark.asyncio
async def test_error_handling(communication_manager, test_message):
    # Test sending to non-existent destination
    with pytest.raises(ValueError):
        await communication_manager.send_message(
            test_message,
            RoutingInfo(
                source="test_sender",
                destination="non_existent",
                route_type=CommunicationType.DIRECT,
                protocol=CommunicationProtocol.ASYNC
            )
        )
    
    # Check error metrics
    metrics = communication_manager.get_metrics()
    assert metrics["errors"] == 1
