"""
Advanced inter-agent communication protocols for IbuthoAGI.
"""
from typing import Dict, List, Optional, Any, Set, Tuple
from datetime import datetime
import uuid
import asyncio
import logging
from enum import Enum
from dataclasses import dataclass, field

from .message import Message, MessagePriority

class CommunicationType(str, Enum):
    """Types of communication between agents."""
    DIRECT = "direct"
    BROADCAST = "broadcast"
    PUBLISH = "publish"
    REQUEST = "request"
    RESPONSE = "response"
    EVENT = "event"

class CommunicationProtocol(str, Enum):
    """Communication protocols for message exchange."""
    SYNC = "sync"
    ASYNC = "async"
    STREAM = "stream"
    BATCH = "batch"

@dataclass
class RoutingInfo:
    """Information for message routing."""
    source: str
    destination: str
    route_type: CommunicationType = CommunicationType.DIRECT
    protocol: CommunicationProtocol = CommunicationProtocol.ASYNC
    ttl: int = 10  # Time-to-live in hops
    path: List[str] = field(default_factory=list)

class CommunicationChannel:
    """Represents a communication channel between agents."""
    
    def __init__(self, channel_id: str):
        self.channel_id = channel_id
        self.subscribers: Set[str] = set()
        self.message_queue: asyncio.Queue[Tuple[Message, RoutingInfo]] = asyncio.Queue()
        self.active = True
        self.logger = logging.getLogger(f"channel.{channel_id}")
        self.metrics = {
            "messages_processed": 0,
            "subscribers": 0,
            "errors": 0
        }
    
    async def subscribe(self, agent_id: str) -> None:
        """Subscribe an agent to this channel."""
        self.subscribers.add(agent_id)
        self.metrics["subscribers"] = len(self.subscribers)
        self.logger.info(f"Agent {agent_id} subscribed to channel {self.channel_id}")
    
    async def unsubscribe(self, agent_id: str) -> None:
        """Unsubscribe an agent from this channel."""
        self.subscribers.discard(agent_id)
        self.metrics["subscribers"] = len(self.subscribers)
        self.logger.info(f"Agent {agent_id} unsubscribed from channel {self.channel_id}")
    
    async def publish(self, message: Message, routing_info: RoutingInfo) -> None:
        """Publish a message to the channel."""
        if self.active:
            await self.message_queue.put((message, routing_info))
            self.metrics["messages_processed"] += 1
            self.logger.debug(f"Message {message.id} published to channel {self.channel_id}")
    
    async def process_messages(self) -> None:
        """Process messages in the channel."""
        while self.active:
            try:
                message, routing_info = await self.message_queue.get()
                
                # Process message based on routing type
                if routing_info.route_type == CommunicationType.BROADCAST:
                    # Send to all subscribers except source
                    for subscriber in self.subscribers:
                        if subscriber != routing_info.source:
                            await self._deliver_message(message, subscriber)
                else:
                    # Direct message to specific destination
                    if routing_info.destination in self.subscribers:
                        await self._deliver_message(message, routing_info.destination)
                    else:
                        self.logger.warning(
                            f"Destination {routing_info.destination} not found in channel {self.channel_id}"
                        )
                
            except Exception as e:
                self.metrics["errors"] += 1
                self.logger.error(f"Error processing message in channel {self.channel_id}: {str(e)}")
    
    async def _deliver_message(self, message: Message, destination: str) -> None:
        """Deliver a message to a specific destination."""
        # In a real implementation, this would handle the actual message delivery
        # For now, we just log it
        self.logger.debug(f"Delivering message {message.id} to {destination}")
    
    def get_metrics(self) -> Dict[str, int]:
        """Get channel metrics."""
        return self.metrics.copy()

class CommunicationManager:
    """Manages communication channels and message routing."""
    
    def __init__(self):
        self.channels: Dict[str, CommunicationChannel] = {}
        self.routing_table: Dict[str, str] = {}  # agent_id -> channel_id
        self.logger = logging.getLogger("communication_manager")
        self.metrics = {
            "channels": 0,
            "total_messages": 0,
            "active_routes": 0,
            "errors": 0
        }
    
    def create_channel(self, channel_id: str) -> CommunicationChannel:
        """Create a new communication channel."""
        if channel_id not in self.channels:
            channel = CommunicationChannel(channel_id)
            self.channels[channel_id] = channel
            self.metrics["channels"] = len(self.channels)
            self.logger.info(f"Created channel {channel_id}")
            
            # Start processing messages in the channel
            asyncio.create_task(channel.process_messages())
            
            return channel
        return self.channels[channel_id]
    
    async def subscribe(self, agent_id: str, channel_id: str) -> None:
        """Subscribe an agent to a channel."""
        if channel_id not in self.channels:
            self.create_channel(channel_id)
        
        channel = self.channels[channel_id]
        await channel.subscribe(agent_id)
        self.routing_table[agent_id] = channel_id
        self.metrics["active_routes"] = len(self.routing_table)
    
    async def unsubscribe(self, agent_id: str, channel_id: str) -> None:
        """Unsubscribe an agent from a channel."""
        if channel_id in self.channels:
            channel = self.channels[channel_id]
            await channel.unsubscribe(agent_id)
            self.routing_table.pop(agent_id, None)
            self.metrics["active_routes"] = len(self.routing_table)
    
    async def send_message(
        self,
        message: Message,
        routing_info: Optional[RoutingInfo] = None
    ) -> None:
        """Send a message through the appropriate channel."""
        try:
            if routing_info is None:
                routing_info = RoutingInfo(
                    source=message.sender,
                    destination=message.receiver
                )
            
            # Find the appropriate channel
            channel_id = self._get_channel_for_destination(routing_info.destination)
            if channel_id and channel_id in self.channels:
                channel = self.channels[channel_id]
                await channel.publish(message, routing_info)
                self.metrics["total_messages"] += 1
            else:
                raise ValueError(f"No route found for destination {routing_info.destination}")
            
        except Exception as e:
            self.metrics["errors"] += 1
            self.logger.error(f"Error sending message: {str(e)}")
            raise
    
    async def broadcast_message(
        self,
        message: Message,
        channel_id: str
    ) -> None:
        """Broadcast a message to all subscribers of a channel."""
        try:
            if channel_id not in self.channels:
                self.create_channel(channel_id)
            
            channel = self.channels[channel_id]
            routing_info = RoutingInfo(
                source=message.sender,
                destination="broadcast",
                route_type=CommunicationType.BROADCAST
            )
            
            await channel.publish(message, routing_info)
            self.metrics["total_messages"] += 1
            
        except Exception as e:
            self.metrics["errors"] += 1
            self.logger.error(f"Error broadcasting message: {str(e)}")
            raise
    
    def _get_channel_for_destination(self, destination: str) -> Optional[str]:
        """Get the channel ID for a destination."""
        # First check if destination is directly mapped
        channel_id = self.routing_table.get(destination)
        if channel_id:
            return channel_id
        
        # Check if destination is a role-based channel
        if destination.startswith("role:"):
            return destination
        
        return None
    
    def get_metrics(self) -> Dict[str, int]:
        """Get communication manager metrics."""
        metrics = self.metrics.copy()
        
        # Add aggregated channel metrics
        channel_metrics = {
            "total_subscribers": 0,
            "total_processed": 0,
            "channel_errors": 0
        }
        
        for channel in self.channels.values():
            channel_stats = channel.get_metrics()
            channel_metrics["total_subscribers"] += channel_stats["subscribers"]
            channel_metrics["total_processed"] += channel_stats["messages_processed"]
            channel_metrics["channel_errors"] += channel_stats["errors"]
        
        metrics.update(channel_metrics)
        return metrics
