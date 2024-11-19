"""
Agent management system for IbuthoAGI.
"""
from typing import Dict, List, Optional, Any, Union
import asyncio
import logging
from datetime import datetime
import uuid

from .agent import Agent, AgentRole, AgentCapability, Message
from .communication import (
    CommunicationType,
    CommunicationProtocol,
    MessagePriority,
    RoutingInfo,
    CommunicationManager
)

class AgentManager:
    """Manages agent lifecycle and communication."""
    
    def __init__(self):
        self.agents: Dict[str, Agent] = {}
        self.role_groups: Dict[AgentRole, List[str]] = {role: [] for role in AgentRole}
        self.comm_manager = CommunicationManager()
        self.logger = logging.getLogger("agent_manager")
        
        # Start communication processing
        self.processing = False
        self.process_task: Optional[asyncio.Task] = None
    
    async def start(self) -> None:
        """Start the agent manager."""
        if not self.processing:
            self.processing = True
            self.process_task = asyncio.create_task(self.comm_manager.start())
            self.logger.info("Agent manager started")
    
    async def stop(self) -> None:
        """Stop the agent manager."""
        if self.processing:
            self.processing = False
            if self.process_task:
                self.process_task.cancel()
                try:
                    await self.process_task
                except asyncio.CancelledError:
                    pass
            await self.comm_manager.stop()
            self.logger.info("Agent manager stopped")
    
    def register_agent(
        self,
        agent_id: str,
        role: AgentRole,
        capabilities: List[AgentCapability]
    ) -> Agent:
        """Register a new agent."""
        if agent_id in self.agents:
            raise ValueError(f"Agent {agent_id} already registered")
        
        agent = Agent(
            agent_id=agent_id,
            role=role,
            capabilities=capabilities,
            comm_manager=self.comm_manager
        )
        
        self.agents[agent_id] = agent
        self.role_groups[role].append(agent_id)
        
        # Subscribe to role-specific channel
        asyncio.create_task(
            agent.subscribe_to_channel(f"role_{role}")
        )
        
        self.logger.info(f"Registered agent {agent_id} with role {role}")
        return agent
    
    def unregister_agent(self, agent_id: str) -> None:
        """Unregister an agent."""
        if agent_id in self.agents:
            agent = self.agents[agent_id]
            
            # Unsubscribe from role-specific channel
            asyncio.create_task(
                agent.unsubscribe_from_channel(f"role_{agent.role}")
            )
            
            # Remove from role group
            self.role_groups[agent.role].remove(agent_id)
            
            # Remove agent
            del self.agents[agent_id]
            
            self.logger.info(f"Unregistered agent {agent_id}")
    
    async def submit_task(
        self,
        task_content: Dict[str, Any],
        target_role: Optional[AgentRole] = None,
        target_agent: Optional[str] = None,
        priority: int = MessagePriority.MEDIUM
    ) -> str:
        """Submit a task to agents."""
        task_id = str(uuid.uuid4())
        task_content["task_id"] = task_id
        
        if target_agent:
            # Direct task to specific agent
            if target_agent not in self.agents:
                raise ValueError(f"Agent {target_agent} not found")
            
            await self.agents[target_agent].send_message(
                target_agent,
                task_content,
                "task_request",
                priority=priority,
                requires_response=True
            )
            
        elif target_role:
            # Broadcast task to role
            if not self.role_groups[target_role]:
                raise ValueError(f"No agents found for role {target_role}")
            
            await self._broadcast_to_role(
                target_role,
                task_content,
                "task_request",
                priority=priority
            )
            
        else:
            # Find suitable agent based on task requirements
            coordinator = self._get_coordinator()
            if not coordinator:
                raise ValueError("No coordinator agent available")
            
            await coordinator.send_message(
                coordinator.agent_id,
                task_content,
                "task_request",
                priority=priority,
                requires_response=True
            )
        
        self.logger.info(f"Submitted task {task_id}")
        return task_id
    
    async def broadcast_message(
        self,
        content: Dict[str, Any],
        msg_type: str,
        target_role: Optional[AgentRole] = None,
        priority: int = MessagePriority.MEDIUM
    ) -> None:
        """Broadcast a message to all agents or agents with a specific role."""
        if target_role:
            await self._broadcast_to_role(target_role, content, msg_type, priority)
        else:
            for agent in self.agents.values():
                await agent.send_message(
                    "all",
                    content,
                    msg_type,
                    route_type=CommunicationType.BROADCAST,
                    priority=priority
                )
    
    async def get_agent_status(
        self,
        agent_id: Optional[str] = None,
        role: Optional[AgentRole] = None
    ) -> Dict[str, Any]:
        """Get status of agents."""
        status = {}
        
        if agent_id:
            if agent_id not in self.agents:
                raise ValueError(f"Agent {agent_id} not found")
            agent = self.agents[agent_id]
            status[agent_id] = {
                "role": agent.role,
                "capabilities": agent.capabilities,
                "state": agent.state.__dict__,
                "metrics": agent.performance_metrics
            }
            
        elif role:
            for agent_id in self.role_groups[role]:
                agent = self.agents[agent_id]
                status[agent_id] = {
                    "role": agent.role,
                    "capabilities": agent.capabilities,
                    "state": agent.state.__dict__,
                    "metrics": agent.performance_metrics
                }
                
        else:
            for agent_id, agent in self.agents.items():
                status[agent_id] = {
                    "role": agent.role,
                    "capabilities": agent.capabilities,
                    "state": agent.state.__dict__,
                    "metrics": agent.performance_metrics
                }
        
        return status
    
    def _get_coordinator(self) -> Optional[Agent]:
        """Get the first available coordinator agent."""
        coordinator_ids = self.role_groups[AgentRole.COORDINATOR]
        return self.agents[coordinator_ids[0]] if coordinator_ids else None
    
    async def _broadcast_to_role(
        self,
        role: AgentRole,
        content: Dict[str, Any],
        msg_type: str,
        priority: int = MessagePriority.MEDIUM
    ) -> None:
        """Broadcast a message to all agents with a specific role."""
        if not self.role_groups[role]:
            raise ValueError(f"No agents found for role {role}")
        
        for agent_id in self.role_groups[role]:
            await self.agents[agent_id].send_message(
                agent_id,
                content,
                msg_type,
                route_type=CommunicationType.BROADCAST,
                priority=priority
            )
