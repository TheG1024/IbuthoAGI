"""
Agent manager for orchestrating multiple agents.
"""
from typing import Dict, List, Optional, Any
import asyncio
from datetime import datetime
import logging
from uuid import uuid4

from .agent import Agent, AgentRole, AgentCapability, Message

class AgentManager:
    """Manages a group of agents and their interactions."""
    
    def __init__(self):
        self.agents: Dict[str, Agent] = {}
        self.message_queue: asyncio.Queue[Message] = asyncio.Queue()
        self.logger = logging.getLogger("agent_manager")
        self.active_tasks: Dict[str, Dict[str, Any]] = {}
    
    def register_agent(self, agent: Agent) -> None:
        """Register a new agent."""
        if agent.agent_id in self.agents:
            raise ValueError(f"Agent {agent.agent_id} already registered")
        self.agents[agent.agent_id] = agent
        self.logger.info(f"Registered agent: {agent.agent_id} ({agent.role})")
    
    def unregister_agent(self, agent_id: str) -> None:
        """Unregister an agent."""
        if agent_id in self.agents:
            del self.agents[agent_id]
            self.logger.info(f"Unregistered agent: {agent_id}")
    
    async def start(self) -> None:
        """Start the agent manager."""
        self.logger.info("Starting agent manager")
        await self._process_message_queue()
    
    async def stop(self) -> None:
        """Stop the agent manager."""
        self.logger.info("Stopping agent manager")
        # Implement cleanup logic
    
    async def submit_task(
        self,
        task: Dict[str, Any],
        priority: int = 1
    ) -> str:
        """Submit a new task to the system."""
        task_id = str(uuid4())
        coordinator = self._get_coordinator()
        
        if not coordinator:
            raise RuntimeError("No coordinator agent available")
        
        message = Message(
            id=str(uuid4()),
            sender="system",
            receiver=coordinator.agent_id,
            content={
                "task_id": task_id,
                "task": task
            },
            type="task_request",
            timestamp=datetime.now(),
            priority=priority
        )
        
        self.active_tasks[task_id] = {
            "status": "submitted",
            "task": task,
            "coordinator": coordinator.agent_id,
            "start_time": datetime.now(),
            "priority": priority
        }
        
        await self.message_queue.put(message)
        return task_id
    
    async def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """Get the status of a task."""
        if task_id not in self.active_tasks:
            raise ValueError(f"Task {task_id} not found")
        return self.active_tasks[task_id]
    
    def get_available_agents(
        self,
        role: Optional[AgentRole] = None,
        capability: Optional[AgentCapability] = None
    ) -> List[Agent]:
        """Get available agents matching criteria."""
        available_agents = []
        
        for agent in self.agents.values():
            if agent.state.busy:
                continue
            
            if role and agent.role != role:
                continue
            
            if capability and capability not in agent.capabilities:
                continue
            
            available_agents.append(agent)
        
        return available_agents
    
    async def broadcast_message(
        self,
        content: Dict[str, Any],
        msg_type: str,
        priority: int = 1,
        role: Optional[AgentRole] = None
    ) -> None:
        """Broadcast a message to all agents or agents with specific role."""
        for agent in self.agents.values():
            if role and agent.role != role:
                continue
            
            message = Message(
                id=str(uuid4()),
                sender="system",
                receiver=agent.agent_id,
                content=content,
                type=msg_type,
                timestamp=datetime.now(),
                priority=priority
            )
            
            await self.message_queue.put(message)
    
    async def _process_message_queue(self) -> None:
        """Process messages in the queue."""
        while True:
            try:
                message = await self.message_queue.get()
                await self._route_message(message)
                self.message_queue.task_done()
            except Exception as e:
                self.logger.error(f"Error processing message: {str(e)}")
    
    async def _route_message(self, message: Message) -> None:
        """Route message to appropriate agent."""
        receiver = self.agents.get(message.receiver)
        if not receiver:
            self.logger.warning(f"Unknown receiver: {message.receiver}")
            return
        
        try:
            response = await receiver.process_message(message)
            if response:
                await self.message_queue.put(response)
        except Exception as e:
            self.logger.error(f"Error routing message: {str(e)}")
            # Create error response
            error_message = Message(
                id=str(uuid4()),
                sender="system",
                receiver=message.sender,
                content={"error": str(e)},
                type="error_response",
                timestamp=datetime.now(),
                priority=3
            )
            await self.message_queue.put(error_message)
    
    def _get_coordinator(self) -> Optional[Agent]:
        """Get an available coordinator agent."""
        for agent in self.agents.values():
            if (
                agent.role == AgentRole.COORDINATOR
                and not agent.state.busy
            ):
                return agent
        return None
    
    async def _update_task_status(
        self,
        task_id: str,
        status: str,
        result: Optional[Dict[str, Any]] = None
    ) -> None:
        """Update task status."""
        if task_id in self.active_tasks:
            self.active_tasks[task_id].update({
                "status": status,
                "last_updated": datetime.now(),
                "result": result
            })
            
            if status in ["completed", "failed"]:
                self.active_tasks[task_id]["end_time"] = datetime.now()
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """Get system-wide performance metrics."""
        metrics = {
            "active_agents": len(self.agents),
            "active_tasks": len(self.active_tasks),
            "message_queue_size": self.message_queue.qsize(),
            "agent_metrics": {}
        }
        
        for agent_id, agent in self.agents.items():
            metrics["agent_metrics"][agent_id] = {
                "role": agent.role,
                "busy": agent.state.busy,
                "metrics": agent.performance_metrics
            }
        
        return metrics
