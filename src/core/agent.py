"""
Core agent implementation for IbuthoAGI.
"""
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime
import uuid
import json
from enum import Enum
import logging
from pydantic import BaseModel

from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory

class AgentRole(str, Enum):
    """Predefined agent roles."""
    COORDINATOR = "coordinator"
    RESEARCHER = "researcher"
    PLANNER = "planner"
    EXECUTOR = "executor"
    CRITIC = "critic"
    INNOVATOR = "innovator"

class AgentCapability(str, Enum):
    """Agent capabilities."""
    TASK_DECOMPOSITION = "task_decomposition"
    INFORMATION_GATHERING = "information_gathering"
    SOLUTION_SYNTHESIS = "solution_synthesis"
    CODE_GENERATION = "code_generation"
    EVALUATION = "evaluation"
    INNOVATION = "innovation"

@dataclass
class AgentState:
    """Agent's current state."""
    busy: bool = False
    current_task: Optional[str] = None
    last_action: Optional[str] = None
    last_action_time: Optional[datetime] = None
    context: Dict[str, Any] = field(default_factory=dict)
    performance_metrics: Dict[str, float] = field(default_factory=dict)

class Message(BaseModel):
    """Message format for agent communication."""
    id: str
    sender: str
    receiver: str
    content: Dict[str, Any]
    type: str
    timestamp: datetime
    priority: int = 1
    requires_response: bool = False
    context: Dict[str, Any] = {}

class Agent:
    """Base agent class with core functionality."""
    
    def __init__(
        self,
        agent_id: str,
        role: AgentRole,
        capabilities: List[AgentCapability],
        llm: Optional[OpenAI] = None,
        memory_size: int = 1000
    ):
        self.agent_id = agent_id
        self.role = role
        self.capabilities = capabilities
        self.state = AgentState()
        self.llm = llm or OpenAI()
        
        # Initialize memory
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            max_tokens=memory_size
        )
        
        # Message handlers
        self.message_handlers: Dict[str, Callable] = {}
        self._register_default_handlers()
        
        # Performance tracking
        self.performance_metrics = {
            "tasks_completed": 0,
            "success_rate": 1.0,
            "average_response_time": 0.0,
            "error_rate": 0.0
        }
        
        # Initialize logger
        self.logger = logging.getLogger(f"agent.{agent_id}")
    
    def _register_default_handlers(self) -> None:
        """Register default message handlers."""
        self.message_handlers.update({
            "task_request": self._handle_task_request,
            "information_request": self._handle_information_request,
            "status_update": self._handle_status_update,
            "error_report": self._handle_error_report
        })
    
    async def process_message(self, message: Message) -> Optional[Message]:
        """Process incoming message."""
        try:
            handler = self.message_handlers.get(message.type)
            if handler:
                start_time = datetime.now()
                response = await handler(message)
                self._update_metrics(start_time)
                return response
            else:
                self.logger.warning(f"No handler for message type: {message.type}")
                return None
        except Exception as e:
            self.logger.error(f"Error processing message: {str(e)}")
            self._update_error_rate()
            return self._create_error_message(str(e), message.sender)
    
    async def _handle_task_request(self, message: Message) -> Message:
        """Handle task request messages."""
        if self.state.busy:
            return self._create_busy_message(message.sender)
        
        self.state.busy = True
        self.state.current_task = message.content.get("task_id")
        
        # Process task based on role
        result = await self._process_task_by_role(message.content)
        
        self.state.busy = False
        self.state.current_task = None
        
        return Message(
            id=str(uuid.uuid4()),
            sender=self.agent_id,
            receiver=message.sender,
            content={"result": result},
            type="task_response",
            timestamp=datetime.now(),
            context=message.context
        )
    
    async def _handle_information_request(self, message: Message) -> Message:
        """Handle information request messages."""
        info_type = message.content.get("info_type")
        if info_type == "status":
            return self._create_status_message(message.sender)
        elif info_type == "capabilities":
            return self._create_capabilities_message(message.sender)
        else:
            return self._create_error_message(
                f"Unknown information type: {info_type}",
                message.sender
            )
    
    async def _handle_status_update(self, message: Message) -> None:
        """Handle status update messages."""
        self.state.context.update(message.content.get("status", {}))
    
    async def _handle_error_report(self, message: Message) -> Message:
        """Handle error report messages."""
        error = message.content.get("error")
        self.logger.error(f"Error reported: {error}")
        self._update_error_rate()
        return self._create_acknowledgment_message(message.sender)
    
    async def _process_task_by_role(self, task_content: Dict[str, Any]) -> Dict[str, Any]:
        """Process task based on agent's role."""
        task_type = task_content.get("type", "")
        
        if self.role == AgentRole.COORDINATOR:
            return await self._coordinate_task(task_content)
        elif self.role == AgentRole.RESEARCHER:
            return await self._research_task(task_content)
        elif self.role == AgentRole.PLANNER:
            return await self._plan_task(task_content)
        elif self.role == AgentRole.EXECUTOR:
            return await self._execute_task(task_content)
        elif self.role == AgentRole.CRITIC:
            return await self._evaluate_task(task_content)
        elif self.role == AgentRole.INNOVATOR:
            return await self._innovate_task(task_content)
        else:
            raise ValueError(f"Unknown role: {self.role}")
    
    async def _coordinate_task(self, task_content: Dict[str, Any]) -> Dict[str, Any]:
        """Coordinate task execution."""
        # Implement coordination logic
        pass
    
    async def _research_task(self, task_content: Dict[str, Any]) -> Dict[str, Any]:
        """Research task implementation."""
        # Implement research logic
        pass
    
    async def _plan_task(self, task_content: Dict[str, Any]) -> Dict[str, Any]:
        """Plan task implementation."""
        # Implement planning logic
        pass
    
    async def _execute_task(self, task_content: Dict[str, Any]) -> Dict[str, Any]:
        """Execute task implementation."""
        # Implement execution logic
        pass
    
    async def _evaluate_task(self, task_content: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate task implementation."""
        # Implement evaluation logic
        pass
    
    async def _innovate_task(self, task_content: Dict[str, Any]) -> Dict[str, Any]:
        """Innovate task implementation."""
        # Implement innovation logic
        pass
    
    def _create_message(
        self,
        receiver: str,
        content: Dict[str, Any],
        msg_type: str,
        priority: int = 1,
        requires_response: bool = False
    ) -> Message:
        """Create a new message."""
        return Message(
            id=str(uuid.uuid4()),
            sender=self.agent_id,
            receiver=receiver,
            content=content,
            type=msg_type,
            timestamp=datetime.now(),
            priority=priority,
            requires_response=requires_response
        )
    
    def _create_error_message(self, error: str, receiver: str) -> Message:
        """Create error message."""
        return self._create_message(
            receiver,
            {"error": error},
            "error_response",
            priority=3
        )
    
    def _create_busy_message(self, receiver: str) -> Message:
        """Create busy status message."""
        return self._create_message(
            receiver,
            {
                "status": "busy",
                "current_task": self.state.current_task
            },
            "status_response"
        )
    
    def _create_status_message(self, receiver: str) -> Message:
        """Create status message."""
        return self._create_message(
            receiver,
            {
                "status": "available" if not self.state.busy else "busy",
                "metrics": self.performance_metrics
            },
            "status_response"
        )
    
    def _create_capabilities_message(self, receiver: str) -> Message:
        """Create capabilities message."""
        return self._create_message(
            receiver,
            {
                "role": self.role,
                "capabilities": self.capabilities
            },
            "capabilities_response"
        )
    
    def _create_acknowledgment_message(self, receiver: str) -> Message:
        """Create acknowledgment message."""
        return self._create_message(
            receiver,
            {"status": "acknowledged"},
            "acknowledgment"
        )
    
    def _update_metrics(self, start_time: datetime) -> None:
        """Update performance metrics."""
        response_time = (datetime.now() - start_time).total_seconds()
        self.performance_metrics["tasks_completed"] += 1
        
        # Update average response time
        prev_avg = self.performance_metrics["average_response_time"]
        n = self.performance_metrics["tasks_completed"]
        new_avg = ((n - 1) * prev_avg + response_time) / n
        self.performance_metrics["average_response_time"] = new_avg
    
    def _update_error_rate(self) -> None:
        """Update error rate metric."""
        n = self.performance_metrics["tasks_completed"]
        if n > 0:
            self.performance_metrics["error_rate"] = (
                self.performance_metrics.get("error_rate", 0) * (n - 1) + 1
            ) / n
            self.performance_metrics["success_rate"] = 1 - self.performance_metrics["error_rate"]
