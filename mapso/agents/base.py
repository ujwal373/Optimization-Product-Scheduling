"""
Base agent class for MAPSO

Defines the abstract interface that all agents must implement.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from datetime import datetime
from dataclasses import dataclass

from mapso.core.models import Schedule
from mapso.utils.logging_config import get_logger


@dataclass
class AgentMessage:
    """
    Message passed between agents

    Attributes:
        sender: Agent ID of sender
        receiver: Agent ID of receiver
        message_type: Type of message (e.g., "validation", "suggestion", "alert")
        payload: Message data
        timestamp: When message was sent
    """

    sender: str
    receiver: str
    message_type: str
    payload: Dict[str, Any]
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


class AbstractAgent(ABC):
    """
    Base class for all scheduling agents

    Each agent owns a specific constraint or objective and operates independently.
    Agents communicate via messages to coordinate and validate schedules.

    Key Principles:
    - Single Responsibility: Each agent focuses on one aspect
    - Autonomy: Agents make independent decisions
    - Communication: Agents share information via messages
    - Validation: Agents validate schedules against their constraints
    """

    def __init__(self, agent_id: str, config: Optional[Dict[str, Any]] = None):
        """
        Initialize agent

        Args:
            agent_id: Unique identifier for this agent
            config: Configuration dictionary
        """
        self.agent_id = agent_id
        self.config = config or {}
        self.inbox: List[AgentMessage] = []
        self.outbox: List[AgentMessage] = []
        self.logger = get_logger(f"agent.{agent_id}")
        self.enabled = True

        self.logger.info(f"Agent {agent_id} initialized")

    @abstractmethod
    def process(self, schedule: Schedule) -> Schedule:
        """
        Process a schedule and potentially modify it

        This is the main processing logic for the agent. The agent may:
        - Validate constraints
        - Suggest improvements
        - Modify job assignments or timings
        - Add metadata/annotations

        Args:
            schedule: Input schedule

        Returns:
            Processed schedule (may be modified)
        """
        pass

    @abstractmethod
    def validate(self, schedule: Schedule) -> tuple[bool, List[str]]:
        """
        Validate schedule against agent's constraints

        Args:
            schedule: Schedule to validate

        Returns:
            Tuple of (is_valid, list_of_violations)
        """
        pass

    @abstractmethod
    def score(self, schedule: Schedule) -> float:
        """
        Score schedule from this agent's perspective

        Lower scores are better (minimization).

        Args:
            schedule: Schedule to score

        Returns:
            Score value (lower is better)
        """
        pass

    def send_message(self, receiver: str, message_type: str, payload: Dict[str, Any]) -> None:
        """
        Send message to another agent

        Args:
            receiver: Agent ID of receiver
            message_type: Type of message
            payload: Message data
        """
        message = AgentMessage(
            sender=self.agent_id, receiver=receiver, message_type=message_type, payload=payload
        )
        self.outbox.append(message)
        self.logger.debug(f"Sent {message_type} message to {receiver}")

    def receive_message(self, message: AgentMessage) -> None:
        """
        Receive message from another agent

        Args:
            message: Message object
        """
        self.inbox.append(message)
        self.logger.debug(f"Received {message.message_type} message from {message.sender}")

    def get_unread_messages(self) -> List[AgentMessage]:
        """Get all unread messages and clear inbox"""
        messages = self.inbox.copy()
        self.inbox.clear()
        return messages

    def get_sent_messages(self) -> List[AgentMessage]:
        """Get all sent messages and clear outbox"""
        messages = self.outbox.copy()
        self.outbox.clear()
        return messages

    def enable(self) -> None:
        """Enable this agent"""
        self.enabled = True
        self.logger.info(f"Agent {self.agent_id} enabled")

    def disable(self) -> None:
        """Disable this agent"""
        self.enabled = False
        self.logger.info(f"Agent {self.agent_id} disabled")

    def is_enabled(self) -> bool:
        """Check if agent is enabled"""
        return self.enabled

    def get_info(self) -> Dict[str, Any]:
        """
        Get agent information

        Returns:
            Dictionary with agent metadata
        """
        return {
            "agent_id": self.agent_id,
            "agent_type": self.__class__.__name__,
            "enabled": self.enabled,
            "config": self.config,
            "inbox_size": len(self.inbox),
            "outbox_size": len(self.outbox),
        }

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(agent_id='{self.agent_id}', enabled={self.enabled})"
