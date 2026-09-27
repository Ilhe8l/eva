"""Values crossing Eva's application boundaries."""

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ActionRequest:
    """A tool call waiting for the user's approval."""

    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class AgentStep:
    """Where a turn stopped: a final reply, or actions awaiting approval."""

    reply: str | None
    pending_actions: tuple[ActionRequest, ...] = ()


@dataclass(frozen=True)
class Speech:
    """A sentence Eva chose to say aloud while working."""

    text: str


@dataclass(frozen=True)
class ToolUse:
    """A tool Eva started using, reported as progress."""

    name: str
    arguments: dict[str, Any]


AgentEvent = Speech | ToolUse
