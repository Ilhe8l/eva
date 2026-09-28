"""Values crossing Eva's application boundaries.

`source` names the background task that produced a value; None is the main
conversation.
"""

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ActionRequest:
    """A tool call waiting for the user's approval."""

    name: str
    arguments: dict[str, Any]
    source: str | None = None


@dataclass(frozen=True)
class AgentStep:
    """Where a turn stopped: a final reply, or actions awaiting approval."""

    reply: str | None
    pending_actions: tuple[ActionRequest, ...] = ()


@dataclass(frozen=True)
class Speech:
    """Words Eva chose to say aloud while working."""

    text: str
    source: str | None = None


@dataclass(frozen=True)
class ToolUse:
    """A tool Eva started using, reported as progress."""

    name: str
    arguments: dict[str, Any]
    source: str | None = None


@dataclass(frozen=True)
class TextDelta:
    """A piece of Eva's written reply, as the model generates it."""

    text: str
    source: str | None = None


AgentEvent = Speech | ToolUse | TextDelta


class TurnCancelled(Exception):
    """The user or Eva stopped a turn before it finished."""
