"""Values crossing Eva's application boundaries.

`source` names the background task that produced a value; None is the main
conversation.
"""

from dataclasses import dataclass
from typing import Any

MOODS = (
    "happy",
    "amused",
    "proud",
    "sad",
    "worried",
    "surprised",
    "curious",
    "confused",
    "thinking",
    "focused",
    "sleepy",
    "wink",
)


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
    """Words Eva chose to tell the user: aloud, or as a status line when speech is off."""

    text: str
    source: str | None = None
    aloud: bool = True
    mood: str | None = None


@dataclass(frozen=True)
class ToolUse:
    """A tool Eva started using, reported as progress."""

    name: str
    arguments: dict[str, Any]
    source: str | None = None
    call_id: str = ""


@dataclass(frozen=True)
class ToolResult:
    """How a tool call ended: `summary` is the first line of its output."""

    name: str
    ok: bool
    summary: str = ""
    source: str | None = None
    call_id: str = ""


@dataclass(frozen=True)
class Reaction:
    """An expression Eva chose to show on her face for a moment (one of MOODS)."""

    mood: str
    source: str | None = None


@dataclass(frozen=True)
class TextDelta:
    """A piece of Eva's written reply, as the model generates it."""

    text: str
    source: str | None = None


AgentEvent = Speech | ToolUse | ToolResult | TextDelta | Reaction


class TurnCancelled(Exception):
    """The user or Eva stopped a turn before it finished."""


class StepLimitReached(Exception):
    """A turn used all the graph steps it was allowed and stopped unfinished."""
