from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ActionRequest:
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class AgentStep:
    reply: str | None
    pending_actions: tuple[ActionRequest, ...] = ()
