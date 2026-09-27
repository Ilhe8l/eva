"""Ports the application layer depends on."""

from typing import Protocol

from eva.domain.models import ActionRequest, AgentStep


class AgentPort(Protocol):
    def ask(self, message: str) -> AgentStep: ...

    def resume(self, decisions: list[dict[str, str]]) -> AgentStep: ...


class ApprovalPort(Protocol):
    def approve(self, action: ActionRequest) -> bool: ...
