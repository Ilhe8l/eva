"""One conversational turn, including the human decisions it needs."""

from datetime import datetime

from eva.application.messages import HEARTBEAT_OK, heartbeat_message
from eva.application.ports import AgentPort, ApprovalPort
from eva.domain.models import AgentStep

MAX_APPROVAL_ROUNDS = 100
REJECTION_MESSAGE = "The user declined this action. Do not retry it without a new request."


class EvaSession:
    def __init__(self, agent: AgentPort, approval: ApprovalPort) -> None:
        self.agent = agent
        self.approval = approval
        self.denied_actions: list[str] = []

    def send(self, message: str) -> str:
        self.denied_actions = []
        step = self.agent.ask(message)
        for _ in range(MAX_APPROVAL_ROUNDS):
            if not step.pending_actions:
                return step.reply or ""
            step = self.agent.resume(self._decide(step))
        raise RuntimeError("Too many approval rounds in one turn")

    def cancel(self) -> None:
        self.agent.cancel()

    def steer(self, message: str) -> None:
        """Add a message to the turn in progress; Eva reads it at her next step."""
        self.agent.steer(message)

    def take_unread(self) -> list[str]:
        return self.agent.take_unread()

    def heartbeat(self, now: datetime) -> str | None:
        """Let Eva act on her own; return her reply, or None if she had nothing to do.

        Idle heartbeats are removed from the conversation so they do not pile up.
        """
        reply = self.send(heartbeat_message(now))
        if reply.strip() == HEARTBEAT_OK:
            self.agent.forget_last_turn()
            return None
        return reply

    def _decide(self, step: AgentStep) -> list[dict[str, str]]:
        decisions = []
        for action in step.pending_actions:
            if self.approval.approve(action):
                decisions.append({"type": "approve"})
            else:
                self.denied_actions.append(action.name)
                decisions.append({"type": "reject", "message": REJECTION_MESSAGE})
        return decisions
