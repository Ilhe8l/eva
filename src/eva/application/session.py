from eva.application.ports import AgentPort, ApprovalPort
from eva.domain.models import AgentStep


class EvaSession:
    def __init__(self, agent: AgentPort, approval: ApprovalPort) -> None:
        self.agent = agent
        self.approval = approval
        self.denied_actions: list[str] = []

    def send(self, message: str) -> str:
        self.denied_actions = []
        step = self.agent.ask(message)
        for _ in range(100):
            if not step.pending_actions:
                if self.denied_actions:
                    names = ", ".join(self.denied_actions)
                    return f"Action denied and not executed: {names}."
                return step.reply or ""
            step = self._review_and_resume(step)
        raise RuntimeError("Too many approval cycles in one turn")

    def _review_and_resume(self, step: AgentStep) -> AgentStep:
        decisions = []
        for action in step.pending_actions:
            if self.approval.approve(action):
                decisions.append({"type": "approve"})
            else:
                self.denied_actions.append(action.name)
                decisions.append(
                    {
                        "type": "reject",
                        "message": "The user declined this action. Do not retry it without a new request.",
                    }
                )
        return self.agent.resume(decisions)
