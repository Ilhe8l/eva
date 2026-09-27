from eva.application.session import EvaSession
from eva.domain.models import ActionRequest, AgentStep

COMMAND = ActionRequest("execute", {"command": "rm build.log"})


class FakeAgent:
    def __init__(self):
        self.decisions = None

    def ask(self, message):
        return AgentStep(reply=None, pending_actions=(COMMAND,))

    def resume(self, decisions):
        self.decisions = decisions
        return AgentStep(reply="Done.")


class FakeApproval:
    def __init__(self, approved):
        self.approved = approved
        self.seen = []

    def approve(self, action):
        self.seen.append(action)
        return self.approved


def test_approved_action_resumes_the_turn():
    agent, approval = FakeAgent(), FakeApproval(True)
    session = EvaSession(agent, approval)
    assert session.send("clean up") == "Done."
    assert approval.seen == [COMMAND]
    assert agent.decisions == [{"type": "approve"}]
    assert session.denied_actions == []


def test_rejected_action_is_reported_to_agent_and_user():
    agent = FakeAgent()
    session = EvaSession(agent, FakeApproval(False))
    session.send("clean up")
    assert agent.decisions[0]["type"] == "reject"
    assert session.denied_actions == ["execute"]
