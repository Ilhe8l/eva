from eva.application.session import EvaSession
from eva.domain.models import ActionRequest, AgentStep


class FakeAgent:
    def __init__(self, reply="The action was handled."):
        self.decisions = None
        self.reply = reply

    def ask(self, message):
        assert message == "Please inspect my files"
        return AgentStep(
            reply=None,
            pending_actions=(
                ActionRequest("run_host_command", {"command": "pwd", "cwd": "/tmp"}),
            ),
        )

    def resume(self, decisions):
        self.decisions = decisions
        return AgentStep(reply=self.reply)


class FakeApproval:
    def __init__(self, approved):
        self.approved = approved

    def approve(self, action):
        assert action.arguments["command"] == "pwd"
        return self.approved


def test_session_approves_pending_command():
    agent = FakeAgent()
    answer = EvaSession(agent, FakeApproval(True)).send("Please inspect my files")
    assert answer == "The action was handled."
    assert agent.decisions == [{"type": "approve"}]


def test_session_rejects_pending_command():
    agent = FakeAgent(reply="I ran pwd and got /tmp")
    answer = EvaSession(agent, FakeApproval(False)).send("Please inspect my files")
    assert agent.decisions[0]["type"] == "reject"
    assert answer == "Action denied and not executed: run_host_command."
