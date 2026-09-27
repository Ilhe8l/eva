from datetime import datetime

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


class ScriptedAgent:
    def __init__(self, reply):
        self.reply = reply
        self.messages = []
        self.forgotten = 0

    def ask(self, message):
        self.messages.append(message)
        return AgentStep(reply=self.reply)

    def forget_last_turn(self):
        self.forgotten += 1


def test_idle_heartbeat_is_silent_and_forgotten():
    agent = ScriptedAgent(" HEARTBEAT_OK ")
    assert EvaSession(agent, FakeApproval(True)).heartbeat(datetime(2026, 1, 1, 9, 0)) is None
    assert agent.messages[0].startswith("[heartbeat 2026-01-01 09:00]")
    assert agent.forgotten == 1


def test_useful_heartbeat_is_kept():
    agent = ScriptedAgent("Your build finished.")
    assert EvaSession(agent, FakeApproval(True)).heartbeat(datetime(2026, 1, 1)) == "Your build finished."
    assert agent.forgotten == 0
