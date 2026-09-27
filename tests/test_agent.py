"""The real Deep Agents graph with a scripted model: approvals must hold."""

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver

from eva.adapters.agent import DeepAgentAdapter, build_backend
from eva.adapters.policy import approval_rules


class ScriptedModel(GenericFakeChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


def _call(name, **args):
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": f"call-{name}"}])


@pytest.fixture()
def make_agent(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    (project / "notes.txt").write_text("keep me\n")

    def make(*messages):
        model = ScriptedModel(messages=iter([*messages, AIMessage(content="All done.")]))
        agent = DeepAgentAdapter(model, build_backend(project, tmp_path / "memory"), InMemorySaver(), "test")
        agent.configure([], approval_rules())
        return agent

    return make, project, tmp_path / "memory"


def test_unsafe_shell_command_waits_for_approval(make_agent):
    make, project, _ = make_agent
    agent = make(_call("execute", command="rm notes.txt"))
    step = agent.ask("delete my notes")
    assert [action.name for action in step.pending_actions] == ["execute"]
    assert (project / "notes.txt").exists()
    step = agent.resume([{"type": "reject", "message": "no"}])
    assert step.reply == "All done."
    assert (project / "notes.txt").exists()


def test_read_only_shell_command_runs_without_approval(make_agent):
    make, _, _ = make_agent
    step = make(_call("execute", command="ls")).ask("what is here?")
    assert step.pending_actions == ()
    assert step.reply == "All done."


def test_project_write_waits_but_journal_write_does_not(make_agent):
    make, project, memory = make_agent
    step = make(_call("write_file", file_path="/memories/AGENTS.md", content="User likes tea.")).ask("remember")
    assert step.pending_actions == ()
    assert (memory / "AGENTS.md").read_text() == "User likes tea."

    step = make(_call("write_file", file_path="/new.py", content="print(1)")).ask("write code")
    assert [action.name for action in step.pending_actions] == ["write_file"]
    assert not (project / "new.py").exists()
