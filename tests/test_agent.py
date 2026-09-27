"""The real Deep Agents graph with a scripted model: approvals must hold."""

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver

from eva.adapters.agent import DeepAgentAdapter, Workspace
from eva.adapters.followups import FollowUpStore
from eva.adapters.lifecycle import SelfUpdater
from eva.adapters.policy import approval_rules
from eva.adapters.tools import build_tools
from eva.adapters.voice import SpeechChannel
from eva.domain.models import Speech, ToolUse


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

    def make(*messages, events=None, speech=None):
        model = ScriptedModel(messages=iter([*messages, AIMessage(content="All done.")]))
        return DeepAgentAdapter(
            model,
            Workspace(project, tmp_path / "data"),
            InMemorySaver(),
            "test",
            tools=build_tools(speech or SpeechChannel(), SelfUpdater(project), FollowUpStore(tmp_path / "f.json")),
            interrupt_on=approval_rules(),
            on_event=events.append if events is not None else lambda event: None,
        )

    return make, project, tmp_path / "data" / "memory"


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


def test_speech_and_tool_use_stream_before_the_reply(make_agent):
    make, _, _ = make_agent
    events, speech = [], SpeechChannel()
    speech.enabled = True
    agent = make(_call("speak_to_user", text="Checking now."), _call("execute", command="ls"), events=events, speech=speech)
    step = agent.ask("look around")
    assert events == [Speech("Checking now."), ToolUse("execute", {"command": "ls"})]
    assert step.reply == "All done."


def _skill_names(agent):
    return {skill["name"] for skill in agent._agent.get_state(agent.config).values.get("skills_metadata") or []}


def test_skills_written_mid_session_appear_on_the_next_message(make_agent, tmp_path):
    make, _, _ = make_agent
    agent = make(AIMessage(content="First."))
    agent.ask("hello")
    assert "skill-authoring" in _skill_names(agent)

    skill = tmp_path / "data" / "skills" / "greeting"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text("---\nname: greeting\ndescription: Greet people warmly.\n---\n\nSay hi.\n")
    agent.ask("again")
    assert {"skill-authoring", "greeting"} <= _skill_names(agent)


def test_writing_a_skill_needs_approval(make_agent):
    make, _, _ = make_agent
    step = make(_call("write_file", file_path="/skills/x/SKILL.md", content="---\n---")).ask("learn")
    assert [action.name for action in step.pending_actions] == ["write_file"]


def test_forgetting_the_last_turn_keeps_earlier_ones(make_agent):
    make, _, _ = make_agent
    agent = make(AIMessage(content="Hi."))
    agent.ask("hello")
    agent.ask("heartbeat")
    agent.forget_last_turn()
    messages = agent._agent.get_state(agent.config).values["messages"]
    assert [message.content for message in messages] == ["hello", "Hi."]
