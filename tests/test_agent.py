"""The real Deep Agents graph with a scripted model: approvals must hold."""

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver

from eva.adapters.agent import DeepAgentAdapter, Workspace
from eva.adapters.followups import FollowUpStore
from eva.adapters.lifecycle import SelfUpdater
from eva.adapters.policy import ApprovalPolicy
from eva.adapters.speech_mode import KICKOFF_NUDGE, SPEECH_OFF, SPEECH_ON, SpeechModeMiddleware
from eva.adapters.steering import STEERING_PREFIX
from eva.adapters.tools import build_tools
from eva.adapters.voice import SpeechChannel
from eva.application.session import EvaSession
from eva.application.tasks import TaskBoard
from eva.domain.models import Reaction, Speech, StepLimitReached, TextDelta, ToolUse, TurnCancelled


class ScriptedModel(GenericFakeChatModel):
    prompts: list = []  # everything the model saw, per call

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, *args, **kwargs):
        self.prompts.append("\n".join(message.text for message in messages))
        return super()._generate(messages, *args, **kwargs)


def _call(name, **args):
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": f"call-{name}"}])


@pytest.fixture()
def make_agent(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    (project / "notes.txt").write_text("keep me\n")

    def make(*messages, events=None, speech=None, policy=None, max_steps=500):
        model = ScriptedModel(messages=iter([*messages, AIMessage(content="All done.")]), prompts=[])
        make.models.append(model)
        speech = speech or SpeechChannel()
        on_event = events.append if events is not None else lambda event: None
        tasks = TaskBoard(lambda task: EvaSession(adapter.thread(f"task-{task.id}", on_event, f"task {task.id}"), None))
        adapter = DeepAgentAdapter(
            model,
            Workspace(project, tmp_path / "data"),
            InMemorySaver(),
            tools=build_tools(speech, SelfUpdater(project), FollowUpStore(tmp_path / "f.json"), tasks),
            interrupt_on=(policy or ApprovalPolicy()).interrupt_on(),
            middleware=[SpeechModeMiddleware(speech)],
            max_steps=max_steps,
        )
        make.tasks = tasks
        make.adapter = adapter
        return adapter.thread("test", on_event)

    make.models = []
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
    agent = make(
        _call("speak_to_user", text="Checking now."), _call("execute", command="ls"), events=events, speech=speech
    )
    step = agent.ask("look around")
    speech_event, tool_use, result = events
    assert speech_event == Speech("Checking now.")
    assert tool_use == ToolUse("execute", {"command": "ls"}, call_id="call-execute")
    assert (result.name, result.ok, result.call_id) == ("execute", True, "call-execute")
    assert "notes.txt" in result.summary
    assert step.reply == "All done."


def test_expressions_reach_the_face_without_showing_as_tool_use(make_agent):
    make, _, _ = make_agent
    events = []
    make(_call("show_expression", mood="happy"), events=events).ask("it works!")
    assert events == [Reaction("happy")]


def _skill_names(agent):
    return {skill["name"] for skill in agent._graph.get_state(agent.config).values.get("skills_metadata") or []}


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
    messages = agent._graph.get_state(agent.config).values["messages"]
    assert [message.content for message in messages] == ["hello", "Hi."]


def test_model_is_told_whether_speech_is_on(make_agent):
    make, _, _ = make_agent
    speech = SpeechChannel()
    agent = make(AIMessage(content="One."), speech=speech)
    agent.ask("first")
    speech.enabled = True
    agent.ask("second")
    first, second = make.models[-1].prompts
    assert SPEECH_OFF in first and SPEECH_ON not in first
    assert SPEECH_ON in second


def test_tool_work_starts_with_a_kickoff_nudge(make_agent):
    make, _, _ = make_agent
    speech = SpeechChannel()
    speech.enabled = True
    calls = [
        AIMessage(content="", tool_calls=[{"name": "execute", "args": {"command": "ls"}, "id": f"ls-{index}"}])
        for index in range(4)
    ]
    agent = make(*calls, speech=speech)
    agent.ask("dig around")
    prompts = make.models[-1].prompts
    assert [KICKOFF_NUDGE in prompt for prompt in prompts] == [False, True, False, False, False]


def test_stop_cancels_the_turn_at_the_next_step(make_agent):
    make, _, _ = make_agent
    holder = {}

    def cancel_on_first_tool(event):
        if isinstance(event, ToolUse):
            holder["agent"].cancel()

    agent = make(_call("execute", command="ls"), _call("execute", command="pwd"), events=[])
    agent.on_event = cancel_on_first_tool
    holder["agent"] = agent
    with pytest.raises(TurnCancelled):
        agent.ask("look around")


def test_threads_share_the_graph_but_not_the_history(make_agent):
    make, _, _ = make_agent
    main = make(AIMessage(content="Main."), AIMessage(content="Task."))
    task = make.adapter.thread("task-1")
    main.ask("hello main")
    task.ask("hello task")

    def contents(thread):
        return [message.content for message in thread._graph.get_state(thread.config).values["messages"]]

    assert contents(main) == ["hello main", "Main."]
    assert contents(task) == ["hello task", "Task."]


def test_autonomous_mode_and_always_allow_skip_approval(make_agent):
    make, project, _ = make_agent
    policy = ApprovalPolicy(autonomous=True)
    step = make(_call("execute", command="rm notes.txt"), policy=policy).ask("delete my notes")
    assert step.pending_actions == ()
    assert not (project / "notes.txt").exists()

    policy = ApprovalPolicy()
    policy.always_allow("write_file", {"file_path": "/a.txt"})
    step = make(_call("write_file", file_path="/a.txt", content="x"), policy=policy).ask("write a")
    assert step.pending_actions == ()
    step = make(_call("write_file", file_path="/b.txt", content="x"), policy=policy).ask("write b")
    assert [action.name for action in step.pending_actions] == ["write_file"]


def test_messages_sent_mid_turn_reach_the_next_model_call(make_agent):
    make, _, _ = make_agent
    holder = {}

    def steer_once(event):
        if isinstance(event, ToolUse) and not holder.get("sent"):
            holder["sent"] = True
            holder["agent"].steer("actually, only list Python files")

    agent = make(_call("execute", command="ls"))
    agent.on_event = steer_once
    holder["agent"] = agent
    agent.ask("list files")

    messages = agent._graph.get_state(agent.config).values["messages"]
    kinds = [type(message).__name__ for message in messages]
    assert kinds == ["HumanMessage", "AIMessage", "ToolMessage", "HumanMessage", "AIMessage"]
    assert messages[3].content == f"{STEERING_PREFIX} actually, only list Python files"
    assert agent.take_unread() == []


def test_messages_sent_after_the_last_step_are_kept_for_a_new_turn(make_agent):
    make, _, _ = make_agent
    agent = make(AIMessage(content="Done."))
    agent.ask("hi")
    agent.steer("one more thing")
    assert agent.take_unread() == ["one more thing"]


def test_main_thread_streams_reply_text(make_agent):
    make, _, _ = make_agent
    events = []
    make(AIMessage(content="Hello there, friend."), events=events)
    streaming = make.adapter.thread("streamed", events.append, stream_text=True)
    streaming.ask("hi")
    text = "".join(event.text for event in events if isinstance(event, TextDelta))
    assert text == "Hello there, friend."


def test_running_out_of_steps_stops_the_turn_cleanly(make_agent):
    make, _, _ = make_agent
    calls = [
        AIMessage(content="", tool_calls=[{"name": "execute", "args": {"command": "ls"}, "id": f"ls-{index}"}])
        for index in range(20)
    ]
    agent = make(*calls, max_steps=12)
    with pytest.raises(StepLimitReached):
        agent.ask("explore everything")


def test_update_reminders_are_never_saved_to_the_thread(make_agent):
    make, _, _ = make_agent
    agent = make(_call("execute", command="ls"))
    agent.ask("look around")
    assert KICKOFF_NUDGE in make.models[-1].prompts[1]
    saved = [message.text for message in agent._graph.get_state(agent.config).values["messages"]]
    assert not any("[reminder]" in text for text in saved)


def test_eva_can_share_a_plan(make_agent):
    make, _, _ = make_agent
    events = []
    todos = [{"content": "Check the forecast", "status": "in_progress"}]
    agent = make(_call("write_todos", todos=todos), events=events)
    agent.ask("plan it")
    assert events[0] == ToolUse("write_todos", {"todos": todos}, call_id="call-write_todos")
    assert agent._graph.get_state(agent.config).values["todos"] == todos
