"""Deep Agents adapter.

Docs: https://docs.langchain.com/oss/python/deepagents/overview,
https://docs.langchain.com/oss/python/deepagents/backends,
https://docs.langchain.com/oss/python/deepagents/human-in-the-loop and
https://docs.langchain.com/oss/python/deepagents/streaming
"""

import os
import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

from deepagents import create_deep_agent
from deepagents.backends import CompositeBackend, FilesystemBackend, LocalShellBackend
from langchain.agents.middleware import AgentMiddleware, InterruptOnConfig, TodoListMiddleware
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, RemoveMessage, ToolMessage
from langchain_core.tools import BaseTool
from langgraph.errors import GraphRecursionError
from langgraph.types import Checkpointer, Command

from eva.adapters.plan_guard import PlanFollowThroughMiddleware
from eva.adapters.policy import BUILTIN_SKILLS_ROUTE, MEMORY_ROUTE, SCRATCH_ROUTE, SKILLS_ROUTE
from eva.adapters.steering import SteeringInbox, SteeringMiddleware
from eva.adapters.tools import REACTION_EVENT
from eva.adapters.voice import SPEECH_EVENT
from eva.domain.models import (
    ActionRequest,
    AgentEvent,
    AgentStep,
    Reaction,
    Speech,
    StepLimitReached,
    TextDelta,
    ToolResult,
    ToolUse,
    TurnCancelled,
)
from eva.prompts import build_system_prompt

BUILTIN_SKILLS_DIR = Path(__file__).resolve().parents[1] / "skills"
MEMORY_INDEX = f"{MEMORY_ROUTE}AGENTS.md"
SHELL_TIMEOUT_SECONDS = 120
SILENT_TOOLS = {"speak_to_user", "show_expression"}
RESULT_SUMMARY_WIDTH = 160
DEFAULT_MAX_STEPS = 500  # graph steps per turn; one tool call takes a few
_SECRET_ENV_MARKERS = ("KEY", "TOKEN", "SECRET", "PASSWORD")


@dataclass(frozen=True)
class Workspace:
    """What Eva's file tools see, and where it lives on disk.

    `/` is the project (with shell execution), `/memories/` the journal,
    `/skills/` her own skills, `/builtin-skills/` the bundled ones and
    `/scratch/` her working space for temporary files. Deep Agents offloads
    large tool results and old conversation there too (`artifacts_root`), so
    none of it lands in the project.
    `virtual_mode=True` confines file tools to each root; it does not confine
    shell commands, which the approval policy covers instead.
    """

    project_root: Path
    data_dir: Path

    @property
    def memory_dir(self) -> Path:
        return self.data_dir / "memory"

    @property
    def skills_dir(self) -> Path:
        return self.data_dir / "skills"

    @property
    def scratch_dir(self) -> Path:
        return self.data_dir / "scratch"

    def backend(self) -> CompositeBackend:
        for directory in (self.memory_dir, self.skills_dir, self.scratch_dir):
            directory.mkdir(parents=True, exist_ok=True)
        return CompositeBackend(
            default=LocalShellBackend(
                root_dir=self.project_root,
                virtual_mode=True,
                timeout=SHELL_TIMEOUT_SECONDS,
                env=_shell_env(),
            ),
            routes={
                MEMORY_ROUTE: FilesystemBackend(root_dir=self.memory_dir, virtual_mode=True),
                SKILLS_ROUTE: FilesystemBackend(root_dir=self.skills_dir, virtual_mode=True),
                BUILTIN_SKILLS_ROUTE: FilesystemBackend(root_dir=BUILTIN_SKILLS_DIR, virtual_mode=True),
                SCRATCH_ROUTE: FilesystemBackend(root_dir=self.scratch_dir, virtual_mode=True),
            },
            artifacts_root=SCRATCH_ROUTE,
        )

    def system_prompt(self) -> str:
        return build_system_prompt(self.skills_dir, BUILTIN_SKILLS_DIR, self.scratch_dir)


def _shell_env() -> dict[str, str]:
    """The user's environment without API keys and other secrets."""
    return {
        name: value
        for name, value in os.environ.items()
        if not any(marker in name.upper() for marker in _SECRET_ENV_MARKERS)
    }


class DeepAgentAdapter:
    """The compiled Deep Agents graph, shared by every conversation thread."""

    def __init__(
        self,
        model: BaseChatModel,
        workspace: Workspace,
        checkpointer: Checkpointer,
        tools: Sequence[BaseTool],
        interrupt_on: dict[str, InterruptOnConfig],
        middleware: Sequence[AgentMiddleware] = (),
        max_steps: int = DEFAULT_MAX_STEPS,
    ) -> None:
        self.inbox = SteeringInbox()
        self.max_steps = max_steps
        self.graph = create_deep_agent(
            model=model,
            tools=list(tools),
            system_prompt=workspace.system_prompt(),
            backend=workspace.backend(),
            memory=[MEMORY_INDEX],
            skills=[BUILTIN_SKILLS_ROUTE, SKILLS_ROUTE],
            interrupt_on=interrupt_on,
            # Deep Agents 0.7 no longer plans with write_todos by default; Eva's terminal shows the plan.
            middleware=[
                TodoListMiddleware(),
                PlanFollowThroughMiddleware(),
                *middleware,
                SteeringMiddleware(self.inbox),
            ],
            checkpointer=checkpointer,
            name="eva",
        )

    def thread(
        self,
        thread_id: str,
        on_event: Callable[[AgentEvent], None] = lambda event: None,
        source: str | None = None,
        stream_text: bool = False,
    ) -> "AgentThread":
        """A conversation on the shared graph; `stream_text` also reports reply text as it is generated."""
        return AgentThread(self.graph, self.inbox, thread_id, on_event, source, stream_text, self.max_steps)


class AgentThread:
    """One checkpointed conversation. Threads run the same graph concurrently.

    `source` labels everything this thread reports, so the terminal can tell a
    background task from the main conversation.
    """

    def __init__(
        self,
        graph,
        inbox: SteeringInbox,
        thread_id: str,
        on_event: Callable[[AgentEvent], None],
        source: str | None,
        stream_text: bool = False,
        max_steps: int = DEFAULT_MAX_STEPS,
    ) -> None:
        self._graph = graph
        self._stream_modes = ["updates", "custom", *(["messages"] if stream_text else [])]
        self._inbox = inbox
        self._thread_id = thread_id
        self.config = {"configurable": {"thread_id": thread_id}, "recursion_limit": max_steps}
        self.on_event = on_event
        self.source = source
        self._pending: list[tuple[str, int]] = []
        self._cancelled = threading.Event()

    def ask(self, message: str) -> AgentStep:
        self._cancelled.clear()
        # A None skills_metadata makes SkillsMiddleware rescan, so new skills appear.
        return self._run({"messages": [{"role": "user", "content": message}], "skills_metadata": None})

    def resume(self, decisions: list[dict[str, str]]) -> AgentStep:
        if not self._pending:
            raise RuntimeError("There is no paused agent turn")
        resume: dict[str, dict] = {}
        start = 0
        for interrupt_id, count in self._pending:
            resume[interrupt_id] = {"decisions": decisions[start : start + count]}
            start += count
        return self._run(Command(resume=resume))

    def steer(self, message: str) -> None:
        """Deliver `message` to the running turn at its next model call."""
        self._inbox.post(self._thread_id, message)

    def take_unread(self) -> list[str]:
        """Messages sent with `steer` that the turn ended before reading."""
        return self._inbox.take(self._thread_id)

    def cancel(self) -> None:
        """Stop the running turn at its next step; a tool already running finishes first."""
        self._cancelled.set()

    def forget_last_turn(self) -> None:
        """Remove the latest user message and everything after it from the thread."""
        messages = self._graph.get_state(self.config).values.get("messages", [])
        starts = [index for index, message in enumerate(messages) if isinstance(message, HumanMessage)]
        if starts:
            removals = [RemoveMessage(id=message.id) for message in messages[starts[-1] :]]
            self._graph.update_state(self.config, {"messages": removals})

    def _run(self, payload) -> AgentStep:
        try:
            for chunk in self._graph.stream(
                payload,
                config=self.config,
                stream_mode=self._stream_modes,
                subgraphs=True,
                version="v2",
            ):
                if self._cancelled.is_set():
                    raise TurnCancelled
                for event in _events(chunk):
                    self.on_event(replace(event, source=self.source))
        except GraphRecursionError:
            raise StepLimitReached from None
        return self._step(self._graph.get_state(self.config))

    def _step(self, state) -> AgentStep:
        self._pending = []
        actions: list[ActionRequest] = []
        for interrupt in state.interrupts:
            requests = interrupt.value["action_requests"]
            self._pending.append((interrupt.id, len(requests)))
            actions.extend(ActionRequest(item["name"], item["args"], self.source) for item in requests)
        if actions:
            return AgentStep(reply=None, pending_actions=tuple(actions))
        return AgentStep(reply=_text(state.values["messages"][-1].content))


def _events(chunk: dict) -> list[AgentEvent]:
    """Translate one v2 stream chunk into progress events."""
    data = chunk["data"]
    if chunk["type"] == "messages":
        message, metadata = data
        is_reply = not chunk["ns"] and metadata.get("langgraph_node") == "model"
        text = message.text if isinstance(message, AIMessageChunk) and is_reply else ""
        return [TextDelta(text)] if text else []
    if chunk["type"] == "custom":
        kind = data.get("type") if isinstance(data, dict) else None
        if kind == SPEECH_EVENT:
            return [Speech(data["text"], aloud=data.get("aloud", True), mood=data.get("mood"))]
        return [Reaction(data["mood"])] if kind == REACTION_EVENT else []
    events: list[AgentEvent] = []
    for update in data.values():
        messages = update.get("messages", []) if isinstance(update, dict) else []
        for message in messages if isinstance(messages, list) else []:
            if isinstance(message, AIMessage):
                events.extend(
                    ToolUse(call["name"], call["args"], call_id=call.get("id") or "")
                    for call in message.tool_calls
                    if call["name"] not in SILENT_TOOLS
                )
            elif isinstance(message, ToolMessage) and message.name not in SILENT_TOOLS:
                ok = message.status != "error"
                events.append(
                    ToolResult(message.name or "", ok, _first_line(message.content), call_id=message.tool_call_id)
                )
    return events


def _first_line(content: str | list) -> str:
    line = next((line.strip() for line in _text(content).splitlines() if line.strip()), "")
    return line if len(line) <= RESULT_SUMMARY_WIDTH else f"{line[: RESULT_SUMMARY_WIDTH - 1]}…"


def _text(content: str | list) -> str:
    if isinstance(content, str):
        return content
    return "\n".join(
        block.get("text", "") for block in content if isinstance(block, dict) and block.get("type") == "text"
    )
