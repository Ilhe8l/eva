"""Deep Agents adapter.

Docs: https://docs.langchain.com/oss/python/deepagents/overview,
https://docs.langchain.com/oss/python/deepagents/backends and
https://docs.langchain.com/oss/python/deepagents/human-in-the-loop
"""

import os
from collections.abc import Sequence
from pathlib import Path

from deepagents import create_deep_agent
from deepagents.backends import CompositeBackend, FilesystemBackend, LocalShellBackend
from langchain.agents.middleware import InterruptOnConfig
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from langgraph.types import Checkpointer, Command

from eva.adapters.policy import MEMORY_ROUTE
from eva.domain.models import ActionRequest, AgentStep
from eva.prompts import SYSTEM_PROMPT

MEMORY_INDEX = f"{MEMORY_ROUTE}AGENTS.md"
SHELL_TIMEOUT_SECONDS = 120
_SECRET_ENV_MARKERS = ("KEY", "TOKEN", "SECRET", "PASSWORD")


def build_backend(project_root: Path, memory_dir: Path) -> CompositeBackend:
    """Project files at `/`, the journal at `/memories/`, shell in the project.

    `virtual_mode=True` confines file tools to each root; it does not confine
    shell commands, which the approval policy covers instead.
    """
    memory_dir.mkdir(parents=True, exist_ok=True)
    return CompositeBackend(
        default=LocalShellBackend(
            root_dir=project_root,
            virtual_mode=True,
            timeout=SHELL_TIMEOUT_SECONDS,
            env=_shell_env(),
        ),
        routes={MEMORY_ROUTE: FilesystemBackend(root_dir=memory_dir, virtual_mode=True)},
    )


def _shell_env() -> dict[str, str]:
    """The user's environment without API keys and other secrets."""
    return {
        name: value
        for name, value in os.environ.items()
        if not any(marker in name.upper() for marker in _SECRET_ENV_MARKERS)
    }


class DeepAgentAdapter:
    def __init__(
        self,
        model: BaseChatModel,
        backend: CompositeBackend,
        checkpointer: Checkpointer,
        thread_id: str,
    ) -> None:
        self.model = model
        self.backend = backend
        self.checkpointer = checkpointer
        self.config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 150}
        self._agent = None
        self._pending: list[tuple[str, int]] = []

    def configure(self, tools: Sequence[BaseTool], interrupt_on: dict[str, InterruptOnConfig]) -> None:
        """(Re)build the agent graph; call again when the tool set changes."""
        self._agent = create_deep_agent(
            model=self.model,
            tools=list(tools),
            system_prompt=SYSTEM_PROMPT,
            backend=self.backend,
            memory=[MEMORY_INDEX],
            interrupt_on=interrupt_on,
            checkpointer=self.checkpointer,
            name="eva",
        )

    def ask(self, message: str) -> AgentStep:
        return self._run({"messages": [{"role": "user", "content": message}]})

    def resume(self, decisions: list[dict[str, str]]) -> AgentStep:
        if not self._pending:
            raise RuntimeError("There is no paused agent turn")
        resume: dict[str, dict] = {}
        start = 0
        for interrupt_id, count in self._pending:
            resume[interrupt_id] = {"decisions": decisions[start : start + count]}
            start += count
        return self._run(Command(resume=resume))

    def _run(self, payload) -> AgentStep:
        if self._agent is None:
            raise RuntimeError("Call configure() before running the agent")
        result = self._agent.invoke(payload, config=self.config, version="v2")
        return self._step(result)

    def _step(self, result) -> AgentStep:
        self._pending = []
        actions: list[ActionRequest] = []
        for interrupt in result.interrupts:
            requests = interrupt.value["action_requests"]
            self._pending.append((interrupt.id, len(requests)))
            actions.extend(ActionRequest(name=item["name"], arguments=item["args"]) for item in requests)
        if actions:
            return AgentStep(reply=None, pending_actions=tuple(actions))
        return AgentStep(reply=_text(result.value["messages"][-1].content))


def _text(content: str | list) -> str:
    if isinstance(content, str):
        return content
    return "\n".join(
        block.get("text", "") for block in content if isinstance(block, dict) and block.get("type") == "text"
    )
