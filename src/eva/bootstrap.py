"""Composition root.

SQLite checkpointer: https://docs.langchain.com/oss/python/langgraph/checkpointers
"""

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

from langgraph.checkpoint.sqlite import SqliteSaver

from eva.adapters.agent import DeepAgentAdapter, build_backend
from eva.adapters.extensions import ExtensionStore
from eva.adapters.models import build_chat_model
from eva.adapters.patches import PatchStore
from eva.adapters.policy import approval_rules
from eva.adapters.summarizer import SessionSummarizer
from eva.adapters.tools import build_tools
from eva.adapters.voice import SpeechOutbox
from eva.application.ports import ApprovalPort
from eva.application.session import EvaSession
from eva.config import Settings


@dataclass
class Application:
    session: EvaSession
    agent: DeepAgentAdapter
    extensions: ExtensionStore
    patches: PatchStore
    speech: SpeechOutbox
    summarizer: SessionSummarizer

    def reload_tools(self) -> None:
        """Rebuild the agent after the set of active extensions changed."""
        tools = build_tools(self.speech, self.extensions, self.patches)
        active = [item.name for item in self.extensions.list_active()]
        self.agent.configure(tools, approval_rules(always_ask=active))


@contextmanager
def bootstrap(settings: Settings, approval: ApprovalPort) -> Iterator[Application]:
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    memory_dir = settings.data_dir / "memory"
    model = build_chat_model(settings.model, settings.lm_studio_url)
    with SqliteSaver.from_conn_string(str(settings.data_dir / "checkpoints.sqlite")) as checkpointer:
        agent = DeepAgentAdapter(
            model=model,
            backend=build_backend(settings.project_root, memory_dir),
            checkpointer=checkpointer,
            thread_id=settings.thread_id,
        )
        app = Application(
            session=EvaSession(agent=agent, approval=approval),
            agent=agent,
            extensions=ExtensionStore(settings.data_dir),
            patches=PatchStore(settings.data_dir, settings.project_root),
            speech=SpeechOutbox(),
            summarizer=SessionSummarizer(model, memory_dir / "sessions"),
        )
        app.reload_tools()
        yield app
