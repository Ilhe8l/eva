"""Composition root.

SQLite checkpointer: https://docs.langchain.com/oss/python/langgraph/checkpointers
"""

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass

from langgraph.checkpoint.sqlite import SqliteSaver

from eva.adapters.agent import DeepAgentAdapter, Workspace
from eva.adapters.lifecycle import SelfUpdater
from eva.adapters.models import build_chat_model
from eva.adapters.policy import approval_rules
from eva.adapters.summarizer import SessionSummarizer
from eva.adapters.tools import build_tools
from eva.adapters.voice import SpeechChannel
from eva.application.ports import ApprovalPort
from eva.application.session import EvaSession
from eva.config import Settings
from eva.domain.models import AgentEvent


@dataclass
class Application:
    session: EvaSession
    speech: SpeechChannel
    updater: SelfUpdater
    summarizer: SessionSummarizer


@contextmanager
def bootstrap(
    settings: Settings,
    approval: ApprovalPort,
    on_event: Callable[[AgentEvent], None],
) -> Iterator[Application]:
    workspace = Workspace(settings.project_root, settings.data_dir)
    model = build_chat_model(settings.model, settings.lm_studio_url)
    speech = SpeechChannel()
    updater = SelfUpdater(settings.project_root)
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    with SqliteSaver.from_conn_string(str(settings.data_dir / "checkpoints.sqlite")) as checkpointer:
        agent = DeepAgentAdapter(
            model=model,
            workspace=workspace,
            checkpointer=checkpointer,
            thread_id=settings.thread_id,
            tools=build_tools(speech, updater),
            interrupt_on=approval_rules(),
            on_event=on_event,
        )
        yield Application(
            session=EvaSession(agent=agent, approval=approval),
            speech=speech,
            updater=updater,
            summarizer=SessionSummarizer(model, workspace.memory_dir / "sessions"),
        )
