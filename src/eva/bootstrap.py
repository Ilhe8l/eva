"""Composition root.

SQLite checkpointer: https://docs.langchain.com/oss/python/langgraph/checkpointers
"""

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass

from langgraph.checkpoint.sqlite import SqliteSaver

from eva.adapters.agent import DeepAgentAdapter, Workspace
from eva.adapters.followups import FollowUpStore
from eva.adapters.lifecycle import SelfUpdater
from eva.adapters.models import build_chat_model
from eva.adapters.policy import ApprovalPolicy
from eva.adapters.speech_mode import SpeechModeMiddleware
from eva.adapters.summarizer import SessionSummarizer
from eva.adapters.tools import build_tools
from eva.adapters.voice import SpeechChannel
from eva.application.ports import ApprovalPort
from eva.application.session import EvaSession
from eva.application.tasks import BackgroundTask, TaskBoard
from eva.config import Settings
from eva.domain.models import AgentEvent


@dataclass
class Application:
    session: EvaSession
    speech: SpeechChannel
    updater: SelfUpdater
    follow_ups: FollowUpStore
    tasks: TaskBoard
    policy: ApprovalPolicy
    summarizer: SessionSummarizer


@contextmanager
def bootstrap(
    settings: Settings,
    policy: ApprovalPolicy,
    approval: ApprovalPort,
    on_event: Callable[[AgentEvent], None],
) -> Iterator[Application]:
    workspace = Workspace(settings.project_root, settings.data_dir)
    model = build_chat_model(settings.model, settings.lm_studio_url)
    speech = SpeechChannel()
    updater = SelfUpdater(settings.project_root)
    follow_ups = FollowUpStore(settings.data_dir / "follow_ups.json")
    settings.data_dir.mkdir(parents=True, exist_ok=True)

    def open_task_session(task: BackgroundTask) -> EvaSession:
        thread = agent.thread(f"{settings.thread_id}-task-{task.id}", on_event, source=f"task {task.id}")
        return EvaSession(agent=thread, approval=approval)

    tasks = TaskBoard(open_task_session)
    with SqliteSaver.from_conn_string(str(settings.data_dir / "checkpoints.sqlite")) as checkpointer:
        agent = DeepAgentAdapter(
            model=model,
            workspace=workspace,
            checkpointer=checkpointer,
            tools=build_tools(speech, updater, follow_ups, tasks),
            interrupt_on=policy.interrupt_on(),
            middleware=[SpeechModeMiddleware(speech)],
        )
        app = Application(
            session=EvaSession(agent=agent.thread(settings.thread_id, on_event), approval=approval),
            speech=speech,
            updater=updater,
            follow_ups=follow_ups,
            tasks=tasks,
            policy=policy,
            summarizer=SessionSummarizer(model, workspace.memory_dir / "sessions"),
        )
        try:
            yield app
        finally:
            tasks.shutdown()  # before the checkpointer closes under them
