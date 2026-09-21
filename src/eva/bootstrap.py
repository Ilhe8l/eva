import os
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

from langgraph.checkpoint.sqlite import SqliteSaver

from eva.adapters.agent import DeepAgentAdapter
from eva.adapters.extensions import ExtensionStore
from eva.adapters.memory import MemoryStore
from eva.adapters.patches import PatchStore
from eva.adapters.summarizer import SessionSummarizer
from eva.adapters.voice import SpeechOutbox
from eva.application.session import EvaSession


@dataclass(frozen=True)
class Settings:
    model: str
    base_url: str = "http://localhost:1234/v1"
    data_dir: Path = field(default_factory=lambda: Path(".eva"))
    thread_id: str = "main"
    project_root: Path = field(default_factory=Path.cwd)
    home_root: Path = field(default_factory=Path.home)

    @classmethod
    def from_env(cls, model: str | None = None) -> "Settings":
        selected = model or os.getenv("EVA_MODEL")
        if not selected:
            raise ValueError("Set EVA_MODEL or pass --model with a loaded LM Studio model ID")
        return cls(
            model=selected,
            base_url=os.getenv("EVA_LM_STUDIO_URL", "http://localhost:1234/v1"),
            data_dir=Path(os.getenv("EVA_DATA_DIR", ".eva")),
            thread_id=os.getenv("EVA_THREAD_ID", "main"),
            project_root=Path(os.getenv("EVA_PROJECT_ROOT", str(Path.cwd()))),
            home_root=Path(os.getenv("EVA_HOME_ROOT", str(Path.home()))),
        )


@dataclass
class Application:
    session: EvaSession
    extensions: ExtensionStore
    memory: MemoryStore
    patches: PatchStore
    speech: SpeechOutbox
    summarizer: SessionSummarizer


@contextmanager
def bootstrap(settings: Settings, approval) -> Iterator[Application]:
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    extensions = ExtensionStore(settings.data_dir)
    memory = MemoryStore(settings.data_dir)
    patches = PatchStore(settings.data_dir, settings.project_root)
    speech = SpeechOutbox()
    summarizer = SessionSummarizer(
        model_name=settings.model,
        base_url=settings.base_url,
        memory=memory,
    )
    checkpoint_path = settings.data_dir / "checkpoints.sqlite"
    with SqliteSaver.from_conn_string(str(checkpoint_path)) as checkpointer:
        agent = DeepAgentAdapter(
            model_name=settings.model,
            base_url=settings.base_url,
            checkpointer=checkpointer,
            thread_id=settings.thread_id,
            extensions=extensions,
            memory=memory,
            patches=patches,
            speech=speech,
            project_root=settings.project_root,
            home_root=settings.home_root,
        )
        yield Application(
            session=EvaSession(agent=agent, approval=approval),
            extensions=extensions,
            memory=memory,
            patches=patches,
            speech=speech,
            summarizer=summarizer,
        )
