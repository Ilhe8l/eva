"""Runtime settings loaded from the environment and `.env`."""

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    model: str
    lm_studio_url: str = "http://localhost:1234/v1"
    data_dir: Path = field(default_factory=lambda: Path(".eva"))
    project_root: Path = field(default_factory=Path.cwd)
    thread_id: str = "main"
    heartbeat_minutes: float = 30
    autonomous: bool = False

    @classmethod
    def from_env(cls, model: str | None = None) -> "Settings":
        load_dotenv()
        selected = model or os.getenv("EVA_MODEL")
        if not selected:
            raise ValueError(
                "Set EVA_MODEL or pass --model, e.g. google_genai:gemini-2.5-flash or lmstudio:<loaded-model-id>"
            )
        project_root = Path(os.getenv("EVA_PROJECT_ROOT", Path.cwd())).resolve()
        return cls(
            model=selected,
            lm_studio_url=os.getenv("EVA_LM_STUDIO_URL", cls.lm_studio_url),
            data_dir=Path(os.getenv("EVA_DATA_DIR", project_root / ".eva")).resolve(),
            project_root=project_root,
            thread_id=os.getenv("EVA_THREAD_ID", cls.thread_id),
            heartbeat_minutes=float(os.getenv("EVA_HEARTBEAT_MINUTES", cls.heartbeat_minutes)),
            autonomous=os.getenv("EVA_AUTONOMOUS", "").strip().lower() in {"1", "true", "yes", "on"},
        )
