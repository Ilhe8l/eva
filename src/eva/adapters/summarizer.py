"""Writes a short summary of each session into Eva's journal."""

import logging
from datetime import datetime
from pathlib import Path

from langchain_core.language_models import BaseChatModel

MIN_EXCHANGES = 2

SUMMARY_PROMPT = """\
Summarize this conversation between a user and Eva, their assistant, as compact Markdown:
- a one-line `##` title
- "What happened": what was asked, discussed or decided
- "Decisions and lessons": choices made, errors found, preferences noted
- "Next steps": anything left pending
Be factual and brief, write in English, and omit empty sections.

Conversation:
{transcript}
"""

logger = logging.getLogger(__name__)


class SessionSummarizer:
    def __init__(self, model: BaseChatModel, sessions_dir: Path) -> None:
        self._model = model
        self._sessions_dir = sessions_dir
        self._exchanges: list[tuple[str, str]] = []

    def record(self, user_message: str, reply: str) -> None:
        self._exchanges.append((user_message.strip(), reply.strip()))

    def save(self) -> Path | None:
        """Summarize the session; return the file written, or None when skipped."""
        if len(self._exchanges) < MIN_EXCHANGES:
            return None
        transcript = "\n\n".join(f"User: {user}\n\nEva: {reply}" for user, reply in self._exchanges)
        try:
            response = self._model.invoke(SUMMARY_PROMPT.format(transcript=transcript))
        except Exception:
            logger.exception("Session summary failed")
            return None
        summary = response.text.strip()
        if not summary:
            return None
        now = datetime.now().astimezone()
        path = self._sessions_dir / f"{now:%Y-%m-%d_%H-%M}.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"# Session {now:%Y-%m-%d %H:%M %Z}\n\n{summary}\n", encoding="utf-8")
        return path
