from __future__ import annotations

import textwrap
from datetime import datetime, timezone

from langchain_openai import ChatOpenAI

from eva.adapters.memory import MemoryStore


MIN_EXCHANGES = 2  # skip trivial sessions (one-liners)

SUMMARIZE_PROMPT = """\
You are summarizing a conversation between a user and Eva (an AI assistant).
Write a compact Markdown summary with:
- A one-line title (##)
- A "What happened" bullet list (what was asked / discussed / decided)
- A "Decisions & lessons" bullet list (choices made, errors found, preferences noted)
- A "Next steps" bullet list (if anything was left pending)

Be factual and brief. Do not editorialize. Write in the same language as the conversation.
Omit sections that would be empty.

Conversation:
{transcript}
"""


class SessionSummarizer:

    def __init__(
        self,
        model_name: str,
        base_url: str,
        memory: MemoryStore,
    ) -> None:
        self._model = ChatOpenAI(
            model=model_name,
            base_url=base_url,
            api_key="lm-studio",
            temperature=0.3,
        )
        self._memory = memory
        self._exchanges: list[tuple[str, str]] = []  # (user, eva)

    def record(self, user_message: str, eva_reply: str) -> None:
        self._exchanges.append((user_message.strip(), eva_reply.strip()))

    def save(self) -> str | None:
        if len(self._exchanges) < MIN_EXCHANGES:
            return None

        transcript = "\n\n".join(
            f"User: {u}\n\nEva: {e}" for u, e in self._exchanges
        )
        prompt = textwrap.dedent(SUMMARIZE_PROMPT).format(transcript=transcript)

        try:
            response = self._model.invoke([{"role": "user", "content": prompt}])
            summary = response.content if isinstance(response.content, str) else ""
        except Exception as exc:
            return None

        if not summary.strip():
            return None

        now = datetime.now(tz=timezone.utc).astimezone()
        filename = f"sessions/{now.strftime('%Y-%m-%d_%H-%M')}.md"
        header = f"# Session - {now.strftime('%Y-%m-%d %H:%M %Z')}\n\n"
        try:
            self._memory.write(filename, header + summary)
        except (OSError, ValueError):
            return None

        return filename
