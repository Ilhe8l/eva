from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from eva.adapters.memory import MemoryStore
from eva.adapters.summarizer import MIN_EXCHANGES, SessionSummarizer


@pytest.fixture()
def memory(tmp_path: Path) -> MemoryStore:
    return MemoryStore(tmp_path)


def _make_summarizer(memory: MemoryStore, reply: str = "## Session\n\n- Did stuff") -> SessionSummarizer:
    s = SessionSummarizer(
        model_name="test-model",
        base_url="http://localhost:1234/v1",
        memory=memory,
    )
    # Stub the LLM so no real network call is made
    fake_response = MagicMock()
    fake_response.content = reply
    s._model = MagicMock()
    s._model.invoke.return_value = fake_response
    return s


def test_skips_trivial_session(memory: MemoryStore) -> None:
    s = _make_summarizer(memory)
    # fewer than MIN_EXCHANGES -> no file written
    for _ in range(MIN_EXCHANGES - 1):
        s.record("hi", "hello")
    result = s.save()
    assert result is None
    assert memory.list() == []


def test_saves_session_after_enough_exchanges(memory: MemoryStore) -> None:
    s = _make_summarizer(memory)
    for i in range(MIN_EXCHANGES):
        s.record(f"question {i}", f"answer {i}")
    path = s.save()
    assert path is not None
    assert path.startswith("sessions/")
    assert path.endswith(".md")
    content = memory.read(path)
    assert "## Session" in content


def test_skips_when_llm_returns_empty(memory: MemoryStore) -> None:
    s = _make_summarizer(memory, reply="")
    for i in range(MIN_EXCHANGES):
        s.record(f"q{i}", f"a{i}")
    result = s.save()
    assert result is None


def test_skips_when_llm_raises(memory: MemoryStore) -> None:
    s = _make_summarizer(memory)
    s._model.invoke.side_effect = RuntimeError("network gone")
    for i in range(MIN_EXCHANGES):
        s.record(f"q{i}", f"a{i}")
    result = s.save()
    assert result is None
