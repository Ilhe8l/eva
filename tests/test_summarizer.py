from langchain_core.language_models.fake_chat_models import FakeListChatModel

from eva.adapters.summarizer import MIN_EXCHANGES, SessionSummarizer


def _summarizer(tmp_path, reply="## Session\n\n- Did things"):
    return SessionSummarizer(FakeListChatModel(responses=[reply]), tmp_path / "sessions")


def _chat(summarizer, exchanges=MIN_EXCHANGES):
    for index in range(exchanges):
        summarizer.record(f"question {index}", f"answer {index}")


def test_trivial_session_is_not_saved(tmp_path):
    summarizer = _summarizer(tmp_path)
    _chat(summarizer, MIN_EXCHANGES - 1)
    assert summarizer.save() is None


def test_session_summary_is_written_to_the_journal(tmp_path):
    summarizer = _summarizer(tmp_path)
    _chat(summarizer)
    path = summarizer.save()
    assert path.parent == tmp_path / "sessions"
    assert "## Session" in path.read_text()


def test_empty_summary_is_skipped(tmp_path):
    summarizer = _summarizer(tmp_path, reply="  ")
    _chat(summarizer)
    assert summarizer.save() is None
