from eva.adapters.voice import SpeechOutbox


def test_only_explicit_queued_text_is_spoken():
    speech = SpeechOutbox()
    assert "disabled" in speech.enqueue("This should not play")
    assert speech.drain() == []
    speech.enabled = True
    speech.enqueue("  Hello there.  ")
    assert speech.drain() == ["Hello there."]
    assert speech.drain() == []
