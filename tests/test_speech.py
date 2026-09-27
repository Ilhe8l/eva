from eva.adapters.voice import MAX_SPOKEN_CHARACTERS, SpeechOutbox


def test_only_chosen_text_is_spoken_while_enabled():
    speech = SpeechOutbox()
    assert "off" in speech.enqueue("Not now")
    assert speech.drain() == []
    speech.enabled = True
    speech.enqueue("  Hello there.  ")
    speech.enqueue("x" * (MAX_SPOKEN_CHARACTERS + 1))
    assert speech.drain() == ["Hello there."]
    assert speech.drain() == []
