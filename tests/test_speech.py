from eva.adapters.voice import MAX_SPOKEN_CHARACTERS, SPEECH_EVENT, SpeechChannel


def test_only_chosen_text_is_spoken_while_enabled():
    emitted = []
    speech = SpeechChannel()
    assert "off" in speech.say("Not now", emitted.append)
    speech.enabled = True
    speech.say("  Hello there.  ", emitted.append)
    speech.say("x" * (MAX_SPOKEN_CHARACTERS + 1), emitted.append)
    speech.say("   ", emitted.append)
    assert emitted == [{"type": SPEECH_EVENT, "text": "Hello there."}]
