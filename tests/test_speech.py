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


def test_stopping_the_player_drops_queued_sentences():
    import threading

    from eva.adapters.voice import SpeechPlayer

    class SlowSpeaker:
        def __init__(self):
            self.spoken, self.started, self.release = [], threading.Event(), threading.Event()

        def speak(self, text):
            self.started.set()
            self.release.wait(timeout=2)
            self.spoken.append(text)

        def stop(self):
            self.release.set()

    speaker = SlowSpeaker()
    player = SpeechPlayer(speaker, on_error=print)
    player.play("first")
    speaker.started.wait(timeout=2)
    player.play("second")
    player.play("third")
    player.stop()
    player.wait()
    assert speaker.spoken == ["first"]
