from eva.adapters.voice import MAX_SPOKEN_CHARACTERS, SPEECH_EVENT, SpeechChannel


def test_updates_are_spoken_with_speech_on_and_shown_with_it_off():
    emitted = []
    speech = SpeechChannel()
    assert "status line" in speech.say("Checking the logs.", emitted.append)
    speech.enabled = True
    speech.say("  Hello there.  ", emitted.append)
    speech.say("x" * (MAX_SPOKEN_CHARACTERS + 1), emitted.append)
    speech.say("   ", emitted.append)
    assert emitted == [
        {"type": SPEECH_EVENT, "text": "Checking the logs.", "aloud": False, "mood": None},
        {"type": SPEECH_EVENT, "text": "Hello there.", "aloud": True, "mood": None},
    ]


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


def test_whisper_uses_the_configured_precision_on_the_gpu():
    import pytest

    pytest.importorskip("numpy", reason="needs the voice extra")
    from eva.adapters.voice import WhisperTranscriber

    calls = []

    class FakeModel:
        def __init__(self, name, device, compute_type):
            calls.append((name, device, compute_type))

        def transcribe(self, audio, language=None):
            return [], None

    transcriber = WhisperTranscriber("tiny", compute_type="float16")
    assert transcriber._on_gpu(FakeModel) is not None
    assert calls == [("tiny", "cuda", "float16")]


def test_a_failed_sentence_does_not_silence_the_player():
    from eva.adapters.voice import SpeechPlayer

    class FlakySpeaker:
        def __init__(self):
            self.spoken = []

        def speak(self, text):
            if text == "boom":
                raise RuntimeError("CUDA out of memory")
            self.spoken.append(text)

        def stop(self):
            pass

    errors, speaker = [], FlakySpeaker()
    player = SpeechPlayer(speaker, on_error=errors.append)
    player.play("boom")
    player.play("still here")
    player.wait()
    assert speaker.spoken == ["still here"]
    assert "out of memory" in str(errors[0])
