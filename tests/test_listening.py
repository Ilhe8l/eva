import pytest

np = pytest.importorskip("numpy", reason="needs the voice extra")

from eva.adapters.voice import MICROPHONE_SAMPLE_RATE, UtteranceDetector  # noqa: E402

FRAME = UtteranceDetector.FRAME_SAMPLES
rng = np.random.default_rng(0)


def loud(frame):
    return float(np.sqrt(np.mean(frame**2))) > 0.05


def frames(seconds, level):
    count = round(seconds / UtteranceDetector.FRAME_SECONDS)
    return [rng.normal(0, level, FRAME).astype("float32") for _ in range(count)]


def silence(seconds):
    return frames(seconds, 0.002)


def voice(seconds):
    return frames(seconds, 0.2)


def run(frames_):
    detector = UtteranceDetector(is_speech=loud)
    return [utterance for frame in frames_ if (utterance := detector.feed(frame)) is not None]


def test_speech_between_pauses_becomes_one_utterance():
    utterances = run(silence(1) + voice(1.5) + silence(1))
    assert len(utterances) == 1
    assert 1.5 <= len(utterances[0]) / MICROPHONE_SAMPLE_RATE <= 2.0  # speech plus the pre-roll


def test_short_clicks_are_ignored():
    assert run(silence(1) + voice(0.1) + silence(1)) == []


def test_two_sentences_with_a_long_pause_are_two_utterances():
    assert len(run(silence(1) + voice(1) + silence(1.2) + voice(1) + silence(1))) == 2


def test_reset_drops_a_partial_utterance():
    detector = UtteranceDetector(is_speech=loud)
    for frame in silence(1) + voice(0.5):
        detector.feed(frame)
    detector.reset()
    assert all(detector.feed(frame) is None for frame in silence(1))


def test_silero_classifier_tells_noise_from_nothing():
    pytest.importorskip("faster_whisper", reason="needs the voice extra")
    from eva.adapters.voice import SileroSpeechClassifier

    classifier = SileroSpeechClassifier()
    assert not any(classifier(frame) for frame in silence(0.5))
