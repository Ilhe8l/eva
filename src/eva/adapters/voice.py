"""Microphone capture, speech recognition and speech synthesis, all local.

Model APIs: https://github.com/SYSTRAN/faster-whisper and
https://github.com/hexgrad/kokoro (voices:
https://huggingface.co/hexgrad/Kokoro-82M/blob/main/VOICES.md).
Heavy imports are deferred so a text-only install still works.
"""

import queue
import shutil
import subprocess
import tempfile
import threading
from collections.abc import Callable
from pathlib import Path

MAX_SPOKEN_CHARACTERS = 1200
KOKORO_REPO = "hexgrad/Kokoro-82M"
KOKORO_SAMPLE_RATE = 24000
MICROPHONE_SAMPLE_RATE = 16000
SPEECH_EVENT = "speech"


class SpeechChannel:
    """Carries the exact sentences Eva chose to say aloud to the stream."""

    def __init__(self) -> None:
        self.enabled = False

    def say(self, text: str, emit: Callable[[dict], None]) -> str:
        if not self.enabled:
            return "Speech is off; the user reads your text reply instead."
        spoken = text.strip()
        if not spoken:
            return "Nothing was spoken because the text was empty."
        if len(spoken) > MAX_SPOKEN_CHARACTERS:
            return f"Not spoken: keep speech under {MAX_SPOKEN_CHARACTERS} characters."
        emit({"type": SPEECH_EVENT, "text": spoken})
        return "Spoken to the user."


class Microphone:
    def record(self, until: Callable[[], object]) -> Path:
        """Record from the default microphone until `until()` returns."""
        import numpy as np
        import sounddevice as sd
        import soundfile as sf

        chunks: list = []
        with sd.InputStream(
            samplerate=MICROPHONE_SAMPLE_RATE,
            channels=1,
            dtype="float32",
            callback=lambda data, frames, time, status: chunks.append(data.copy()),
        ):
            until()
        if not chunks:
            raise ValueError("Nothing was recorded")
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as file:
            path = Path(file.name)
        sf.write(path, np.concatenate(chunks), MICROPHONE_SAMPLE_RATE)
        return path


class WhisperTranscriber:
    def __init__(self, model_name: str = "large-v3-turbo") -> None:
        self.model_name = model_name
        self._model = None

    def transcribe(self, audio_path: Path) -> str:
        from faster_whisper import WhisperModel

        if self._model is None:
            self._model = WhisperModel(self.model_name, device="auto", compute_type="default")
        # Greedy decoding (beam_size=1) keeps latency low for short commands.
        segments, _ = self._model.transcribe(str(audio_path), beam_size=1, vad_filter=True)
        return " ".join(segment.text.strip() for segment in segments).strip()


class KokoroSpeaker:
    def __init__(self, voice: str = "af_heart", language: str = "a") -> None:
        self.voice = voice
        self.language = language
        self._pipeline = None

    def speak(self, text: str) -> None:
        """Synthesize and play `text`; blocks until playback ends."""
        import numpy as np
        from kokoro import KPipeline

        if not text.strip():
            return
        if self._pipeline is None:
            self._pipeline = KPipeline(lang_code=self.language, repo_id=KOKORO_REPO)
        chunks = [audio for _, _, audio in self._pipeline(text, voice=self.voice)]
        if chunks:
            _play(np.concatenate(chunks), KOKORO_SAMPLE_RATE)


def _play(audio, sample_rate: int) -> None:
    """Play through the system sound server, falling back to PortAudio."""
    player = next(
        ([name, *flags] for name, flags in (("paplay", []), ("aplay", ["-q"])) if shutil.which(name)),
        None,
    )
    if player is None:
        import sounddevice as sd

        sd.play(audio, samplerate=sample_rate)
        sd.wait()
        return
    import soundfile as sf

    with tempfile.NamedTemporaryFile(suffix=".wav") as file:
        sf.write(file.name, audio, sample_rate)
        subprocess.run([*player, file.name], check=False)


class SpeechPlayer:
    """Plays sentences in order on a background thread, so Eva keeps working."""

    def __init__(self, speaker: KokoroSpeaker, on_error: Callable[[Exception], None]) -> None:
        self._speaker = speaker
        self._on_error = on_error
        self._queue: queue.Queue[str] = queue.Queue()
        threading.Thread(target=self._work, daemon=True, name="eva-speech").start()

    def play(self, text: str) -> None:
        self._queue.put(text)

    def wait(self) -> None:
        self._queue.join()

    def _work(self) -> None:
        while True:
            text = self._queue.get()
            try:
                self._speaker.speak(text)
            except (ImportError, OSError, ValueError) as exc:
                self._on_error(exc)
            finally:
                self._queue.task_done()
