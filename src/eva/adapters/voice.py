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
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Protocol

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
        self._lock = threading.Lock()

    def warm_up(self) -> None:
        """Load the model ahead of the first recording; errors surface on use instead."""
        try:
            self._load()
        except Exception:  # noqa: BLE001
            pass

    def transcribe(self, audio_path: Path) -> str:
        # Greedy decoding (beam_size=1) keeps latency low for short commands.
        segments, _ = self._load().transcribe(str(audio_path), beam_size=1, vad_filter=True)
        return " ".join(segment.text.strip() for segment in segments).strip()

    def _load(self):
        from faster_whisper import WhisperModel

        with self._lock:
            if self._model is None:
                self._model = WhisperModel(self.model_name, device="auto", compute_type="default")
            return self._model


class Speaker(Protocol):
    """A text-to-speech engine. Swap Kokoro for another model by implementing this."""

    def warm_up(self) -> None: ...

    def speak(self, text: str) -> None: ...


class KokoroSpeaker:
    """Kokoro-82M: small, fast (~40x real time on a consumer GPU), streams by sentence."""

    def __init__(self, voice: str = "af_heart", language: str = "a") -> None:
        self.voice = voice
        self.language = language
        self._pipeline = None

    def warm_up(self) -> None:
        """Load the model and run CUDA kernels once, so the first reply is not slow."""
        for _ in self._synthesize("Ready."):
            pass

    def speak(self, text: str) -> None:
        """Play `text`, starting with the first sentence while the rest is synthesized."""
        if text.strip():
            _stream(self._synthesize(text), KOKORO_SAMPLE_RATE)

    def _synthesize(self, text: str) -> Iterator:
        from kokoro import KPipeline

        if self._pipeline is None:
            self._pipeline = KPipeline(lang_code=self.language, repo_id=KOKORO_REPO)
        for _, _, audio in self._pipeline(text, voice=self.voice):
            yield audio.numpy()


def _stream(chunks: Iterator, sample_rate: int) -> None:
    """Stream float32 mono audio to the sound server, falling back to PortAudio."""
    players = (
        ("paplay", ["--raw", "--format=float32le", f"--rate={sample_rate}", "--channels=1"]),
        ("aplay", ["-q", "-t", "raw", "-f", "FLOAT_LE", "-r", str(sample_rate), "-c", "1"]),
    )
    command = next(([path, *flags] for name, flags in players if (path := shutil.which(name))), None)
    if command is None:
        import sounddevice as sd

        with sd.OutputStream(samplerate=sample_rate, channels=1, dtype="float32") as stream:
            for chunk in chunks:
                stream.write(chunk)
        return
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    try:
        for chunk in chunks:
            process.stdin.write(chunk.astype("float32").tobytes())
    finally:
        process.stdin.close()
        process.wait()


class SpeechPlayer:
    """Plays sentences in order on a background thread, so Eva keeps working."""

    def __init__(self, speaker: Speaker, on_error: Callable[[Exception], None]) -> None:
        self._speaker = speaker
        self._on_error = on_error
        self._queue: queue.Queue[Callable[[], None]] = queue.Queue()
        threading.Thread(target=self._work, daemon=True, name="eva-speech").start()

    def warm_up(self) -> None:
        self._queue.put(self._speaker.warm_up)

    def play(self, text: str) -> None:
        self._queue.put(lambda: self._speaker.speak(text))

    def wait(self) -> None:
        self._queue.join()

    def _work(self) -> None:
        while True:
            job = self._queue.get()
            try:
                job()
            except (ImportError, OSError, ValueError) as exc:
                self._on_error(exc)
            finally:
                self._queue.task_done()
