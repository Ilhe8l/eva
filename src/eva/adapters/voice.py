"""Microphone capture, speech recognition and speech synthesis, all local.

Model APIs: https://github.com/SYSTRAN/faster-whisper and
https://github.com/hexgrad/kokoro (voices:
https://huggingface.co/hexgrad/Kokoro-82M/blob/main/VOICES.md).
Heavy imports are deferred so a text-only install still works.
"""

import contextlib
import ctypes
import importlib.util
import queue
import shutil
import subprocess
import tempfile
import threading
import time
import warnings
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Protocol

MAX_SPOKEN_CHARACTERS = 1200
KOKORO_REPO = "hexgrad/Kokoro-82M"
KOKORO_SAMPLE_RATE = 24000
MICROPHONE_SAMPLE_RATE = 16000
VOICE_MODULES = ("numpy", "sounddevice", "faster_whisper", "kokoro")
VOICE_MISSING = "Voice is not installed. Run `uv sync --extra voice`, then restart Eva."
MAX_NO_SPEECH_PROB = 0.6
ECHO_SECONDS = 0.5
SPEECH_EVENT = "speech"


def voice_available() -> bool:
    """Whether the optional voice extra is installed."""
    return all(importlib.util.find_spec(name) is not None for name in VOICE_MODULES)


class SpeechChannel:
    """Carries what Eva chose to tell the user to the stream.

    With speech on it is said aloud; with speech off it is shown as a status line.
    """

    def __init__(self) -> None:
        self.enabled = False

    def say(self, text: str, emit: Callable[[dict], None]) -> str:
        spoken = text.strip()
        if not spoken:
            return "Nothing was said because the text was empty."
        if len(spoken) > MAX_SPOKEN_CHARACTERS:
            return f"Not said: keep it under {MAX_SPOKEN_CHARACTERS} characters."
        emit({"type": SPEECH_EVENT, "text": spoken, "aloud": self.enabled})
        return "Spoken to the user." if self.enabled else "Shown to the user as a status line (speech is off)."


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


class SileroSpeechClassifier:
    """Is this frame speech? Silero VAD, bundled with faster-whisper, run on CPU.

    The model is fed a short window of recent frames for context and judged on
    the newest one. Unlike a loudness threshold, it ignores fans, music and hum.
    """

    CONTEXT_FRAMES = 8

    def __init__(self, threshold: float = 0.5) -> None:
        from faster_whisper.vad import get_vad_model

        self._model = get_vad_model()
        self._threshold = threshold
        self._window: list = []

    def __call__(self, frame) -> bool:
        import numpy as np

        self._window = [*self._window, frame][-self.CONTEXT_FRAMES :]
        probabilities = self._model(np.concatenate(self._window))
        return float(np.ravel(probabilities)[-1]) > self._threshold


class UtteranceDetector:
    """Cuts a stream of frames into utterances, given a speech/non-speech classifier.

    An utterance starts after a few speech frames in a row and ends after a
    pause. A little audio before the start is kept, so first syllables are not
    clipped.
    """

    FRAME_SAMPLES = 512  # what Silero VAD expects at 16 kHz
    FRAME_SECONDS = FRAME_SAMPLES / MICROPHONE_SAMPLE_RATE

    def __init__(
        self,
        is_speech: Callable,
        start_frames: int = 3,
        pause_seconds: float = 0.8,
        min_seconds: float = 0.4,
        max_seconds: float = 30.0,
        preroll_seconds: float = 0.3,
    ) -> None:
        frames = lambda seconds: max(1, round(seconds / self.FRAME_SECONDS))  # noqa: E731
        self._is_speech = is_speech
        self._start_frames = start_frames
        self._pause_frames = frames(pause_seconds)
        self._min_frames = frames(min_seconds)
        self._max_frames = frames(max_seconds)
        self._preroll_frames = frames(preroll_seconds)
        self.reset()

    def reset(self) -> None:
        """Drop any partial utterance, e.g. while Eva is talking."""
        self._recent: list = []
        self._utterance: list = []
        self._onset = 0  # index where speech starts, after the pre-roll
        self._speech_run = 0
        self._pause_run = 0

    def feed(self, frame):
        """Add one frame of float32 samples; return a finished utterance or None."""
        import numpy as np

        speech = self._is_speech(frame)
        if not self._utterance:
            self._recent = [*self._recent, frame][-(self._preroll_frames + self._start_frames) :]
            self._speech_run = self._speech_run + 1 if speech else 0
            if self._speech_run >= self._start_frames:
                self._utterance, self._recent = self._recent, []
                self._onset = len(self._utterance) - self._start_frames
            return None
        self._utterance.append(frame)
        self._pause_run = 0 if speech else self._pause_run + 1
        if self._pause_run >= self._pause_frames or len(self._utterance) >= self._max_frames:
            utterance = self._utterance[: len(self._utterance) - self._pause_run]
            spoken_frames = len(utterance) - self._onset
            self.reset()
            return np.concatenate(utterance) if spoken_frames >= self._min_frames else None
        return None


class HandsFreeListener:
    """Listens continuously and hands each utterance to `on_utterance`.

    `is_muted()` is checked on every frame; while it is true (Eva is talking),
    audio is ignored so Eva does not hear herself.
    """

    def __init__(self, on_utterance: Callable, is_muted: Callable[[], bool]) -> None:
        self._on_utterance = on_utterance
        self._is_muted = is_muted
        self._running = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def active(self) -> bool:
        return self._running.is_set()

    def start(self) -> None:
        if not self.active:
            self._running.set()
            self._thread = threading.Thread(target=self._listen, daemon=True, name="eva-listen")
            self._thread.start()

    def stop(self) -> None:
        self._running.clear()

    def _listen(self) -> None:
        import sounddevice as sd

        detector = UtteranceDetector(SileroSpeechClassifier())
        frame_size = UtteranceDetector.FRAME_SAMPLES
        with sd.InputStream(
            samplerate=MICROPHONE_SAMPLE_RATE, channels=1, dtype="float32", blocksize=frame_size
        ) as stream:
            while self._running.is_set():
                frame, _ = stream.read(frame_size)
                if self._is_muted():
                    detector.reset()
                    continue
                utterance = detector.feed(frame[:, 0])
                if utterance is not None:
                    self._on_utterance(utterance)


class WhisperTranscriber:
    def __init__(
        self,
        model_name: str = "large-v3-turbo",
        compute_type: str = "int8_float16",
        notify: Callable[[str], None] = lambda message: None,
    ) -> None:
        self.model_name = model_name
        self.compute_type = compute_type  # precision on the GPU; the CPU fallback always uses int8
        self.notify = notify
        self._model = None
        self._lock = threading.Lock()

    def warm_up(self) -> None:
        """Load the model ahead of the first recording; errors surface on use instead."""
        try:
            self._load()
        except Exception:  # noqa: BLE001
            pass

    def transcribe(self, audio) -> str:
        """Text of `audio` (a file path or 16 kHz float32 samples), without guesses made on noise."""
        if self._model is None and self._lock.locked():
            self.notify("Waiting for the speech recognition model to finish loading...")
        source = str(audio) if isinstance(audio, Path) else audio
        # Greedy decoding (beam_size=1) keeps latency low for short commands.
        segments, _ = self._load().transcribe(source, beam_size=1, vad_filter=True)
        # On noise, Whisper tends to hallucinate "Thank you." with a high no-speech probability.
        spoken = [segment.text.strip() for segment in segments if segment.no_speech_prob < MAX_NO_SPEECH_PROB]
        return " ".join(spoken).strip()

    def _load(self):
        from faster_whisper import WhisperModel

        with self._lock:
            if self._model is None:
                if not self._is_downloaded():
                    self.notify(f"Downloading speech recognition model '{self.model_name}' (first run only)...")
                self._model = self._on_gpu(WhisperModel) or WhisperModel(
                    self.model_name, device="cpu", compute_type="int8"
                )
                self.notify("Speech recognition ready.")
            return self._model

    def _on_gpu(self, model_class):
        """The model on CUDA, or None when the GPU or its libraries are unavailable.

        CTranslate2 is built for CUDA 12. The pip wheel `nvidia-cublas-cu12`
        provides cuBLAS; it is loaded here because it is not on the linker path.
        """
        import numpy as np

        _load_cublas_12()
        try:
            model = model_class(self.model_name, device="cuda", compute_type=self.compute_type)
            segments, _ = model.transcribe(np.zeros(MICROPHONE_SAMPLE_RATE, dtype=np.float32), language="en")
            list(segments)  # runs the encoder now, so missing CUDA libraries fail here
        except (RuntimeError, ValueError) as exc:
            self.notify(f"Speech recognition runs on the CPU (slower): {exc}")
            return None
        return model

    def _is_downloaded(self) -> bool:
        from faster_whisper.utils import download_model

        try:
            download_model(self.model_name, local_files_only=True)
        except Exception:  # noqa: BLE001 - any failure means it is not usable offline yet
            return False
        return True


def _load_cublas_12() -> None:
    spec = importlib.util.find_spec("nvidia.cublas")
    if spec is None or not spec.submodule_search_locations:
        return
    lib_dir = Path(next(iter(spec.submodule_search_locations))) / "lib"
    for name in ("libcublasLt.so.12", "libcublas.so.12"):
        if (lib_dir / name).exists():
            ctypes.CDLL(str(lib_dir / name), mode=ctypes.RTLD_GLOBAL)


class Speaker(Protocol):
    """A text-to-speech engine. Swap Kokoro for another model by implementing this."""

    def warm_up(self) -> None: ...

    def speak(self, text: str) -> None: ...

    def stop(self) -> None: ...


class KokoroSpeaker:
    """Kokoro-82M: small, fast (~40x real time on a consumer GPU), streams by sentence."""

    def __init__(self, voice: str = "af_heart", language: str = "a") -> None:
        self.voice = voice
        self.language = language
        self._pipeline = None
        self._stopped = threading.Event()
        self._player: subprocess.Popen | None = None

    def warm_up(self) -> None:
        """Load the model and run CUDA kernels once, so the first reply is not slow."""
        for _ in self._synthesize("Ready."):
            pass

    def speak(self, text: str) -> None:
        """Play `text`, starting with the first sentence while the rest is synthesized."""
        self._stopped.clear()
        if text.strip():
            self._play(self._synthesize(text))

    def stop(self) -> None:
        """Cut the current utterance short; safe to call from any thread."""
        self._stopped.set()
        if (player := self._player) is not None:
            player.terminate()

    def _synthesize(self, text: str) -> Iterator:
        if self._pipeline is None:
            with warnings.catch_warnings():
                # Kokoro's model code triggers torch deprecation warnings on load.
                warnings.simplefilter("ignore", category=UserWarning)
                warnings.simplefilter("ignore", category=FutureWarning)
                from kokoro import KPipeline

                self._pipeline = KPipeline(lang_code=self.language, repo_id=KOKORO_REPO)
        for _, _, audio in self._pipeline(text, voice=self.voice):
            if self._stopped.is_set():
                return
            yield audio.numpy()

    def _play(self, chunks: Iterator) -> None:
        """Stream float32 mono audio to the sound server, falling back to PortAudio."""
        command = _player_command(KOKORO_SAMPLE_RATE)
        if command is None:
            import sounddevice as sd

            with sd.OutputStream(samplerate=KOKORO_SAMPLE_RATE, channels=1, dtype="float32") as stream:
                for chunk in chunks:
                    stream.write(chunk)
            return
        self._player = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
        try:
            for chunk in chunks:
                self._player.stdin.write(chunk.astype("float32").tobytes())
        except BrokenPipeError:
            pass  # stopped mid-sentence
        finally:
            with contextlib.suppress(BrokenPipeError):
                self._player.stdin.close()
            self._player.wait()
            self._player = None


def _player_command(sample_rate: int) -> list[str] | None:
    players = (
        ("paplay", ["--raw", "--format=float32le", f"--rate={sample_rate}", "--channels=1"]),
        ("aplay", ["-q", "-t", "raw", "-f", "FLOAT_LE", "-r", str(sample_rate), "-c", "1"]),
    )
    return next(([path, *flags] for name, flags in players if (path := shutil.which(name))), None)


class SpeechPlayer:
    """Plays sentences in order on a background thread, so Eva keeps working."""

    def __init__(self, speaker: Speaker, on_error: Callable[[Exception], None]) -> None:
        self._speaker = speaker
        self._on_error = on_error
        self._queue: queue.Queue[Callable[[], None]] = queue.Queue()
        self._generation = 0  # bumped by stop(), so sentences queued before it are skipped
        self._speaking = False
        self._finished_at = 0.0
        threading.Thread(target=self._work, daemon=True, name="eva-speech").start()

    @property
    def busy(self) -> bool:
        """True while speaking, with speech queued, or just after (the room still echoes)."""
        echoing = time.monotonic() - self._finished_at < ECHO_SECONDS
        return self._speaking or self._queue.unfinished_tasks > 0 or echoing

    def warm_up(self) -> None:
        self._queue.put(self._speaker.warm_up)

    def play(self, text: str) -> None:
        generation = self._generation
        self._queue.put(lambda: generation == self._generation and self._speaker.speak(text))

    def wait(self) -> None:
        self._queue.join()

    def stop(self) -> None:
        """Drop everything queued and cut off what is playing now."""
        self._generation += 1
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
            self._queue.task_done()
        self._speaker.stop()

    def _work(self) -> None:
        while True:
            job = self._queue.get()
            self._speaking = True
            try:
                job()
            except Exception as exc:  # noqa: BLE001 - a failed sentence must not silence Eva for good
                self._on_error(exc)
            finally:
                self._speaking = False
                self._finished_at = time.monotonic()
                self._queue.task_done()
