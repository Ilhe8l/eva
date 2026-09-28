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
import warnings
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
    def __init__(self, model_name: str = "large-v3-turbo", notify: Callable[[str], None] = lambda message: None) -> None:
        self.model_name = model_name
        self.notify = notify
        self._model = None
        self._lock = threading.Lock()

    def warm_up(self) -> None:
        """Load the model ahead of the first recording; errors surface on use instead."""
        try:
            self._load()
        except Exception:  # noqa: BLE001
            pass

    def transcribe(self, audio_path: Path) -> str:
        if self._model is None and self._lock.locked():
            self.notify("Waiting for the speech recognition model to finish loading...")
        # Greedy decoding (beam_size=1) keeps latency low for short commands.
        segments, _ = self._load().transcribe(str(audio_path), beam_size=1, vad_filter=True)
        return " ".join(segment.text.strip() for segment in segments).strip()

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
            model = model_class(self.model_name, device="cuda", compute_type="float16")
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
        threading.Thread(target=self._work, daemon=True, name="eva-speech").start()

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
            try:
                job()
            except (ImportError, OSError, ValueError) as exc:
                self._on_error(exc)
            finally:
                self._queue.task_done()
