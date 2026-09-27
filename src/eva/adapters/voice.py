"""Microphone capture, speech recognition and speech synthesis, all local.

Model APIs: https://github.com/SYSTRAN/faster-whisper and
https://github.com/hexgrad/kokoro (voices:
https://huggingface.co/hexgrad/Kokoro-82M/blob/main/VOICES.md).
Heavy imports are deferred so a text-only install still works.
"""

import shutil
import subprocess
import tempfile
import threading
from pathlib import Path

MAX_SPOKEN_CHARACTERS = 1200
KOKORO_REPO = "hexgrad/Kokoro-82M"
KOKORO_SAMPLE_RATE = 24000
MICROPHONE_SAMPLE_RATE = 16000


class SpeechOutbox:
    """Holds the exact sentences Eva chose to say aloud."""

    def __init__(self) -> None:
        self.enabled = False
        self._messages: list[str] = []

    def enqueue(self, text: str) -> str:
        if not self.enabled:
            return "Speech is off; the user reads your text reply instead."
        spoken = text.strip()
        if not spoken:
            return "Nothing was spoken because the text was empty."
        if len(spoken) > MAX_SPOKEN_CHARACTERS:
            return f"Not spoken: keep speech under {MAX_SPOKEN_CHARACTERS} characters."
        self._messages.append(spoken)
        return "Spoken to the user."

    def drain(self) -> list[str]:
        messages, self._messages = self._messages, []
        return messages


class Microphone:
    def record(self, seconds: float = 6.0) -> Path:
        import sounddevice as sd
        import soundfile as sf

        if not 0.5 <= seconds <= 60:
            raise ValueError("Recording length must be between 0.5 and 60 seconds")
        audio = sd.rec(
            int(seconds * MICROPHONE_SAMPLE_RATE),
            samplerate=MICROPHONE_SAMPLE_RATE,
            channels=1,
            dtype="float32",
        )
        sd.wait()
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as file:
            path = Path(file.name)
        sf.write(path, audio, MICROPHONE_SAMPLE_RATE)
        return path


class WhisperTranscriber:
    def __init__(self, model_name: str = "small") -> None:
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
        self._lock = threading.Lock()

    def speak(self, text: str) -> None:
        """Synthesize and play `text`, one utterance at a time."""
        import numpy as np
        from kokoro import KPipeline

        if not text.strip():
            return
        with self._lock:
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
