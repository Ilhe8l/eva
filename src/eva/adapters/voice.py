import tempfile
from pathlib import Path


class SpeechOutbox:

    def __init__(self) -> None:
        self.enabled = False
        self._messages: list[str] = []

    def enqueue(self, text: str) -> str:
        if not self.enabled:
            return "Speech is disabled in the terminal. Continue with a text reply."
        spoken = text.strip()
        if not spoken:
            return "No speech was queued because the text was empty."
        if len(spoken) > 1200:
            return "Speech was not queued: keep spoken content under 1200 characters."
        self._messages.append(spoken)
        return "Speech queued for playback after this turn."

    def drain(self) -> list[str]:
        messages, self._messages = self._messages, []
        return messages


class Microphone:
    def record(self, seconds: float = 6.0) -> str:
        import sounddevice as sd
        import soundfile as sf

        if not 0.5 <= seconds <= 60:
            raise ValueError("Recording length must be between 0.5 and 60 seconds")
        sample_rate = 16000
        audio = sd.rec(
            int(seconds * sample_rate),
            samplerate=sample_rate,
            channels=1,
            dtype="float32",
        )
        sd.wait()
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as file:
            path = Path(file.name)
        sf.write(path, audio, sample_rate)
        return str(path)


class WhisperTranscriber:
    def __init__(self, model_name: str = "small") -> None:
        self.model_name = model_name
        self._model = None

    def transcribe(self, audio_path: str) -> str:
        from faster_whisper import WhisperModel

        if self._model is None:
            # Full GPU available now!
            self._model = WhisperModel(
                self.model_name, device="cuda", compute_type="float16"
            )
        segments, _ = self._model.transcribe(audio_path, beam_size=5, vad_filter=True)
        return " ".join(segment.text.strip() for segment in segments).strip()


class KokoroSpeaker:
    def __init__(self, voice: str = "af_heart", language: str = "a") -> None:
        self.voice = voice
        self.language = language
        self._pipeline = None

    def speak(self, text: str) -> None:
        import sounddevice as sd
        from kokoro import KPipeline

        if not text.strip():
            return
        if self._pipeline is None:
            # Uses CUDA now that VRAM is free
            self._pipeline = KPipeline(lang_code=self.language, repo_id="hexgrad/Kokoro-82M", device="cuda")

        # Texto puro, sem filtros artificiais
        for _, _, audio in self._pipeline(text, voice=self.voice):
            sd.play(audio, samplerate=24000)
            sd.wait()
