import tempfile
from pathlib import Path


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
            self._model = WhisperModel(
                self.model_name, device="auto", compute_type="auto"
            )
        segments, _ = self._model.transcribe(audio_path, beam_size=5, vad_filter=True)
        return " ".join(segment.text.strip() for segment in segments).strip()


class KokoroSpeaker:
    def __init__(self, voice: str = "pf_dora", language: str = "p") -> None:
        self.voice = voice
        self.language = language
        self._pipeline = None

    def speak(self, text: str) -> None:
        import sounddevice as sd
        from kokoro import KPipeline

        if not text.strip():
            return
        if self._pipeline is None:
            self._pipeline = KPipeline(lang_code=self.language)
        for _, _, audio in self._pipeline(text, voice=self.voice):
            sd.play(audio, samplerate=24000)
            sd.wait()
