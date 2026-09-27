import tempfile
from pathlib import Path


class SpeechOutbox:

    def __init__(self) -> None:
        self.enabled = False
        self._messages: list[str] = []
        self.on_speak = None

    def enqueue(self, text: str) -> str:
        if not self.enabled:
            return "Speech is disabled in the terminal. Continue with a text reply."
        spoken = text.strip()
        if not spoken:
            return "No speech was queued because the text was empty."
        if len(spoken) > 1200:
            return "Speech was not queued: keep spoken content under 1200 characters."
            
        if self.on_speak:
            self.on_speak(spoken)
            return "Status update spoken to user in real-time."
            
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
    def __init__(self, model_name: str = "tiny") -> None:
        self.model_name = model_name
        self._model = None

    def transcribe(self, audio_path: str) -> str:
        from faster_whisper import WhisperModel

        if self._model is None:
            # Full GPU available now!
            self._model = WhisperModel(
                self.model_name, device="cuda", compute_type="float16"
            )
        # beam_size=1 força decodificação "greedy" (imediata) para latência quase zero
        segments, _ = self._model.transcribe(audio_path, beam_size=1, vad_filter=True)
        return " ".join(segment.text.strip() for segment in segments).strip()


class KokoroSpeaker:
    def __init__(self, voice: str = "af_heart", language: str = "a") -> None:
        import threading
        self.voice = voice
        self.language = language
        self._pipeline = None
        self._lock = threading.Lock()

    def speak_async(self, text: str) -> None:
        import threading
        threading.Thread(target=self.speak, args=(text,), daemon=True).start()

    def speak(self, text: str) -> None:
        with self._lock:
            import os
            import tempfile
            import subprocess
            import numpy as np
            import soundfile as sf
            from kokoro import KPipeline
    
            if not text.strip():
                return
            if self._pipeline is None:
                # Força explicitamente a VRAM (CUDA) para máxima velocidade
                self._pipeline = KPipeline(lang_code=self.language, device="cuda")
    
            audio_chunks = []
            for _, _, audio in self._pipeline(text, voice=self.voice):
                audio_chunks.append(audio)
                
            if audio_chunks:
                full_audio = np.concatenate(audio_chunks)
                # salva num wav temporário e toca com o player do sistema
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                    temp_path = f.name
                
                sf.write(temp_path, full_audio, 24000)
                
                # Toca via player nativo do Linux (PulseAudio/PipeWire)
                import shutil
                if shutil.which("paplay"):
                    subprocess.run(["paplay", temp_path])
                elif shutil.which("aplay"):
                    subprocess.run(["aplay", "-q", temp_path])
                else:
                    import sounddevice as sd
                    sd.play(full_audio, samplerate=24000)
                    sd.wait()
                    
                os.remove(temp_path)
