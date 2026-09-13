from typing import Protocol

from eva.domain.models import ActionRequest, AgentStep


class AgentPort(Protocol):
    def ask(self, message: str) -> AgentStep: ...

    def resume(self, decisions: list[dict[str, str]]) -> AgentStep: ...


class ApprovalPort(Protocol):
    def approve(self, action: ActionRequest) -> bool: ...


class SpeechToTextPort(Protocol):
    def transcribe(self, audio_path: str) -> str: ...


class TextToSpeechPort(Protocol):
    def speak(self, text: str) -> None: ...
