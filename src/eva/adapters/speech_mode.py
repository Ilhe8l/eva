"""Tells the model, on every call, whether the user is listening."""

from langchain.agents.middleware import AgentMiddleware, ModelRequest
from langchain_core.messages import SystemMessage

from eva.adapters.voice import SpeechChannel

SPEECH_ON = (
    "Speech output is ON: the user is listening. For every reply, first call speak_to_user with "
    "the sentence a person would say out loud (the key point, in English), then write your text "
    "reply. During slow work, add short spoken updates."
)
SPEECH_OFF = "Speech output is OFF: do not call speak_to_user."


class SpeechModeMiddleware(AgentMiddleware):
    def __init__(self, speech: SpeechChannel) -> None:
        super().__init__()
        self.speech = speech

    def wrap_model_call(self, request: ModelRequest, handler):
        return handler(self._with_mode(request))

    async def awrap_model_call(self, request: ModelRequest, handler):
        return await handler(self._with_mode(request))

    def _with_mode(self, request: ModelRequest) -> ModelRequest:
        note = SPEECH_ON if self.speech.enabled else SPEECH_OFF
        blocks = list(request.system_message.content_blocks) if request.system_message else []
        return request.override(system_message=SystemMessage(content=[*blocks, {"type": "text", "text": note}]))
