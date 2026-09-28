"""Tells the model, on every call, whether the user is listening and when to speak up."""

from langchain.agents.middleware import AgentMiddleware, ModelRequest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from eva.adapters.voice import SpeechChannel

SPEECH_ON = (
    "Speech output is ON: the user is listening. Talk to them with speak_to_user as a person "
    "would, in English: for conversation, say your whole answer; for technical results, say a "
    "spoken summary and leave details to the text reply. On long work, keep them posted."
)
SPEECH_OFF = "Speech output is OFF: do not call speak_to_user."
SILENT_STEPS_BEFORE_UPDATE = 3
PROGRESS_NUDGE = (
    "You have been working for several steps without telling the user anything. Before going on, "
    "give a short spoken progress update: what you did, what you found, what comes next."
)


class SpeechModeMiddleware(AgentMiddleware):
    def __init__(self, speech: SpeechChannel) -> None:
        super().__init__()
        self.speech = speech

    def wrap_model_call(self, request: ModelRequest, handler):
        return handler(self._with_guidance(request))

    async def awrap_model_call(self, request: ModelRequest, handler):
        return await handler(self._with_guidance(request))

    def _with_guidance(self, request: ModelRequest) -> ModelRequest:
        notes = [SPEECH_ON if self.speech.enabled else SPEECH_OFF]
        if self.speech.enabled and silent_steps(request.messages) >= SILENT_STEPS_BEFORE_UPDATE:
            notes.append(PROGRESS_NUDGE)
        blocks = list(request.system_message.content_blocks) if request.system_message else []
        text_blocks = [{"type": "text", "text": note} for note in notes]
        return request.override(system_message=SystemMessage(content=[*blocks, *text_blocks]))


def silent_steps(messages: list) -> int:
    """Tool-calling steps since the user's last message or Eva's last spoken words."""
    count = 0
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            break
        if isinstance(message, AIMessage) and message.tool_calls:
            if any(call["name"] == "speak_to_user" for call in message.tool_calls):
                break
            count += 1
    return count
