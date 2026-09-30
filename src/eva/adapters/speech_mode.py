"""Tells the model, on every call, what time it is, whether the user is listening and when to post an update."""

import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime

from langchain.agents.middleware import AgentMiddleware, ModelRequest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.config import get_config

from eva.adapters.voice import SpeechChannel

SPEECH_ON = (
    "Speech output is ON: the user is listening. Talk with speak_to_user as a person would, in "
    "English: for conversation, say your whole answer; for technical results, a spoken summary, "
    "leaving details to the text reply."
)
SPEECH_OFF = (
    "Speech output is OFF: speak_to_user shows your words as a short status line on screen. Use "
    "it for progress updates on longer work, not for your answers."
)
KICKOFF_NUDGE = (
    "[reminder] If this will take more than a couple of steps, call speak_to_user now, before any "
    "other tool, with one short sentence on what you are about to do. If you are about to answer, "
    "just answer."
)
PROGRESS_NUDGE = (
    "[reminder] The user has heard nothing from you for a while. Call speak_to_user now, before "
    "any other tool, with a one- or two-sentence update: what you finished or found (numbers "
    "help) and what comes next. Not just that you are still working."
)


@dataclass(frozen=True)
class UpdatePace:
    """How long Eva may work silently before she is nudged for an update."""

    quiet_seconds: float
    max_silent_steps: int


CONVERSATION_PACE = UpdatePace(quiet_seconds=20, max_silent_steps=6)
BACKGROUND_PACE = UpdatePace(quiet_seconds=60, max_silent_steps=10)  # tasks run beside the chat


def local_now() -> datetime:
    return datetime.now().astimezone()


def time_note(now: datetime) -> str:
    """The date and time in the user's time zone, to the minute so the prompt stays cacheable."""
    offset = now.strftime("%z")
    return f"Now: {now:%A, %Y-%m-%d %H:%M} (UTC{offset[:3]}:{offset[3:]})."


class SpeechModeMiddleware(AgentMiddleware):
    def __init__(
        self,
        speech: SpeechChannel,
        clock: Callable[[], float] = time.monotonic,
        now: Callable[[], datetime] = local_now,
    ) -> None:
        super().__init__()
        self.speech = speech
        self._clock = clock
        self._now = now
        self._quiet_since: dict[str, float] = {}

    def wrap_model_call(self, request: ModelRequest, handler):
        return handler(self._with_guidance(request))

    async def awrap_model_call(self, request: ModelRequest, handler):
        return await handler(self._with_guidance(request))

    def _with_guidance(self, request: ModelRequest) -> ModelRequest:
        mode = SPEECH_ON if self.speech.enabled else SPEECH_OFF
        blocks = list(request.system_message.content_blocks) if request.system_message else []
        guidance = f"{time_note(self._now())}\n{mode}"
        system = SystemMessage(content=[*blocks, {"type": "text", "text": guidance}])
        messages = request.messages
        if nudge := self.nudge(messages, _thread_id()):
            # Models heed the latest message far more than the end of a long system prompt.
            # The reminder only exists in this request; it is never saved to the thread.
            messages = [*messages, HumanMessage(content=nudge)]
        return request.override(system_message=system, messages=messages)

    def nudge(self, messages: list, thread_id: str) -> str | None:
        """The update Eva should post before her next step, if any.

        A turn that turns into tool work gets one kickoff nudge. After that, she is
        nudged when she has been quiet too long or for too many steps; an update
        resets both.
        """
        now = self._clock()
        steps = silent_steps(messages)
        if steps == 0:
            self._quiet_since[thread_id] = now
            return None
        pace = BACKGROUND_PACE if "-task-" in thread_id else CONVERSATION_PACE
        quiet = now - self._quiet_since.setdefault(thread_id, now)
        if steps == 1 and pace is CONVERSATION_PACE and not spoke_this_turn(messages):
            return KICKOFF_NUDGE
        if quiet >= pace.quiet_seconds or steps >= pace.max_silent_steps:
            return PROGRESS_NUDGE
        return None


def _thread_id() -> str:
    try:
        return str(get_config().get("configurable", {}).get("thread_id", ""))
    except RuntimeError:  # outside a graph run
        return ""


def _turn_steps(messages: list) -> list[AIMessage]:
    """Tool-calling steps since the user's last message, newest first."""
    steps = []
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            break
        if isinstance(message, AIMessage) and message.tool_calls:
            steps.append(message)
    return steps


def _speaks(message: AIMessage) -> bool:
    return any(call["name"] == "speak_to_user" for call in message.tool_calls)


def silent_steps(messages: list) -> int:
    """Tool-calling steps since the user's last message or Eva's last update."""
    count = 0
    for message in _turn_steps(messages):
        if _speaks(message):
            break
        count += 1
    return count


def spoke_this_turn(messages: list) -> bool:
    return any(_speaks(message) for message in _turn_steps(messages))
