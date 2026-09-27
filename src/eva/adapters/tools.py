"""Eva-specific tools added to the Deep Agents built-ins."""

from langchain_core.tools import BaseTool, tool
from langgraph.config import get_stream_writer

from eva.adapters.lifecycle import SelfUpdater
from eva.adapters.voice import SpeechChannel


def build_tools(speech: SpeechChannel, updater: SelfUpdater) -> list[BaseTool]:
    @tool
    def speak_to_user(text: str) -> str:
        """Say one or two short sentences aloud to the user, right now, even mid-task."""
        return speech.say(text, get_stream_writer())

    @tool
    def restart_eva() -> str:
        """Run the test suite and, if it passes, restart Eva to load her edited source."""
        return updater.request_restart()

    return [speak_to_user, restart_eva]
