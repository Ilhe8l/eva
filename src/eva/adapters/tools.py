"""Eva-specific tools added to the Deep Agents built-ins."""

from datetime import datetime, timedelta

from langchain_core.tools import BaseTool, tool
from langgraph.config import get_stream_writer

from eva.adapters.followups import FollowUpStore
from eva.adapters.lifecycle import SelfUpdater
from eva.adapters.voice import SpeechChannel


def build_tools(speech: SpeechChannel, updater: SelfUpdater, follow_ups: FollowUpStore) -> list[BaseTool]:
    @tool
    def speak_to_user(text: str) -> str:
        """Say one or two short sentences aloud to the user, right now, even mid-task."""
        return speech.say(text, get_stream_writer())

    @tool
    def restart_eva() -> str:
        """Run the test suite and, if it passes, restart Eva to load her edited source."""
        return updater.request_restart()

    @tool
    def schedule_follow_up(minutes: int, note: str) -> str:
        """Wake yourself up after `minutes` with `note` as the message, e.g. to check on a
        long-running job or remind the user of something. Works across restarts."""
        try:
            item = follow_ups.add(timedelta(minutes=minutes), note, datetime.now().astimezone())
        except ValueError as exc:
            return f"Not scheduled: {exc}"
        return f"Scheduled follow-up {item.id} for {item.due:%Y-%m-%d %H:%M}.\n{_list(follow_ups)}"

    @tool
    def cancel_follow_up(follow_up_id: str) -> str:
        """Cancel a follow-up you scheduled earlier."""
        cancelled = follow_ups.cancel(follow_up_id)
        return f"{'Cancelled' if cancelled else 'No follow-up with that id'}.\n{_list(follow_ups)}"

    return [speak_to_user, restart_eva, schedule_follow_up, cancel_follow_up]


def _list(follow_ups: FollowUpStore) -> str:
    items = follow_ups.pending()
    if not items:
        return "No follow-ups pending."
    return "Pending follow-ups:\n" + "\n".join(f"- {item.id} at {item.due:%Y-%m-%d %H:%M}: {item.note}" for item in items)
