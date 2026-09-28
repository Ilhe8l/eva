"""Eva-specific tools added to the Deep Agents built-ins."""

from datetime import datetime, timedelta

from langchain_core.tools import BaseTool, tool
from langgraph.config import get_stream_writer

from eva.adapters.followups import FollowUpStore
from eva.adapters.lifecycle import SelfUpdater
from eva.adapters.voice import SpeechChannel
from eva.application.tasks import TaskBoard


def build_tools(
    speech: SpeechChannel,
    updater: SelfUpdater,
    follow_ups: FollowUpStore,
    tasks: TaskBoard,
) -> list[BaseTool]:
    @tool
    def speak_to_user(text: str) -> str:
        """Say `text` aloud to the user right now, even mid-task. Speak naturally, as a person
        would: a whole conversational answer, a spoken summary of technical results, or a
        short progress update. Never code, paths or raw lists."""
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

    @tool
    def start_background_task(title: str, instructions: str) -> str:
        """Hand long work (big searches, builds, downloads, multi-step research) to a
        background copy of yourself, so you can keep talking with the user meanwhile.
        It cannot see this conversation: make `instructions` self-contained. Its report
        comes back to you as a message when it finishes."""
        try:
            task = tasks.start(title, instructions)
        except ValueError as exc:
            return f"Not started: {exc}"
        return f"Started background task {task.id}: {task.title}."

    @tool
    def list_background_tasks() -> str:
        """List background tasks and their status."""
        items = tasks.all()
        if not items:
            return "No background tasks."
        return "\n".join(f"- {task.id} [{task.status.value}] {task.title}" for task in items)

    @tool
    def cancel_background_task(task_id: str) -> str:
        """Stop a running background task."""
        return "Cancelling." if tasks.cancel(task_id) else "No running task with that id."

    return [
        speak_to_user,
        restart_eva,
        schedule_follow_up,
        cancel_follow_up,
        start_background_task,
        list_background_tasks,
        cancel_background_task,
    ]


def _list(follow_ups: FollowUpStore) -> str:
    items = follow_ups.pending()
    if not items:
        return "No follow-ups pending."
    return "Pending follow-ups:\n" + "\n".join(f"- {item.id} at {item.due:%Y-%m-%d %H:%M}: {item.note}" for item in items)
