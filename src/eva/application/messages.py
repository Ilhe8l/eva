"""How non-typed messages are framed for Eva: voice, timers and restarts."""

from datetime import datetime

HEARTBEAT_OK = "HEARTBEAT_OK"


def heartbeat_message(now: datetime) -> str:
    return (
        f"[heartbeat {now:%Y-%m-%d %H:%M}] The user has been quiet for a while. Check your journal "
        "for pending tasks or anything worth doing or saying now. If there is something, do it "
        f"(briefly; you may speak). If there is nothing, reply exactly {HEARTBEAT_OK} and nothing else."
    )


def follow_up_message(note: str, scheduled_at: datetime) -> str:
    return f"[follow-up you scheduled at {scheduled_at:%Y-%m-%d %H:%M}] {note}"


RESUMED_MESSAGE = "[restart] You were just restarted with your edited source. Confirm briefly and carry on."


def voice_message(transcript: str) -> str:
    return f"[voice] {transcript}"


def background_brief(task_id: str, title: str, instructions: str) -> str:
    return (
        f"[background task {task_id}: {title}] You are running in the background while the user keeps "
        "talking to you in the main conversation, which you cannot see. Work on this autonomously:\n"
        f"{instructions}\n"
        "Give short spoken progress updates on long work. Finish with a concise report of what you did "
        "and found; it is handed to the main conversation."
    )


def task_report_message(task_id: str, title: str, status: str, report: str) -> str:
    return f"[background task {task_id} '{title}' {status}] {report}\nTell the user the outcome briefly."


STEP_LIMIT_MESSAGE = (
    "[step limit] You used every step allowed for one turn and had to stop. Briefly tell the "
    "user what you did, what is left, and ask whether to continue."
)
