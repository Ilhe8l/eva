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
