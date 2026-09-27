"""Follow-ups Eva schedules for herself, persisted across restarts."""

import json
import threading
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path

MAX_DELAY = timedelta(days=30)


@dataclass(frozen=True)
class FollowUp:
    id: str
    due: datetime
    created: datetime
    note: str


class FollowUpStore:
    """A small JSON-backed schedule; safe to use from the agent and timer threads."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()

    def add(self, delay: timedelta, note: str, now: datetime) -> FollowUp:
        if not timedelta(minutes=1) <= delay <= MAX_DELAY:
            raise ValueError("Schedule between 1 minute and 30 days ahead")
        if not note.strip():
            raise ValueError("A follow-up needs a note")
        item = FollowUp(id=uuid.uuid4().hex[:8], due=now + delay, created=now, note=note.strip())
        with self._lock:
            self._save([*self._load(), item])
        return item

    def cancel(self, follow_up_id: str) -> bool:
        with self._lock:
            items = self._load()
            remaining = [item for item in items if item.id != follow_up_id]
            self._save(remaining)
        return len(remaining) < len(items)

    def pending(self) -> list[FollowUp]:
        with self._lock:
            return sorted(self._load(), key=lambda item: item.due)

    def pop_due(self, now: datetime) -> list[FollowUp]:
        with self._lock:
            items = self._load()
            due = [item for item in items if item.due <= now]
            if due:
                self._save([item for item in items if item.due > now])
        return sorted(due, key=lambda item: item.due)

    def _load(self) -> list[FollowUp]:
        if not self.path.exists():
            return []
        return [
            FollowUp(
                id=raw["id"],
                due=datetime.fromisoformat(raw["due"]),
                created=datetime.fromisoformat(raw["created"]),
                note=raw["note"],
            )
            for raw in json.loads(self.path.read_text(encoding="utf-8"))
        ]

    def _save(self, items: list[FollowUp]) -> None:
        records = [{**asdict(item), "due": item.due.isoformat(), "created": item.created.isoformat()} for item in items]
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(records, indent=2), encoding="utf-8")
