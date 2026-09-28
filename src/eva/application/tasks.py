"""Background work Eva runs in parallel with the conversation."""

import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum

from eva.application.messages import background_brief
from eva.application.session import EvaSession
from eva.domain.models import TurnCancelled

MAX_RUNNING = 3


class TaskStatus(Enum):
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class BackgroundTask:
    id: str
    title: str
    instructions: str
    status: TaskStatus = TaskStatus.RUNNING
    report: str = ""
    session: EvaSession | None = field(default=None, repr=False)
    thread: threading.Thread | None = field(default=None, repr=False)


class TaskBoard:
    """Starts each task on its own agent thread and reports when it ends.

    `open_session` gives a task its own conversation thread, so tasks never
    see or disturb the main conversation. `on_finish` is called from the task's
    thread.
    """

    def __init__(self, open_session: Callable[[BackgroundTask], EvaSession]) -> None:
        self._open_session = open_session
        self.on_finish: Callable[[BackgroundTask], None] = lambda task: None
        self._tasks: dict[str, BackgroundTask] = {}
        self._lock = threading.Lock()

    def start(self, title: str, instructions: str) -> BackgroundTask:
        with self._lock:
            if len(self.running()) >= MAX_RUNNING:
                raise ValueError(f"Already running {MAX_RUNNING} background tasks; wait or cancel one")
            task = BackgroundTask(id=uuid.uuid4().hex[:6], title=title.strip(), instructions=instructions.strip())
            self._tasks[task.id] = task
        task.session = self._open_session(task)
        task.thread = threading.Thread(target=self._run, args=(task,), daemon=True, name=f"eva-task-{task.id}")
        task.thread.start()
        return task

    def shutdown(self, timeout: float = 10) -> None:
        """Cancel running tasks and give them a moment to stop before Eva exits."""
        self.on_finish = lambda task: None
        running = self.running()
        for task in running:
            task.session.cancel()
        deadline = time.monotonic() + timeout
        for task in running:
            task.thread.join(max(0.0, deadline - time.monotonic()))

    def cancel(self, task_id: str) -> bool:
        task = self._tasks.get(task_id)
        if task is None or task.status is not TaskStatus.RUNNING or task.session is None:
            return False
        task.session.cancel()
        return True

    def running(self) -> list[BackgroundTask]:
        return [task for task in self._tasks.values() if task.status is TaskStatus.RUNNING]

    def all(self) -> list[BackgroundTask]:
        return list(self._tasks.values())

    def _run(self, task: BackgroundTask) -> None:
        try:
            task.report = task.session.send(background_brief(task.id, task.title, task.instructions))
            task.status = TaskStatus.DONE
        except TurnCancelled:
            task.report, task.status = "Cancelled before finishing.", TaskStatus.CANCELLED
        except Exception as exc:  # noqa: BLE001 - a failed task is reported, not raised
            task.report, task.status = f"Failed: {exc}", TaskStatus.FAILED
        self.on_finish(task)
