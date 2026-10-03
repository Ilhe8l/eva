import threading

import pytest

from eva.application.tasks import MAX_RUNNING, TaskBoard, TaskStatus
from eva.domain.models import StepLimitReached, TurnCancelled


class FakeSession:
    def __init__(self, outcome, release=None):
        self.outcome = outcome
        self.release = release or threading.Event()
        self.cancelled = False

    def send(self, message):
        self.release.wait(timeout=2)
        if self.cancelled:
            raise TurnCancelled
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return self.outcome

    def cancel(self):
        self.cancelled = True
        self.release.set()


def _board(session_for):
    finished = []
    done = threading.Event()
    board = TaskBoard(lambda task: session_for(task))
    board.on_finish = lambda task: (finished.append(task), done.set())
    return board, finished, done


def test_a_finished_task_reports_back():
    release = threading.Event()
    board, finished, done = _board(lambda task: FakeSession("Found 3 large files.", release))
    task = board.start("Scan downloads", "Find large files in ~/Downloads")
    assert board.running() == [task]
    release.set()
    assert done.wait(2)
    assert finished[0].status is TaskStatus.DONE
    assert finished[0].report == "Found 3 large files."


def test_tasks_can_be_cancelled_and_failures_are_reported():
    board, finished, done = _board(lambda task: FakeSession("never"))
    task = board.start("Long job", "Wait forever")
    assert board.cancel(task.id)
    assert done.wait(2)
    assert finished[0].status is TaskStatus.CANCELLED

    board, finished, done = _board(lambda task: FakeSession(RuntimeError("disk full"), threading.Event()))
    board.start("Doomed", "Fail").session.release.set()
    assert done.wait(2)
    assert finished[0].status is TaskStatus.FAILED and "disk full" in finished[0].report


def test_running_tasks_are_capped():
    board, _, _ = _board(lambda task: FakeSession("x"))
    for index in range(MAX_RUNNING):
        board.start(f"task {index}", "work")
    with pytest.raises(ValueError, match="Already running"):
        board.start("one too many", "work")


def test_shutdown_cancels_running_tasks_quietly():
    board, finished, _ = _board(lambda task: FakeSession("never"))
    task = board.start("Long job", "Wait forever")
    board.shutdown(timeout=2)
    assert task.status is TaskStatus.CANCELLED
    assert finished == []


def test_a_task_that_runs_out_of_steps_says_so():
    board, finished, done = _board(lambda task: FakeSession(StepLimitReached(), threading.Event()))
    board.start("Huge job", "Explore everything").session.release.set()
    assert done.wait(2)
    assert finished[0].status is TaskStatus.FAILED and "step limit" in finished[0].report
