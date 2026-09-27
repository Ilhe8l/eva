import asyncio

import pytest

from eva.terminal import Console, in_daemon_thread, summarize


def test_worker_thread_questions_are_answered_by_the_next_line():
    async def scenario():
        console = Console()
        console.bind(asyncio.get_running_loop())
        pending = asyncio.ensure_future(in_daemon_thread(console.ask, "Approve? "))
        while not console.answer("y"):
            await asyncio.sleep(0.01)
        return await pending

    assert asyncio.run(scenario()) == "y"


def test_closed_console_declines_instead_of_blocking():
    console = Console()
    console.close()
    assert console.ask("Approve? ") == ""


def test_worker_errors_reach_the_awaiting_task():
    def fail():
        raise ValueError("boom")

    with pytest.raises(ValueError, match="boom"):
        asyncio.run(in_daemon_thread(fail))


def test_progress_lines_are_short():
    assert summarize("execute", {"command": "ls -la"}) == "execute ls -la"
    assert len(summarize("write_file", {"file_path": "/" + "x" * 300})) == 100
