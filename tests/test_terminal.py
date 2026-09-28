import asyncio
import re

import pytest

from eva.domain.models import TextDelta
from eva.terminal import Console, ProgressView, Terminal, Turn, TurnKind, in_daemon_thread, split_blocks, summarize


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


class FakeMicrophone:
    def __init__(self, path):
        self.path = path

    def record(self, until):
        until()
        self.path.write_bytes(b"")
        return self.path


class SilentPlayer:
    def stop(self):
        pass


class FakeTranscriber:
    def transcribe(self, path):
        return "olá, eva"


def test_enter_stops_a_recording_and_sends_the_transcript(tmp_path):
    async def scenario():
        console = Console()
        console.bind(asyncio.get_running_loop())
        terminal = Terminal(
            app=None, console=console, transcriber=FakeTranscriber(), player=SilentPlayer(), heartbeat=None
        )
        terminal.microphone = FakeMicrophone(tmp_path / "take.wav")
        terminal._command(":record")
        while not console.answer(""):  # the user presses Enter
            await asyncio.sleep(0.01)
        await asyncio.wait_for(terminal._recording, timeout=2)
        return terminal._turns.get_nowait()

    turn = asyncio.run(scenario())
    assert turn == Turn(TurnKind.USER, "[voice] olá, eva")


def plain(text):
    return re.sub(r"\x1b\[[0-9;]*m", "", text)


def test_reply_markdown_is_rendered_block_by_block(capsys):
    view = ProgressView(player=SilentPlayer())
    for piece in ["Hello **the", "re**.\n", "\n- one\n- two"]:
        view(TextDelta(piece))
    assert plain(capsys.readouterr().out) == "\nEva> Hello there.\n"
    assert view.end_turn() is True
    assert [line.strip() for line in plain(capsys.readouterr().out).splitlines()] == ["• one", "• two"]
    assert view.end_turn() is False


def test_code_blocks_are_not_split_at_blank_lines():
    blocks, rest = split_blocks("Intro.\n\n```py\nx = 1\n\ny = 2\n```\n\nOutro")
    assert blocks == ["Intro.", "```py\nx = 1\n\ny = 2\n```"]
    assert rest == "Outro"


def test_muted_text_is_not_shown(capsys):
    view = ProgressView(player=SilentPlayer())
    view.muted = True
    view(TextDelta("HEARTBEAT_OK"))
    assert view.end_turn() is False
    assert capsys.readouterr().out == ""
