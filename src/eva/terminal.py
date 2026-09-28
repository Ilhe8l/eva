"""Terminal interface.

Eva works on a background thread while the user keeps typing, and background
tasks run on threads of their own. The keyboard has a single reader
(`Console`), shared by messages and approval answers. A clock wakes Eva for
follow-ups and heartbeats.
"""

import asyncio
import contextlib
import difflib
import sys
import threading
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import Any

from prompt_toolkit import PromptSession
from prompt_toolkit.patch_stdout import patch_stdout

from eva.adapters.policy import ApprovalPolicy
from eva.adapters.voice import Microphone, SpeechPlayer, WhisperTranscriber
from eva.application.messages import RESUMED_MESSAGE, follow_up_message, task_report_message, voice_message
from eva.application.tasks import BackgroundTask
from eva.bootstrap import Application
from eva.domain.models import ActionRequest, AgentEvent, Speech, ToolUse, TurnCancelled

YES = {"y", "yes", "s", "sim"}
ALWAYS = {"a", "always", "sempre"}
HELP = (
    "Commands: :record (talk, then Enter), :speak on|off, :shh (stop talking), "
    ":stop (stop working), :tasks, :cancel ID, :auto on|off, :quit"
)
AUTONOMOUS_WARNING = "Autonomous mode: Eva runs commands and edits files without asking. `:auto off` to stop."
PREVIEW_LINES = 40
SUMMARY_WIDTH = 100
TICK_SECONDS = 15


def now() -> datetime:
    return datetime.now().astimezone()


async def in_daemon_thread(function: Callable[..., Any], *args: Any) -> Any:
    """Run `function` on a daemon thread, so quitting never waits for it."""
    loop = asyncio.get_running_loop()
    future = loop.create_future()

    def settle(setter: Callable[[Any], None], value: Any) -> None:
        if not future.cancelled():
            setter(value)

    def target() -> None:
        try:
            outcome = (future.set_result, function(*args))
        except BaseException as exc:  # noqa: BLE001 - re-raised in the awaiting task
            outcome = (future.set_exception, exc)
        with contextlib.suppress(RuntimeError):  # the loop may already be closed
            loop.call_soon_threadsafe(settle, *outcome)

    threading.Thread(target=target, daemon=True).start()
    return await future


class Console:
    """The only reader of the keyboard. Worker threads ask questions through it."""

    def __init__(self) -> None:
        self._interactive = sys.stdin.isatty()
        self._session: PromptSession[str] | None = PromptSession() if self._interactive else None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._answer: asyncio.Future[str] | None = None
        self._question = ""
        self._closed = False
        self._asking = threading.Lock()  # one question at a time, even from parallel tasks

    def bind(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    async def read(self) -> str:
        if self._session is not None:
            return await self._session.prompt_async(lambda: self._question or "You> ")
        line = await in_daemon_thread(sys.stdin.readline)
        if not line:
            raise EOFError
        return line

    def answer(self, line: str) -> bool:
        """Deliver `line` to a pending question; False if nothing is being asked."""
        if self._answer is None or self._answer.done():
            return False
        self._answer.set_result(line)
        return True

    def ask(self, question: str, context: str = "") -> str:
        """Ask from a worker thread and block until the user answers.

        `context` is printed right before the question, inside the same turn,
        so parallel tasks never interleave their prompts.
        """
        with self._asking:
            if self._closed or self._loop is None:
                return ""
            if context:
                print(context)
            try:
                return asyncio.run_coroutine_threadsafe(self._ask(question), self._loop).result()
            except (RuntimeError, asyncio.CancelledError):
                return ""

    def close(self) -> None:
        self._closed = True
        self.answer("")

    async def _ask(self, question: str) -> str:
        self._answer = asyncio.get_running_loop().create_future()
        self._question = question
        self._redraw()
        if not self._interactive:
            print(question, end="", flush=True)
        try:
            return await self._answer
        finally:
            self._answer = None
            self._question = ""
            self._redraw()

    def _redraw(self) -> None:
        if self._session is not None and self._session.app.is_running:
            self._session.app.invalidate()


class ConsoleApproval:
    def __init__(self, console: Console, policy: ApprovalPolicy) -> None:
        self.console = console
        self.policy = policy

    def approve(self, action: ActionRequest) -> bool:
        header = f"\n── {speaker_label(action.source)} wants to run {action.name} ──"
        question = "Approve? [y]es / [N]o / [a]lways this session: "
        answer = self.console.ask(question, context=f"{header}\n{describe(action)}").strip().lower()
        if answer in ALWAYS:
            self.policy.always_allow(action.name, action.arguments)
        return answer in YES | ALWAYS


def describe(action: ActionRequest) -> str:
    args = action.arguments
    if action.name == "execute":
        return f"  $ {args.get('command', '')}"
    if action.name == "edit_file":
        diff = difflib.unified_diff(
            str(args.get("old_string", "")).splitlines(),
            str(args.get("new_string", "")).splitlines(),
            fromfile=args.get("file_path", ""),
            tofile=args.get("file_path", ""),
            lineterm="",
        )
        return _preview("\n".join(diff))
    if action.name == "write_file":
        return f"  file: {args.get('file_path', '')}\n{_preview(str(args.get('content', '')))}"
    if action.name in {"read_file", "delete"}:
        return f"  file: {args.get('file_path', '')}"
    return _preview("\n".join(f"  {key}: {value}" for key, value in args.items()))


def _preview(text: str) -> str:
    lines = text.splitlines()
    if len(lines) <= PREVIEW_LINES:
        return text
    return "\n".join([*lines[:PREVIEW_LINES], f"... ({len(lines) - PREVIEW_LINES} more lines)"])


class ProgressView:
    """Shows what Eva is doing and plays what she says, as it happens."""

    def __init__(self, player: SpeechPlayer) -> None:
        self.player = player

    def __call__(self, event: AgentEvent) -> None:
        match event:
            case Speech(text, source):
                print(f"\n{speaker_label(source)} (aloud)> {text}")
                self.player.play(text)
            case ToolUse(name, arguments, source):
                print(f"  {'[' + source + '] ' if source else ''}· {summarize(name, arguments)}")


def speaker_label(source: str | None) -> str:
    return f"Eva [{source}]" if source else "Eva"


def summarize(name: str, arguments: dict) -> str:
    keys = ("command", "file_path", "pattern", "path", "description")
    detail = next((str(arguments[key]) for key in keys if arguments.get(key)), "")
    line = f"{name} {detail}".strip().replace("\n", " ")
    return line if len(line) <= SUMMARY_WIDTH else f"{line[: SUMMARY_WIDTH - 1]}…"


class TurnKind(Enum):
    USER = auto()
    FOLLOW_UP = auto()
    HEARTBEAT = auto()
    SYSTEM = auto()


@dataclass(frozen=True)
class Turn:
    kind: TurnKind
    message: str = ""


class Terminal:
    def __init__(
        self,
        app: Application,
        console: Console,
        transcriber: WhisperTranscriber,
        player: SpeechPlayer,
        heartbeat: timedelta | None,
    ) -> None:
        self.app = app
        self.console = console
        self.transcriber = transcriber
        self.player = player
        self.microphone = Microphone()
        self.heartbeat = heartbeat
        self._turns: asyncio.Queue[Turn] = asyncio.Queue()
        self._stop = asyncio.Event()
        self._busy = False
        self._last_activity = now()
        self._recording: asyncio.Task | None = None

    @property
    def restart(self) -> bool:
        return self.app.updater.restart_requested

    async def run(self, resumed: bool = False) -> None:
        loop = asyncio.get_running_loop()
        self.console.bind(loop)
        self.app.tasks.on_finish = lambda task: loop.call_soon_threadsafe(self._task_finished, task)
        print(f"Eva is ready. {HELP}")
        if self.app.policy.autonomous:
            print(AUTONOMOUS_WARNING)
        if resumed:
            await self._turns.put(Turn(TurnKind.SYSTEM, RESUMED_MESSAGE))
        with patch_stdout(raw=True):
            tasks = [asyncio.create_task(job) for job in (self._read(), self._work(), self._tick())]
            await self._stop.wait()
            self.app.tasks.on_finish = lambda task: None  # the loop is about to close
            self.console.close()
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _read(self) -> None:
        while True:
            try:
                line = (await self.console.read()).strip()
            except (EOFError, KeyboardInterrupt):
                break
            if self.console.answer(line) or not line:
                continue
            if line in {":quit", ":exit"}:
                break
            self._last_activity = now()
            if line.startswith(":"):
                self._command(line)
            else:
                await self._submit(line)
        self._stop.set()

    async def _submit(self, message: str) -> None:
        if self._busy:
            print("(Eva will read this after her current task.)")
        await self._turns.put(Turn(TurnKind.USER, message))

    def _task_finished(self, task: BackgroundTask) -> None:
        print(f"\n[task {task.id} {task.status.value}: {task.title}]")
        report = task_report_message(task.id, task.title, task.status.value, task.report)
        self._turns.put_nowait(Turn(TurnKind.SYSTEM, report))

    async def _work(self) -> None:
        while True:
            turn = await self._turns.get()
            self._busy = True
            try:
                await in_daemon_thread(self._take, turn)
            finally:
                self._busy = False
                self._last_activity = now()
            if self.restart:
                self._stop.set()

    async def _tick(self) -> None:
        while True:
            await asyncio.sleep(TICK_SECONDS)
            moment = now()
            for item in self.app.follow_ups.pop_due(moment):
                await self._turns.put(Turn(TurnKind.FOLLOW_UP, follow_up_message(item.note, item.created)))
            idle = not self._busy and self._turns.empty()
            if self.heartbeat and idle and moment - self._last_activity >= self.heartbeat:
                self._last_activity = moment
                await self._turns.put(Turn(TurnKind.HEARTBEAT))

    def _take(self, turn: Turn) -> None:
        """Run one turn on the worker thread and print the outcome."""
        session = self.app.session
        try:
            if turn.kind is TurnKind.HEARTBEAT:
                reply = session.heartbeat(now())
            else:
                reply = session.send(turn.message)
        except TurnCancelled:
            print("(Stopped.)")
            return
        except Exception as exc:  # noqa: BLE001 - keep Eva alive on model errors
            print(f"Eva error: {exc}")
            return
        if reply is None:
            return
        print(f"\nEva> {reply}")
        if session.denied_actions:
            print(f"[Declined: {', '.join(session.denied_actions)}]")
        if turn.kind is TurnKind.USER:
            self.app.summarizer.record(turn.message, reply)

    def _command(self, line: str) -> None:
        name, _, argument = line.partition(" ")
        argument = argument.strip()
        if name == ":record":
            self.player.stop()  # the user starts talking: Eva stops
            # Record in the background: the keyboard reader must stay free to see Enter.
            if self._recording is None or self._recording.done():
                self._recording = asyncio.create_task(self._record())
        elif name == ":shh":
            self.player.stop()
        elif name == ":stop":
            self.player.stop()
            self.app.session.cancel()
            print("Stopping after the current step..." if self._busy else "Nothing to stop.")
        elif name == ":auto" and argument in {"on", "off"}:
            self.app.policy.autonomous = argument == "on"
            print(AUTONOMOUS_WARNING if self.app.policy.autonomous else "Eva asks before acting again.")
        elif name == ":tasks":
            tasks = self.app.tasks.all()
            print("\n".join(f"  {task.id} [{task.status.value}] {task.title}" for task in tasks) or "No tasks.")
        elif name == ":cancel" and argument:
            print("Cancelling..." if self.app.tasks.cancel(argument) else "No running task with that id.")
        elif name == ":speak" and argument in {"on", "off"}:
            self.app.speech.enabled = argument == "on"
            if self.app.speech.enabled:
                self.player.warm_up()
            print(f"Speech {argument}.")
        else:
            print(HELP)

    async def _record(self) -> None:
        message = await in_daemon_thread(self._listen)
        if message:
            await self._submit(message)

    def _listen(self) -> str | None:
        """Record until Enter, transcribe, and show the user what Eva will read."""
        try:
            path = self.microphone.record(until=lambda: self.console.ask("Listening... press Enter to stop. "))
            print("Transcribing...")
            try:
                text = self.transcriber.transcribe(path)
            finally:
                path.unlink(missing_ok=True)
        except Exception as exc:  # noqa: BLE001 - audio devices and model downloads fail in many ways
            print(f"Voice input unavailable: {exc}")
            return None
        if not text:
            print("(Nothing understood.)")
            return None
        print(f"You (voice)> {text}")
        return voice_message(text)
