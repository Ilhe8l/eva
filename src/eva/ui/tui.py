"""Split-screen terminal interface: the conversation on the left, Eva's face
and her work (plan, files, commands) on the right.

Built with Textual (https://textual.textualize.io/guide/app/). The app runs on
the same asyncio loop as `Terminal`; worker threads reach it through
`loop.call_soon_threadsafe`, since Textual widgets are not thread-safe.
"""

import asyncio
import re
import threading
import time
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import PurePosixPath
from typing import TYPE_CHECKING, Any

from rich.console import Group, RenderableType
from rich.markdown import Markdown
from rich.panel import Panel
from rich.syntax import Syntax
from rich.text import Text
from textual import events
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.message import Message
from textual.widgets import Label, RichLog, Static, TextArea

from eva.adapters.voice import SpeechPlayer
from eva.domain.models import ActionRequest, AgentEvent, Reaction, Speech, TextDelta, ToolResult, ToolUse
from eva.terminal import PREVIEW_LINES, Console, describe, speaker_label, split_blocks, summarize
from eva.ui.face import FaceAnimator, draw, to_text

if TYPE_CHECKING:
    from eva.terminal import Terminal

BLUE, GREEN, DIM, YELLOW, RED = "#79c0ff", "#7ee787", "#8b949e", "#e3b341", "#ff7b72"
FPS = 12
SLEEP_AFTER_SECONDS = 300
WORDS_PER_SECOND = 2.7  # roughly how fast Kokoro speaks, to animate the eyes meanwhile
MAX_FILES = 5
TOOL_GLYPHS = {
    "execute": "$",
    "write_file": "+",
    "edit_file": "~",
    "read_file": "·",
    "ls": "·",
    "glob": "·",
    "grep": "·",
    "write_todos": "☰",
    "task": "»",
    "start_background_task": "»",
}
FILE_VERBS = {"write_file": "+", "edit_file": "~", "read_file": " "}
QUIET_WHEN_OK = {
    "read_file",
    "ls",
    "glob",
    "grep",
    "write_file",
    "edit_file",
    "write_todos",
}  # success says nothing new
TODO_MARKS = {"completed": ("✓", DIM), "in_progress": ("▸", "bold"), "pending": ("○", "")}
APPROVAL_KEYS = {"y", "n", "a", "s"}
LIST_ITEM = re.compile(r"([-*+]|\d+[.)])\s")


class PromptBox(TextArea):
    """Where the user types: long lines wrap and the box grows, and Enter sends."""

    MAX_LINES = 6

    class Submitted(Message):
        def __init__(self, value: str) -> None:
            super().__init__()
            self.value = value

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(soft_wrap=True, compact=True, highlight_cursor_line=False, tab_behavior="focus", **kwargs)

    async def _on_key(self, event: events.Key) -> None:
        if event.key == "enter":
            event.stop()
            event.prevent_default()
            self.post_message(self.Submitted(self.text.replace("\n", " ")))
            return
        await super()._on_key(event)

    def on_text_area_changed(self) -> None:
        self._fit()

    def on_resize(self) -> None:
        self._fit()

    def _fit(self) -> None:
        self.styles.height = max(1, min(self.MAX_LINES, self.wrapped_document.height))


class ConversationLog(RichLog):
    """The chat. Anything a library prints while the app runs lands here too."""

    def on_print(self, event: events.Print) -> None:
        if text := event.text.strip():
            self.write(Text(text, style=DIM))


class FaceWidget(Static):
    def __init__(self, animator: FaceAnimator, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.animator = animator
        self._shown: list | None = None

    def on_mount(self) -> None:
        self.set_interval(1 / FPS, self._draw)
        self._draw()

    def _draw(self) -> None:
        pixels = draw(self.animator.frame())
        if pixels != self._shown:  # most ticks move nothing, and redrawing costs terminal bandwidth
            self._shown = pixels
            self.update(to_text(pixels))


class EvaApp(App[None]):
    CSS = f"""
    Screen {{ background: #0d1117; }}
    #body {{ height: 1fr; }}
    #conversation {{
        width: 1fr; min-width: 40; border: round #30363d; border-title-color: {DIM};
        padding: 0 1; background: #0d1117; scrollbar-size-vertical: 1;
    }}
    #side {{ width: 54; }}
    #face {{ width: 100%; height: 11; content-align: center middle; margin-top: 1; }}
    #state {{ width: 100%; height: 2; content-align: center top; color: {DIM}; }}
    .panel {{ border: round #30363d; border-title-color: {DIM}; padding: 0 1; background: #0d1117; }}
    #plan {{ height: auto; max-height: 8; display: none; }}
    #plan.shown {{ display: block; }}
    #files {{ height: auto; max-height: 7; display: none; }}
    #files.shown {{ display: block; }}
    #activity {{ height: 1fr; min-height: 6; scrollbar-size-vertical: 1; }}
    #prompt-row {{ height: auto; border: round #30363d; background: #0d1117; }}
    #prompt-row.asking {{ border: round {YELLOW}; }}
    #prompt-label {{ width: auto; padding: 0 1; color: {GREEN}; text-style: bold; }}
    #prompt-row.asking #prompt-label {{ color: {YELLOW}; }}
    #prompt {{ border: none; background: #0d1117; padding: 0; height: 1; width: 1fr; }}
    #hints {{ height: 1; color: {DIM}; padding: 0 2; }}
    """
    BINDINGS = [
        Binding("ctrl+c", "quit", "Quit", priority=True),
        Binding("pageup", "scroll_chat(-1)", show=False, priority=True),
        Binding("pagedown", "scroll_chat(1)", show=False, priority=True),
    ]

    def __init__(self, view: "TuiView") -> None:
        super().__init__()
        self.view = view

    def compose(self) -> ComposeResult:
        with Horizontal(id="body"):
            yield ConversationLog(id="conversation", wrap=True, min_width=20)
            with Vertical(id="side"):
                yield FaceWidget(self.view.face, id="face")
                yield Static(id="state")
                yield Static(id="plan", classes="panel")
                yield Static(id="files", classes="panel")
                yield RichLog(id="activity", wrap=True, min_width=20, classes="panel")
        with Horizontal(id="prompt-row"):
            yield Label("You ›", id="prompt-label")
            yield PromptBox(id="prompt")
        yield Static("enter send · :help commands · pgup/pgdn scroll · ctrl+c quit", id="hints")

    def on_mount(self) -> None:
        self.title = "Eva"
        self.query_one("#conversation").border_title = "Conversation"
        self.query_one("#plan").border_title = "Plan"
        self.query_one("#files").border_title = "Files"
        self.query_one("#activity").border_title = "Activity"
        self.begin_capture_print(self.query_one(ConversationLog), stdout=True, stderr=False)
        self.query_one(PromptBox).focus()
        self.set_interval(0.5, self.view.refresh_state)
        self.view.mounted(self)

    def on_unmount(self) -> None:
        self.view.console.feed(None)

    def on_prompt_box_submitted(self, event: PromptBox.Submitted) -> None:
        self.query_one(PromptBox).clear()
        self.view.console.feed(event.value)

    def on_text_area_changed(self, event: TextArea.Changed) -> None:
        """One key answers an approval: y, n or a, no Enter needed."""
        answer = event.text_area.text.strip().lower()
        if self.view.console.asking_approval and answer in APPROVAL_KEYS:
            event.text_area.clear()
            self.view.console.feed(answer)

    def action_scroll_chat(self, pages: int) -> None:
        log = self.query_one(ConversationLog)
        log.scroll_to(y=log.scroll_y + pages * max(1, log.size.height - 2), animate=False)


class TuiView:
    """The `View` for the split screen. Safe to call from any thread."""

    def __init__(self, player: SpeechPlayer, clock: Callable[[], float] = time.monotonic) -> None:
        self.player = player
        self.muted = False
        self.face = FaceAnimator(clock)
        self.console = TuiConsole(self)
        self.app = EvaApp(self)
        self._clock = clock
        self._loop: asyncio.AbstractEventLoop | None = None
        self._ready = asyncio.Event()
        self._early: list[tuple[Callable[..., None], tuple]] = []
        self._lock = threading.Lock()
        self._pending = ""
        self._streamed = False
        self._busy = False
        self._last_activity = clock()
        self._terminal: Terminal | None = None
        self._files: dict[str, str] = {}
        self._plan: list[dict] = []
        self._draft = ""  # what the user was typing when a question interrupted them

    # Lifecycle

    @asynccontextmanager
    async def screen(self, terminal: "Terminal") -> AsyncIterator[None]:
        self._terminal = terminal
        self._loop = asyncio.get_running_loop()
        running = asyncio.create_task(self.app.run_async())
        ready = asyncio.create_task(self._ready.wait())
        await asyncio.wait({running, ready}, return_when=asyncio.FIRST_COMPLETED)
        if running.done():
            ready.cancel()
            running.result()  # the app failed to start: raise why
            return
        try:
            yield
        finally:
            if self.app.is_running:
                self.app.exit()
            await running

    def mounted(self, app: EvaApp) -> None:
        self._ready.set()
        for function, args in self._early:
            function(*args)
        self._early.clear()

    def _ui(self, function: Callable[..., None], *args: Any) -> None:
        """Run `function` on the app's loop, in call order."""
        if self._loop is None or not self._ready.is_set():
            self._early.append((function, args))
            return
        try:
            self._loop.call_soon_threadsafe(function, *args)
        except RuntimeError:  # the loop is closed: Eva is exiting
            pass

    # View

    def __call__(self, event: AgentEvent) -> None:
        with self._lock:
            match event:
                case TextDelta(text):
                    if not self.muted:
                        self.face.mood = "neutral"
                        blocks, self._pending = split_blocks(self._pending + text)
                        for block in blocks:
                            self._write_block(block)
                case Speech(text, source, aloud, mood):
                    self._flush()
                    self._ui(self._chat, _speech_line(text, source, aloud))
                    duration = len(text.split()) / WORDS_PER_SECOND
                    if aloud:
                        self.player.play(text)
                        self.face.speak(duration)
                    if mood:
                        self.face.react(mood, max(3.5, duration))
                        self._ui(self.refresh_state)
                case ToolUse(name, arguments, source):
                    self._flush()
                    self.face.mood = "loading" if name == "restart_eva" else "focused"
                    self._ui(self._activity, _tool_line(name, arguments, source))
                    self._track(name, arguments)
                case ToolResult(name, ok, summary, source):
                    if not (ok and name in QUIET_WHEN_OK):
                        self._ui(self._activity, _result_line(ok, summary))
                    if not ok:
                        self.face.react("worried", 3)
                case Reaction(mood):
                    self.face.react(mood, 4)
                    self._ui(self.refresh_state)
        self._last_activity = self._clock()

    def end_turn(self) -> bool:
        with self._lock:
            self._flush()
            streamed, self._streamed = self._streamed, False
            return streamed

    def reply(self, text: str) -> None:
        self._ui(self._chat, Group(Text("Eva", style=f"bold {BLUE}"), Markdown(text.strip())), True)

    def notice(self, text: str, tone: str = "info") -> None:
        style = {"warning": YELLOW, "error": RED}.get(tone, DIM)
        self._ui(self._chat, Text(text, style=style))
        if tone == "error":
            self.face.react("glitch", 2.5)
        elif tone == "warning":
            self.face.react("worried", 3)

    def user_said(self, text: str, voice: bool = False) -> None:
        label = Text.assemble(("You", f"bold {GREEN}"), (" (voice)" if voice else "", DIM), " ")
        self._ui(self._chat, Text.assemble(label, text), True)
        self._last_activity = self._clock()
        if self.face.mood == "sleepy":
            self.face.mood = "neutral"
            self.face.react("surprised", 1.2)  # woken up

    def status(self, state: str) -> None:
        self._busy = state not in {"idle"}
        if state == "idle":
            listening = self._terminal is not None and self._terminal.listener.active
            self.face.mood = "listening" if listening else "neutral"
        else:
            self.face.mood = state
        self._last_activity = self._clock()
        self._ui(self.refresh_state)

    # Approvals

    def show_request(self, action: ActionRequest | None, context: str) -> None:
        self.face.mood = "curious"
        title = f"{speaker_label(action.source)} wants to run {action.name}" if action else "Eva asks"
        body = _request_body(action) if action else Text(context)
        panel = Panel(body, title=title, title_align="left", border_style=YELLOW, subtitle="y yes · n no · a always")
        self._ui(self._activity, panel)
        if action:
            line = f"{title}: {_detail(action.name, action.arguments)}. Approve? (details on the right)"
            self._ui(self._chat, Text(line, style=YELLOW))

    def ask(self, question: str) -> None:
        self._ui(self._set_question, question)

    def answered(self, question: str, answer: str) -> None:
        if question.startswith("Approve"):
            verdict = {"y": "approved", "s": "approved", "a": "always allowed"}.get(answer.strip().lower()[:1])
            self._ui(self._activity, Text(f"  {verdict or 'declined'}", style=GREEN if verdict else RED))
            self.face.mood = "focused" if verdict else "neutral"

    # Drawing (on the app's loop)

    def _chat(self, renderable: RenderableType, gap: bool = False) -> None:
        log = self.app.query_one(ConversationLog)
        if gap and log.lines:
            log.write(Text(""))
        log.write(renderable)

    def _activity(self, renderable: RenderableType) -> None:
        self.app.query_one("#activity", RichLog).write(renderable)

    def _set_question(self, question: str) -> None:
        row = self.app.query_one("#prompt-row")
        row.set_class(bool(question), "asking")
        self.app.query_one("#prompt-label", Label).update(Text(question.strip() or "You ›"))
        box = self.app.query_one(PromptBox)
        if question:  # an empty box, so a single key answers
            self._draft, _ = box.text, box.clear()
        elif self._draft:
            box.insert(self._draft)
            self._draft = ""

    def refresh_state(self) -> None:
        now = self._clock()
        if not self._busy and self.face.mood == "neutral" and now - self._last_activity > SLEEP_AFTER_SECONDS:
            self.face.mood = "sleepy"
        if not self._ready.is_set():
            return
        self.app.query_one("#state", Static).update(self._state_text())
        self._show_panels()

    def _state_text(self) -> Text:
        showing, _ = self.face.expression
        doing = {
            "thinking": "thinking",
            "focused": "working",
            "curious": "waiting for you",
            "loading": "loading",
            "listening": "listening",
            "sleepy": "resting",
        }.get(self.face.mood, "ready")
        if showing != self.face.mood:  # a reaction is playing: name what the face shows
            doing = showing
        text = Text.assemble(("Eva", f"bold {BLUE}"), (f" · {doing}", DIM))
        terminal = self._terminal
        if terminal is not None and getattr(terminal, "app", None) is not None:
            app = terminal.app
            flags = [
                "speech on" if app.speech.enabled else "speech off",
                "autonomous" if app.policy.autonomous else "asks first",
            ]
            if running := len(app.tasks.running()):
                flags.append(f"{running} task{'s' if running > 1 else ''}")
            text.append("\n" + " · ".join(flags), style=DIM)
        return text

    def _show_panels(self) -> None:
        plan = self.app.query_one("#plan", Static)
        plan.set_class(bool(self._plan), "shown")
        if self._plan:
            lines = Text()
            for item in self._plan:
                mark, style = TODO_MARKS.get(item.get("status", ""), ("○", ""))
                lines.append(f"{mark} {item.get('content', '')}\n", style=style)
            plan.update(lines.rstrip() or lines)
        files = self.app.query_one("#files", Static)
        files.set_class(bool(self._files), "shown")
        if self._files:
            lines = Text(no_wrap=True, overflow="ellipsis")  # one line per file, however long its path
            for path, verb in list(self._files.items())[:MAX_FILES]:
                lines.append(f"{verb} {path}\n", style=DIM if verb == " " else "")
            files.update(lines.rstrip() or lines)

    # Bookkeeping

    def _track(self, name: str, arguments: dict) -> None:
        if name == "write_todos" and isinstance(arguments.get("todos"), list):
            self._plan = [item for item in arguments["todos"] if isinstance(item, dict)]
        elif name in FILE_VERBS and (path := arguments.get("file_path")):
            verb = FILE_VERBS[name]
            if self._files.get(path, " ") != " " and verb == " ":
                verb = self._files[path]  # reading a file does not undo writing it
            self._files.pop(path, None)
            self._files = {path: verb, **self._files}  # newest first
        self._ui(self.refresh_state)

    def _write_block(self, block: str) -> None:
        first = not self._streamed
        self._streamed = True
        content = Markdown(block.strip())
        if first:
            self._ui(self._chat, Group(Text("Eva", style=f"bold {BLUE}"), content), True)
        else:  # rich already puts a blank line above a list
            self._ui(self._chat, content if LIST_ITEM.match(block.lstrip()) else Group(Text(""), content))

    def _flush(self) -> None:
        if self._pending.strip():
            self._write_block(self._pending)
        self._pending = ""


class TuiConsole(Console):
    """Reads lines from the app's input box instead of prompt_toolkit."""

    def __init__(self, view: TuiView) -> None:
        super().__init__(keyboard=False)
        self._view = view
        self._lines: asyncio.Queue[str | None] = asyncio.Queue()

    @property
    def asking_approval(self) -> bool:
        return self._question.startswith("Approve")

    def feed(self, line: str | None) -> None:
        self._lines.put_nowait(line)

    async def read(self) -> str:
        line = await self._lines.get()
        if line is None:
            raise EOFError
        return line

    def answer(self, line: str) -> bool:
        question = self._question
        answered = super().answer(line)
        if answered:
            self._view.answered(question, line)
        return answered

    def _show(self, context: str, action: ActionRequest | None) -> None:
        self._view.show_request(action, context)

    def _redraw(self) -> None:
        self._view.ask(self._question)


def _speech_line(text: str, source: str | None, aloud: bool) -> Text:
    mark = " ♪ " if aloud else " · "
    return Text.assemble((speaker_label(source), f"bold {BLUE}"), (mark, DIM), (text, "" if aloud else "italic"))


def _tool_line(name: str, arguments: dict, source: str | None) -> Text:
    glyph = TOOL_GLYPHS.get(name, "•")
    stamp = datetime.now().strftime("%H:%M")
    line = Text.assemble((f"{stamp} ", DIM), (f"{glyph} ", BLUE))
    if source:
        line.append(f"[{source}] ", style=DIM)
    line.append(_detail(name, arguments))
    return line


def _detail(name: str, arguments: dict) -> str:
    """What a tool call is about: the command or path alone when its glyph already names the tool."""
    if name == "execute" and arguments.get("command"):
        return str(arguments["command"])
    if name in FILE_VERBS and arguments.get("file_path"):
        return str(arguments["file_path"])
    if name == "write_todos":
        return "updated the plan"
    return summarize(name, arguments)


def _result_line(ok: bool, summary: str) -> Text:
    mark, style = ("✓", GREEN) if ok else ("✗", RED)
    return Text.assemble((f"      {mark} ", style), (summary, DIM if ok else RED))


def _request_body(action: ActionRequest) -> RenderableType:
    arguments = action.arguments
    if action.name == "execute":
        return Syntax(str(arguments.get("command", "")), "bash", theme="github-dark", word_wrap=True)
    if action.name == "write_file":
        path = str(arguments.get("file_path", ""))
        lines = str(arguments.get("content", "")).splitlines()
        more = f"\n... ({len(lines) - PREVIEW_LINES} more lines)" if len(lines) > PREVIEW_LINES else ""
        code = Syntax(
            "\n".join(lines[:PREVIEW_LINES]) + more,
            Syntax.guess_lexer(PurePosixPath(path).name or "x.txt"),
            theme="github-dark",
            word_wrap=True,
        )
        return Group(Text(path, style="bold"), code)
    if action.name == "edit_file":
        return Syntax(describe(action), "diff", theme="github-dark", word_wrap=True)
    return Text(describe(action))
