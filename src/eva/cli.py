"""Terminal interface: the user's side of every approval."""

import argparse
import difflib
import os
import sys

from eva.adapters.voice import KokoroSpeaker, Microphone, SpeechPlayer, WhisperTranscriber
from eva.bootstrap import Application, bootstrap
from eva.config import Settings
from eva.domain.models import ActionRequest, AgentEvent, Speech, ToolUse

YES = {"y", "yes", "s", "sim"}
PREVIEW_LINES = 40
SUMMARY_WIDTH = 100
HELP = "Commands: :record [seconds], :speak on|off, :quit"
RESUMED_MESSAGE = "(You were just restarted with your edited source. Confirm briefly and carry on.)"


def confirm(question: str) -> bool:
    try:
        return input(f"{question} [y/N] ").strip().lower() in YES
    except (EOFError, KeyboardInterrupt):
        return False


def _preview(text: str) -> str:
    lines = text.splitlines()
    if len(lines) <= PREVIEW_LINES:
        return text
    return "\n".join([*lines[:PREVIEW_LINES], f"... ({len(lines) - PREVIEW_LINES} more lines)"])


class TerminalApproval:
    def approve(self, action: ActionRequest) -> bool:
        print(f"\n── Eva wants to run {action.name} ──")
        print(describe(action))
        return confirm("Approve?")


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


class ProgressView:
    """Shows what Eva is doing and plays what she says, as it happens."""

    def __init__(self, player: SpeechPlayer) -> None:
        self.player = player

    def __call__(self, event: AgentEvent) -> None:
        match event:
            case Speech(text):
                print(f"\nEva (aloud)> {text}")
                self.player.play(text)
            case ToolUse(name, arguments):
                print(f"  · {summarize(name, arguments)}")


def summarize(name: str, arguments: dict) -> str:
    keys = ("command", "file_path", "pattern", "path", "description")
    detail = next((str(arguments[key]) for key in keys if arguments.get(key)), "")
    line = f"{name} {detail}".strip().replace("\n", " ")
    return line if len(line) <= SUMMARY_WIDTH else f"{line[: SUMMARY_WIDTH - 1]}…"


class Terminal:
    def __init__(self, app: Application, transcriber: WhisperTranscriber) -> None:
        self.app = app
        self.transcriber = transcriber
        self.microphone = Microphone()

    @property
    def restart(self) -> bool:
        return self.app.updater.restart_requested

    def run(self, resumed: bool = False) -> None:
        print(f"Eva is ready. {HELP}")
        if resumed:
            self.converse(RESUMED_MESSAGE)
        while True:
            try:
                line = input("\nYou> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                return
            if line in {":quit", ":exit"}:
                return
            message = self.command(line) if line.startswith(":") else line
            if message:
                self.converse(message)
            if self.restart:
                return

    def command(self, line: str) -> str | None:
        """Handle a terminal command; return a message to send to Eva, if any."""
        name, _, argument = line.partition(" ")
        argument = argument.strip()
        if name == ":record":
            try:
                return self.listen(float(argument or 6))
            except ValueError as exc:
                print(f"Voice input unavailable: {exc}")
                return None
        if name == ":speak" and argument in {"on", "off"}:
            self.app.speech.enabled = argument == "on"
            print(f"Speech {argument}.")
        else:
            print(HELP)
        return None

    def listen(self, seconds: float) -> str | None:
        try:
            path = self.microphone.record(seconds)
            try:
                text = self.transcriber.transcribe(path)
            finally:
                path.unlink(missing_ok=True)
        except (ImportError, OSError) as exc:
            print(f"Voice input unavailable: {exc}")
            return None
        print(f"You said: {text}")
        return text or None

    def converse(self, message: str) -> None:
        try:
            reply = self.app.session.send(message)
        except Exception as exc:  # noqa: BLE001 - keep the terminal alive on model errors
            print(f"Eva error: {exc}")
            return
        print(f"\nEva> {reply}")
        if self.app.session.denied_actions:
            print(f"[Declined: {', '.join(self.app.session.denied_actions)}]")
        self.app.summarizer.record(message, reply)


def main() -> None:
    parser = argparse.ArgumentParser(description="Eva, a local terminal assistant")
    parser.add_argument("--model", help="provider:model, e.g. google_genai:gemini-2.5-flash or lmstudio:<id>")
    parser.add_argument("--speak", action="store_true", help="Start with speech output on")
    parser.add_argument("--whisper-model", default=os.getenv("EVA_WHISPER_MODEL", "small"))
    parser.add_argument("--voice", default=os.getenv("EVA_VOICE", "af_heart"))
    parser.add_argument("--voice-language", default=os.getenv("EVA_VOICE_LANGUAGE", "a"))
    parser.add_argument("--resumed", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    try:
        settings = Settings.from_env(args.model)
    except ValueError as exc:
        parser.error(str(exc))

    player = SpeechPlayer(
        KokoroSpeaker(voice=args.voice, language=args.voice_language),
        on_error=lambda exc: print(f"Voice output unavailable: {exc}"),
    )
    with bootstrap(settings, TerminalApproval(), ProgressView(player)) as app:
        app.speech.enabled = args.speak
        terminal = Terminal(app, WhisperTranscriber(model_name=args.whisper_model))
        terminal.run(resumed=args.resumed)
        player.wait()
        speaking = app.speech.enabled
        if not terminal.restart and (saved := app.summarizer.save()):
            print(f"\n[Session summary saved to {saved}]")

    if terminal.restart:
        print("Restarting Eva with the updated source...")
        options = [option for option in sys.argv[1:] if option not in {"--resumed", "--speak"}]
        options += ["--resumed", *(["--speak"] if speaking else [])]
        os.execv(sys.executable, [sys.executable, "-m", "eva.cli", *options])


if __name__ == "__main__":
    main()
