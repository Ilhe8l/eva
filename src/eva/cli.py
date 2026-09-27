"""Terminal interface: the user's side of every approval."""

import argparse
import difflib
import os
import sys

from eva.adapters.voice import KokoroSpeaker, Microphone, WhisperTranscriber
from eva.bootstrap import Application, bootstrap
from eva.config import Settings
from eva.domain.models import ActionRequest

YES = {"y", "yes", "s", "sim"}
PREVIEW_LINES = 40
HELP = "Commands: :record [seconds], :speak on|off, :tools, :activate NAME, :patches, :apply NAME, :quit"


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


def activate_extension(app: Application, name: str) -> None:
    try:
        proposal = app.extensions.read_proposal(name)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Cannot open proposal: {exc}")
        return
    print(f"\n{name}: {proposal.description}\n\n{proposal.source}")
    if not confirm("Activate this exact Python source?"):
        print("Proposal remains inactive.")
        return
    app.extensions.activate(name, expected=proposal)
    app.reload_tools()
    print("Activated. Eva can use it now; each call still asks for approval.")


def apply_patch(app: Application, name: str) -> bool:
    try:
        proposal = app.patches.read(name)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Cannot open patch: {exc}")
        return False
    print(f"\n{name}: {proposal.description}\n\n{proposal.diff}")
    if not confirm("Apply this exact patch and run tests?"):
        print("Patch remains unapplied.")
        return False
    try:
        success, detail = app.patches.apply(name, expected=proposal)
    except (OSError, ValueError) as exc:
        print(f"Patch could not be applied: {exc}")
        return False
    print(detail)
    return success


class Terminal:
    def __init__(self, app: Application, speaker: KokoroSpeaker, transcriber: WhisperTranscriber) -> None:
        self.app = app
        self.speaker = speaker
        self.transcriber = transcriber
        self.microphone = Microphone()
        self.restart = False

    def run(self) -> None:
        print(f"Eva is ready. {HELP}")
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
        elif name == ":tools":
            print("Active:", ", ".join(item.name for item in self.app.extensions.list_active()) or "none")
            print("Proposals:", ", ".join(self.app.extensions.list_proposals()) or "none")
        elif name == ":activate" and argument:
            activate_extension(self.app, argument)
        elif name == ":patches":
            print("Source patches:", ", ".join(self.app.patches.list()) or "none")
        elif name == ":apply" and argument:
            self.restart = apply_patch(self.app, argument)
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
            self.app.speech.drain()
            print(f"Eva error: {exc}")
            return
        print(f"\nEva> {reply}")
        if self.app.session.denied_actions:
            print(f"[Declined: {', '.join(self.app.session.denied_actions)}]")
        self.app.summarizer.record(message, reply)
        for sentence in self.app.speech.drain():
            try:
                self.speaker.speak(sentence)
            except (ImportError, OSError, ValueError) as exc:
                print(f"Voice output unavailable: {exc}")
                break


def main() -> None:
    parser = argparse.ArgumentParser(description="Eva, a local terminal assistant")
    parser.add_argument("--model", help="provider:model, e.g. google_genai:gemini-2.5-flash or lmstudio:<id>")
    parser.add_argument("--speak", action="store_true", help="Start with speech output on")
    parser.add_argument("--whisper-model", default=os.getenv("EVA_WHISPER_MODEL", "small"))
    parser.add_argument("--voice", default=os.getenv("EVA_VOICE", "af_heart"))
    parser.add_argument("--voice-language", default=os.getenv("EVA_VOICE_LANGUAGE", "a"))
    args = parser.parse_args()
    try:
        settings = Settings.from_env(args.model)
    except ValueError as exc:
        parser.error(str(exc))

    speaker = KokoroSpeaker(voice=args.voice, language=args.voice_language)
    transcriber = WhisperTranscriber(model_name=args.whisper_model)
    with bootstrap(settings, TerminalApproval()) as app:
        app.speech.enabled = args.speak
        terminal = Terminal(app, speaker, transcriber)
        terminal.run()
        if not terminal.restart and (saved := app.summarizer.save()):
            print(f"\n[Session summary saved to {saved}]")

    if terminal.restart:
        print("Restarting Eva with the updated source...")
        os.execv(sys.executable, [sys.executable, "-m", "eva.cli", *sys.argv[1:]])


if __name__ == "__main__":
    main()
