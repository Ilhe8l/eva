"""Command-line entry point."""

import argparse
import asyncio
import importlib.util
import os
import sys
import threading
from datetime import timedelta

from eva.adapters.notifier import DesktopNotifier
from eva.adapters.policy import ApprovalPolicy
from eva.adapters.voice import VOICE_MISSING, KokoroSpeaker, SpeechPlayer, WhisperTranscriber, voice_available
from eva.bootstrap import bootstrap
from eva.config import Settings
from eva.terminal import Console, ConsoleApproval, ProgressView, Terminal, View


def main() -> None:
    parser = argparse.ArgumentParser(description="Eva, a personal assistant in your terminal")
    parser.add_argument("--model", help="provider:model, e.g. lmstudio:<id> or google_genai:gemini-3.8-flash")
    parser.add_argument("--speak", action="store_true", help="Start with speech output on")
    parser.add_argument("--listen", action="store_true", help="Start listening hands-free")
    parser.add_argument("--whisper-model", default=os.getenv("EVA_WHISPER_MODEL", "large-v3-turbo"))
    parser.add_argument(
        "--whisper-compute-type",
        default=os.getenv("EVA_WHISPER_COMPUTE_TYPE", "int8_float16"),
        help="Whisper precision on the GPU, e.g. int8_float16 or float16",
    )
    parser.add_argument("--voice", default=os.getenv("EVA_VOICE", "af_heart"))
    parser.add_argument("--voice-language", default=os.getenv("EVA_VOICE_LANGUAGE", "a"))
    parser.add_argument(
        "--plain",
        action="store_true",
        default=os.getenv("EVA_UI", "").lower() == "plain",
        help="Line-by-line output instead of the split screen (EVA_UI=plain)",
    )
    parser.add_argument("--resumed", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    try:
        settings = Settings.from_env(args.model)
    except ValueError as exc:
        parser.error(str(exc))

    player = SpeechPlayer(
        KokoroSpeaker(voice=args.voice, language=args.voice_language),
        on_error=lambda exc: view.notice(f"Voice output unavailable: {exc}", tone="error"),
    )
    view: View
    if args.plain or not (sys.stdin.isatty() and sys.stdout.isatty()):
        console, view = Console(), ProgressView(player)
    else:
        from eva.ui.tui import TuiView

        view = TuiView(player)
        console = view.console
    heartbeat = timedelta(minutes=settings.heartbeat_minutes) if settings.heartbeat_minutes > 0 else None
    transcriber = WhisperTranscriber(args.whisper_model, args.whisper_compute_type, notify=view.notice)
    if importlib.util.find_spec("faster_whisper"):
        threading.Thread(target=transcriber.warm_up, daemon=True).start()
    if (args.speak or args.listen) and not voice_available():
        view.notice(VOICE_MISSING, tone="warning")
        args.speak = args.listen = False
    if args.speak:
        player.warm_up()
    policy = ApprovalPolicy(autonomous=settings.autonomous)
    notifier = DesktopNotifier()
    with bootstrap(settings, policy, ConsoleApproval(console, policy, notifier), view) as app:
        app.speech.enabled = args.speak
        terminal = Terminal(app, console, transcriber, player, heartbeat, notifier, view)
        asyncio.run(terminal.run(resumed=args.resumed, listen=args.listen))
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
