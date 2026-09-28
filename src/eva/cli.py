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
from eva.adapters.voice import KokoroSpeaker, SpeechPlayer, WhisperTranscriber
from eva.bootstrap import bootstrap
from eva.config import Settings
from eva.terminal import Console, ConsoleApproval, ProgressView, Terminal


def main() -> None:
    parser = argparse.ArgumentParser(description="Eva, a personal assistant in your terminal")
    parser.add_argument("--model", help="provider:model, e.g. google_genai:gemini-2.5-flash or lmstudio:<id>")
    parser.add_argument("--speak", action="store_true", help="Start with speech output on")
    parser.add_argument("--whisper-model", default=os.getenv("EVA_WHISPER_MODEL", "large-v3-turbo"))
    parser.add_argument("--voice", default=os.getenv("EVA_VOICE", "af_heart"))
    parser.add_argument("--voice-language", default=os.getenv("EVA_VOICE_LANGUAGE", "a"))
    parser.add_argument("--resumed", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    try:
        settings = Settings.from_env(args.model)
    except ValueError as exc:
        parser.error(str(exc))

    console = Console()
    player = SpeechPlayer(
        KokoroSpeaker(voice=args.voice, language=args.voice_language),
        on_error=lambda exc: print(f"Voice output unavailable: {exc}"),
    )
    heartbeat = timedelta(minutes=settings.heartbeat_minutes) if settings.heartbeat_minutes > 0 else None
    transcriber = WhisperTranscriber(model_name=args.whisper_model, notify=print)
    if importlib.util.find_spec("faster_whisper"):
        threading.Thread(target=transcriber.warm_up, daemon=True).start()
    if args.speak:
        player.warm_up()
    policy = ApprovalPolicy(autonomous=settings.autonomous)
    notifier = DesktopNotifier()
    with bootstrap(settings, policy, ConsoleApproval(console, policy, notifier), ProgressView(player)) as app:
        app.speech.enabled = args.speak
        terminal = Terminal(app, console, transcriber, player, heartbeat, notifier)
        asyncio.run(terminal.run(resumed=args.resumed))
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
