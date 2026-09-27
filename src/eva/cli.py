import argparse
import json
import os
import sys
from pathlib import Path

from eva.adapters.voice import KokoroSpeaker, Microphone, WhisperTranscriber
from eva.bootstrap import Settings, bootstrap
from eva.domain.models import ActionRequest

# Tools that always require approval (shell + writes outside project)
_ALWAYS_INTERRUPT = {"execute", "write_file", "edit_file", "delete"}


class TerminalApproval:
    def approve(self, action: ActionRequest) -> bool:
        tool = action.name
        args = action.arguments
        print(f"\nEva requests: {tool}")
        if "command" in args:
            print(f"  command : {args['command']}")
            print(f"  cwd     : {args.get('cwd', args.get('path', '?'))}")
        elif "file_path" in args:
            print(f"  path    : {args['file_path']}")
        elif "path" in args:
            print(f"  path    : {args['path']}")
        else:
            print(json.dumps(args, ensure_ascii=False, indent=2))
        try:
            answer = input("Approve? [y/N] ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            return False
        return answer in {"y", "yes", "s", "sim"}


def _activate(extensions, name: str) -> None:
    try:
        proposal = extensions.read_proposal(name)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Cannot open proposal: {exc}")
        return
    print(f"\n{name}: {proposal.description}\n")
    print(proposal.source)
    try:
        answer = input("Activate this exact Python source? [y/N] ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        answer = ""
    if answer in {"y", "yes", "s", "sim"}:
        extensions.activate(name, expected=proposal)
        print("Activated. Eva can use the tool on the next turn, with approval.")
    else:
        print("Proposal remains inactive.")


def _apply_patch(patches, name: str) -> bool:
    try:
        proposal = patches.read(name)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Cannot open patch: {exc}")
        return False
    print(f"\n{name}: {proposal.description}\n")
    print(proposal.diff)
    try:
        answer = input("Apply this exact patch and run tests? [y/N] ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        answer = ""
    if answer not in {"y", "yes", "s", "sim"}:
        print("Patch remains unapplied.")
        return False
    try:
        success, detail = patches.apply(name, expected=proposal)
    except (OSError, ValueError) as exc:
        print(f"Patch could not be applied: {exc}")
        return False
    print(detail)
    return success


def main() -> None:
    parser = argparse.ArgumentParser(description="Eva local terminal assistant")
    parser.add_argument("--model", help="Loaded LM Studio model ID")
    parser.add_argument("--speak", action="store_true", help="Speak each final reply")
    parser.add_argument("--whisper-model", default="small")
    parser.add_argument("--voice", default=os.getenv("EVA_VOICE", "af_heart"))
    parser.add_argument("--voice-language", default=os.getenv("EVA_VOICE_LANGUAGE", "a"))
    args = parser.parse_args()

    try:
        settings = Settings.from_env(args.model)
    except ValueError as exc:
        parser.error(str(exc))

    speaker = KokoroSpeaker(voice=args.voice, language=args.voice_language)
    microphone = Microphone()
    transcriber = WhisperTranscriber(model_name=args.whisper_model)
    speak = args.speak

    print("Eva ready. Commands: :record [seconds], :speak on|off, :tools, :activate NAME, :patches, :apply NAME, :quit")
    restart = False
    with bootstrap(settings, TerminalApproval()) as app:
        app.speech.enabled = speak
        app.speech.on_speak = speaker.speak_async
        while True:
            try:
                message = input("\nYou> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not message:
                continue
            if message in {":quit", ":exit"}:
                break
            if message == ":tools":
                print("Active:", ", ".join(item.name for item in app.extensions.list_active()) or "none")
                print("Proposals:", ", ".join(app.extensions.list_proposals()) or "none")
                continue
            if message == ":patches":
                print("Source proposals:", ", ".join(app.patches.list()) or "none")
                continue
            if message.startswith(":apply "):
                if _apply_patch(app.patches, message.split(maxsplit=1)[1]):
                    restart = True
                    break
                continue
            if message.startswith(":activate "):
                _activate(app.extensions, message.split(maxsplit=1)[1])
                continue
            if message in {":speak on", ":speak off"}:
                speak = message.endswith("on")
                app.speech.enabled = speak
                print(f"Speech {'enabled' if speak else 'disabled'}")
                continue
            if message.startswith(":record"):
                try:
                    seconds = float(message.split(maxsplit=1)[1]) if " " in message else 6.0
                    path = Path(microphone.record(seconds))
                    try:
                        message = transcriber.transcribe(str(path))
                    finally:
                        path.unlink(missing_ok=True)
                    print(f"You said: {message}")
                except (ImportError, OSError, ValueError) as exc:
                    print(f"Voice input unavailable: {exc}")
                    continue
                if not message:
                    continue
            try:
                reply = app.session.send(message)
            except Exception as exc:
                app.speech.drain()
                print(f"Eva error: {exc}")
                continue
            print(f"\nEva> {reply}")
            app.summarizer.record(message, reply)
            
            if app.session.denied_actions:
                app.speech.drain()
            
            # Limpa o que foi enviado pro outbox (se o LLM usou a tool por acaso)
            tool_spoken = app.speech.drain()
            
            # Se a voz está ativa, fala a própria resposta de texto (melhor para modelos leves)
            if app.speech.enabled:
                import re
                # Remove blocos de código para a Eva não tentar ler python
                clean_reply = re.sub(r'```.*?```', '', reply, flags=re.DOTALL).strip()
                if clean_reply:
                    try:
                        speaker.speak(clean_reply)
                    except (ImportError, OSError, ValueError) as exc:
                        print(f"Voice output unavailable: {exc}")

    if not restart:
        saved = app.summarizer.save()
        if saved:
            print(f"\n[Eva saved a session summary -> {saved}]")

    if restart:
        print("Restarting Eva with the updated source...")
        os.execv(sys.executable, [sys.executable, *sys.argv])


if __name__ == "__main__":
    main()
