# Eva

Eva is a personal assistant that lives on your computer, named after EVE (EVA
in Brazil) from WALL-E. You talk to her in a terminal by text or voice. She
works with your files and shell, keeps her own journal, and can extend her own
abilities. Anything with side effects outside her journal waits for your
approval.

- **Reasoning:** [Deep Agents](https://docs.langchain.com/oss/python/deepagents/overview) on LangGraph, with any LangChain chat model (Gemini, or a local model through LM Studio).
- **Speech:** local speech-to-text with [faster-whisper](https://github.com/SYSTRAN/faster-whisper) and text-to-speech with [Kokoro](https://github.com/hexgrad/kokoro). Eva chooses what to say aloud; she never reads everything.
- **Safety:** read-only commands run immediately. Other commands and file changes ask first; answer `a` to stop asking about that exact command for the session, or set `EVA_AUTONOMOUS=1` to never ask. See [docs/decisions.md](docs/decisions.md).

Eva replies in English whatever language you write in.

## Quick start

```bash
cp .env.example .env          # set EVA_MODEL and, for Gemini, GEMINI_API_KEY
uv sync --extra voice --extra test
uv run eva
```

Models are named `provider:model`:

| Model | `EVA_MODEL` |
| --- | --- |
| Gemini | `google_genai:gemini-2.5-flash` |
| LM Studio | `lmstudio:<loaded-model-id>` (server at `EVA_LM_STUDIO_URL`) |

For a text-only install, use `uv sync --extra test`. Voice needs `espeak-ng`, a
microphone and speakers. It uses the GPU when one is available.

## Terminal commands

| Command | Action |
| --- | --- |
| any text | Talk to Eva |
| `:record` | Talk; press Enter to stop. The transcript is shown and sent |
| `:speak on` / `:speak off` | Let Eva speak aloud |
| `:shh` | Stop talking now |
| `:stop` | Stop what Eva is doing, at the next step |
| `:tasks` / `:cancel ID` | List / stop background tasks |
| `:auto on` / `:auto off` | Let Eva act without asking / ask again |
| `:quit` | Exit and save a session summary |

You can keep typing while Eva works; your message reaches her after her
current task. For long jobs she starts background tasks, keeps talking with you
while they run, and tells you when they finish. After `EVA_HEARTBEAT_MINUTES` of silence (default 30), she checks
her journal and may act on her own. She can also schedule follow-ups for
herself.

## Growing Eva

When Eva lacks an ability, she writes a
[skill](https://docs.langchain.com/oss/python/deepagents/skills): a
`SKILL.md` with instructions, plus optional `uv` scripts. Skills live in
`.eva/skills/` and become available from the next message. To change her core
behavior, she edits `src/eva/` and `tests/`. You approve each diff, and then
`restart_eva` runs the tests and restarts her if they pass.

## Data

Everything Eva keeps lives in `.eva/`:

- `memory/`: her journal. `AGENTS.md` is loaded into every conversation, and
  `sessions/` holds session summaries.
- `skills/`: skills Eva wrote for you.
- `checkpoints.sqlite`: the conversation, which survives restarts.
- `follow_ups.json`: reminders Eva scheduled for herself.

## Container

```bash
docker compose build
docker compose run --rm eva
```

In the container, shell commands run inside the container, with the repository
mounted at `/workspace`. Run Eva on the host when she needs the rest of your
computer. Voice in a container requires audio device setup; the host is the
supported path.

## Docs

- [Architecture](docs/architecture.md)
- [Decisions](docs/decisions.md)
- [Roadmap](docs/roadmap.md)
