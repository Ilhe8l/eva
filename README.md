# Eva

Eva is a personal assistant that lives on your computer, named after EVE (EVA
in Brazil) from WALL-E. You talk to her by voice or text. She works with your
files and shell, keeps her own journal, runs long jobs in the background, and
teaches herself new abilities. Anything with side effects asks you first.

She is built to run locally: the model is served by
[LM Studio](https://lmstudio.ai) on your machine, speech recognition is
[faster-whisper](https://github.com/SYSTRAN/faster-whisper), and her voice is
[Kokoro](https://github.com/hexgrad/kokoro). Nothing has to leave your computer.

## What she does

- **Talks.** Push-to-talk (`:record`) or hands-free (`:listen on`). She picks
  what to say aloud (a greeting, the gist of an answer, a heads-up before slow
  work) and leaves code and lists to the screen. You can talk over her.
- **Works while you talk.** Long jobs go to background tasks, and she tells you
  when they finish. A message sent mid-task reaches her at her next step, so
  "actually, stop" works.
- **Remembers.** She keeps a Markdown journal that she organizes herself, and
  the conversation survives restarts.
- **Grows.** When she lacks an ability, she writes a
  [skill](https://docs.langchain.com/oss/python/deepagents/skills) for it. She
  can also edit her own code: you approve each diff, and she restarts only if
  the tests pass.
- **Takes initiative.** After a quiet spell she checks her journal for pending
  work, and she can schedule follow-ups for herself.

## Quick start

You need Linux, Python 3.12, [uv](https://docs.astral.sh/uv/), and
[LM Studio](https://lmstudio.ai) running its local server with a model that
supports tool calling (Qwen3 works well).

```bash
git clone https://github.com/Ilhe8l/eva.git && cd eva
cp .env.example .env    # set EVA_MODEL to the model loaded in LM Studio
uv sync --extra voice
uv run eva --speak
```

Voice needs a microphone and speakers. A GPU with about 6 GB fits Whisper
`large-v3-turbo` and Kokoro together; without one, both run on the CPU, more
slowly. For a text-only install, use `uv sync` and skip `--speak`.

Eva answers in English whatever language you speak.

### Models

Models are named `provider:model`, and any LangChain chat model works:

| Where | `EVA_MODEL` |
| --- | --- |
| LM Studio (local) | `lmstudio:<loaded-model-id>`, served at `EVA_LM_STUDIO_URL` |
| Gemini (cloud) | `google_genai:gemini-3.8-flash`, with `GEMINI_API_KEY` |
| Other providers | e.g. `openai:...` or `anthropic:...`, with that provider's LangChain package |

## Using Eva

| Command | Action |
| --- | --- |
| any text | Talk to Eva |
| `:record` | Talk, then press Enter; the transcript is shown and sent |
| `:listen on` / `:listen off` | Hands-free: just talk, she notices when you stop (also `--listen`) |
| `:speak on` / `:speak off` | Let her speak aloud |
| `:shh` | Stop talking now |
| `:stop` | Stop what she is doing, at the next step |
| `:tasks` / `:cancel ID` | List or stop background tasks |
| `:auto on` / `:auto off` | Let her act without asking, or ask again |
| `:quit` | Exit and save a session summary |

## Safety

Read-only commands (`ls`, `cat`, `grep`, `git status`, ...) run right away.
Other commands and file changes wait for your approval. Answer `a` to allow
that exact command or file for the rest of the session. `EVA_AUTONOMOUS=1`
turns approvals off; read [SECURITY.md](SECURITY.md) before using it.

## Configuration

Set these in `.env`:

| Variable | Default | Meaning |
| --- | --- | --- |
| `EVA_MODEL` | (required) | Chat model, `provider:model` |
| `EVA_LM_STUDIO_URL` | `http://localhost:1234/v1` | LM Studio server |
| `EVA_WHISPER_MODEL` | `large-v3-turbo` | Speech recognition model |
| `EVA_VOICE` / `EVA_VOICE_LANGUAGE` | `af_heart` / `a` | [Kokoro voice](https://huggingface.co/hexgrad/Kokoro-82M/blob/main/VOICES.md) |
| `EVA_AUTONOMOUS` | off | Act without asking |
| `EVA_HEARTBEAT_MINUTES` | `30` | Quiet time before she checks in (`0` disables) |
| `EVA_DATA_DIR` | `.eva` | Where her journal, skills and history live |
| `HF_TOKEN` | none | Faster model downloads |

Everything Eva keeps lives in `.eva/`: her journal (`memory/`, with
`AGENTS.md` loaded into every conversation), the skills she wrote (`skills/`),
the conversation (`checkpoints.sqlite`) and her follow-ups
(`follow_ups.json`).

## How it works

Eva is a [Deep Agents](https://docs.langchain.com/oss/python/deepagents/overview)
graph on LangGraph, wrapped in a small hexagonal core: the domain and the
application layer know nothing about models, audio or the terminal. Approvals
use LangChain's human-in-the-loop middleware, progress and speech are streamed,
and each background task runs as its own thread on the same graph. See
[docs/architecture.md](docs/architecture.md) and
[docs/decisions.md](docs/decisions.md).

## Container

```bash
docker compose build
docker compose run --rm eva
```

Inside the container, shell commands run in the container, with the repository
mounted at `/workspace`. Run Eva on the host when she needs the rest of your
computer, or for voice.

## Development

```bash
uv sync --extra voice --extra test
uv run pytest
uv run ruff check . && uv run ruff format --check .
```

Commits follow [Conventional Commits](https://www.conventionalcommits.org);
release-please turns them into versions and the changelog.

## License

[MIT](LICENSE)
