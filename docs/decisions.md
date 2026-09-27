# Decision log

Newest first. Each entry records what was decided and why, so later changes can
revisit the reasoning instead of rediscovering it.

## Autonomy: background turns, heartbeats and follow-ups

- The terminal is asynchronous (`prompt_toolkit`). Eva works on a background
  thread while the user keeps typing. Messages typed meanwhile are queued for
  her next turn. The keyboard has one reader, shared by messages and approval
  answers.
- **Heartbeat:** after `EVA_HEARTBEAT_MINUTES` of silence (default 30, 0
  disables), Eva gets a `[heartbeat]` message and may act or speak. If she has
  nothing to do, she replies `HEARTBEAT_OK`, and that exchange is removed from
  the thread with `RemoveMessage` so idle checks do not fill the context.
- **Follow-ups:** `schedule_follow_up` stores a note in
  `.eva/follow_ups.json`, which survives restarts. A clock delivers it as a
  `[follow-up]` message when due.
- Deep Agents
  [async subagents](https://docs.langchain.com/oss/python/deepagents/async-subagents)
  need an Agent Protocol server (`langgraph dev` or a deployment). An
  in-process loop covers the current needs with no extra service. They remain
  an option for long parallel jobs.

## The model is told, on every call, whether speech is on

A model cannot know whether anyone is listening unless it is told.
`SpeechModeMiddleware` appends the current speech
state to the system prompt on each model call. With speech on, Eva says the
key sentence of each reply with `speak_to_user`, and she still chooses the
words. The language rule explicitly forbids greetings in other languages,
because the English Kokoro voice cannot pronounce them.

## Kokoro stays; speech streams by sentence

Measured on a consumer GPU with 6 GB: Kokoro-82M synthesizes 7.3 s
of audio in 0.19 s once warm, with the first sentence ready in about 0.1 s.
Among open-weights models it ranks close to much larger ones
([Artificial Analysis](https://artificialanalysis.ai/text-to-speech/leaderboard/provider-voice/open-weights)).
The higher-ranked models (Voxtral 4B, Fish S2 Pro) would not fit next to
Whisper large-v3-turbo on 6 GB, or would add latency. Chatterbox mainly adds
voice cloning. To cut latency further:

- speech is streamed to `paplay`/`aplay` sentence by sentence while the rest
  is synthesized;
- Kokoro warms up in the background when speech is turned on, and Whisper
  warms up at startup. The cold start (about 6 s) is paid before the first
  reply, not during it.

`Speaker` is a small protocol (`warm_up`, `speak`), so another engine can
replace Kokoro without touching the rest of Eva.

## Voice input is push-to-talk and framed for Eva

`:record` listens until Enter. The transcript is shown to the user and sent as
`[voice] ...`, so Eva knows it was spoken, may contain transcription errors,
and deserves a spoken answer. The default Whisper model is `large-v3-turbo`.
It is multilingual and fits on a 6 GB GPU next to
Kokoro, since the chat model runs remotely.

## Self-extension through skills and reviewed source edits

Eva extends herself with the mechanisms Deep Agents already provides:

- **Skills** ([docs](https://docs.langchain.com/oss/python/deepagents/skills)):
  Eva writes `/skills/<name>/SKILL.md` and optional scripts, stored in
  `.eva/skills/`, so they are personal and outside Git. Bundled skills live in
  `src/eva/skills/` and are mounted at `/builtin-skills/`. Writing a skill needs
  approval, and so does running its scripts, like any non-read-only command.
  Scripts are standalone `uv` scripts with inline dependencies, so they never
  change Eva's own environment. Each message passes `skills_metadata: None`,
  which makes skills created mid-session appear on the next message.
- **Source edits:** Eva edits `/src/eva/` and `/tests/` with the normal file
  tools. Every edit is shown as a diff for approval. `restart_eva` runs the
  test suite and restarts the process only if it passes. The checkpointed
  conversation continues after the restart. Git remains the undo mechanism.

A skill covers a new ability. A source edit covers Eva's core behavior.

## Progress is streamed, speech plays while Eva works

The adapter runs `agent.stream(stream_mode=["updates", "custom"],
subgraphs=True, version="v2")`
([streaming](https://docs.langchain.com/oss/python/deepagents/streaming)).
`speak_to_user` emits a custom event through `get_stream_writer()`, and the
terminal plays it immediately on a background thread, so Eva can say "checking
the logs now" before a slow step instead of after the whole turn. Tool calls
are shown as one-line progress. Interrupts and the final reply are read from
`get_state()` after the stream ends. Subagent events arrive through
`subgraphs=True`.

## Approval policy lives in `interrupt_on`, not `permissions`

Deep Agents' `FilesystemPermission` rules only cover file tools, never
`execute`, and they cannot target the default route of a backend that runs
commands. Eva therefore builds `interrupt_on` for `HumanInTheLoopMiddleware`,
with `when` predicates (`adapters/policy.py`):

| Action | Approval |
| --- | --- |
| Read-only shell commands (`ls`, `cat`, `grep`, `git status`, ...), pipelines of them | none |
| Any other command, redirects, `$`/backticks, secret paths (`.env`, `.ssh`, ...) | required |
| Writes, edits and deletes under `/memories/` (the journal) | none |
| Any other write, edit or delete | required |
| Reading secret files with `read_file` | required |

`tests/test_agent.py` runs the real graph with a scripted model to keep this true.

## One backend, two roots

A `CompositeBackend` exposes the project at `/` through `LocalShellBackend`,
which also runs shell commands from the project directory. The journal is
exposed at `/memories/` through `FilesystemBackend`. Both use
`virtual_mode=True`, so file tools cannot escape their roots. Shell commands
can reach the whole computer, and the approval policy governs them. API keys
and other secret-looking variables are removed from the shell environment.

## Journal is Deep Agents memory

`/memories/AGENTS.md` is loaded into every turn through the native `memory=`
parameter. Eva writes the rest of `/memories/` freely, using any structure she
likes, and reads those files on demand. Session summaries go to
`/memories/sessions/`.

## Models are named `provider:model`

`init_chat_model` builds the model, so any LangChain provider works, e.g.
`google_genai:gemini-2.5-flash`. The `lmstudio:<id>` prefix targets LM
Studio's OpenAI-compatible server. The session summarizer uses the same model.

## English only

Code, docs and prompts are in English, and Eva always replies in English,
whatever language the user writes in.
