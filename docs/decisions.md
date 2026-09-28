# Decision log

Newest first. Each entry records what was decided and why, so later changes can
revisit the reasoning instead of rediscovering it.

## Running out of steps ends a turn with a report

A LangGraph run stops after `recursion_limit` graph steps, and one tool call
takes a few of them. Long autonomous work used to hit the old limit of 150
and end with a raw `GraphRecursionError`. The limit is now `EVA_MAX_STEPS`
(default 500). When a turn still runs out, the adapter raises
`StepLimitReached`, and the terminal sends Eva a `[step limit]` message so she
summarizes what she did and asks whether to continue. She is asked once, never
in a loop. The conversation is checkpointed, so "continue" picks up where she
stopped. A background task that runs out reports it as its result.

## A local setup for a 6 GB GPU

Measured with Qwen3.5-4B (Q4_K_M) served by LM Studio, next to Whisper and
Kokoro on the same GPU:

| Setting | Value | Why |
| --- | --- | --- |
| Model | `unsloth/qwen3.5-4b`, all layers on the GPU | ~44 tokens/s; a 9B split with the CPU ran at ~11 |
| Context | 65536 tokens | Qwen3.5's hybrid attention keeps the cache small |
| KV cache | q4_0 (K and V), flash attention on | 64K fits in about 3.8 GB |
| Parallel slots | 2 | background tasks run next to the conversation |
| Thinking | on | better tool choice on multi-step requests, at some latency |
| Whisper | `int8_float16` on the GPU (`EVA_WHISPER_COMPUTE_TYPE`) | about 1 GB instead of 1.6, same accuracy for speech |

Everything together uses about 5.6 GB. Kokoro stays on the GPU: on the CPU its
first sentence takes about 2 s instead of 0.1 s. The vision projector (F32,
1.3 GB) does not fit alongside the voice models, so the model runs text-only.

## Hands-free listening

`:listen on` (or `--listen`) keeps the microphone open. `UtteranceDetector`
cuts the stream into utterances: it starts after three speech frames and ends
after 0.8 s of pause, with 0.3 s of pre-roll so first syllables survive. Each
utterance is transcribed and sent as `[voice] ...`. Each frame is classified
by Silero VAD, which ships with faster-whisper and runs on CPU. A neural VAD is
used rather than a loudness threshold, which breaks down under steady
background noise that is as loud as speech. While Eva speaks, and for 0.5 s after, the microphone
is ignored so she does not hear herself.

Transcripts also drop segments Whisper marks as probably not speech
(`no_speech_prob` ≥ 0.6), which removes the "Thank you." it hallucinates on
noise.

## The reply streams as it is written

The main thread adds `messages` to its stream modes. Text chunks from the
top-level `model` node become `TextDelta` events. The terminal renders the
reply as Markdown with `rich`, one block at a time: a paragraph, list or code
block is printed as soon as it is complete. Half-written Markdown does not
render, and prompt_toolkit redraws the input line on every write, so partial
lines would be torn apart. Models write Markdown naturally, so Eva is not told
to avoid it. The final reply is not printed again once it was streamed.
Background tasks do not stream, because only their report matters. Heartbeat
turns are muted, so an idle `HEARTBEAT_OK` never shows.

## Steering: messages reach Eva mid-task

Messages typed while the main turn runs are not queued behind it.
They go to a per-thread `SteeringInbox`, and `SteeringMiddleware.before_model`
adds them to the history as `[sent while you were working] ...` before the
next model call. The user can therefore redirect or cancel work in progress
with plain words. A message that arrives after the last model call becomes the
next turn. The thread is found through the run's `thread_id`, so the same
middleware serves the main conversation and background tasks.

## Desktop notifications

When a background task ends or Eva needs an approval, `notify-send` shows a
desktop notification, so the user does not have to watch the terminal. Where
`notify-send` is missing, nothing happens.

## Autonomous mode and "always allow"

`ApprovalPolicy` holds the approval rules and state that can change during a
session. The `when` predicates read it on every call, so changes apply
immediately without rebuilding the graph.

- `EVA_AUTONOMOUS=1`, or `:auto on` at runtime, turns every approval off. Eva
  runs commands and edits files, her own source included, without asking. The
  terminal shows a warning while it is on. Secrets are still stripped from the
  shell environment.
- At an approval prompt, `a` approves and stops asking about that exact
  command or file for the rest of the session. It is not persisted, so each
  session starts cautious.

## Background tasks, interruptions and fuller speech

- **Background tasks:** `start_background_task` hands long work to another
  conversation thread of the same compiled graph, so Eva keeps talking with the
  user meanwhile. `DeepAgentAdapter` holds the graph, and each `AgentThread`
  has its own checkpointed history, pending approvals and cancel flag. A task
  cannot see the main conversation, so its instructions must be
  self-contained. When it ends, its report reaches the main conversation as a
  `[background task ...]` message and Eva relays it. At most three run at
  once. Task output and approvals are labeled `[task <id>]`, and approval
  questions are serialized so parallel tasks never interleave prompts. On exit,
  running tasks are cancelled before the checkpointer closes. This stays
  in-process: Deep Agents async subagents need an Agent Protocol server.
- **Stopping:** `:stop` cancels the main turn at its next graph step. A tool
  that is already running finishes first. `:cancel ID` does the same for a
  task. `:shh` drops queued speech and kills playback, and `:record` does it
  automatically, so the user can talk over Eva.
- **Speech:** there is no fixed length. With speech on, Eva says
  whole conversational answers aloud, gives spoken summaries of technical
  results, and keeps the user posted during long work. If she runs three
  tool-calling steps without speaking, the speech middleware adds a nudge for
  a spoken progress update. The check is based on state, so it works for every
  thread.

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

## Whisper on the GPU needs cuBLAS for CUDA 12

`faster-whisper` runs on CTranslate2, whose wheels are built for CUDA 12. The
torch build Kokoro uses ships CUDA 13, so `libcublas.so.12` has to come from
somewhere else. The voice extra installs
`nvidia-cublas-cu12`, and `WhisperTranscriber` loads it with `ctypes` before
creating the model. If the GPU still fails, Eva says so and falls back to the
CPU (int8). On a consumer GPU, a short phrase takes about 1.2 s on the GPU
and about 8 s on the CPU.

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
`google_genai:gemini-3.8-flash`. The `lmstudio:<id>` prefix targets LM
Studio's OpenAI-compatible server, the default for running everything locally. The session summarizer uses the same model.

## English only

Code, docs and prompts are in English, and Eva always replies in English,
whatever language the user writes in.
