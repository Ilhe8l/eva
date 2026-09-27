# Decision log

Newest first. Each entry records what was decided and why, so later changes can
revisit the reasoning instead of rediscovering it.

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
