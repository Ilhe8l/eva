# Decision log

Newest first. Each entry records what was decided and why, so later changes can
revisit the reasoning instead of rediscovering it.

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
| Activated extension tools | required |

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
