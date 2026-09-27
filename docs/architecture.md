# Architecture

## Goal

Eva is a personal assistant that lives on the user's computer. She talks by
text or voice and works on tasks with the computer's tools. She keeps her own
journal and can extend her abilities. Every action with side effects outside
her journal waits for the user's approval.

## Layers

| Layer | Module | Responsibility |
| --- | --- | --- |
| Domain | `eva.domain` | Values crossing boundaries: `ActionRequest`, `AgentStep` |
| Application | `eva.application` | A conversational turn and its approvals (`EvaSession`), ports |
| Adapters | `eva.adapters` | Deep Agents, models, approval policy, tools, voice, restart, summaries |
| Composition | `eva.bootstrap`, `eva.config` | Settings and wiring |
| Interface | `eva.cli` | Terminal, microphone, speaker and approval prompts |

The application layer depends only on its ports. Deep Agents, LangGraph and the
voice models stay in adapters.

## Agent

`DeepAgentAdapter` wraps `create_deep_agent` and uses:

- a `CompositeBackend` built by `Workspace`: the project at `/` (with shell
  execution), the journal at `/memories/`, Eva's skills at `/skills/` and
  bundled skills at `/builtin-skills/`;
- `memory=["/memories/AGENTS.md"]`, which is loaded into every turn;
- `skills=["/builtin-skills/", "/skills/"]`, which are rescanned on every
  message;
- `interrupt_on` from `adapters/policy.py`;
- a SQLite checkpointer, so a turn paused for approval, and the whole
  conversation, survive restarts.

The graph is built once per session. The adapter streams each run and
reports tool use and chosen speech as events.

## Trust boundaries

1. Model output is untrusted. The terminal shows each action that needs
   approval before it runs.
2. The approval policy is described in [decisions.md](decisions.md). Rejected
   actions are reported to the model, and it is told not to retry them.
3. File tools are confined to their roots (`virtual_mode=True`). Shell commands
   are not confined, so they are governed by approval. Secrets are stripped
   from the shell environment.
4. Skills and source edits are file writes, so they need approval, and so does
   running scripts. A restart after source edits requires a passing test suite.
5. In a container, shell commands run inside the container.
