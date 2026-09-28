# Security

Eva runs shell commands and edits files on the computer she lives on, so her
safety model matters.

## How Eva is kept in check

- Read-only shell commands (`ls`, `cat`, `grep`, `git status`, ...) run
  immediately. Anything else waits for your approval in the terminal: other
  commands, redirects, command substitution, and paths that look like secrets
  (`.env`, `.ssh`, ...).
- File tools are confined to the project and to her journal. Writes outside
  the journal ask first, and so do skills and edits to her own source.
- API keys and other secret-looking variables are removed from the environment
  of every command she runs.
- Changes to her own code must pass the test suite before she restarts.

The full policy is in [docs/decisions.md](docs/decisions.md) and
`src/eva/adapters/policy.py`.

## Autonomous mode

`EVA_AUTONOMOUS=1` (or `:auto on`) turns approvals off: Eva then runs any
command and edits any file without asking. Use it only on a machine, or in a
container, where that is acceptable.

The model's output is untrusted. A web page, a file or a tool result can
contain instructions aimed at Eva (prompt injection). Approvals are the
safeguard against that, and autonomous mode removes them.

## Reporting a vulnerability

Please report security issues privately through
[GitHub's security advisories](https://github.com/Ilhe8l/eva/security/advisories/new)
rather than in a public issue.
