# Contributing

Thanks for your interest in Eva! Issues and pull requests are welcome.

## Setup

```bash
git clone https://github.com/Ilhe8l/eva.git && cd eva
uv sync --extra voice --extra test   # or --extra test for a text-only setup
cp .env.example .env                  # then set EVA_MODEL
```

## Before opening a pull request

```bash
uv run pytest
uv run ruff check . && uv run ruff format --check .
```

CI runs the same checks, and a pull request can merge only when they pass.

- Branch from `master` and open the pull request against it.
- Write commit messages with [Conventional Commits](https://www.conventionalcommits.org)
  (`feat:`, `fix:`, `docs:`, ...). Release-please builds the changelog from them.
- Add or update tests with your change. Changes to the approval policy need tests
  in `tests/test_policy.py` or `tests/test_agent.py`, which run the real agent graph.
- For a design change, add a short entry to [docs/decisions.md](docs/decisions.md)
  explaining what was decided and why.

## Where things live

[docs/architecture.md](docs/architecture.md) maps the code: the domain and the
application layer stay free of models, audio and the terminal, and everything
else lives in adapters.

## Security

Please report vulnerabilities privately, as described in [SECURITY.md](SECURITY.md).
