---
name: skill-authoring
description: How to give yourself a new ability by writing a skill. Read this before creating or changing anything under /skills/.
---

# Writing a skill

A skill is a folder under `/skills/` that teaches you how to do one kind of
task. The name and description of every skill are listed in your prompt. You
read the full `SKILL.md` only when a task needs it.

## Layout

```
/skills/<name>/
├── SKILL.md          # required: frontmatter + instructions
└── scripts/          # optional: code you run with `execute`
    └── <script>.py
```

- `<name>`: lowercase letters, digits and hyphens, at most 64 characters. It
  must match the `name` field.
- `description`: say what the skill does *and* when to use it. It is all you
  will see before opening the skill, so make it specific.

```markdown
---
name: image-metadata
description: Read EXIF metadata (camera, date, GPS) from photos. Use when the user asks where or when a photo was taken.
---

# Image metadata

Run `uv run <skills dir>/image-metadata/scripts/exif.py <image path>`.
It prints JSON; summarize the fields the user asked about.
```

## Scripts

- Prefer instructions and existing tools. Write a script only when you need
  code, for example to parse a format or call a library.
- Write Python scripts as standalone `uv` scripts, with dependencies declared
  inline, so they never touch your own environment:

  ```python
  # /// script
  # requires-python = ">=3.12"
  # dependencies = ["pillow"]
  # ///
  import sys
  ...
  ```

- Run them with `uv run <absolute path on disk>/scripts/<script>.py args`. The
  system prompt tells you where `/skills/` is on disk. Running a script needs
  the user's approval, like any command that is not read-only.
- Take input as arguments, print results to stdout (JSON for structured data),
  and exit non-zero on failure with a clear message on stderr.

## Workflow

1. Check whether an existing skill or tool already covers the task.
2. Write `SKILL.md`, then any scripts. Each write asks for the user's approval.
3. Try the skill on the real task. Fix it until it works.
4. Note in `/memories/AGENTS.md` what you learned, if it matters later.

New and edited skills are picked up from the next user message.

## Skills or source code?

Use a skill for a new ability built from instructions and scripts. Change
`/src/eva/` only for your core behavior (tools, prompts, the terminal); those
changes need tests and `restart_eva`.
