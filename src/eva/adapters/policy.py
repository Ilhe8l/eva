"""Which tool calls pause for human approval.

Deep Agents' `FilesystemPermission` rules do not cover `execute`, and they
cannot target the default route of a backend that runs commands
(https://docs.langchain.com/oss/python/deepagents/permissions). Eva therefore
builds `interrupt_on` entries with `when` predicates for
`HumanInTheLoopMiddleware`
(https://docs.langchain.com/oss/python/deepagents/human-in-the-loop).

Read-only shell commands and writes to the journal or the scratch space run
freely. Everything else pauses.
"""

import posixpath
import re
import shlex
from collections.abc import Callable
from pathlib import PurePosixPath

from langchain.agents.middleware import InterruptOnConfig

MEMORY_ROUTE = "/memories/"
SKILLS_ROUTE = "/skills/"
BUILTIN_SKILLS_ROUTE = "/builtin-skills/"
SCRATCH_ROUTE = "/scratch/"
FREE_WRITE_PREFIXES = (MEMORY_ROUTE, SCRATCH_ROUTE)
SENSITIVE_MARKERS = (".env", ".ssh", ".gnupg", ".aws", ".netrc", "id_rsa", "id_ed25519", "credentials")
_CHAIN_OPERATORS = {"|", "&&", "||"}
_FORBIDDEN_CHARACTERS = ("$", "`", "\n")
# Discarding output is harmless, so these redirects do not make a command unsafe.
_DISCARD_REDIRECTS = re.compile(r"(?<!\S)(?:[12]?>|&>)\s*/dev/null(?!\S)|(?<!\S)2>&1(?!\S)")


def _no_flags(*flags: str) -> Callable[[list[str]], bool]:
    def check(args: list[str]) -> bool:
        return not any(arg == flag or arg.startswith(f"{flag}=") for arg in args for flag in flags)

    return check


def _any_args(_: list[str]) -> bool:
    return True


def _date_args(args: list[str]) -> bool:
    return all(arg.startswith("+") for arg in args)


_SAFE_GIT_SUBCOMMANDS = {"status", "log", "diff", "show", "ls-files", "rev-parse", "blame"}


def _git_args(args: list[str]) -> bool:
    if not args or args[0] not in _SAFE_GIT_SUBCOMMANDS:
        return False
    return not any(arg.startswith(("--output", "--ext-diff", "--textconv")) for arg in args)


SAFE_COMMANDS: dict[str, Callable[[list[str]], bool]] = {
    "basename": _any_args,
    "cat": _any_args,
    "cut": _any_args,
    "date": _date_args,
    "df": _any_args,
    "dirname": _any_args,
    "du": _any_args,
    "echo": _any_args,
    "file": _any_args,
    "find": _no_flags("-exec", "-execdir", "-ok", "-okdir", "-delete", "-fprint", "-fprint0", "-fprintf", "-fls"),
    "free": _any_args,
    "git": _git_args,
    "grep": _any_args,
    "head": _any_args,
    "hostname": lambda args: not args,
    "ls": _any_args,
    "nvidia-smi": lambda args: not args,
    "ps": _any_args,
    "pwd": _any_args,
    "realpath": _any_args,
    "rg": _no_flags("--pre", "--pre-glob"),
    "sort": _no_flags("-o", "--output"),
    "stat": _any_args,
    "tail": _any_args,
    "tree": _no_flags("-o"),
    "uname": _any_args,
    "uptime": _any_args,
    "wc": _any_args,
    "which": _any_args,
    "whoami": _any_args,
}


def is_safe_command(command: str) -> bool:
    """Return True for a read-only command that may run without approval.

    Pipelines and `&&`/`||` chains are safe when every part is safe. Redirects
    (other than to /dev/null), substitutions, background jobs and sensitive paths
    always need approval.
    """
    command = _DISCARD_REDIRECTS.sub(" ", command)
    if not command.strip() or any(char in command for char in _FORBIDDEN_CHARACTERS):
        return False
    lexer = shlex.shlex(command, posix=True, punctuation_chars=True)
    lexer.whitespace_split = True
    try:
        tokens = list(lexer)
    except ValueError:
        return False
    segments: list[list[str]] = [[]]
    for token in tokens:
        if token in _CHAIN_OPERATORS:
            segments.append([])
        elif token and all(char in "();<>|&" for char in token):
            return False
        else:
            segments[-1].append(token)
    return all(_is_safe_segment(segment) for segment in segments)


def _is_safe_segment(words: list[str]) -> bool:
    if not words:
        return False
    name, args = words[0], words[1:]
    check = SAFE_COMMANDS.get(name)
    return check is not None and check(args) and not any(is_sensitive_path(arg) for arg in args)


def is_sensitive_path(path: str) -> bool:
    return any(marker in part for part in PurePosixPath(path).parts for marker in SENSITIVE_MARKERS)


def needs_write_approval(path: str) -> bool:
    normalized = posixpath.normpath("/" + path.lstrip("/")) + "/"
    return not normalized.startswith(FREE_WRITE_PREFIXES)


class ApprovalPolicy:
    """Decides which tool calls pause for the user; the user can widen it live.

    In autonomous mode nothing pauses. Otherwise read-only commands and writes
    to the journal or scratch space run freely, anything the user chose to "always allow" this session
    runs freely, and everything else pauses.
    """

    RULES: dict[str, Callable[[dict], bool]] = {
        "execute": lambda args: not is_safe_command(str(args.get("command", ""))),
        "read_file": lambda args: is_sensitive_path(str(args.get("file_path", ""))),
        "write_file": lambda args: needs_write_approval(str(args.get("file_path", ""))),
        "edit_file": lambda args: needs_write_approval(str(args.get("file_path", ""))),
        "delete": lambda args: needs_write_approval(str(args.get("file_path", ""))),
    }

    def __init__(self, autonomous: bool = False) -> None:
        self.autonomous = autonomous
        self._always: set[tuple[str, str]] = set()

    def needs_approval(self, tool: str, args: dict) -> bool:
        if self.autonomous or (tool, subject(tool, args)) in self._always:
            return False
        rule = self.RULES.get(tool)
        return rule is not None and rule(args)

    def always_allow(self, tool: str, args: dict) -> None:
        """Stop asking about this exact command or file for the rest of the session."""
        self._always.add((tool, subject(tool, args)))

    def interrupt_on(self) -> dict[str, InterruptOnConfig]:
        """The `interrupt_on` mapping for `create_deep_agent`."""
        return {tool: self._rule(tool) for tool in self.RULES}

    def _rule(self, tool: str) -> InterruptOnConfig:
        return InterruptOnConfig(
            allowed_decisions=["approve", "reject"],
            when=lambda request: self.needs_approval(tool, request.tool_call["args"]),
        )


def subject(tool: str, args: dict) -> str:
    """What an approval is about: the exact command, or the file."""
    return str(args.get("command") if tool == "execute" else args.get("file_path", ""))
