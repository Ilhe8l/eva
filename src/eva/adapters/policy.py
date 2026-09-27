"""Which tool calls pause for human approval.

Deep Agents' `FilesystemPermission` rules do not cover `execute`, and they
cannot target the default route of a backend that runs commands
(https://docs.langchain.com/oss/python/deepagents/permissions). Eva therefore
builds `interrupt_on` entries with `when` predicates for
`HumanInTheLoopMiddleware`
(https://docs.langchain.com/oss/python/deepagents/human-in-the-loop).

Read-only shell commands and memory writes run freely. Everything else pauses.
"""

import posixpath
import shlex
from collections.abc import Callable, Iterable
from pathlib import PurePosixPath

from langchain.agents.middleware import InterruptOnConfig
from langchain.tools.tool_node import ToolCallRequest

MEMORY_ROUTE = "/memories/"
FREE_WRITE_PREFIXES = (MEMORY_ROUTE,)
SENSITIVE_MARKERS = (".env", ".ssh", ".gnupg", ".aws", ".netrc", "id_rsa", "id_ed25519", "credentials")
_CHAIN_OPERATORS = {"|", "&&", "||"}
_FORBIDDEN_CHARACTERS = ("$", "`", "\n")


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
    "cat": _any_args,
    "date": _date_args,
    "df": _any_args,
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
    "rg": _no_flags("--pre", "--pre-glob"),
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

    Pipelines and `&&`/`||` chains are safe when every part is safe. Redirects,
    substitutions, background jobs and sensitive paths always need approval.
    """
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


def _path_arg(request: ToolCallRequest) -> str:
    return str(request.tool_call["args"].get("file_path", ""))


def _approval(when: Callable[[ToolCallRequest], bool] | None = None) -> InterruptOnConfig:
    config = InterruptOnConfig(allowed_decisions=["approve", "reject"])
    if when is not None:
        config["when"] = when
    return config


def approval_rules(always_ask: Iterable[str] = ()) -> dict[str, InterruptOnConfig]:
    """Build the `interrupt_on` mapping for `create_deep_agent`."""
    write_rule = _approval(lambda request: needs_write_approval(_path_arg(request)))
    rules = {
        "execute": _approval(lambda request: not is_safe_command(str(request.tool_call["args"].get("command", "")))),
        "read_file": _approval(lambda request: is_sensitive_path(_path_arg(request))),
        "write_file": write_rule,
        "edit_file": write_rule,
        "delete": write_rule,
    }
    rules.update({name: _approval() for name in always_ask})
    return rules
