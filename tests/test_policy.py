import pytest

from eva.adapters.policy import is_safe_command, is_sensitive_path, needs_write_approval


@pytest.mark.parametrize(
    "command",
    ["ls -la", "pwd", "git status", "git log --oneline | head -5", "cat README.md && wc -l README.md",
     'grep -rn "a|b" src', "date +%F", "find . -name '*.py'"],
)
def test_read_only_commands_run_freely(command):
    assert is_safe_command(command)


@pytest.mark.parametrize(
    "command",
    ["rm -rf build", "ls > out.txt", "ls; rm x", "echo $HOME", "echo `id`", "cat .env", "cat ~/.ssh/id_rsa",
     "find . -delete", "find . -exec rm {} ;", "git push", "git -c core.pager=sh log", "git diff --output=x",
     "python script.py", "sleep 5 &", "", "date -s 2020-01-01", "rg --pre=sh x"],
)
def test_other_commands_need_approval(command):
    assert not is_safe_command(command)


@pytest.mark.parametrize(
    ("path", "expected"),
    [("/memories/AGENTS.md", False), ("memories/people/user.md", False), ("/src/eva/cli.py", True),
     ("/memories/../src/eva/cli.py", True), ("/memories", False), ("/README.md", True)],
)
def test_only_journal_writes_are_free(path, expected):
    assert needs_write_approval(path) is expected


def test_secret_files_are_sensitive():
    assert is_sensitive_path("/.env")
    assert is_sensitive_path("/home/user/.ssh/config")
    assert not is_sensitive_path("/src/eva/environment.py")
