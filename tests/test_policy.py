import pytest

from eva.adapters.policy import is_safe_command, is_sensitive_path, needs_write_approval


@pytest.mark.parametrize(
    "command",
    [
        "ls -la",
        "pwd",
        "git status",
        "git log --oneline | head -5",
        "cat README.md && wc -l README.md",
        'grep -rn "a|b" src',
        "date +%F",
        "find . -name '*.py'",
        "du -h -d 1 . 2>/dev/null | sort -h -r | head",
        "grep -r TODO src 2>&1 | wc -l",
    ],
)
def test_read_only_commands_run_freely(command):
    assert is_safe_command(command)


@pytest.mark.parametrize(
    "command",
    [
        "rm -rf build",
        "ls > out.txt",
        "ls; rm x",
        "echo $HOME",
        "echo `id`",
        "cat .env",
        "cat ~/.ssh/id_rsa",
        "find . -delete",
        "find . -exec rm {} ;",
        "git push",
        "git -c core.pager=sh log",
        "git diff --output=x",
        "python script.py",
        "sleep 5 &",
        "sort -o sorted.txt data.txt",
        "ls 2>/tmp/errors.log",
        "cat notes.txt >/dev/nullx",
        "",
        "date -s 2020-01-01",
        "rg --pre=sh x",
    ],
)
def test_other_commands_need_approval(command):
    assert not is_safe_command(command)


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("/memories/AGENTS.md", False),
        ("memories/people/user.md", False),
        ("/src/eva/cli.py", True),
        ("/memories/../src/eva/cli.py", True),
        ("/memories", False),
        ("/README.md", True),
        ("/scratch/download.zip", False),
        ("/scratch/../src/eva/cli.py", True),
    ],
)
def test_only_journal_and_scratch_writes_are_free(path, expected):
    assert needs_write_approval(path) is expected


def test_secret_files_are_sensitive():
    assert is_sensitive_path("/.env")
    assert is_sensitive_path("/home/user/.ssh/config")
    assert not is_sensitive_path("/src/eva/environment.py")
