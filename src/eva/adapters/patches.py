import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


_NAME = re.compile(r"[a-z][a-z0-9_]{2,63}\Z")
_ALLOWED = ("src/eva/", "tests/")


@dataclass(frozen=True)
class PatchProposal:
    name: str
    description: str
    diff: str


class PatchStore:
    def __init__(self, data_dir: Path, project_root: Path) -> None:
        self.root = data_dir / "patches"
        self.root.mkdir(parents=True, exist_ok=True)
        self.project_root = project_root.resolve()

    def propose(self, name: str, description: str, diff: str) -> str:
        self._validate(name, description, diff)
        self._path(name).write_text(
            json.dumps(
                PatchProposal(name, description, diff).__dict__,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return f"Source patch {name} is staged for human review. It has not been applied."

    def list(self) -> list[str]:
        return sorted(path.stem for path in self.root.glob("*.json"))

    def read(self, name: str) -> PatchProposal:
        proposal = PatchProposal(**json.loads(self._path(name).read_text(encoding="utf-8")))
        self._validate(proposal.name, proposal.description, proposal.diff)
        return proposal

    def apply(
        self, name: str, expected: PatchProposal | None = None
    ) -> tuple[bool, str]:
        proposal = self.read(name)
        if expected is not None and proposal != expected:
            return False, "Patch changed since review; inspect it again."
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=self.project_root,
            capture_output=True,
            text=True,
            check=False,
        )
        if status.returncode != 0 or status.stdout.strip():
            return False, "The project has uncommitted changes. Review them before applying a patch."
        check = self._git_apply("--check", proposal.diff)
        if check.returncode != 0:
            return False, f"Patch check failed:\n{check.stderr}"
        applied = self._git_apply("", proposal.diff)
        if applied.returncode != 0:
            return False, f"Patch apply failed:\n{applied.stderr}"
        try:
            tests = subprocess.run(
                [sys.executable, "-m", "pytest", "-q"],
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=180,
                check=False,
            )
            failure = tests.returncode != 0
            detail = tests.stdout + tests.stderr
        except subprocess.TimeoutExpired:
            failure = True
            detail = "Tests timed out after 180 seconds."
        if failure:
            rollback = self._git_apply("--reverse", proposal.diff)
            if rollback.returncode != 0:
                detail += f"\nRollback failed: {rollback.stderr}"
            return False, f"Tests failed; patch rolled back.\n{detail[:12000]}"
        self._path(name).unlink()
        return True, f"Patch applied; tests passed.\n{tests.stdout[:12000]}"

    def _git_apply(self, flag: str, diff: str) -> subprocess.CompletedProcess[str]:
        args = ["git", "apply"]
        if flag:
            args.append(flag)
        args.append("-")
        return subprocess.run(
            args,
            cwd=self.project_root,
            input=diff,
            capture_output=True,
            text=True,
            check=False,
        )

    def _path(self, name: str) -> Path:
        if not _NAME.fullmatch(name):
            raise ValueError("Invalid patch name")
        return self.root / f"{name}.json"

    @staticmethod
    def _validate(name: str, description: str, diff: str) -> None:
        if not _NAME.fullmatch(name):
            raise ValueError("Invalid patch name")
        if not 10 <= len(description) <= 500 or len(diff) > 100000:
            raise ValueError("Invalid patch description or size")
        headers = re.findall(r"^diff --git a/(\S+) b/(\S+)$", diff, re.MULTILINE)
        if not headers or len(headers) != diff.count("diff --git "):
            raise ValueError("Expected a standard Git unified diff")
        for before, after in headers:
            if before != after or not before.startswith(_ALLOWED):
                raise ValueError("Patches may only change Eva source or tests")
        for line in diff.splitlines():
            if line.startswith(("--- ", "+++ ")):
                path = line[4:]
                if path != "/dev/null" and (
                    not path.startswith(("a/", "b/"))
                    or not path[2:].startswith(_ALLOWED)
                ):
                    raise ValueError("Patch contains a path outside Eva source or tests")
            if line.startswith(("new file mode 120000", "old mode 120000", "rename from ", "rename to ")):
                raise ValueError("Symlinks and renames are not accepted in source proposals")
