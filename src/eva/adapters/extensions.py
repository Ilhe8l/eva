import ast
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


_NAME = re.compile(r"[a-z][a-z0-9_]{2,63}\Z")


@dataclass(frozen=True)
class Extension:
    name: str
    description: str
    source: str


class ExtensionStore:
    def __init__(self, data_dir: Path) -> None:
        self.root = data_dir / "extensions"
        self.proposals = self.root / "proposals"
        self.active = self.root / "active"
        self.proposals.mkdir(parents=True, exist_ok=True)
        self.active.mkdir(parents=True, exist_ok=True)

    def propose(self, name: str, description: str, source: str) -> str:
        self._validate(name, description, source)
        if self._path(self.active, name).exists():
            raise ValueError(f"An active extension named {name} already exists")
        path = self._path(self.proposals, name)
        path.write_text(
            json.dumps(
                {"name": name, "description": description, "source": source},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return f"Proposal saved as {name}. The user must review and activate it."

    def list_proposals(self) -> list[str]:
        return sorted(path.stem for path in self.proposals.glob("*.json"))

    def read_proposal(self, name: str) -> Extension:
        return self._read(self._path(self.proposals, name))

    def activate(self, name: str) -> Extension:
        proposal = self.read_proposal(name)
        self._validate(proposal.name, proposal.description, proposal.source)
        destination = self._path(self.active, name)
        destination.write_text(
            json.dumps(proposal.__dict__, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        self._path(self.proposals, name).unlink()
        return proposal

    def list_active(self) -> list[Extension]:
        return [self._read(path) for path in sorted(self.active.glob("*.json"))]

    def run(self, extension: Extension, input_text: str) -> str:
        result = subprocess.run(
            [sys.executable, "-c", extension.source],
            input=json.dumps({"input": input_text}),
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        return f"Exit code: {result.returncode}\n{(result.stdout + result.stderr)[:12000]}"

    @staticmethod
    def _path(directory: Path, name: str) -> Path:
        if not _NAME.fullmatch(name):
            raise ValueError("Extension names must use 3-64 lowercase letters, numbers, or underscores")
        return directory / f"{name}.json"

    @staticmethod
    def _validate(name: str, description: str, source: str) -> None:
        if not _NAME.fullmatch(name):
            raise ValueError("Invalid extension name")
        if len(description) < 10 or len(description) > 500:
            raise ValueError("Description must have 10-500 characters")
        if len(source) > 50000:
            raise ValueError("Extension source is too large")
        ast.parse(source, filename=f"{name}.py")
        compile(source, f"{name}.py", "exec")

    @staticmethod
    def _read(path: Path) -> Extension:
        data = json.loads(path.read_text(encoding="utf-8"))
        extension = Extension(**data)
        ExtensionStore._validate(
            extension.name, extension.description, extension.source
        )
        return extension
