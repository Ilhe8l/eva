from pathlib import Path


class MemoryStore:
    def __init__(self, data_dir: Path) -> None:
        self.root = (data_dir / "memory").resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def list(self) -> list[str]:
        return sorted(path.relative_to(self.root).as_posix() for path in self.root.rglob("*.md"))

    def read(self, path: str) -> str:
        return self._resolve(path).read_text(encoding="utf-8")

    def write(self, path: str, content: str) -> str:
        target = self._resolve(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return f"Saved {path}"

    def edit(self, path: str, old: str, new: str) -> str:
        target = self._resolve(path)
        content = target.read_text(encoding="utf-8")
        if not old or content.count(old) != 1:
            raise ValueError("The old text must occur exactly once")
        target.write_text(content.replace(old, new, 1), encoding="utf-8")
        return f"Updated {path}"

    def delete(self, path: str) -> str:
        self._resolve(path).unlink()
        return f"Deleted {path}"

    def _resolve(self, path: str) -> Path:
        candidate = (self.root / path).resolve()
        if not candidate.is_relative_to(self.root) or candidate.suffix != ".md":
            raise ValueError("Memory path must be a Markdown file inside the memory directory")
        return candidate
