import pytest

from eva.adapters.memory import MemoryStore


def test_journal_can_be_edited_and_survives_new_store(tmp_path):
    memory = MemoryStore(tmp_path)
    memory.write("people/user.md", "Prefers short replies")
    memory.edit("people/user.md", "short", "detailed")
    assert MemoryStore(tmp_path).read("people/user.md") == "Prefers detailed replies"
    assert memory.list() == ["people/user.md"]
    memory.delete("people/user.md")
    assert memory.list() == []


@pytest.mark.parametrize("path", ["../escape.md", "/tmp/escape.md", "notes.txt"])
def test_journal_rejects_paths_outside_markdown_folder(tmp_path, path):
    with pytest.raises(ValueError):
        MemoryStore(tmp_path).write(path, "content")
