import subprocess
from pathlib import Path

import pytest

from eva.adapters.patches import PatchStore


DIFF = """diff --git a/src/eva/example.py b/src/eva/example.py
--- a/src/eva/example.py
+++ b/src/eva/example.py
@@ -1 +1 @@
-value = 1
+value = 2
"""


@pytest.mark.parametrize(
    "diff",
    [
        DIFF.replace("src/eva/example.py", "../../outside.py"),
        DIFF.replace("src/eva/example.py", ".git/config"),
        DIFF.replace("src/eva/example.py", "docs/architecture.md"),
    ],
)
def test_source_patch_rejects_paths_outside_source(tmp_path, diff):
    with pytest.raises(ValueError):
        PatchStore(tmp_path / "data", tmp_path).propose(
            "bad_patch", "A long enough description", diff
        )


def test_reviewed_source_patch_runs_tests_before_staying_applied(tmp_path):
    root = tmp_path / "project"
    (root / "src/eva").mkdir(parents=True)
    (root / "tests").mkdir()
    (root / "src/eva/example.py").write_text("value = 1\n")
    (root / "tests/test_example.py").write_text(
        "from pathlib import Path\n"
        "def test_example():\n"
        "    path = Path(__file__).parents[1] / 'src/eva/example.py'\n"
        "    assert path.read_text() == 'value = 2\\n'\n"
    )
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    subprocess.run(["git", "add", "."], cwd=root, check=True)
    subprocess.run(
        ["git", "-c", "user.name=Eva Test", "-c", "user.email=test@example.invalid", "commit", "-qm", "initial"],
        cwd=root,
        check=True,
    )
    store = PatchStore(tmp_path / "data", root)
    store.propose("fix_example", "Set expected example value", DIFF)
    success, detail = store.apply("fix_example")
    assert success, detail
    assert (root / "src/eva/example.py").read_text() == "value = 2\n"
    assert store.list() == []
