import pytest

from eva.adapters.extensions import ExtensionStore


SCRIPT = "import json, sys\nprint(json.loads(sys.stdin.read())['input'].upper())\n"


def test_extension_is_inert_until_activation(tmp_path):
    store = ExtensionStore(tmp_path)
    store.propose("uppercase", "Convert text to uppercase", SCRIPT)
    assert store.list_proposals() == ["uppercase"]
    assert store.list_active() == []
    store.activate("uppercase")
    active = ExtensionStore(tmp_path).list_active()
    assert [item.name for item in active] == ["uppercase"]
    assert "HELLO" in store.run(active[0], "hello")


def test_extension_rejects_invalid_source_and_path(tmp_path):
    store = ExtensionStore(tmp_path)
    with pytest.raises(SyntaxError):
        store.propose("bad_code", "A long enough description", "def nope(")
    with pytest.raises(ValueError):
        store.propose("../bad", "A long enough description", SCRIPT)
