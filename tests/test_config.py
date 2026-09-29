from pathlib import Path

from eva.config import Settings


def test_eva_keeps_her_data_in_the_home_directory(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("EVA_DATA_DIR", raising=False)
    monkeypatch.chdir(tmp_path)
    assert Settings.from_env("lmstudio:any").data_dir == tmp_path / ".eva"


def test_the_data_directory_can_be_moved(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("EVA_DATA_DIR", "~/elsewhere")
    assert Settings.from_env("lmstudio:any").data_dir == Path(tmp_path / "elsewhere")
