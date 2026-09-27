from eva.adapters.lifecycle import SelfUpdater


def _project(tmp_path, assertion):
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_sample.py").write_text(f"def test_sample():\n    assert {assertion}\n")
    return tmp_path


def test_restart_is_requested_only_when_tests_pass(tmp_path):
    updater = SelfUpdater(_project(tmp_path, "True"))
    assert "Tests passed" in updater.request_restart()
    assert updater.restart_requested


def test_failing_tests_block_the_restart(tmp_path):
    updater = SelfUpdater(_project(tmp_path, "False"))
    assert "tests failed" in updater.request_restart()
    assert not updater.restart_requested
