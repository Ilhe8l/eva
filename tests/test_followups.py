from datetime import UTC, datetime, timedelta

import pytest

from eva.adapters.followups import FollowUpStore

NOW = datetime(2026, 1, 1, 9, 0, tzinfo=UTC)


def test_follow_ups_become_due_once_and_survive_reloads(tmp_path):
    path = tmp_path / "follow_ups.json"
    early = FollowUpStore(path).add(timedelta(minutes=5), "check the build", NOW)
    FollowUpStore(path).add(timedelta(hours=2), "stretch", NOW)
    store = FollowUpStore(path)
    assert store.pop_due(NOW + timedelta(minutes=4)) == []
    assert store.pop_due(NOW + timedelta(minutes=6)) == [early]
    assert store.pop_due(NOW + timedelta(minutes=6)) == []
    assert [item.note for item in store.pending()] == ["stretch"]


def test_follow_ups_can_be_cancelled(tmp_path):
    store = FollowUpStore(tmp_path / "follow_ups.json")
    item = store.add(timedelta(minutes=5), "call mom", NOW)
    assert store.cancel(item.id)
    assert not store.cancel(item.id)
    assert store.pending() == []


@pytest.mark.parametrize(
    ("delay", "note"), [(timedelta(seconds=10), "x"), (timedelta(days=31), "x"), (timedelta(minutes=5), " ")]
)
def test_invalid_follow_ups_are_rejected(tmp_path, delay, note):
    with pytest.raises(ValueError):
        FollowUpStore(tmp_path / "follow_ups.json").add(delay, note, NOW)
