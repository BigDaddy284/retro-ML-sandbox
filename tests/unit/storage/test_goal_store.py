"""T-407: goals on disk."""

from __future__ import annotations

from pathlib import Path

import pytest

from retro_ai_lab.core import ConfigurationError
from retro_ai_lab.rewards import Goal, Objective, Trigger
from retro_ai_lab.storage import GoalStore
from retro_ai_lab.storage.goals import export_goal, load_goal
from retro_ai_lab.understanding import EventKind


@pytest.fixture
def store(tmp_path: Path) -> GoalStore:
    return GoalStore(tmp_path / "goals")


def sample() -> Goal:
    g = Goal("toyracer-0", title="Win")
    g.add(Objective("a", "Pass a checkpoint", Trigger(EventKind.PROGRESS, "v1")))
    g.add(Objective("b", "Finish a lap", Trigger(EventKind.PROGRESS, "v2"), depends_on=("a",)))
    g.complete("a", 5)
    return g


def test_store_round_trips_lists_and_deletes(store: GoalStore) -> None:
    assert store.folder.is_dir() and store.games() == ()
    fresh = store.load_or_new("toyracer-0", title="Try")
    assert fresh.title == "Try" and len(fresh) == 0 and not store.exists("toyracer-0")
    path = store.save(sample())
    assert path == store.path_for("toyracer-0") and store.games() == ("toyracer-0",)
    again = store.load("toyracer-0")
    assert again.to_dict() == sample().to_dict()
    assert store.load_or_new("toyracer-0").title == "Win"
    assert store.delete("toyracer-0") and not store.delete("toyracer-0")
    with pytest.raises(ConfigurationError):
        store.load("toyracer-0")


def test_export_load_and_errors(tmp_path: Path) -> None:
    target = tmp_path / "goal.json"
    export_goal(sample(), target)
    assert load_goal(target).get("a").completed
    (tmp_path / "bad.json").write_text("{", "utf-8")
    with pytest.raises(ConfigurationError):
        load_goal(tmp_path / "bad.json")
    with pytest.raises(ConfigurationError):
        load_goal(tmp_path / "missing.json")
    (tmp_path / "file").write_text("x", "utf-8")
    with pytest.raises(ConfigurationError):
        GoalStore(tmp_path / "file")
