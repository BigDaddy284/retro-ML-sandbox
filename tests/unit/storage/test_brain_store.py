"""T-506: brains on disk."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from retro_ai_lab.core import ConfigurationError
from retro_ai_lab.emulator.fake.toy_racer import BUTTONS
from retro_ai_lab.learning import ActionSpace, QLearningAgent
from retro_ai_lab.storage import BrainStore
from retro_ai_lab.storage.brains import export_brain, load_brain
from retro_ai_lab.training import Brain


@pytest.fixture
def store(tmp_path: Path) -> BrainStore:
    return BrainStore(tmp_path / "brains")


def sample(game_id: str = "toyracer-0") -> dict[str, object]:
    data = Brain(game_id, QLearningAgent(ActionSpace.from_buttons(BUTTONS), seed=1)).to_dict()
    return json.loads(json.dumps(data))  # as it comes back from disk (tuples become lists)


def test_store_creates_folder_lists_games_and_deletes(store: BrainStore) -> None:
    assert store.folder.is_dir() and store.games() == ()
    path = store.save(sample())
    assert path == store.path_for("toyracer-0") and path.exists()
    store.save(sample("other/game:1"))
    assert store.games() == ("other/game:1", "toyracer-0")
    assert store.exists("toyracer-0") and not store.exists("nope")
    assert store.load("toyracer-0") == sample()
    assert Brain.from_dict(store.load("toyracer-0")).game_id == "toyracer-0"
    assert store.delete("toyracer-0") and not store.delete("toyracer-0")
    assert store.games() == ("other/game:1",)
    with pytest.raises(ConfigurationError, match="No saved brain"):
        store.load("toyracer-0")
    with pytest.raises(ConfigurationError, match="name its game"):
        store.save({"schema": 1})


def test_damaged_and_foreign_files(store: BrainStore, tmp_path: Path) -> None:
    store.path_for("bad").write_text("{not json", "utf-8")
    with pytest.raises(ConfigurationError, match="not valid JSON"):
        store.load("bad")
    (store.folder / "notes.json").write_text(json.dumps([1, 2]), "utf-8")
    assert store.games() == ()  # a list isn't a brain; left alone
    with pytest.raises(ConfigurationError, match="does not hold"):
        load_brain(store.folder / "notes.json")
    out = export_brain(sample(), tmp_path / "exported.json")
    assert load_brain(out) == sample()
    with pytest.raises(ConfigurationError, match="Can't read"):
        load_brain(tmp_path / "missing.json")
