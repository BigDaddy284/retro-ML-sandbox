"""T-404: reward profiles on disk."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from retro_ai_lab.core import ConfigurationError
from retro_ai_lab.rewards import Behaviour, Category, RewardProfile, Trigger, apply_personality
from retro_ai_lab.storage import RewardStore
from retro_ai_lab.storage.rewards import export_profile, load_profile
from retro_ai_lab.understanding import EventKind


@pytest.fixture
def store(tmp_path: Path) -> RewardStore:
    return RewardStore(tmp_path / "rewards")


def sample() -> RewardProfile:
    p = RewardProfile("toyracer-0")
    p.add(
        Behaviour("reset:*", "Starting over", Category.SURVIVAL, Trigger(EventKind.RESET), -5, -5)
    )
    apply_personality(p, "survivor")
    p.increase("reset:*")
    return p


def test_store_creates_folder_and_round_trips(store: RewardStore) -> None:
    assert store.folder.is_dir()
    assert store.games() == () and not store.exists("toyracer-0")
    path = store.save(sample())
    assert path == store.path_for("toyracer-0") and path.exists()
    assert store.games() == ("toyracer-0",) and store.exists("toyracer-0")
    again = store.load("toyracer-0")
    assert again.to_dict() == sample().to_dict()
    assert again.get("reset:*").value == -9.0 and again.personality == "survivor"


def test_load_or_new_and_delete(store: RewardStore) -> None:
    fresh = store.load_or_new("new-game")
    assert fresh.game_id == "new-game" and len(fresh) == 0 and not store.exists("new-game")
    store.save(sample())
    assert len(store.load_or_new("toyracer-0")) == 1
    assert store.delete("toyracer-0") and not store.delete("toyracer-0")
    with pytest.raises(ConfigurationError):
        store.load("toyracer-0")


def test_awkward_game_ids_become_safe_file_names(store: RewardStore) -> None:
    p = RewardProfile("NES:Super Game/1 (U) [!]")
    path = store.save(p)
    assert path.name == "NES_Super_Game_1_U.json" and path.parent == store.folder
    assert store.load(p.game_id).game_id == p.game_id
    with pytest.raises(ConfigurationError):
        store.path_for("///")


def test_games_ignores_foreign_and_damaged_files(store: RewardStore) -> None:
    (store.folder / "notes.json").write_text("[1, 2]", "utf-8")
    (store.folder / "broken.json").write_text("{", "utf-8")
    store.save(sample())
    assert store.games() == ("toyracer-0",)
    with pytest.raises(ConfigurationError):
        load_profile(store.folder / "broken.json")
    with pytest.raises(ConfigurationError):
        load_profile(store.folder / "notes.json")


def test_export_and_load_anywhere(tmp_path: Path) -> None:
    target = tmp_path / "share" / "rewards.json"
    target.parent.mkdir()
    export_profile(sample(), target)
    data = json.loads(target.read_text("utf-8"))
    assert data["game_id"] == "toyracer-0" and data["personality"] == "survivor"
    assert load_profile(target).to_dict() == sample().to_dict()
    with pytest.raises(ConfigurationError):
        load_profile(tmp_path / "missing.json")
    with pytest.raises(ConfigurationError):
        export_profile(sample(), tmp_path / "no" / "such" / "folder" / "x.json")


def test_store_rejects_a_file_as_folder(tmp_path: Path) -> None:
    (tmp_path / "file").write_text("x", "utf-8")
    with pytest.raises(ConfigurationError):
        RewardStore(tmp_path / "file")
