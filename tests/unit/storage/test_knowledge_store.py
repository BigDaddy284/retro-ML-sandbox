"""T-301: knowledge files on disk."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from retro_ai_lab.core import ConfigurationError
from retro_ai_lab.storage import KnowledgeStore
from retro_ai_lab.storage.knowledge import export, load
from retro_ai_lab.understanding import Evidence, Knowledge, VariableKind


@pytest.fixture
def store(tmp_path: Path) -> KnowledgeStore:
    return KnowledgeStore(tmp_path / "knowledge")


def sample() -> Knowledge:
    k = Knowledge("toyracer-0", title="Toy Racer")
    k.propose("ram", 8, kind=VariableKind.SCORE, evidence=[Evidence("a", "b", 2.0)])
    return k


def test_store_creates_folder_and_round_trips(store: KnowledgeStore) -> None:
    assert store.folder.is_dir()
    assert store.games() == () and not store.exists("toyracer-0")
    path = store.save(sample())
    assert path == store.path_for("toyracer-0") and path.name == "toyracer-0.json"
    assert store.exists("toyracer-0") and store.games() == ("toyracer-0",)
    loaded = store.load("toyracer-0")
    assert loaded.to_dict() == sample().to_dict()
    assert json.loads(path.read_text("utf-8"))["game_id"] == "toyracer-0"  # human readable
    assert not list(store.folder.glob("*.tmp"))


def test_load_or_new_and_delete(store: KnowledgeStore) -> None:
    fresh = store.load_or_new("new-game", title="New")
    assert len(fresh) == 0 and fresh.title == "New"
    store.save(sample())
    assert len(store.load_or_new("toyracer-0")) == 1
    assert store.delete("toyracer-0") and not store.delete("toyracer-0")
    with pytest.raises(ConfigurationError, match="No saved knowledge"):
        store.load("toyracer-0")


def test_awkward_game_ids_become_safe_file_names(store: KnowledgeStore) -> None:
    k = Knowledge("Pokémon Red (U) [!]/v1.0:abc")
    path = store.save(k)
    assert path.parent == store.folder and "/" not in path.name and ":" not in path.name
    assert store.load(k.game_id).game_id == k.game_id
    with pytest.raises(ConfigurationError):
        store.path_for("///")


def test_games_ignores_foreign_files(store: KnowledgeStore) -> None:
    (store.folder / "notes.json").write_text("[1, 2]", "utf-8")
    (store.folder / "broken.json").write_text("{not json", "utf-8")
    store.save(sample())
    assert store.games() == ("toyracer-0",)


def test_export_and_load_anywhere(tmp_path: Path) -> None:
    target = tmp_path / "shared" / "racer.json"
    target.parent.mkdir()
    export(sample(), target)
    assert load(target).to_dict() == sample().to_dict()
    with pytest.raises(ConfigurationError, match="valid JSON"):
        target.write_text("{", "utf-8")
        load(target)
    with pytest.raises(ConfigurationError, match="knowledge object"):
        target.write_text("[]", "utf-8")
        load(target)
    with pytest.raises(ConfigurationError, match="Can't read"):
        load(tmp_path / "missing.json")
    with pytest.raises(ConfigurationError, match="Can't write"):
        export(sample(), tmp_path / "nope" / "deeper" / "x.json")


def test_store_rejects_a_file_as_folder(tmp_path: Path) -> None:
    f = tmp_path / "file"
    f.write_text("x")
    with pytest.raises(ConfigurationError):
        KnowledgeStore(f)
