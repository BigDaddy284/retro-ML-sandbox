"""T-1064: the way the AI remembers, in a JSON file per game."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from retro_ai_lab.core import ConfigurationError
from retro_ai_lab.storage import WayStore

GAME = "nes/Super Game (World)/0123abcd"
WAY = {
    "format": 1,
    "game_id": GAME,
    "slot": "way_0123456789ab",
    "spot": {"ram": "00ff"},
    "frames_per_step": 4,
    "actions": [[], ["right"], ["right", "a"]],
    "way": {"moves": [1, 2], "paid": [1.0, 2.5], "by": [[["x", 1.0]], [["x", 1.5]]], "stood": [9]},
}


def test_a_way_is_saved_and_loaded(tmp_path: Path) -> None:
    store = WayStore(tmp_path / "ways")
    assert store.folder == tmp_path / "ways" and store.folder.is_dir()
    assert not store.exists(GAME) and store.games() == ()
    path = store.save(WAY)
    assert path == store.path_for(GAME) and path.parent == store.folder
    assert path.name == "nes_Super_Game_World_0123abcd.json"  # as the brain store names it
    assert store.exists(GAME) and store.games() == (GAME,)
    assert store.load(GAME) == WAY
    assert json.loads(path.read_text("utf-8")) == WAY
    assert not list(store.folder.glob("*.tmp"))


def test_a_way_saved_again_takes_the_place_of_the_one_before(tmp_path: Path) -> None:
    store = WayStore(tmp_path)
    store.save(WAY)
    longer = {**WAY, "way": {"moves": [1, 2, 2], "paid": [1.0, 2.5, 3.0], "by": [[], [], []]}}
    store.save(longer)
    assert store.load(GAME) == longer and store.games() == (GAME,)


def test_each_game_has_its_own(tmp_path: Path) -> None:
    store = WayStore(tmp_path)
    other = {**WAY, "game_id": "gba/Other/ffff"}
    store.save(WAY)
    store.save(other)
    assert store.games() == ("gba/Other/ffff", GAME)
    assert store.load("gba/Other/ffff") == other and store.load(GAME) == WAY
    assert store.delete(GAME)
    assert not store.exists(GAME) and store.exists("gba/Other/ffff")


def test_deleting(tmp_path: Path) -> None:
    store = WayStore(tmp_path)
    assert not store.delete(GAME)  # nothing there: nothing done, and no error
    store.save(WAY)
    assert store.delete(GAME) and not store.exists(GAME)
    assert not store.delete(GAME)
    with pytest.raises(ConfigurationError, match="No way written down"):
        store.load(GAME)


@pytest.mark.parametrize("game", [None, "", 7])
def test_a_way_to_save_names_its_game(tmp_path: Path, game: object) -> None:
    store = WayStore(tmp_path)
    with pytest.raises(ConfigurationError, match="must name its game"):
        store.save({**WAY, "game_id": game})
    with pytest.raises(ConfigurationError, match="must name its game"):
        store.save({"way": {}})
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("text", ["", "{", "[1, 2]", '"way"', "null"])
def test_a_file_that_is_not_a_way_is_refused(tmp_path: Path, text: str) -> None:
    store = WayStore(tmp_path)
    store.path_for(GAME).write_text(text, "utf-8")
    assert store.exists(GAME)
    with pytest.raises(ConfigurationError):
        store.load(GAME)
    assert store.games() == ()  # not one of ours: left alone and not listed


def test_what_json_cannot_hold_leaves_no_half_file(tmp_path: Path) -> None:
    store = WayStore(tmp_path)
    store.save(WAY)
    with pytest.raises(TypeError):
        store.save({**WAY, "spot": {"ram": b"\x00"}})
    assert store.load(GAME) == WAY and not list(tmp_path.glob("*.tmp"))


def test_the_folder_must_be_a_folder(tmp_path: Path) -> None:
    file = tmp_path / "ways"
    file.write_text("x", "utf-8")
    with pytest.raises(ConfigurationError, match="is not a folder"):
        WayStore(file)
    with pytest.raises(ConfigurationError, match="can't be turned into a file name"):
        WayStore(tmp_path / "w").path_for("///")
