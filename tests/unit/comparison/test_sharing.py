"""T-805: saved AIs as shareable files, imported safely."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from retro_ai_lab import __version__
from retro_ai_lab.comparison import (
    SavedAi,
    SavedAiChanged,
    SavedAiService,
    export_saved_ai,
    import_saved_ai,
)
from retro_ai_lab.core import ConfigurationError, UnsupportedCapabilityError
from retro_ai_lab.storage import SavedAiStore
from tests.unit.comparison.test_compare import GAME, go, saved
from tests.unit.training.test_service import Lab


def test_export_and_import_round_trip(tmp_path: Path) -> None:
    a = saved("AI A", (go(1, 5.0), go(2, 7.0, 10, 1)), personality="explorer")
    path = export_saved_ai(a, tmp_path / "ai-a.json")
    data = json.loads(path.read_text("utf-8"))
    assert data["kind"] == "saved_ai" and data["schema"] == 1
    assert data["app_version"] == __version__ and data["name"] == "AI A"
    back = import_saved_ai(path)
    assert back == a
    with pytest.raises(ConfigurationError, match="Can't write the shared AI"):
        export_saved_ai(a, tmp_path / "nope" / "deep" / "x.json")


def _write(tmp_path: Path, name: str, text: str) -> Path:
    path = tmp_path / name
    path.write_text(text, "utf-8")
    return path


@pytest.mark.parametrize(
    ("make", "match"),
    [
        (lambda d: "not json at all", "is not a shared AI file"),
        (lambda d: "[1, 2, 3]", "is not a shared AI file"),
        (lambda d: json.dumps(d).replace('"schema": 1', '"schema": NaN'), "NaN is not allowed"),
        (lambda d: json.dumps({**d, "kind": "knowledge"}), "is not a shared AI file"),
        (lambda d: json.dumps({k: v for k, v in d.items() if k != "kind"}), "is not a shared AI"),
        (lambda d: json.dumps({**d, "schema": 9}), "newer version"),
        (lambda d: json.dumps({**d, "name": ""}), "Give the AI a name"),
        (
            lambda d: json.dumps({**d, "brain": {**d["brain"], "agent": {"algorithm": "magic"}}}),
            "brain can't be used",
        ),
        (
            lambda d: json.dumps({**d, "rewards": {**d["rewards"], "behaviours": [{"id": 1}]}}),
            "rewards can't be used",
        ),
        (
            lambda d: json.dumps({**d, "goal": {**d["goal"], "objectives": [{"bad": True}]}}),
            "goal can't be used",
        ),
        (lambda d: json.dumps({**d, "notes": [1]}), "Notes must be text"),
        (
            lambda d: json.dumps(
                {**d, "goal": {**d["goal"], "deep": json.loads("[" * 40 + "]" * 40)}}
            ),
            "nested far deeper",
        ),
    ],
)
def test_bad_files_are_refused(tmp_path: Path, make: Any, match: str) -> None:
    good = saved("AI A", (go(1, 5.0),)).to_dict()
    path = _write(tmp_path, "shared.json", make(json.loads(json.dumps(good))))
    with pytest.raises(ConfigurationError, match=match):
        import_saved_ai(path)


def test_missing_and_oversized_files_are_refused(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError, match="Can't read the shared AI"):
        import_saved_ai(tmp_path / "missing.json")
    path = export_saved_ai(saved("AI A"), tmp_path / "big.json")
    with pytest.raises(ConfigurationError, match="too big to be a shared AI"):
        import_saved_ai(path, max_bytes=100)
    assert import_saved_ai(path, max_bytes=path.stat().st_size).name == "AI A"
    path.write_bytes(b"\xff\xfe not utf-8 \x00")
    with pytest.raises(ConfigurationError, match="Can't read the shared AI"):
        import_saved_ai(path)


def test_service_imports_for_the_connected_game_without_replacing(lab: Lab, tmp_path: Path) -> None:
    ais = SavedAiService(
        lab.rewards, lab.goals, lab.training, lab.bus, store=SavedAiStore(tmp_path / "ais")
    )
    shared = tmp_path / "friend.json"
    with pytest.raises(UnsupportedCapabilityError):
        ais.import_from(shared)
    with pytest.raises(UnsupportedCapabilityError):
        ais.export_to("AI A", shared)
    lab.connect()
    game_id = lab.training.game_id
    assert game_id is not None
    events: list[SavedAiChanged] = []
    lab.bus.subscribe(SavedAiChanged, events.append)
    lab.training.train(1)
    mine = ais.save_as("AI A", notes="mine")
    assert ais.export_to("AI A", shared) == shared
    with pytest.raises(ConfigurationError, match="No saved AI called 'Nope'"):
        ais.export_to("Nope", shared)
    # a friend's AI with the same name sits next to mine, never over it
    friend = import_saved_ai(shared)
    friend = SavedAi(
        friend.name, friend.game_id, friend.rewards, friend.goal, friend.brain, notes="theirs"
    )
    export_saved_ai(friend, shared)
    got = ais.import_from(shared)
    assert got.name == "AI A (2)" and got.notes == "theirs"
    assert ais.import_from(shared).name == "AI A (3)"
    assert ais.names() == ("AI A", "AI A (2)", "AI A (3)")
    assert ais.get("AI A").notes == "mine" and ais.get("AI A").created == mine.created
    assert events[-1] == SavedAiChanged(game_id, "AI A (3)", "imported")
    # a name that would land on another AI's file is skipped too ("AI_A (2)" and
    # "AI_A (3)" would share files with "AI A (2)" and "AI A (3)")
    export_saved_ai(SavedAi("AI_A", game_id, friend.rewards, friend.goal, friend.brain), shared)
    assert ais.import_from(shared).name == "AI_A (4)"
    # another game's AI is refused, and nothing is written
    other_path = tmp_path / "other.json"
    other_path.write_text(json.dumps(saved("Other").to_dict()).replace(GAME, "other-game"), "utf-8")
    with pytest.raises(ConfigurationError, match="for another game"):
        ais.import_from(other_path)
    assert len(ais.names()) == 4
