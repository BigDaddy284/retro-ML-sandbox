"""T-1064: wherever the training service saves the checkpoint, the spot is kept.

A Start is not the only thing that saves the spot the goes start from: a game
that had to be woken, a wait for a game that does not listen, a start that
loses a life by itself and is moved on, and a race all save it. After each the
game's own slot holds that state, so a way written down never names a slot that
holds another.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from retro_ai_lab.learning import ActionSpace, QLearningAgent, Settings
from retro_ai_lab.storage import WayStore
from retro_ai_lab.training import Mode, Status, TrainingService
from retro_ai_lab.training import service as module
from retro_ai_lab.understanding.wakeup import InputProbe
from tests.unit.training import test_service as in_service
from tests.unit.training.test_service import Lab

lab = in_service.lab  # the practice racer behind the real services (pytest fixture)

SLOT = module.CHECKPOINT_SLOT


def kept(lab: Lab, folder: Path) -> tuple[TrainingService, WayStore, list[str]]:
    """The lab's training service with a way store, and the slots saved from here on."""
    ways = WayStore(folder / "ways")
    t = lab.training
    t._ways = ways
    lab.connect()
    saved: list[str] = []
    adapter = lab.adapter()
    save = adapter.save_state
    adapter.save_state = lambda s: (saved.append(s), save(s))[1]  # type: ignore[method-assign]
    return t, ways, saved


def pairs(saved: list[str], slot: str) -> int:
    """How often the checkpoint was saved with the game's own slot right after it;
    neither is saved any other way."""
    both = [s for s in saved if s in (SLOT, slot)]
    assert both == [SLOT, slot] * (len(both) // 2), both
    return len(both) // 2


def slot_of(t: TrainingService) -> str:
    game = t.game_id
    assert game is not None
    return module.way_slot(game)


def test_a_start_moved_on_keeps_the_new_spot(
    lab: Lab, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    t, _, saved = kept(lab, tmp_path)
    in_service._the_game_takes_a_life_every(lab, 6, monkeypatch)
    game = lab.adapter().game
    lives = game.state.lives
    t.start(Mode.INDEPENDENT)
    slot = slot_of(t)
    assert pairs(saved, slot) == 1 and t._kept is not None and t._kept[0] == (SLOT, 1)
    in_service._goes_lose_their_first_life_at(monkeypatch, [6, 6, 6])
    in_service._run_goes(lab, 3)
    assert t.step()  # go 4 begins a life further on
    assert pairs(saved, slot) == 2 and t._kept is not None and t._kept[0] == (SLOT, 2)
    harness = t.harness
    assert harness is not None and harness.taken == 2
    lab.adapter().load_state(slot)
    assert game.state.lives == lives - 1  # the spot kept is the one moved to
    t.stop()


def test_a_game_that_was_woken_keeps_the_spot_it_woke_at(
    lab: Lab, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        module, "probe_inputs", lambda *args, **kwargs: InputProbe(True, frozenset())
    )
    t, _, saved = kept(lab, tmp_path)
    in_service._drive_to_game_over(lab)
    game = lab.adapter().game
    t.start(Mode.INDEPENDENT)  # the checkpoint is the game-over screen
    slot = slot_of(t)
    assert pairs(saved, slot) == 1
    while t.episodes < 1:
        assert t.step()
    assert t.step()  # the next go starts by working out how to wake the game
    assert t.wake_button == "start" and pairs(saved, slot) == 2
    assert t._kept is not None and t._kept[0] == (SLOT, 2)
    in_service._drive_to_game_over(lab)
    lab.adapter().load_state(slot)
    assert not game.state.game_over
    t.stop()


def test_waiting_for_a_game_that_does_not_listen_keeps_each_spot(
    lab: Lab, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(module, "find_wake_button", lambda *a, **k: None)
    t, _, saved = kept(lab, tmp_path)
    in_service._drive_to_game_over(lab)
    t.start(Mode.INDEPENDENT)
    assert not t.step() and t.status is Status.IDLE
    # The Start, and one after each wait between the searches.
    assert pairs(saved, slot_of(t)) == module.DEAF_TRIES
    assert t._kept is not None and t._kept[0] == (SLOT, module.DEAF_TRIES)


def test_a_race_keeps_the_spot_it_was_run_from(lab: Lab, tmp_path: Path) -> None:
    t, ways, saved = kept(lab, tmp_path)
    game = lab.adapter().game
    lab.play(["up"], 3)
    frame = game.state.frame
    space = ActionSpace.from_buttons(lab.adapter().buttons())
    t.compete([("A", QLearningAgent(space, Settings(), seed=1))], goes=1, max_steps=5)
    slot = slot_of(t)
    assert pairs(saved, slot) == 1 and t._kept is not None and t._kept[0] == (SLOT, 1)
    lab.play(["up"], 3)
    lab.adapter().load_state(slot)
    assert game.state.frame == frame
    assert ways.games() == ()


def test_the_app_hands_the_training_service_its_way_store(tmp_path: Path) -> None:
    from retro_ai_lab.ui.context import AppContext

    context = AppContext(data_dir=tmp_path)
    assert isinstance(context.way_store, WayStore)
    assert context.way_store.folder == tmp_path / "ways" and context.way_store.folder.is_dir()
    assert context.training._ways is context.way_store
    assert context.brain_store.folder == tmp_path / "brains"
