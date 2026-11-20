"""T-1067: the way written down is kept until a Start has had its say.

T-1064 writes the way the AI remembers beside its brain, and T-1065 takes it up at
the first Start after the game is connected. Until that Start nothing in memory
speaks for the way, and every writing of the brain took the file for stale and
removed it: closing the app, a link that dropped, a setting changed, a saved AI
loaded. Now the file is left alone until a Start has either taken the way up or
saved its own spot over the one the way starts from.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

import pytest

from retro_ai_lab.core import ConfigurationError, EmulatorDisconnectedError
from retro_ai_lab.learning import Settings
from retro_ai_lab.learning.learner import AUTO, algorithm_choices
from retro_ai_lab.observation.memory import MemoryCapture, MemorySnapshot
from retro_ai_lab.training import DEFAULT_MAX_STEPS, Mode, Status
from retro_ai_lab.training import service as module
from tests.unit.training import test_way_taken_up_again as taken_up
from tests.unit.training.test_pinned_spot_keeps_the_way import played_on
from tests.unit.training.test_remembered_way_in_training import counting
from tests.unit.training.test_way_taken_up_again import (
    NOT_TAKEN,
    SLOT,
    TOOK_UP,
    Opener,
    Sitting,
    a_sitting_with_a_way,
    game_of,
    put,
    rewritten,
)

opened = taken_up.opened  # the app opened on the folders it keeps its files in (fixture)


def the_file(sitting: Sitting) -> bytes | None:
    """The file written down for the connected game as it lies on disk, or None."""
    game_id = sitting.t.game_id
    assert game_id is not None
    path = sitting.ways.path_for(game_id)
    return path.read_bytes() if sitting.ways.exists(game_id) else None


def says(data: dict[str, object]) -> dict[str, object]:
    """What a file says, with each move's buttons in one order. A move's buttons are
    a set: written again from memory they may be listed in another order, and the
    file is read with them in any (T-1065)."""
    actions = data["actions"]
    assert isinstance(actions, list)
    return {**data, "actions": [sorted(buttons) for buttons in actions]}


def written_for(sitting: Sitting) -> dict[str, object] | None:
    """What is written down for the connected game (see ``says``), or None."""
    game_id = sitting.t.game_id
    assert game_id is not None
    return says(sitting.ways.load(game_id)) if sitting.ways.exists(game_id) else None


def the_brain_written(s: Sitting) -> None:
    assert s.t.save() is not None


def learning_settings(s: Sitting) -> None:
    s.t.set_settings(Settings(alpha=0.25))


def go_length(s: Sitting) -> None:
    s.t.set_go_length(DEFAULT_MAX_STEPS + 1)


def giving_up(s: Sitting) -> None:
    s.t.set_give_up_after(40)


def until_over(s: Sitting) -> None:
    s.t.set_go_until_over(True)


def another_learner(s: Sitting) -> None:
    agent = s.t.agent
    assert agent is not None
    other = next(a for a in algorithm_choices() if a not in (AUTO, agent.algorithm))
    assert s.t.set_algorithm(other)


def a_saved_ai_loaded(s: Sitting) -> None:
    brain = s.t.brain()
    assert brain is not None
    s.t.replace_brain(brain.to_dict())


def the_child_played_first(s: Sitting) -> None:
    s.t.start(Mode.MANUAL)
    played_on(s.lab, 40)
    assert s.t.save() is not None  # as the end of a go of the child's writes the brain
    s.t.stop()


def the_link_dropped(s: Sitting) -> None:
    states = dict(s.adapter._slots)
    s.lab.connection.disconnect()
    s.lab.connect()
    s.adapter._slots.update(states)  # the emulator ran on


BEFORE_A_START: list[Callable[[Sitting], None]] = [
    the_brain_written,
    learning_settings,
    go_length,
    giving_up,
    until_over,
    another_learner,
    a_saved_ai_loaded,
    the_child_played_first,
    the_link_dropped,
]


@pytest.mark.parametrize("done", BEFORE_A_START, ids=lambda f: f.__name__.replace("_", " "))
def test_what_is_done_before_a_start_leaves_the_way_written(
    opened: Opener,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    done: Callable[[Sitting], None],
) -> None:
    states, spot, way, _ = a_sitting_with_a_way(opened, monkeypatch)
    again = opened(states)
    t = again.t
    saves, file = t._saves, the_file(again)
    assert file is not None
    done(again)
    assert t._saves > saves or done is the_link_dropped  # the brain was written
    assert the_file(again) == file and t.remembered == 0  # not a byte of it changed
    played_on(again.lab, 40)
    with caplog.at_level(logging.INFO):
        counting(t, monkeypatch)  # Start
    assert game_of(again).state.x == spot and t.remembered == way
    assert TOOK_UP in caplog.text and NOT_TAKEN not in caplog.text
    assert t.step()
    go = t.episode
    assert go is not None and go.remembered == way
    t.stop()


def test_the_app_closed_with_no_start_keeps_the_way(
    opened: Opener, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Opened to look and closed again, twice: the third sitting has the way."""
    states, spot, way, _ = a_sitting_with_a_way(opened, monkeypatch)
    file = None
    for _ in range(2):
        looked = opened(states)
        file = file or the_file(looked)
        assert file is not None
        played_on(looked.lab, 16)
        assert looked.close() == states  # no save state of the game was made or changed
        assert the_file(looked) == file  # nor a byte of the file
    again = opened(states)
    counting(again.t, monkeypatch)
    assert game_of(again).state.x == spot and again.t.remembered == way
    again.t.stop()


def test_starting_from_scratch_before_a_start_still_removes_the_way(
    opened: Opener, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    states, spot, _, _ = a_sitting_with_a_way(opened, monkeypatch)
    again = opened(states)
    t = again.t
    assert the_file(again) is not None
    t.forget()
    assert the_file(again) is None
    played_on(again.lab, 40)
    here = game_of(again).state.x
    with caplog.at_level(logging.INFO):
        counting(t, monkeypatch)
    assert game_of(again).state.x == here != spot and t.remembered == 0
    assert TOOK_UP not in caplog.text and NOT_TAKEN not in caplog.text
    assert not any(s.startswith("way_") for s in again.lab.loaded)
    t.stop()


def pinned(s: Sitting, monkeypatch: pytest.MonkeyPatch) -> None:
    s.t.pin_spot()
    counting(s.t, monkeypatch)


def handed_over(s: Sitting, monkeypatch: pytest.MonkeyPatch) -> None:
    s.t.start(Mode.MANUAL)
    played_on(s.lab, 40)
    s.t.set_mode(Mode.INDEPENDENT)


def refused(s: Sitting, monkeypatch: pytest.MonkeyPatch) -> None:
    rewritten(s, put("format", 0))  # what is written cannot be used
    counting(s.t, monkeypatch)


@pytest.mark.parametrize(
    "start",
    [pinned, handed_over, refused],
    ids=["a pinned spot", "handed over by the child", "a way that cannot be used"],
)
def test_a_start_that_takes_its_own_spot_removes_the_way_as_before(
    opened: Opener,
    monkeypatch: pytest.MonkeyPatch,
    start: Callable[[Sitting, pytest.MonkeyPatch], None],
) -> None:
    """The spot the way starts from is saved over at such a Start: the file may not
    go on saying a way starts there. And it does not wait for the brain to be written."""
    states, _, _, written = a_sitting_with_a_way(opened, monkeypatch)
    again = opened(states)
    t = again.t
    assert the_file(again) is not None
    saves = t._saves
    start(again, monkeypatch)
    assert the_file(again) is None and t._saves == saves and t.remembered == 0
    assert again.lab.saved.count(written["slot"]) == 1
    # From here on a writing of the brain with no way in memory leaves none on disk.
    again.ways.save(written)
    assert t.save() is not None and the_file(again) is None
    t.stop()


def test_once_a_start_has_taken_the_way_up_a_way_forgotten_is_removed(
    opened: Opener, monkeypatch: pytest.MonkeyPatch
) -> None:
    states, _, way, written = a_sitting_with_a_way(opened, monkeypatch)
    again = opened(states)
    t = again.t
    counting(t, monkeypatch)
    assert t.remembered == way
    t.stop()
    assert written_for(again) == says(written)
    t._route.forget()  # as a reward row that went, or a place that keeps costing a life
    assert t.save() is not None and the_file(again) is None


@pytest.mark.parametrize("during", ["the load", "the read", "the putting back"])
def test_a_link_that_drops_at_the_start_leaves_the_way_for_the_next_one(
    opened: Opener, monkeypatch: pytest.MonkeyPatch, during: str
) -> None:
    states, spot, way, written = a_sitting_with_a_way(opened, monkeypatch)
    again = opened(states)
    t, lab = again.t, again.lab
    file = the_file(again)

    def gone() -> None:
        lab.connection.disconnect()
        raise EmulatorDisconnectedError()

    with monkeypatch.context() as m:
        load = again.adapter.load_state
        if during == "the load":
            m.setattr(
                again.adapter, "load_state", lambda s: gone() if s.startswith("way_") else load(s)
            )
        elif during == "the putting back":
            # The way is taken up by then, and the link goes as the game is put back.
            m.setattr(again.adapter, "load_state", lambda s: gone() if s == SLOT else load(s))
        else:

            class Dropping(MemoryCapture):
                def capture(self, frame: int) -> MemorySnapshot:
                    gone()
                    raise AssertionError

            m.setattr(module, "MemoryCapture", Dropping)
        with pytest.raises(ConfigurationError, match="went away"):
            t.start(Mode.INDEPENDENT)
    assert t.status is Status.IDLE and written_for(again) == says(written)
    # Where the way was not yet taken up the file was not written again at all. Where
    # it was, the link's going wrote it again from memory: it says the same.
    assert the_file(again) == file or during == "the putting back"
    # The link comes back, and the emulator has run on: this Start is the first again.
    lab.connect()
    again.adapter._slots.update(states)
    assert t.remembered == 0
    played_on(lab, 40)
    counting(t, monkeypatch)
    assert game_of(again).state.x == spot and t.remembered == way
    t.stop()
