"""T-1065: the way the AI remembers is taken up again when the app is opened.

T-1064 writes the way down beside the brain and keeps the spot it starts from in
a save state of the game's own. At the first Start after a game is connected the
training service loads that spot, makes sure by the game's memory that it is the
one the way was played from, and the goes start there with the way. Where it
cannot be sure, the game is put back, the log says why, and Start does what it
always did.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

import pytest

from retro_ai_lab.core import ConfigurationError, EmulatorDisconnectedError
from retro_ai_lab.core.events import EventBus
from retro_ai_lab.emulator.fake import FakeEmulatorAdapter, toy_hopper
from retro_ai_lab.emulator.fake.toy_hopper import ToyHopper
from retro_ai_lab.emulator.registry import default_registry
from retro_ai_lab.observation.memory import MemoryCapture, MemorySnapshot
from retro_ai_lab.services import (
    ConnectionService,
    DiscoveryService,
    ObservationService,
    RewardService,
)
from retro_ai_lab.storage import BrainStore, RewardStore, WayStore
from retro_ai_lab.training import Mode, Status, TrainingService
from retro_ai_lab.training import service as module
from tests.unit.training.test_going_back_in_training import GAP_ENDS, Lab
from tests.unit.training.test_pinned_spot_keeps_the_way import played_on
from tests.unit.training.test_remembered_way_in_training import counting, goes

SLOT = module.CHECKPOINT_SLOT
TOOK_UP = "Took up the way the AI remembered for this game from before"
NOT_TAKEN = "A way is written down for this game from before, but it is not taken up"
CHANGED = "The spot the goes start from has changed"


@dataclass
class Sitting:
    """The app, opened on the folders it keeps its files in."""

    lab: Lab
    t: TrainingService
    ways: WayStore

    @property
    def adapter(self) -> FakeEmulatorAdapter:
        adapter = self.lab.connection.adapter
        assert isinstance(adapter, FakeEmulatorAdapter)
        return adapter

    def close(self) -> dict[str, bytes]:
        """Close the app. Returns the save states the emulator has: a console that
        keeps them as files, or that runs on, still has them the next time."""
        states = dict(self.adapter._slots)
        if self.t.status is not Status.IDLE:
            self.t.stop()
        self.lab.connection.disconnect()
        return states


Opener = Callable[..., Sitting]


@pytest.fixture
def opened(tmp_path: Path) -> Iterator[Opener]:
    """Open the app: new services on the same folders, and the practice game
    connected from its beginning with the emulator's save states as they were."""
    labs: list[Lab] = []

    def opening(states: dict[str, bytes] | None = None, **kw: object) -> Sitting:
        bus = EventBus(on_error=lambda exc, event, handler: (_ for _ in ()).throw(exc))
        connection = ConnectionService(default_registry(), bus)
        observation = ObservationService(connection, bus)
        discovery = DiscoveryService(connection, observation, bus, experiment_every=None)
        rewards = RewardService(connection, discovery, bus, store=RewardStore(tmp_path / "r"))
        lab = Lab(bus, connection, observation, discovery, rewards)
        labs.append(lab)
        ways = WayStore(tmp_path / "ways")
        kw.setdefault("store", BrainStore(tmp_path / "brains"))
        kw.setdefault("ways", ways)
        t = lab.training(**kw)
        lab.connect()
        sitting = Sitting(lab, t, ways)
        sitting.adapter._slots.update(states or {})
        return sitting

    yield opening
    for lab in labs:
        lab.connection.disconnect()


def game_of(sitting: Sitting) -> ToyHopper:
    game = sitting.adapter.game
    assert isinstance(game, ToyHopper)
    return game


def a_sitting_with_a_way(
    opened: Opener, monkeypatch: pytest.MonkeyPatch, **kw: object
) -> tuple[dict[str, bytes], int, int, dict[str, object]]:
    """A first sitting that leaves a way across the first gap. Returns the save
    states the emulator is left with, the place the way starts at, its length, and
    the file as written."""
    first = opened(**kw)
    spot = game_of(first).state.x
    counting(first.t, monkeypatch)
    goes(first.t, 1)
    way = first.t.remembered
    assert way > 0
    game_id = first.t.game_id
    assert game_id is not None
    states = first.close()
    written = first.ways.load(game_id)
    assert len(written["way"]["moves"]) == way
    return states, spot, way, written


def test_the_way_is_taken_up_when_the_app_is_opened_again(
    opened: Opener, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    states, spot, way, written = a_sitting_with_a_way(opened, monkeypatch)
    again = opened(states)
    t, game = again.t, game_of(again)
    played_on(again.lab, 40)  # the game is somewhere else when Start is pressed
    assert game.state.x != spot and t.remembered == 0
    with caplog.at_level(logging.INFO):
        asked = counting(t, monkeypatch)  # Start
    assert game.state.x == spot and t.remembered == way
    assert f"{TOOK_UP}: {way} steps. The goes start from the spot that way starts from" in (
        caplog.text
    )
    assert NOT_TAKEN not in caplog.text
    # The first go plays the way before the learner is asked for anything, and is
    # across the gap that cost the first sitting a life, with every life.
    assert t.step()
    go = t.episode
    assert go is not None and go.remembered == way
    for _ in range(way - 1):
        assert t.step() and t.episode is go
    assert asked == []
    assert game.state.x > GAP_ENDS and game.state.lives == toy_hopper.START_LIVES
    assert t.step() and asked == [1]
    assert CHANGED not in caplog.text
    # It is written down again as any way is, from the same spot.
    goes(t, 1)
    game_id = t.game_id
    assert game_id is not None
    now = again.ways.load(game_id)
    assert now["spot"] == written["spot"] and now["slot"] == written["slot"]
    assert len(now["way"]["moves"]) == t.remembered >= way
    t.stop()


def test_the_way_is_taken_up_when_the_link_comes_back(
    opened: Opener, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """The same services: only the link dropped, and the emulator ran on."""
    sitting = opened()
    t, lab = sitting.t, sitting.lab
    spot = game_of(sitting).state.x
    counting(t, monkeypatch)
    goes(t, 1)
    way = t.remembered
    assert way > 0
    states = dict(sitting.adapter._slots)
    lab.connection.disconnect()
    lab.connect()
    sitting.adapter._slots.update(states)
    assert t.remembered == 0
    played_on(lab, 40)
    with caplog.at_level(logging.INFO):
        counting(t, monkeypatch)
    assert game_of(sitting).state.x == spot and t.remembered == way
    assert TOOK_UP in caplog.text
    assert t.step()
    go = t.episode
    assert go is not None and go.remembered == way
    t.stop()


def test_what_taking_up_saves_and_loads(opened: Opener, monkeypatch: pytest.MonkeyPatch) -> None:
    states, _, way, written = a_sitting_with_a_way(opened, monkeypatch)
    again = opened(states)
    t, lab = again.t, again.lab
    played_on(lab, 40)
    here = game_of(again).state.x
    slot = written["slot"]
    assert isinstance(slot, str) and lab.saved == [] and lab.loaded == []
    counting(t, monkeypatch)
    # Where the game is, to come back to; the spot kept; the checkpoint from it; and
    # the checkpoint once more, which tells the observation of the jump.
    assert lab.saved == [SLOT, SLOT] and lab.loaded == [slot, SLOT]
    harness = t.harness
    assert harness is not None and harness.taken == 2
    assert t._kept == ((SLOT, 2), written["spot"])
    assert t._route.start == (SLOT, 2) and len(t._route.moves) == way
    assert game_of(again).state.x != here
    t.stop()


def not_taken(
    again: Sitting, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, why: str
) -> list[str]:
    """Start, and see that the way was left: the game is where it was, the log says
    why once, the file is gone, and the goes start from there with no way. Returns
    the save states the Start loaded."""
    t, game = again.t, game_of(again)
    played_on(again.lab, 40)
    here = game.state.x
    game_id = t.game_id
    assert game_id is not None and again.ways.path_for(game_id).exists()
    with caplog.at_level(logging.INFO):
        counting(t, monkeypatch)
        loaded = list(again.lab.loaded)
        assert game.state.x == here and t.remembered == 0
        assert not again.ways.path_for(game_id).exists()
        assert t.step()
        go = t.episode
        assert go is not None and go.remembered == 0
    said = [r.getMessage() for r in caplog.records if NOT_TAKEN in r.getMessage()]
    assert said == [f"{NOT_TAKEN}: {why}. The goes start from where the game is, with no way."], (
        said
    )
    assert TOOK_UP not in caplog.text
    # The spot kept is now the one the game is at, as after any Start.
    harness = t.harness
    assert harness is not None and t._kept is not None and t._kept[0] == (SLOT, harness.taken)
    again.adapter.load_state(module.way_slot(game_id))
    assert game.state.x == here
    t.stop()
    return loaded


def test_a_spot_that_cannot_be_loaded_leaves_the_game_where_it_is(
    opened: Opener, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """An emulator that was closed in between, where it keeps its states in memory."""
    _, _, _, written = a_sitting_with_a_way(opened, monkeypatch)
    again = opened({})
    slot = written["slot"]
    loaded = not_taken(
        again,
        monkeypatch,
        caplog,
        f"the saved spot it starts from could not be loaded (No save state in slot {slot!r}.)",
    )
    assert loaded == [slot, SLOT]  # tried, and put back


def test_a_spot_that_is_not_as_it_was_leaves_the_game_where_it_is(
    opened: Opener, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Something else saved over the slot: the way does not lead from there."""
    states, spot, _, written = a_sitting_with_a_way(opened, monkeypatch)
    again = opened(states)
    slot = written["slot"]
    assert isinstance(slot, str)
    played_on(again.lab, 12)
    again.adapter.save_state(slot)
    assert game_of(again).state.x != spot
    again.lab.saved.clear()
    loaded = not_taken(
        again,
        monkeypatch,
        caplog,
        "the saved spot it starts from is not as it was when the way was played: "
        "the game's memory differs in ram",
    )
    assert loaded == [slot, SLOT]


def rewritten(sitting: Sitting, change: Callable[[dict[str, object]], object]) -> None:
    """Change the file written down for the connected game."""
    game_id = sitting.t.game_id
    assert game_id is not None
    path = sitting.ways.path_for(game_id)
    data = json.loads(path.read_text("utf-8"))
    changed = change(data)
    path.write_text(changed if isinstance(changed, str) else json.dumps(data), "utf-8")


def put(key: str, value: object) -> Callable[[dict[str, object]], object]:
    return lambda data: data.__setitem__(key, value)


def in_way(key: str, value: object) -> Callable[[dict[str, object]], object]:
    def change(data: dict[str, object]) -> None:
        way = data["way"]
        assert isinstance(way, dict)
        way[key] = value

    return change


def in_spot(key: str, value: object) -> Callable[[dict[str, object]], object]:
    def change(data: dict[str, object]) -> None:
        spot = data["spot"]
        assert isinstance(spot, dict)
        spot[key] = value

    return change


def only_spot(key: str) -> Callable[[dict[str, object]], object]:
    def change(data: dict[str, object]) -> None:
        spot = data["spot"]
        assert isinstance(spot, dict)
        data["spot"] = {key: spot["ram"]}

    return change


MEMORY = (
    "the saved spot it starts from is not as it was when the way was played: "
    "the game's memory differs in "
)
NO_USE = "what is written cannot be used "


@pytest.mark.parametrize(
    ("change", "why"),
    [
        (in_spot("vram", "00"), MEMORY + "vram"),  # a region the game no longer offers
        (only_spot("oam"), MEMORY + "oam, ram"),  # and one it offers that was not written
        (in_spot("ram", "00"), MEMORY + "ram"),
    ],
    ids=["a region fewer", "another region", "another digest"],
)
def test_memory_that_is_not_as_written_leaves_the_way(
    opened: Opener,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    change: Callable[[dict[str, object]], object],
    why: str,
) -> None:
    states, _, _, written = a_sitting_with_a_way(opened, monkeypatch)
    again = opened(states)
    rewritten(again, change)
    assert not_taken(again, monkeypatch, caplog, why) == [written["slot"], SLOT]


@pytest.mark.parametrize(
    ("change", "why"),
    [
        (put("format", 2), "(it is written in another format)"),
        (lambda data: data.pop("format"), "(it is written in another format)"),
        (put("game_id", "nes/Another/00"), "(it is another game's)"),
        (put("slot", "way_000000000000"), "(it names another save state)"),
        (put("slot", SLOT), "(it names another save state)"),
        (
            put("frames_per_step", 2),
            "(it was played at 2 frames a step and the AI now plays at 4)",
        ),
        (
            lambda data: data.pop("frames_per_step"),
            "(it was played at None frames a step and the AI now plays at 4)",
        ),
        (put("actions", [[], ["right"]]), "(the AI's moves are not the ones it was played with)"),
        (put("actions", None), "(the AI's moves are not the ones it was played with)"),
        (put("spot", {}), "(it does not say what the game's memory held)"),
        (put("spot", ["ram"]), "(it does not say what the game's memory held)"),
        (in_spot("ram", 7), "(it does not say what the game's memory held)"),
        (lambda data: data.pop("spot"), "(it does not say what the game's memory held)"),
        (in_way("moves", "right"), "(A way written down has a list of 'moves'.)"),
        (in_way("stood", [-1]), "(The step a loss stood at must be a whole number, 0 or more.)"),
        (put("way", None), "(A way written down is a dictionary.)"),
        (
            put("way", {"moves": [], "paid": [], "by": [], "stood": []}),
            "(it holds no moves)",
        ),
        (lambda data: "[1, 2]", None),
        (lambda data: "{", None),
    ],
    ids=[
        "another format",
        "no format",
        "another game",
        "another slot",
        "the checkpoint's slot",
        "other frames a step",
        "no frames a step",
        "other moves",
        "no moves' buttons",
        "an empty spot",
        "a spot that is a list",
        "a digest that is a number",
        "no spot",
        "moves that are text",
        "a place before the start",
        "no way",
        "a way with no moves",
        "a list",
        "not JSON",
    ],
)
def test_a_file_that_cannot_be_used_leaves_the_game_alone(
    opened: Opener,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    change: Callable[[dict[str, object]], object],
    why: str | None,
) -> None:
    states, _, _, _ = a_sitting_with_a_way(opened, monkeypatch)
    again = opened(states)
    rewritten(again, change)
    game_id = again.t.game_id
    assert game_id is not None
    if why is None:  # the store's own words for a file it cannot read
        with pytest.raises(ConfigurationError) as refused:
            again.ways.load(game_id)
        why = f"({refused.value})"
    # The file said enough by itself: nothing was loaded to find out.
    assert not_taken(again, monkeypatch, caplog, NO_USE + why) == []


def test_a_pinned_spot_decides_and_nothing_is_read(
    opened: Opener, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    states, spot, _, _ = a_sitting_with_a_way(opened, monkeypatch)
    again = opened(states)
    t, game = again.t, game_of(again)
    played_on(again.lab, 40)
    here = game.state.x
    t.pin_spot()
    played_on(again.lab, 8)
    with caplog.at_level(logging.INFO):
        counting(t, monkeypatch)
    assert game.state.x == here != spot and t.remembered == 0
    assert not any(s.startswith("way_") for s in again.lab.loaded)
    assert TOOK_UP not in caplog.text and NOT_TAKEN not in caplog.text
    t.stop()
    # The chance is gone with the first Start: unpinned, the next one reads nothing.
    t.unpin_spot()
    counting(t, monkeypatch)
    assert not any(s.startswith("way_") for s in again.lab.loaded) and t.remembered == 0
    t.stop()


def test_handing_over_to_the_ai_while_the_child_plays_reads_nothing(
    opened: Opener, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """The game is where the child took it on purpose."""
    states, spot, _, _ = a_sitting_with_a_way(opened, monkeypatch)
    again = opened(states)
    t, game = again.t, game_of(again)
    t.start(Mode.MANUAL)
    played_on(again.lab, 40)
    here = game.state.x
    with caplog.at_level(logging.INFO):
        t.set_mode(Mode.INDEPENDENT)
    assert game.state.x == here != spot and t.remembered == 0
    assert not any(s.startswith("way_") for s in again.lab.loaded)
    assert TOOK_UP not in caplog.text and NOT_TAKEN not in caplog.text
    t.stop()
    counting(t, monkeypatch)  # and a Start after that is not the first
    assert not any(s.startswith("way_") for s in again.lab.loaded) and t.remembered == 0
    t.stop()


def test_a_start_at_which_the_child_plays_is_not_the_one(
    opened: Opener, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The first Start that prepares the AI's goes takes the way up, whenever it comes."""
    states, spot, way, _ = a_sitting_with_a_way(opened, monkeypatch)
    again = opened(states)
    t = again.t
    t.start(Mode.MANUAL)
    played_on(again.lab, 40)
    t.stop()
    assert t.remembered == 0 and again.lab.loaded == []
    counting(t, monkeypatch)
    assert t.remembered == way and game_of(again).state.x == spot
    t.stop()


def test_only_the_first_start_takes_a_way_up(
    opened: Opener, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Later in the sitting a Start with no pinned spot is from where the game is, as
    it always was: somebody left the game there."""
    states, spot, way, written = a_sitting_with_a_way(opened, monkeypatch)
    again = opened(states)
    t, game = again.t, game_of(again)
    counting(t, monkeypatch)
    assert t.remembered == way
    goes(t, 1)
    t.stop()
    slot = written["slot"]
    game_id = t.game_id
    assert game_id is not None and again.ways.exists(game_id)
    assert again.lab.loaded.count(slot) == 1
    played_on(again.lab, 16)
    here = game.state.x
    remembered = t.remembered
    with caplog.at_level(logging.INFO):
        counting(t, monkeypatch)
        assert game.state.x == here != spot
        assert again.lab.loaded.count(slot) == 1  # not loaded again
        assert t.step()
        go = t.episode
        assert go is not None and go.remembered == 0
    assert f"{CHANGED}, so the way the AI remembered from the old one ({remembered} steps)" in (
        caplog.text
    )
    assert TOOK_UP not in caplog.text and NOT_TAKEN not in caplog.text
    t.stop()


@pytest.mark.parametrize(
    "kw",
    [{"ways": None}, {"remember": False}, {"store": None}],
    ids=["no way store", "not remembering", "no brain store"],
)
def test_where_no_way_is_kept_none_is_taken_up(
    opened: Opener, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, kw: dict
) -> None:
    states, spot, _, _ = a_sitting_with_a_way(opened, monkeypatch)
    again = opened(states, **kw)
    t, game = again.t, game_of(again)
    played_on(again.lab, 40)
    here = game.state.x
    with caplog.at_level(logging.INFO):
        counting(t, monkeypatch)
    assert game.state.x == here != spot and t.remembered == 0
    assert again.lab.saved == [SLOT] and again.lab.loaded == []
    assert TOOK_UP not in caplog.text and NOT_TAKEN not in caplog.text
    t.stop()


@pytest.mark.parametrize("during", ["the load", "the read"])
def test_a_link_that_drops_while_the_spot_is_looked_for_refuses_to_start(
    opened: Opener, monkeypatch: pytest.MonkeyPatch, during: str
) -> None:
    states, _, _, _ = a_sitting_with_a_way(opened, monkeypatch)
    again = opened(states)
    t, lab = again.t, again.lab

    def gone() -> None:
        lab.connection.disconnect()
        raise EmulatorDisconnectedError()

    if during == "the load":
        load = again.adapter.load_state
        monkeypatch.setattr(
            again.adapter, "load_state", lambda s: gone() if s.startswith("way_") else load(s)
        )
    else:

        class Dropping(MemoryCapture):
            def capture(self, frame: int) -> MemorySnapshot:
                gone()
                raise AssertionError

        monkeypatch.setattr(module, "MemoryCapture", Dropping)
    with pytest.raises(ConfigurationError, match="went away"):
        t.start(Mode.INDEPENDENT)
    assert t.status is Status.IDLE and not t.enabled and t.remembered == 0


def test_pay_from_a_reward_that_is_gone_is_put_right_by_the_first_go(
    opened: Opener, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A way is worth what it is paid now (T-1026): a row that paid in the sitting
    before and is no longer there pays nothing when the way is played again, and
    the way takes that go's figures."""
    states, _, way, written = a_sitting_with_a_way(opened, monkeypatch)
    again = opened(states)

    def more(data: dict[str, object]) -> None:
        inner = data["way"]
        assert isinstance(inner, dict)
        inner["paid"] = [paid + 5.0 * (at + 1) for at, paid in enumerate(inner["paid"])]
        inner["by"] = [[*parts, ["gone", 5.0]] for parts in inner["by"]]

    rewritten(again, more)
    t = again.t
    counting(t, monkeypatch)
    route = t._route
    was = written["way"]
    assert isinstance(was, dict)
    assert len(route.moves) == way and route.paid == was["paid"][-1] + 5.0 * way
    goes(t, 1)
    assert route.moves and route.paid < 5.0 * way
    assert all(row != "gone" for parts in route.paid_by for row, _ in parts)
    t.stop()


def test_the_buttons_of_a_move_are_told_in_any_order(
    opened: Opener, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A move's buttons are a set, and the order they are written in changes from one
    run of the app to the next: the same moves all the same."""
    states, spot, way, written = a_sitting_with_a_way(opened, monkeypatch)
    again = opened(states)
    actions = written["actions"]
    assert isinstance(actions, list) and any(len(buttons) > 1 for buttons in actions)
    rewritten(again, put("actions", [list(reversed(buttons)) for buttons in actions]))
    counting(again.t, monkeypatch)
    assert again.t.remembered == way and game_of(again).state.x == spot
    again.t.stop()


@pytest.mark.parametrize(
    "change",
    [
        lambda actions: actions[:-1],
        lambda actions: [*actions, ["select"]],
        lambda actions: [actions[1], actions[0], *actions[2:]],
        lambda actions: [[*actions[0], "b"], *actions[1:]],
        lambda actions: [*actions[:-1], actions[-1][:1]],
        lambda actions: [*actions[:-1], "right"],
        lambda actions: [*actions[:-1], [["right"]]],
        lambda actions: [*actions[:-1], [7]],
        lambda actions: {"moves": actions},
    ],
    ids=[
        "a move fewer",
        "a move more",
        "two moves the other way round",
        "a button more",
        "a button fewer",
        "a move that is text",
        "a button that is a list",
        "a button that is a number",
        "no list",
    ],
)
def test_other_moves_than_the_ais_leave_the_way(
    opened: Opener,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    change: Callable[[list[list[str]]], object],
) -> None:
    states, _, _, written = a_sitting_with_a_way(opened, monkeypatch)
    again = opened(states)
    actions = written["actions"]
    assert isinstance(actions, list)
    rewritten(again, put("actions", change(actions)))
    loaded = not_taken(
        again,
        monkeypatch,
        caplog,
        NO_USE + "(the AI's moves are not the ones it was played with)",
    )
    assert loaded == []
