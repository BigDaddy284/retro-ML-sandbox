"""T-1064: the way the AI remembers, and the spot it starts from, are written down.

Closing the app lost the way: it was in the app's memory, and what said where it
starts was a count of saves and a save state in a slot every game shares. The
training service now keeps the spot the goes start from in a slot of the game's
own, with a digest of the game's memory to tell it by, and writes the way beside
the brain whenever it writes the brain. Nothing reads the file yet (T-1065).
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from retro_ai_lab.core import (
    ConfigurationError,
    EmulatorDisconnectedError,
    UnsupportedCapabilityError,
)
from retro_ai_lab.emulator.fake import FakeEmulatorAdapter
from retro_ai_lab.observation.memory import MemoryCapture, MemorySnapshot
from retro_ai_lab.storage import BrainStore, WayStore
from retro_ai_lab.training import Mode, Status, TrainingService
from retro_ai_lab.training import service as module
from retro_ai_lab.training.route import Route
from tests.unit.training import test_going_back_in_training as in_training
from tests.unit.training.test_going_back_in_training import Lab
from tests.unit.training.test_pinned_spot_keeps_the_way import a_way, played_on
from tests.unit.training.test_remembered_way_in_training import counting, goes

lab = in_training.lab  # the practice game behind the real services (pytest fixture)

SLOT = module.CHECKPOINT_SLOT


def service(lab: Lab, folder: Path, **kw: object) -> tuple[TrainingService, WayStore]:
    ways = WayStore(folder / "ways")
    kw.setdefault("store", BrainStore(folder / "brains"))
    kw.setdefault("ways", ways)
    return lab.training(**kw), ways


def reads_counted(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Every read of the game's memory the training service asks for is counted."""
    made: list[int] = []

    class Counted(MemoryCapture):
        def capture(self, frame: int) -> MemorySnapshot:
            made.append(frame)
            return super().capture(frame)

    monkeypatch.setattr(module, "MemoryCapture", Counted)
    return made


def digests(lab: Lab) -> dict[str, str]:
    """A digest of each region of the game's memory as it is now."""
    regions = MemoryCapture(lab.connection).capture(0).regions
    assert regions
    return {name: hashlib.blake2b(regions[name], digest_size=16).hexdigest() for name in regions}


def the_slot(t: TrainingService) -> str:
    game = t.game_id
    assert game is not None
    return module.way_slot(game)


def the_way(t: TrainingService) -> Route:
    return t._route


def test_the_slot_is_the_games_own() -> None:
    one, other = module.way_slot("nes/Game One/00ff"), module.way_slot("nes/Game Two/00ff")
    assert one != other and one == module.way_slot("nes/Game One/00ff")
    for slot in (one, other, module.way_slot(""), module.way_slot("Ünï cödé / 漢字")):
        assert slot.startswith("way_") and len(slot) == 16
        assert slot.replace("_", "").isalnum() and slot.isascii()  # a name a slot may have
    assert one not in (SLOT, module.PRACTISE_SLOT)
    assert module.WAY_FORMAT == 1


def test_a_start_keeps_the_spot_in_the_games_own_slot(
    lab: Lab, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    t, ways = service(lab, tmp_path)
    game = lab.connect()
    reads = reads_counted(monkeypatch)
    spot, mark = game.state.x, digests(lab)
    counting(t, monkeypatch)  # Start
    slot = the_slot(t)
    # One save of the game's own slot after the checkpoint, and one read of memory.
    assert lab.saved == [SLOT, slot] and reads == [0]
    assert t._kept == ((SLOT, 1), mark)
    assert ways.games() == ()  # no way yet: nothing written
    assert t.step()  # a go has begun from the spot kept, and has no way yet either
    assert t.save() is not None and ways.games() == ()
    goes(t, 1)
    assert lab.saved.count(slot) == 1 and reads == [0]  # a go saves and reads no more
    # The slot holds the state the goes start from.
    played_on(lab)
    adapter = lab.connection.adapter
    assert isinstance(adapter, FakeEmulatorAdapter)
    adapter.load_state(slot)
    assert game.state.x == spot and digests(lab) == mark
    t.stop()


def test_after_a_go_the_way_is_written_beside_the_brain(
    lab: Lab, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    t, ways = service(lab, tmp_path)
    lab.connect()
    mark = digests(lab)
    counting(t, monkeypatch)
    game_id = t.game_id
    assert game_id is not None and not ways.exists(game_id)
    goes(t, 1)
    way = t.remembered
    assert way > 0
    data = ways.load(game_id)
    agent = t.agent
    assert agent is not None
    space = agent.space
    assert data == {
        "format": 1,
        "game_id": game_id,
        "slot": the_slot(t),
        "spot": mark,
        "frames_per_step": 4,
        "actions": [list(space.buttons_for(i)) for i in range(len(space))],
        "way": the_way(t).to_dict(),
    }
    assert ["right"] in data["actions"] and len(data["actions"]) == len(space) > 1
    back = Route.from_dict(data["way"], ("anywhere", 1))
    assert back.moves == the_way(t).moves and len(back.moves) == way
    assert back.paid == the_way(t).paid > 0 and back.paid_by == the_way(t).paid_by
    assert ways.path_for(game_id).parent == tmp_path / "ways"
    assert (tmp_path / "brains").is_dir() and len(list((tmp_path / "brains").iterdir())) == 1
    # It is written again with the brain after each go, and is what is remembered.
    for _ in range(3):
        goes(t, 1)
        assert ways.load(game_id)["way"] == the_way(t).to_dict()
        assert len(ways.load(game_id)["way"]["moves"]) == t.remembered
    assert t.remembered > way  # the way has grown, and so has the file
    t.stop()
    assert ways.load(game_id)["way"] == the_way(t).to_dict()
    assert ways.load(game_id)["spot"] == mark


def test_a_start_from_another_spot_takes_the_file_away(
    lab: Lab, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The slot is saved over with the new spot, so nothing may go on saying a way
    starts from it."""
    t, ways = service(lab, tmp_path)
    game = lab.connect()
    old = digests(lab)
    way = a_way(t, monkeypatch)
    game_id = t.game_id
    assert game_id is not None and ways.load(game_id)["spot"] == old
    played_on(lab)
    here, new = game.state.x, digests(lab)
    assert new != old
    counting(t, monkeypatch)  # Start, with no pinned spot: from where the game is
    slot = the_slot(t)
    assert not ways.exists(game_id) and lab.saved.count(slot) == 2
    assert t._kept == ((SLOT, 2), new)
    # Until a go begins the AI still holds the old way; writing the brain does not
    # write that way down as one that starts from the new spot.
    assert t.remembered == way
    assert t.save() is not None and not ways.exists(game_id)
    goes(t, 1)
    if t.remembered:
        assert ways.load(game_id)["spot"] == new
        assert ways.load(game_id)["way"] == the_way(t).to_dict()
    else:
        assert not ways.exists(game_id)
    played_on(lab)
    adapter = lab.connection.adapter
    assert isinstance(adapter, FakeEmulatorAdapter)
    adapter.load_state(slot)
    assert game.state.x == here
    t.stop()


def test_a_way_that_is_forgotten_is_taken_off_the_disk(
    lab: Lab, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    t, ways = service(lab, tmp_path)
    lab.connect()
    a_way(t, monkeypatch)
    game_id = t.game_id
    assert game_id is not None and ways.exists(game_id)
    the_way(t).forget()  # as a reward row that went, or a place that keeps costing a life
    assert t.save() is not None
    assert not ways.exists(game_id)
    assert t.save() is not None and not ways.exists(game_id)  # nothing there: no harm


def test_a_pinned_spot_held_keeps_the_spot_and_the_file(
    lab: Lab, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """T-1063's Start that leaves the checkpoint as it is saves and reads nothing."""
    t, ways = service(lab, tmp_path)
    lab.connect()
    t.pin_spot()
    mark = digests(lab)
    reads = reads_counted(monkeypatch)
    way = a_way(t, monkeypatch)
    game_id = t.game_id
    assert game_id is not None
    slot = the_slot(t)
    before = ways.load(game_id)
    assert before["spot"] == mark and lab.saved.count(slot) == 1 and reads == [0]
    counting(t, monkeypatch)  # Start again: the same spot
    assert lab.saved.count(slot) == 1 and reads == [0] and t._kept == ((SLOT, 1), mark)
    assert ways.load(game_id) == before and t.remembered == way
    goes(t, 1)
    t.stop()
    after = ways.load(game_id)
    assert after["spot"] == mark and after["way"] == the_way(t).to_dict()
    assert len(after["way"]["moves"]) >= way


def test_forgetting_the_brain_removes_the_way(
    lab: Lab, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    t, ways = service(lab, tmp_path)
    lab.connect()
    a_way(t, monkeypatch)
    game_id = t.game_id
    assert game_id is not None and ways.exists(game_id)
    t.forget()
    assert not ways.exists(game_id) and t.remembered == 0
    t.forget()  # again, with nothing written: no harm


def test_the_way_is_on_disk_when_the_link_goes(
    lab: Lab, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    t, ways = service(lab, tmp_path)
    lab.connect()
    counting(t, monkeypatch)
    goes(t, 1)
    game_id = t.game_id
    assert game_id is not None
    written = the_way(t).to_dict()
    assert written["moves"]
    lab.connection.disconnect()
    assert ways.load(game_id)["way"] == written
    # A new link is a new harness with nothing kept; the file stays until a Start
    # saves the slot again.
    lab.connect()
    assert t._kept is None and ways.exists(game_id)
    assert t.save() is not None and ways.load(game_id)["way"] == written  # T-1067: kept


@pytest.mark.parametrize(
    "kw",
    [{"ways": None}, {"remember": False}, {"store": None}],
    ids=["no way store", "not remembering", "no brain store"],
)
def test_where_no_way_can_be_written_nothing_more_is_saved_or_read(
    lab: Lab, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kw: dict[str, object]
) -> None:
    t, ways = service(lab, tmp_path, **kw)
    lab.connect()
    reads = reads_counted(monkeypatch)
    counting(t, monkeypatch)
    goes(t, 2)
    t.stop()
    assert SLOT in lab.saved and not any(s.startswith("way_") for s in lab.saved)
    assert reads == [] and t._kept is None
    assert ways.games() == () and list(ways.folder.iterdir()) == []


@pytest.mark.parametrize("error", [ConfigurationError, UnsupportedCapabilityError])
def test_where_the_memory_cannot_be_read_nothing_is_kept(
    lab: Lab, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, error: type[Exception]
) -> None:
    t, ways = service(lab, tmp_path)
    lab.connect()

    class Unreadable(MemoryCapture):
        def capture(self, frame: int) -> MemorySnapshot:
            raise error("save_states") if error is UnsupportedCapabilityError else error("no")

    monkeypatch.setattr(module, "MemoryCapture", Unreadable)
    t.pin_spot()
    counting(t, monkeypatch)
    assert lab.saved == [module.PRACTISE_SLOT, SLOT] and t._kept is None
    goes(t, 1)
    way = t.remembered
    assert way > 0  # the AI remembers as before; it is only not written
    t.stop()
    assert ways.games() == ()
    # The Start itself went on as it does: the pinned spot is held (T-1063), and the
    # next Start leaves the checkpoint and the way as they are.
    counting(t, monkeypatch)
    harness = t.harness
    assert harness is not None and harness.taken == 1 and t.remembered == way
    assert lab.saved.count(SLOT) == 1 and t._kept is None
    t.stop()
    assert ways.games() == ()


def test_an_older_file_goes_when_the_spot_cannot_be_told(
    lab: Lab, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    t, ways = service(lab, tmp_path)
    lab.connect()
    a_way(t, monkeypatch)
    game_id = t.game_id
    assert game_id is not None and ways.exists(game_id)

    class Unreadable(MemoryCapture):
        def capture(self, frame: int) -> MemorySnapshot:
            raise ConfigurationError("no")

    monkeypatch.setattr(module, "MemoryCapture", Unreadable)
    played_on(lab)
    counting(t, monkeypatch)
    assert not ways.exists(game_id) and t._kept is None
    assert lab.saved.count(the_slot(t)) == 1  # the old state is still in the slot, unnamed
    t.stop()


@pytest.mark.parametrize("during", ["the save", "the read"])
def test_a_link_that_drops_while_the_spot_is_kept_refuses_to_start(
    lab: Lab, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, during: str
) -> None:
    """As a link that drops during the checkpoint itself does."""
    t, ways = service(lab, tmp_path)
    lab.connect()
    adapter = lab.connection.adapter
    assert isinstance(adapter, FakeEmulatorAdapter)

    def gone() -> None:
        lab.connection.disconnect()
        raise EmulatorDisconnectedError()

    if during == "the save":
        save = adapter.save_state
        monkeypatch.setattr(
            adapter, "save_state", lambda s: gone() if s.startswith("way_") else save(s)
        )
    else:

        class Dropping(MemoryCapture):
            def capture(self, frame: int) -> MemorySnapshot:
                gone()
                raise AssertionError

        monkeypatch.setattr(module, "MemoryCapture", Dropping)
    with pytest.raises(ConfigurationError, match="went away"):
        t.start(Mode.INDEPENDENT)
    assert t.status is Status.IDLE and not t.enabled and t._kept is None
    assert ways.games() == ()


def test_a_file_that_cannot_be_written_is_said_and_training_goes_on(
    lab: Lab, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    t, ways = service(lab, tmp_path)
    lab.connect()

    def full(data: dict[str, object]) -> Path:
        raise ConfigurationError("Can't write way: the disk is full")

    monkeypatch.setattr(ways, "save", full)
    counting(t, monkeypatch)
    goes(t, 2)
    assert t.remembered > 0 and t.status is Status.RUNNING
    assert "Couldn't write down the way the AI remembers: Can't write way" in caplog.text
    t.stop()


def test_a_file_that_cannot_be_removed_is_said_and_training_goes_on(
    lab: Lab, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    t, ways = service(lab, tmp_path)
    lab.connect()
    a_way(t, monkeypatch)

    def locked(game_id: str) -> bool:
        raise PermissionError("in use")

    monkeypatch.setattr(ways, "delete", locked)
    played_on(lab)
    counting(t, monkeypatch)
    goes(t, 1)
    assert t.status is Status.RUNNING
    assert "Couldn't remove the way written down for this game: in use" in caplog.text
    t.stop()
