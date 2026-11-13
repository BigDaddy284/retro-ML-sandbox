"""T-1020: the watch finds the game's own count of its camera, and goes by it where
the screen cannot tell how far it slid."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass

import pytest

from retro_ai_lab.core import ConfigurationError, Frame
from retro_ai_lab.emulator.fake import toy_hopper
from retro_ai_lab.emulator.fake.toy_hopper import LAYOUT, ToyHopper
from retro_ai_lab.observation.camera_memory import NEEDS, CameraMemory
from retro_ai_lab.observation.tracking import ObjectTracker
from retro_ai_lab.training.vision import JOLT, ScreenWatch
from retro_ai_lab.understanding.player import find_player_in_game
from tests.unit.training import test_going_back as going_back
from tests.unit.training.test_going_back import steps
from tests.unit.training.test_trial import FRAMES, Hopper

lab = going_back.lab  # the practice game behind the real services (pytest fixture)

LOOK_EVERY = 2  # steps between two looks, as in a go
STEP = toy_hopper.PLAYER_H  # "a step" of ground
BYTE = ("ram", LAYOUT.scroll)  # where the practice game keeps the low half of its camera

Move = tuple[str, ...]
STAND: Move = ()
RIGHT: Move = ("right",)
LEFT: Move = ("left",)
JUMP: Move = ("right", "a")

# Standing, then to and fro on the first stretch of level 0, where nothing but the
# player moves: twenty still looks and twenty the picture measures right.
LEARN: list[Move] = [STAND] * 40 + [RIGHT] * 16 + ([LEFT] * 8 + [RIGHT] * 8) * 3
# On over the first gap and across the bare ground the first walker paces on, over it.
ON: list[Move] = [RIGHT] * 3 + [JUMP] * 4 + [RIGHT] * 24 + [JUMP] * 4 + [RIGHT] * 12


def camera(game: ToyHopper) -> int:
    return int.from_bytes(bytes(game.ram[LAYOUT.scroll : LAYOUT.scroll + 2]), "little")


def memory_of(game: ToyHopper) -> dict[str, bytes]:
    return {"ram": bytes(game.ram)}


@dataclass(frozen=True)
class Looked:
    """One look of a walk: what the game did, and what each watch made of it."""

    moved: int  # the game's own camera since the last look
    walker: bool  # one was on the screen
    lives: int
    read: tuple[int, int]  # the camera's move as the watch given memory had it
    sighted: tuple[int, int]  # ... and as the picture alone said to that watch
    plain: tuple[int, int]  # ... and to the watch that is given no memory
    found: tuple[tuple[str, int] | None, tuple[str, int] | None]


def a_watch(hopper: Hopper) -> ScreenWatch:
    names = [b.name for b in hopper.adapter.buttons() if b.name not in ("start", "select")]
    player = find_player_in_game(hopper.connection, names)
    assert player is not None and player.colours is not None
    hopper.game.reset()
    return ScreenWatch(player)


def frame_of(hopper: Hopper) -> Frame:
    return hopper.adapter.capture_frame()


def walk(hopper: Hopper, given: ScreenWatch, plain: ScreenWatch, moves: list[Move]) -> list[Looked]:
    """Play ``moves`` with a look every other step: ``given`` is handed the game's
    memory with each screen, ``plain`` the screen alone."""
    game, buttons = hopper.game, {b.name: b for b in hopper.adapter.buttons()}
    out: list[Looked] = []
    was = camera(game)
    for at, move in enumerate(moves):
        hopper.adapter.step(FRAMES, pressed=[buttons[name] for name in move])
        if at % LOOK_EVERY != LOOK_EVERY - 1:
            continue
        frame = frame_of(hopper)
        given.look(frame, memory=memory_of(game))
        plain.look(frame)
        assert given.me == plain.me  # it sees itself where it would without the memory
        now = camera(game)
        out.append(
            Looked(
                now - was,
                bool(game.ram[LAYOUT.walkers]),
                game.state.lives,
                given.tracker.scroll,
                given.tracker.sighted,
                plain.tracker.scroll,
                given.camera_bytes,
            )
        )
        was = now
    return out


def two_watches(hopper: Hopper) -> tuple[ScreenWatch, ScreenWatch]:
    given = a_watch(hopper)
    plain = ScreenWatch(given.player)
    for watch in (given, plain):
        watch.begin_go()
    frame = frame_of(hopper)
    given.look(frame, memory=memory_of(hopper.game))
    plain.look(frame)
    return given, plain


def test_the_watch_finds_the_cameras_byte_and_goes_by_it(
    hopper: Hopper, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO, logger="retro_ai_lab.training.vision")
    given, plain = two_watches(hopper)
    assert given.camera_bytes == (None, None)

    learnt = walk(hopper, given, plain, LEARN)
    # Nothing walks here, and the picture measures every look right, with memory or not.
    assert not any(look.walker for look in learnt)
    assert all(look.read == look.sighted == look.plain == (look.moved, 0) for look in learnt)
    # The byte is taken at the look that makes it twenty of each kind, and not before.
    moving = [at for at, look in enumerate(learnt) if look.moved]
    still = [at for at, look in enumerate(learnt) if not look.moved]
    at = max(moving[NEEDS - 1], still[NEEDS - 1])
    assert [look.found for look in learnt[:at]] == [(None, None)] * at
    assert [look.found for look in learnt[at:]] == [(BYTE, None)] * (len(learnt) - at)

    def said() -> list[str]:
        return [r.getMessage() for r in caplog.records if "keeps count of its camera" in r.message]

    assert len(said()) == 1
    assert f"ram[{LAYOUT.scroll}] moves as the screen slides across" in said()[0]
    assert f"(in {at + 1} of the {at + 1} looks it has compared)" in said()[0]

    on = walk(hopper, given, plain, ON)
    assert all(look.lives == toy_hopper.START_LIVES for look in on)  # nothing put it back
    assert all(look.found == (BYTE, None) for look in on)  # never dropped
    assert len(said()) == 1  # said once
    # With the memory, every look is the game's own move.
    assert all(look.read == (look.moved, 0) for look in on)
    # The picture alone is fooled beside the walker, as it is today: it takes the
    # walker's four pixels a look for the ground's.
    fooled = [look for look in on if look.plain != (look.moved, 0)]
    assert len(fooled) >= 5
    assert all(look.walker for look in fooled)
    assert all(look.plain == (look.moved + 4, 0) for look in fooled)
    # ... and it said the same to the watch that was given the memory.
    assert all(look.sighted == look.plain for look in on)
    # The furthest ground counted: the game's own with the memory, not without.
    gone = sum(look.moved for look in learnt + on)  # the game's camera at the last look
    assert gone > 400
    assert abs(given.ground.reach[0] - gone) <= STEP
    assert given.ground.reach[0] == gone
    assert plain.ground.reach[0] - gone > STEP
    assert plain.camera_bytes == (None, None)


class Spy:
    """Notes what the camera's memory is shown and what the tracker is told."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.noted: list[tuple[int, int]] = []
        self.told: list[tuple[int | None, int | None] | None] = []
        note, update = CameraMemory.note, ObjectTracker.update

        def noting(
            memory: CameraMemory, before: bytes, after: bytes, seen: tuple[int, int]
        ) -> None:
            self.noted.append(seen)
            note(memory, before, after, seen)

        def updating(tracker: ObjectTracker, frame: Frame, **kw: object) -> object:
            self.told.append(kw.get("expect"))  # type: ignore[arg-type]
            return update(tracker, frame, **kw)  # type: ignore[arg-type]

        monkeypatch.setattr(CameraMemory, "note", noting)
        monkeypatch.setattr(ObjectTracker, "update", updating)

    def since(self) -> tuple[int, list[tuple[int | None, int | None] | None]]:
        """How many looks were learnt from, and what the tracker was told, since last asked."""
        got = (len(self.noted), self.told)
        self.noted, self.told = [], []
        return got


def test_nothing_is_taken_across_a_jump_and_a_try_is_as_before(
    hopper: Hopper, monkeypatch: pytest.MonkeyPatch
) -> None:
    given, plain = two_watches(hopper)
    spy = Spy(monkeypatch)
    game, adapter = hopper.game, hopper.adapter
    right = [b for b in adapter.buttons() if b.name == "right"]

    def look(*, memory: bool = True, lost: int = 0, steps: int = 2) -> None:
        if steps:
            adapter.step(FRAMES * steps, pressed=right)
        given.look(frame_of(hopper), lost=lost, memory=memory_of(game) if memory else None)

    # The first look of the go (made above) had nothing before it to compare with.
    # From the second on, each look is learnt from; nothing is found yet to tell.
    look()
    look()
    assert spy.since() == (2, [(None, None), (None, None)])
    # A look without memory is not learnt from, and neither is the one after it:
    # there is no memory from just before to set against.
    look(memory=False)
    look()
    assert spy.since() == (0, [None, None])
    look()
    assert spy.since() == (1, [(None, None)])
    # The game put back to a saved moment, and a new go: no move is read across either.
    mark = given.mark()
    adapter.save_state("here")
    look()
    adapter.load_state("here")
    given.back_to(mark)
    assert spy.since() == (1, [(None, None)])
    look()
    assert spy.since() == (0, [None])
    look()
    assert spy.since() == (1, [(None, None)])
    game.reset()
    given.begin_go()
    look(steps=0)
    assert spy.since() == (0, [None])

    # Once the byte is found the tracker is told its change with every look.
    walk(hopper, given, plain, LEARN)
    assert given.camera_bytes == (BYTE, None)
    spy.since()
    was = camera(game)
    look()
    moved = camera(game) - was
    assert moved == 16
    assert spy.since() == (1, [(moved, None)])
    # ... but not at a look in which a life went: the game may have put the player
    # back, and its count's jump is no move to try. The picture decides alone.
    look(lost=1)
    assert spy.since() == (1, [None])
    # A look the picture could not measure (a jolt: another screen altogether) is not
    # learnt from.
    game.reset()
    given.look(frame_of(hopper), memory=memory_of(game))
    assert given.tracker.scroll == (0, 0) and given.tracker.unexplained >= JOLT
    noted, told = spy.since()
    assert noted == 0 and len(told) == 1 and told[0] is not None and told[0][1] is None
    assert given.camera_bytes == (BYTE, None)

    # Beside the walker the tracker takes the byte's word, and such a look is not
    # learnt from either way: the picture could not tell there.
    given.begin_go()
    plain.begin_go()
    frame = frame_of(hopper)
    given.look(frame, memory=memory_of(game))
    plain.look(frame)
    spy.since()
    on = walk(hopper, given, plain, LEARN + ON)
    assert all(look.lives == toy_hopper.START_LIVES for look in on)
    taken = [look for look in on if look.read != look.sighted]
    assert len(taken) >= 5
    noted, told = spy.since()
    assert len(told) == 2 * len(on)  # both watches' trackers
    assert noted == len(on) - len(taken)
    assert given.camera_bytes == (BYTE, None)

    # A watch for a try is as before: it is shown no memory, has found nothing, and
    # its tracker is told nothing.
    trying = given.trial(given.mark())
    assert trying.camera_bytes == (None, None)
    trying.look(frame_of(hopper))
    adapter.step(FRAMES * 2, pressed=right)
    trying.look(frame_of(hopper))
    assert spy.since() == (0, [None, None])
    assert given.camera_bytes == (BYTE, None)

    # Memory laid out another way is another game: what was learnt is void.
    given.look(frame_of(hopper), memory={"ram": bytes(game.ram), "more": b"\x00"})
    assert given.camera_bytes == (None, None)
    assert spy.since() == (0, [None])
    # And what is no memory is refused, with the reason.
    for nonsense in ({}, {"ram": b""}, {1: b"\x00"}, {"ram": "text"}, [b"\x00"], b"\x00"):
        with pytest.raises(ConfigurationError, match="memory"):
            given.look(frame_of(hopper), memory=nonsense)  # type: ignore[arg-type]


def test_the_runner_hands_the_watch_the_memory(
    lab: going_back.Lab, monkeypatch: pytest.MonkeyPatch
) -> None:
    lab.teach()
    watch = lab.watch()
    handed: list[tuple[Mapping[str, bytes] | None, bytes, bool]] = []
    look = watch.look

    def looking(frame: Frame, **kw: object) -> None:
        memory = kw.get("memory")
        assert memory is None or isinstance(memory, Mapping)
        current = lab.adapter.capture_frame()
        handed.append((memory, bytes(lab.game.ram), frame.pixels == current.pixels))
        look(frame, **kw)  # type: ignore[arg-type]

    monkeypatch.setattr(watch, "look", looking)
    script = [lab.move(*move) for move in LEARN + ON]
    lab.agent.script.extend(script)
    r = lab.runner(watch=watch, unstick_after=None)
    r.begin()
    steps(r, len(script))
    # A look every other step, each with the memory read with the screen it is shown.
    assert len(handed) in (len(script) // LOOK_EVERY, len(script) // LOOK_EVERY + 1)
    for memory, ram, same_screen in handed:
        assert memory is not None and same_screen
        assert memory["ram"] == ram
    # So a go's own watch finds the byte, and counts the ground as the game has it.
    assert lab.game.state.lives == toy_hopper.START_LIVES
    assert watch.camera_bytes == (BYTE, None)
    assert camera(lab.game) > 400
    assert abs(watch.ground.reach[0] - camera(lab.game)) <= STEP
    r.finish()
