"""T-1001: a try from a saved moment, off the record, and whether it got past."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import pytest

from retro_ai_lab.core import ConfigurationError, EmulatorError
from retro_ai_lab.core.events import EventBus
from retro_ai_lab.emulator.fake import toy_hopper
from retro_ai_lab.emulator.fake.adapter import FakeEmulatorAdapter
from retro_ai_lab.emulator.fake.toy_hopper import LAYOUT, ToyHopper
from retro_ai_lab.emulator.registry import default_registry
from retro_ai_lab.learning.actions import ActionSpace
from retro_ai_lab.services import ConnectionService, ObservationService
from retro_ai_lab.training.trial import PAST, Outcome, play
from retro_ai_lab.training.vision import DEFAULT_LOOK_EVERY, Mark, ScreenWatch
from retro_ai_lab.understanding.knowledge import Variable, VariableKind
from retro_ai_lab.understanding.player import find_player_in_game

FRAMES = 4  # a step
SPOT = "spot"
LIVES = Variable("lives", "ram", LAYOUT.lives, kind=VariableKind.LIVES)


@dataclass
class Hopper:
    bus: EventBus
    connection: ConnectionService
    adapter: FakeEmulatorAdapter
    game: ToyHopper
    space: ActionSpace
    events: list[object]
    observation: ObservationService

    def move(self, *buttons: str) -> int:
        return self.space.index_of(buttons)

    def play(self, moves: list[int], **kw: object) -> Outcome:
        kw.setdefault("lives", [LIVES])
        return play(self.connection, SPOT, moves, self.space, frames_per_step=FRAMES, **kw)  # type: ignore[arg-type]


@pytest.fixture
def hopper() -> Iterator[Hopper]:
    bus = EventBus(on_error=lambda exc, event, handler: (_ for _ in ()).throw(exc))
    connection = ConnectionService(default_registry(), bus)
    observation = ObservationService(connection, bus)
    connection.connect("fake", game="hopper", seed=0)
    adapter = connection.adapter
    assert isinstance(adapter, FakeEmulatorAdapter)
    game = adapter.game
    assert isinstance(game, ToyHopper)
    game.reset()
    adapter.save_state(SPOT)
    events: list[object] = []
    bus.subscribe(object, events.append)
    space = ActionSpace.from_buttons(adapter.buttons())
    yield Hopper(bus, connection, adapter, game, space, events, observation)
    connection.disconnect()


def test_walking_into_the_gap_is_a_life_lost(hopper: Hopper) -> None:
    right = hopper.move("right")
    out = hopper.play([right] * 200)
    assert out.lost and not out.past and out.ground == 0
    assert 20 < out.steps < 200
    # It stopped at the step the game's own life count dropped.
    assert hopper.game.state.lives == toy_hopper.START_LIVES - 1
    hopper.adapter.load_state(SPOT)
    buttons = {b.name: b for b in hopper.adapter.buttons()}
    hopper.adapter.step(FRAMES * (out.steps - 1), pressed=[buttons["right"]])
    assert hopper.game.state.lives == toy_hopper.START_LIVES
    hopper.adapter.step(FRAMES, pressed=[buttons["right"]])
    assert hopper.game.state.lives == toy_hopper.START_LIVES - 1
    # A walk that stops short of the gap loses nothing.
    short = hopper.play([right] * 10)
    assert short == Outcome(steps=10, lost=False, ground=0) and not short.past


def a_go_that_fell_in(hopper: Hopper) -> tuple[ScreenWatch, Mark, int]:
    """The watch of a go that walked right from the save into the first gap, the mark
    of its start, and the step the life went at."""
    names = [b.name for b in hopper.adapter.buttons() if b.name not in ("start", "select")]
    hopper.adapter.load_state(SPOT)
    player = find_player_in_game(hopper.connection, names)
    assert player is not None and player.colours is not None
    watch = ScreenWatch(player)
    watch.begin_go()
    mark = watch.mark()
    buttons = {b.name: b for b in hopper.adapter.buttons()}
    hopper.adapter.load_state(SPOT)
    for step in range(200):
        look = step % DEFAULT_LOOK_EVERY == 0
        result = hopper.adapter.step(FRAMES, pressed=[buttons["right"]], screenshot=look)
        if result.frame is not None:
            watch.look(result.frame)
        if hopper.game.state.lives < toy_hopper.START_LIVES:
            return watch, mark, step + 1
    raise AssertionError("it never fell in")


def test_a_jump_at_the_edge_gets_past_and_standing_still_does_not(hopper: Hopper) -> None:
    watch, mark, died = a_go_that_fell_in(hopper)
    right, jump, still = hopper.move("right"), hopper.move("a", "right"), hopper.move()
    before = (watch.looks, watch.ground.place, watch.ground.reach, watch.ground.steps, watch.me)
    assert watch.ground.steps >= 5  # the go itself got somewhere
    # What it did, again: the life goes at the same step, and nothing is new.
    again = hopper.play([right] * (died + 20), watch=watch.trial(mark))
    assert again == Outcome(steps=died, lost=True, ground=0) and not again.past
    # The same walk with a jump somewhere before the edge, and on for a while.
    total = died + 20
    crossings = []
    for at in range(died - 1, 0, -1):
        moves = ([right] * at + [jump] * 8 + [right] * total)[:total]
        if hopper.play(moves).lost:
            continue
        seen = hopper.play(moves, watch=watch.trial(mark))
        assert not seen.lost and seen.steps == total
        crossings.append((at, seen))
    assert crossings
    assert all(seen.past and seen.ground >= PAST == 2 for _, seen in crossings)
    # Alive is not enough. Standing still: nothing new.
    stood = hopper.play([still] * total, watch=watch.trial(mark))
    assert stood == Outcome(steps=total, lost=False, ground=0) and not stood.past
    # Walking to the edge and stopping there: alive, and no further than the go got.
    stop = max(
        at for at in range(1, died) if not hopper.play([right] * at + [still] * (total - at)).lost
    )
    edge = hopper.play([right] * stop + [still] * (total - stop), watch=watch.trial(mark))
    assert not edge.lost and edge.ground == 0 and not edge.past
    # None of it changed the go's own watch.
    assert before == (
        watch.looks,
        watch.ground.place,
        watch.ground.reach,
        watch.ground.steps,
        watch.me,
    )


def test_the_same_try_twice_is_the_same_and_the_pad_is_let_go(
    hopper: Hopper, monkeypatch: pytest.MonkeyPatch
) -> None:
    right, jump = hopper.move("right"), hopper.move("a", "right")
    moves = [right] * 30 + [jump] * 8 + [right] * 30
    first = hopper.play(moves)
    place = (hopper.game.state.x, hopper.game.state.y, hopper.game.state.lives)
    assert hopper.adapter.held == frozenset()
    assert hopper.play(moves) == first
    assert place == (hopper.game.state.x, hopper.game.state.y, hopper.game.state.lives)
    # The game is left where the try ended; the save is still the save.
    hopper.adapter.load_state(SPOT)
    assert hopper.game.state.x == toy_hopper.START_X
    # The emulator fails part-way: the error comes out and the pad is let go.
    real, calls = hopper.adapter.step, [0]

    def failing(*args: object, **kw: object) -> object:
        calls[0] += 1
        if calls[0] == 5:
            raise EmulatorError("The link dropped.")
        return real(*args, **kw)  # type: ignore[arg-type]

    monkeypatch.setattr(hopper.adapter, "step", failing)
    with pytest.raises(EmulatorError):
        hopper.play(moves)
    assert hopper.adapter.held == frozenset()


def test_values_that_rise_and_values_that_fall(hopper: Hopper) -> None:
    still = hopper.move()
    low = Variable("low", "ram", LAYOUT.clock, kind=VariableKind.HEALTH)
    high = Variable("high", "ram", LAYOUT.clock + 1, kind=VariableKind.HEALTH)
    # The clock's lower byte counts up and starts over: the step it does is a value
    # going down, and the try ends there.
    hopper.adapter.load_state(SPOT)
    was, wrapped = hopper.adapter.read_memory("ram", LAYOUT.clock, 1)[0], None
    for step in range(1, 100):
        hopper.adapter.step(FRAMES)
        now = hopper.adapter.read_memory("ram", LAYOUT.clock, 1)[0]
        if now < was:
            wrapped = step
            break
        was = now
    assert wrapped is not None and wrapped > 10
    assert hopper.play([still] * 100, lives=[low]) == Outcome(wrapped, True, 0)
    # Its upper byte only goes up in that time: no loss.
    assert hopper.play([still] * 100, lives=[high]) == Outcome(100, False, 0)
    # Any one of the values watched going down is a loss.
    assert hopper.play([still] * 100, lives=[high, low, LIVES]) == Outcome(wrapped, True, 0)
    # With nothing to read lives from, nothing can be seen to be lost.
    right = hopper.move("right")
    assert hopper.play([right] * 120, lives=[]) == Outcome(120, False, 0)


def test_a_try_is_off_the_record(hopper: Hopper) -> None:
    observation = hopper.observation
    assert observation.enabled and observation.observe() is not None
    observation.advance(4)
    frame, seen = observation.frame, len(hopper.events)
    assert frame > 0 and seen > 0  # it is listening, and events do reach the list
    out = hopper.play([hopper.move("right")] * 200)
    assert out.lost
    assert observation.frame == frame and len(hopper.events) == seen


def test_nonsense_is_refused(hopper: Hopper) -> None:
    right = hopper.move("right")
    with pytest.raises(ConfigurationError):
        hopper.play([])
    with pytest.raises(ConfigurationError):
        hopper.play([right, len(hopper.space)])
    with pytest.raises(ConfigurationError):
        play(hopper.connection, SPOT, [right], hopper.space, frames_per_step=0, lives=[LIVES])
    with pytest.raises(ConfigurationError):
        hopper.play([right], look_every=0)
    with pytest.raises(ConfigurationError):
        hopper.play([right], watch="a watch")
    with pytest.raises(ConfigurationError):
        play(hopper.connection, "", [right], hopper.space, frames_per_step=4, lives=[LIVES])
    with pytest.raises(ConfigurationError):
        hopper.play([right], lives=["lives"])
    # A move that needs a button this controller does not have.
    odd = ActionSpace([frozenset({"turbo"})])
    with pytest.raises(ConfigurationError):
        play(hopper.connection, SPOT, [0], odd, frames_per_step=4, lives=[LIVES])
    # A save that was never made is the emulator's to refuse (the practice one does).
    with pytest.raises((ConfigurationError, EmulatorError)):
        play(hopper.connection, "nowhere", [right], hopper.space, frames_per_step=4, lives=[LIVES])
