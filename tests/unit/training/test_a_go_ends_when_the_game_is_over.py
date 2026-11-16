"""T-1050: the game being over is read from the count lives lost are counted on.

Boss's log of 16 Nov: one go ran 17 minutes through 64 lives lost. His game counts
its lives 2, 1, 0 and then below nought (255 in a byte) when the last one goes, and
sets the count to 2 when a new game begins. Neither read as the game starting over,
and a reward of his own that pays on nearly every step kept the wait for "nothing
paid" from running out.
"""

from __future__ import annotations

import logging
from itertools import pairwise

import pytest

from retro_ai_lab.core.events import EventBus
from retro_ai_lab.emulator.fake import toy_hopper
from retro_ai_lab.emulator.fake.toy_hopper import LAYOUT, ToyHopper
from retro_ai_lab.services import DiscoveryUpdated
from retro_ai_lab.training.episode import EpisodeSignals
from retro_ai_lab.ui.learning_panel import ENDED_BY
from retro_ai_lab.understanding import VariableKind
from retro_ai_lab.understanding.events import EventKind, GameEvent
from tests.unit.training import test_going_back as going_back
from tests.unit.training.test_going_back import DIED_AT, SURE, Lab, steps
from tests.unit.training.test_search_at_a_loss import GAP_ENDS, searching

lab = going_back.lab  # the practice game behind the real services (pytest fixture)

LIFES, HEALTH, OTHER = "ram:1882", "ram:40", "ram:7"
NAME = "Lifes at ram[1882]"
BELOW = "it fell below nought, so every life is lost"
ANEW = "it was set back to what it was when the go began"
WENT_HOME = "Progress, Flag all went back to their starting values."
LIVES = f"Lives at ram[{LAYOUT.lives}]"  # the practice game's count, as the lab names it


def a_change(value: str, before: int, after: int, *, frame: int = 7) -> GameEvent:
    """A value a person's reward is on changing, as the detector reports it."""
    kind = EventKind.UP if after > before else EventKind.DOWN
    return GameEvent(kind, frame, value, "", before, after)


def tell(bus: EventBus, *events: GameEvent) -> None:
    bus.publish(DiscoveryUpdated((), (), tuple(events), ()))


def counts(bus: EventBus, value: str, *values: int) -> None:
    """The value reads each of ``values`` in turn, one update each."""
    for before, after in pairwise(values):
        tell(bus, a_change(value, before, after))


@pytest.fixture
def bus() -> EventBus:
    return EventBus(on_error=lambda exc, event, handler: (_ for _ in ()).throw(exc))


@pytest.fixture
def signals(bus: EventBus) -> EpisodeSignals:
    """A go's signals with lives lost counted on a count of lives and a health, of
    which only the count is read for the game being over."""
    made = EpisodeSignals(bus)
    made.watch_setbacks(frozenset({LIFES, HEALTH}))
    made.watch_lives({LIFES: NAME})
    made.start()
    return made


def taken(signals: EpisodeSignals) -> tuple[int, int, bool]:
    """Restarts and lives lost since last asked, and whether the count told them all."""
    _, restarts = signals.take()
    return restarts, signals.take_setbacks(), signals.over


# ------------------------------------------------------------ the two signs


def test_below_nought_is_the_game_over_and_a_life_lost(
    bus: EventBus, signals: EpisodeSignals
) -> None:
    counts(bus, LIFES, 2, 1, 0)
    assert taken(signals) == (0, 2, False) and signals.last_reset == ""
    # The last life goes: the count leaves nought for the top of its byte. That read
    # as a rise, so nothing else counted the life.
    counts(bus, LIFES, 0, 255)
    assert taken(signals) == (1, 1, True)
    assert signals.last_reset == f"{NAME} went from 0 to 255: {BELOW} (frame 7)"
    assert signals.last_reset_ids == ()
    # Asked again with nothing new, it is still how the last restart was told.
    assert taken(signals) == (0, 0, True)


@pytest.mark.parametrize("top", [3, 99, 255])
def test_below_nought_is_anything_above_what_the_go_began_with(
    bus: EventBus, signals: EpisodeSignals, top: int
) -> None:
    counts(bus, LIFES, 2, 1, 0, top)
    assert taken(signals) == (1, 3, True)


def test_set_anew_is_the_game_over(bus: EventBus, signals: EpisodeSignals) -> None:
    counts(bus, LIFES, 3, 2, 1, 0)
    assert taken(signals) == (0, 3, False)
    # The game begins again and gives its lives anew: no life is lost by that.
    counts(bus, LIFES, 0, 3)
    assert taken(signals) == (1, 0, True)
    assert signals.last_reset == f"{NAME} went from 0 to 3: {ANEW} (frame 7)"


def test_set_anew_from_wherever_it_stood(bus: EventBus, signals: EpisodeSignals) -> None:
    # A game that is over with the count left at 1, and one that counts on from the
    # top of its byte (the count Boss's game shows when a new game begins).
    counts(bus, LIFES, 5, 4, 3, 2, 1, 5)
    assert taken(signals) == (1, 4, True)
    signals.start()
    counts(bus, LIFES, 2, 1, 0, 255)
    assert taken(signals) == (1, 3, True)
    counts(bus, LIFES, 255, 2)  # (down, to the eye: counted as a life lost as it was)
    assert taken(signals) == (1, 1, True)
    assert signals.last_reset == f"{NAME} went from 255 to 2: {ANEW} (frame 7)"


# ------------------------------------------------------------ what is not one


def test_a_step_of_one_is_never_it(bus: EventBus, signals: EpisodeSignals) -> None:
    # A life lost, a life given back (to what the go began with, too), the last life
    # going where the count stops at nought, and a life given there.
    counts(bus, LIFES, 2, 1, 2, 3, 2, 1, 0, 1, 0)
    assert taken(signals) == (0, 5, False) and signals.last_reset == ""
    # (And it is still read: the next change is the game over.)
    counts(bus, LIFES, 0, 2)
    assert taken(signals) == (1, 0, True)


@pytest.mark.parametrize(
    "first",
    [
        (2, 5),  # more than one before any fall
        (2, 0),  # ... downwards
        (0, 255),  # a go that begins at nought: no fall has been seen
    ],
)
def test_a_jump_before_any_fall_rules_the_value_out(
    bus: EventBus, signals: EpisodeSignals, first: tuple[int, int]
) -> None:
    counts(bus, LIFES, *first)
    assert taken(signals)[0] == 0
    # Nothing it does for the rest of the go is read, whatever it looks like.
    counts(bus, LIFES, 2, 1, 0, 255)
    counts(bus, LIFES, 1, 0, first[0] or 3)
    assert taken(signals)[0] == 0 and signals.over is False and signals.last_reset == ""


@pytest.mark.parametrize(
    "went",
    [
        (4, 3, 7),  # up by more than one, past what it began with
        (4, 3, 1),  # down by more than one
        (4, 3, 2, 1, 0, 2),  # from nought to less than it began with
        (4, 3, 2, 1, 0, 3),
        (4, 3, 2, 4, 3, 2, 0),  # (set anew once: the game over; then down by two)
    ],
)
def test_a_jump_to_elsewhere_rules_the_value_out(
    bus: EventBus, signals: EpisodeSignals, went: tuple[int, ...]
) -> None:
    # It has fallen, and then changes by more than one to what is neither above its
    # beginning from nought nor its beginning: health filled part-way, or no count.
    counts(bus, LIFES, *went)
    assert taken(signals)[0] == (1 if went[3:4] == (4,) else 0)
    counts(bus, LIFES, 1, 0, 255)
    counts(bus, LIFES, 1, 0, 4)
    assert taken(signals)[0] == 0 and signals.over == (went[3:4] == (4,))


def test_only_the_values_lives_are_counted_on_and_no_health(
    bus: EventBus, signals: EpisodeSignals
) -> None:
    # A health lives lost are counted on, and a value they are not: neither is read,
    # and what counted as a life lost before still does.
    counts(bus, HEALTH, 3, 2, 1, 0, 255, 3)
    counts(bus, OTHER, 3, 2, 1, 0, 255, 3)
    assert taken(signals) == (0, 4, False)  # (the health's falls, and 255 to 3)
    # A change with no values to it is not read.
    tell(bus, GameEvent(EventKind.DOWN, 7, LIFES, ""), GameEvent(EventKind.UP, 8, LIFES, ""))
    assert taken(signals) == (0, 1, False)
    # Other values' events in the same update do not get in the way.
    counts(bus, LIFES, 2, 1, 0)
    tell(bus, a_change(OTHER, 9, 0), a_change(HEALTH, 3, 9), a_change(LIFES, 0, 255))
    assert taken(signals) == (1, 3, True)


def test_a_change_told_twice_is_read_once(bus: EventBus, signals: EpisodeSignals) -> None:
    # A person's value the AI also takes for a life count comes as two events.
    def both(before: int, after: int) -> None:
        down = after < before
        tell(
            bus,
            a_change(LIFES, before, after),
            GameEvent(
                EventKind.SETBACK if down else EventKind.RECOVERY, 7, LIFES, "", before, after
            ),
        )

    both(2, 1)
    both(1, 0)
    assert taken(signals) == (0, 2, False)
    both(0, 255)
    assert taken(signals) == (1, 1, True)


def test_nothing_is_read_while_no_go_is_on(bus: EventBus) -> None:
    made = EpisodeSignals(bus)
    made.watch_setbacks(frozenset({LIFES}))
    made.watch_lives({LIFES: NAME})
    counts(bus, LIFES, 2, 1, 0, 255)
    assert taken(made) == (0, 0, False)
    made.start()
    counts(bus, LIFES, 2, 1)
    made.stop()
    counts(bus, LIFES, 1, 0, 255)
    assert taken(made) == (0, 1, False)


# ------------------------------------------------------------ the go's own


def test_a_new_go_begins_anew(bus: EventBus, signals: EpisodeSignals) -> None:
    # What it read when the last go began, and that it fell then, are that go's.
    counts(bus, LIFES, 2, 1)
    signals.start()
    counts(bus, LIFES, 0, 255)  # no fall seen in this go
    assert taken(signals) == (0, 0, False)
    # A value ruled out in one go is read in the next.
    signals.start()
    counts(bus, LIFES, 2, 1, 0, 255)
    assert taken(signals) == (1, 3, True)
    # How the last restart was told is forgotten with the go.
    signals.start()
    assert signals.over is False and signals.last_reset == ""


def test_with_the_detectors_restart_it_is_a_restart(bus: EventBus, signals: EpisodeSignals) -> None:
    # The detector says the game started over in the same update: its word stands.
    counts(bus, LIFES, 3, 2, 1, 0)
    signals.take_setbacks()
    tell(
        bus, a_change(LIFES, 0, 3), GameEvent(EventKind.RESET, 9, None, WENT_HOME, variables=("a",))
    )
    assert taken(signals) == (2, 0, False)
    assert signals.last_reset == f"{WENT_HOME} (frame 9)" and signals.last_reset_ids == ("a",)
    # A restart the detector alone tells, after one the count told, is not the count's.
    signals.start()
    counts(bus, LIFES, 2, 1, 0, 255)
    assert taken(signals) == (1, 3, True)
    tell(bus, GameEvent(EventKind.RESET, 9, None, WENT_HOME))
    assert taken(signals) == (1, 0, False)


# ------------------------------------------------------------ the runner


def as_bosss_game_counts(game: ToyHopper) -> None:
    """The practice game's dying pause as it ends on a game that counts the lives in
    reserve: the count goes below nought with the last life (255 in its byte), and
    only then is the game over. (Otherwise the game's own `_die_a_little`.)"""
    ram = game._ram
    ram[LAYOUT.dying] -= 1
    if ram[LAYOUT.dying] > 0:
        return
    ram[LAYOUT.lives] = (ram[LAYOUT.lives] - 1) & 0xFF
    if ram[LAYOUT.lives] == 0xFF:
        ram[LAYOUT.mode] = toy_hopper.GAME_OVER
        return
    game._to_start()


@pytest.fixture
def one_in_reserve(monkeypatch: pytest.MonkeyPatch) -> None:
    """The practice game with one life in reserve, counted as Boss's game counts."""
    monkeypatch.setattr(toy_hopper, "START_LIVES", 1)
    monkeypatch.setattr(ToyHopper, "_die_a_little", as_bosss_game_counts)


def test_the_go_ends_by_game_over(
    lab: Lab, one_in_reserve: None, caplog: pytest.LogCaptureFixture
) -> None:
    lab.teach()
    lab.game.reset()
    assert lab.game.state.lives == 1
    r = lab.runner(max_steps=4000)
    r.begin()
    # Walking right it falls in the first gap, is put back, and falls in again.
    steps(r, DIED_AT)
    assert lab.game.state.lives == 0 and not lab.game.state.game_over
    played = 0
    while r.step():
        played += 1
        assert played < 200
    assert lab.game.state.lives == 255 and lab.game.state.game_over
    result = r.finish()
    assert (result.ended_by, result.setbacks) == ("game_over", 2)
    assert result.steps == DIED_AT + played + 1 < 2 * DIED_AT + 10
    assert r.last_reset.startswith(f"{LIVES} went from 0 to 255: {BELOW} (frame ")
    assert ENDED_BY["game_over"] == "the game was over"
    # The step the last life went on was learned as the end of the story.
    assert lab.agent.learned[-1][4] is True


def test_without_the_count_nothing_is_read(lab: Lab, one_in_reserve: None) -> None:
    # The same go with no life count known: nothing says the game is over. (On this
    # game the dead screen ends it some steps later, as a screen that stopped changing.)
    lab.teach(lives=False)
    lab.game.reset()
    r = lab.runner(max_steps=4000)
    r.begin()
    played = 0
    while r.step():
        played += 1
        assert played < 400
    assert lab.game.state.game_over and played > 2 * DIED_AT
    result = r.finish()
    assert (result.ended_by, result.setbacks, r.last_reset) == ("stuck", 0, "")


def test_a_count_nobody_named_is_named_as_the_ai_names_it(lab: Lab, one_in_reserve: None) -> None:
    lab.teach(lives=False)
    count = lab.knowledge.propose("ram", LAYOUT.lives, kind=VariableKind.LIVES, evidence=SURE)
    lab.bus.publish(DiscoveryUpdated((count,), (), (), ()))
    assert not count.name and count.display_name
    lab.game.reset()
    r = lab.runner(max_steps=4000)
    r.begin()
    while r.step():
        pass
    assert r.finish().ended_by == "game_over"
    assert r.last_reset.startswith(f"{count.display_name} went from 0 to 255: {BELOW} (frame ")


class LastLife:
    """Tells the bus that the count went below nought in the very step a life is seen
    to go, as the engine does when that life was the last."""

    def __init__(self, lab: Lab) -> None:
        self.lab = lab
        self.told = 0
        best = lab.knowledge.best(VariableKind.LIVES)
        assert best is not None
        self.count = best.id
        lab.bus.subscribe(DiscoveryUpdated, self._on_update)

    def _on_update(self, update: DiscoveryUpdated) -> None:
        if any(e.kind is EventKind.SETBACK for e in update.events):
            self.told += 1
            below = GameEvent(
                EventKind.RECOVERY, self.lab.observation.frame, self.count, "", 0, 255
            )
            self.lab.bus.publish(DiscoveryUpdated((), (), (below,), ()))


def test_a_search_goes_first_and_a_way_found_carries_on(
    lab: Lab, caplog: pytest.LogCaptureFixture
) -> None:
    r, moments = searching(lab)
    last = LastLife(lab)
    r.begin()
    steps(r, DIED_AT - 1)
    start = moments.kept[0]
    with caplog.at_level(logging.INFO, logger="retro_ai_lab.training.episode"):
        assert r.step()  # the last life goes, and the go is not over: it looks first
    assert last.told == 1 and r.last_reset.startswith(f"{LIVES} went from 0 to 255: {BELOW}")
    assert (r.ways_sought, r.ways_found, r.times_back) == (1, 1, 1)
    assert lab.loaded[-1] == start.slot and lab.game.state.lives == toy_hopper.START_LIVES
    # The way is played and gets the player across with every life.
    while lab.game.state.x <= GAP_ENDS:
        assert r.step()
    assert lab.game.state.lives == toy_hopper.START_LIVES and last.told == 1
    # Its one search is used: the next life told to be the last ends the go.
    while r.step():
        pass
    result = r.finish()
    assert last.told == 2 and result.ended_by == "game_over"
    assert (result.ways_sought, result.ways_found) == (1, 1)


def test_with_no_way_found_the_go_ends_by_game_over(lab: Lab) -> None:
    r, _ = searching(lab, way_tries=3)
    last = LastLife(lab)
    r.begin()
    steps(r, DIED_AT - 1)
    assert not r.step()  # searched, nothing found: the game is over
    assert last.told == 1 and (r.ways_sought, r.ways_found) == (1, 0)
    result = r.finish()
    assert (result.ended_by, result.steps) == ("game_over", DIED_AT)
