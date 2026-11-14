"""T-1025: the top of a clock is judged at every go's start, not only at a probe.

T-996's rule (the upper byte of a number that ticks by itself at one pace is no
progress) was asked only inside ``doubt_self_moving``, which the training service
calls from a probe, and the probe runs at the first go and after a go that ended
stuck or by a reset. On most practice levels the guess stayed and went on being
paid for standing still.
"""

from __future__ import annotations

import logging

import pytest

from retro_ai_lab.core.events import EventBus
from retro_ai_lab.emulator.registry import default_registry
from retro_ai_lab.observation import MemorySnapshot
from retro_ai_lab.services import (
    ConnectionService,
    DiscoveryService,
    DiscoveryUpdated,
    ObservationService,
)
from retro_ai_lab.services.discovery import CLOCK_TOP, MOVES_ALONE
from retro_ai_lab.understanding import Knowledge, Variable, VariableKind

LOWS = {("ram", 0), ("ram", 2), ("ram", 4), ("ram", 6)}  # what a probe sees move alone


@pytest.fixture
def bus() -> EventBus:
    return EventBus(on_error=lambda exc, event, handler: (_ for _ in ()).throw(exc))


@pytest.fixture
def connection(bus: EventBus) -> ConnectionService:
    return ConnectionService(default_registry(), bus)


@pytest.fixture
def observation(connection: ConnectionService, bus: EventBus) -> ObservationService:
    return ObservationService(connection, bus)


@pytest.fixture
def discovery(
    connection: ConnectionService, observation: ObservationService, bus: EventBus
) -> DiscoveryService:
    return DiscoveryService(connection, observation, bus, review_every=100, experiment_every=250)


def play(observation: ObservationService, first: int, upto: int) -> None:
    """The game's memory for steps ``first`` to ``upto``, as T-996's test has it, and a
    second clock at bytes 8 and 9 that no probe is told of: a clock that stops now and
    then and starts again with each go (bytes 0-1 and 4-5), a position that moves by
    varying amounts and its page (2-3), and a slower clock (6-7)."""
    for i in range(first, upto):
        go = i - i % 450
        ticks = sum(1 for j in range(go + 1, i + 1) if j % 10 != 9)
        clock, slow = 4 * ticks, 2 * ticks
        place = sum((0, 3, 6, 9, 6, 3, 9, 0)[j % 8] for j in range(go + 1, i + 1))
        pairs = (clock, place, clock, slow, clock)
        row = bytes(b for n in pairs for b in (n & 255, (n >> 8) & 255))
        observation.tracker.observe(MemorySnapshot(i, {"ram": row}))


def guesses(k: Knowledge) -> tuple[Variable, Variable, Variable, Variable, Variable]:
    top = k.propose("ram", 1, kind=VariableKind.PROGRESS, name="Clock top")
    page = k.propose("ram", 3, kind=VariableKind.PROGRESS, name="Page")
    sworn = k.propose("ram", 5, kind=VariableKind.PROGRESS, name="Sworn")
    k.confirm(sworn.id)
    lives = k.propose("ram", 7, kind=VariableKind.LIVES, name="Lives")
    unseen = k.propose("ram", 9, kind=VariableKind.PROGRESS, name="Unseen")
    return top, page, sworn, lives, unseen


def clock_doubts(k: Knowledge, var: Variable) -> int:
    return sum(1 for e in k.get(var.id).evidence if e.kind == CLOCK_TOP)


def test_the_top_of_a_clock_is_judged_at_every_go(
    discovery: DiscoveryService,
    connection: ConnectionService,
    observation: ObservationService,
    bus: EventBus,
    caplog: pytest.LogCaptureFixture,
) -> None:
    connection.connect("fake", seed=5)
    k = discovery.knowledge
    assert k is not None
    observation.tracker.reset()
    play(observation, 0, 100)  # the upper byte has moved once: nothing to judge it by yet
    top, page, sworn, lives, unseen = guesses(k)
    seen: list[DiscoveryUpdated] = []
    bus.subscribe(DiscoveryUpdated, seen.append)
    caplog.set_level(logging.INFO, logger="retro_ai_lab.services.discovery")
    # The first go's probe sees the lower bytes move by themselves, and that is all.
    assert discovery.doubt_self_moving(LOWS, frame=5) == ()
    # A later probe that happens to see less takes nothing away from what is known.
    assert discovery.doubt_self_moving({("ram", 7)}) == ()
    discovery.begin_go()
    discovery.begin_go()
    assert clock_doubts(k, top) == 0 and not seen
    # The goes run on with no probe. Once the rule holds, the next go's start doubts
    # the guess and the one after drops it.
    play(observation, 100, 1400)
    discovery.begin_go()
    now = k.get(top.id)
    assert clock_doubts(k, top) == 1 and not now.rejected
    last = now.evidence[-1]
    assert last.kind == CLOCK_TOP and last.frame == observation.frame
    assert "the top of a clock" in last.detail and "not something the player earns" in last.detail
    assert len(seen) == 1 and [v.id for v in seen[0].changed] == [top.id]
    assert (
        "1 guesses count the roll-overs of a clock: 1 doubted (Clock top), 0 dropped" in caplog.text
    )
    discovery.begin_go()
    assert k.get(top.id).rejected and clock_doubts(k, top) == 2
    assert "0 doubted (none), 1 dropped (Clock top)" in caplog.text
    assert len(seen) == 2 and [v.id for v in seen[1].changed] == [top.id]
    discovery.begin_go()  # dropped is dropped: nothing more is said
    assert clock_doubts(k, top) == 2 and len(seen) == 2
    # The page of a position, a person's word and a life count are left as they were;
    # so is the top of a clock whose lower byte no probe saw move.
    for other in (page, sworn, lives, unseen):
        assert clock_doubts(k, other) == 0 and not k.get(other.id).rejected
    assert k.get(sworn.id).confirmed
    assert "while nobody pressed anything" not in caplog.text  # only a probe says that
    assert not any(e.kind == MOVES_ALONE for v in k for e in v.evidence)


def test_a_probe_and_its_go_are_one_sighting(
    discovery: DiscoveryService,
    connection: ConnectionService,
    observation: ObservationService,
) -> None:
    """A go that starts with a probe: the probe judges the tops, and the go's start
    right after it is the same look at the same numbers, not a second one."""
    connection.connect("fake", seed=5)
    k = discovery.knowledge
    assert k is not None
    observation.tracker.reset()
    play(observation, 0, 1400)
    top, *_, again = guesses(k)
    changed = discovery.doubt_self_moving(LOWS, frame=5)
    assert [v.id for v in changed] == [top.id]
    discovery.begin_go()
    assert clock_doubts(k, top) == 1 and not k.get(top.id).rejected
    # The next go has no probe: its start is the second sighting.
    discovery.begin_go()
    assert k.get(top.id).rejected
    # And the other way about: doubted at a go's start, dropped by the next go's
    # probe, with that go's start adding nothing.
    # (The clock at bytes 8 and 9, whose lower byte no probe had seen till now.)
    wider = LOWS | {("ram", 8)}
    assert discovery.doubt_self_moving({("ram", 8)}) != ()  # a probe sees its lower byte
    assert clock_doubts(k, again) == 1
    discovery.begin_go()
    assert clock_doubts(k, again) == 1 and not k.get(again.id).rejected
    assert [v.id for v in discovery.doubt_self_moving(wider)] == [again.id]
    assert k.get(again.id).rejected
    discovery.begin_go()
    assert clock_doubts(k, again) == 2


def test_a_new_game_remembers_no_probe(
    discovery: DiscoveryService,
    connection: ConnectionService,
    observation: ObservationService,
) -> None:
    connection.connect("fake", seed=5)
    k = discovery.knowledge
    assert k is not None
    observation.tracker.reset()
    play(observation, 0, 1400)
    top, *_ = guesses(k)
    # No probe has been made: a go's start has nothing to go by, however the bytes look.
    discovery.begin_go()
    discovery.begin_go()
    assert clock_doubts(k, top) == 0
    # A probe that saw nothing move alone is no probe to remember either.
    assert discovery.doubt_self_moving(()) == ()
    discovery.begin_go()
    assert clock_doubts(k, top) == 0
    assert discovery.doubt_self_moving(LOWS) != ()
    assert clock_doubts(k, top) == 1
    # Another game: what a probe saw move in the last one says nothing about this one.
    connection.disconnect()
    connection.connect("fake", seed=6)
    other = discovery.knowledge
    assert other is not None and other is not k
    observation.tracker.reset()
    play(observation, 0, 1400)
    top, *_ = guesses(other)
    discovery.begin_go()
    discovery.begin_go()
    assert clock_doubts(other, top) == 0 and not other.get(top.id).rejected
