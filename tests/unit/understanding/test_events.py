"""T-304: events from variable changes."""

from __future__ import annotations

import pytest

from retro_ai_lab.core import ConfigurationError
from retro_ai_lab.emulator.fake import ToyRacer
from retro_ai_lab.emulator.fake.toy_racer import LAYOUT
from retro_ai_lab.observation import MemorySnapshot
from retro_ai_lab.understanding import (
    Confidence,
    Encoding,
    EventDetector,
    EventKind,
    EventLog,
    Evidence,
    GameEvent,
    Knowledge,
    VariableKind,
)
from retro_ai_lab.understanding.events import CYCLES

RAM = "ram"
SURE = [Evidence("test", "", 3.0)]  # Medium


def snap(frame: int, *values: int) -> MemorySnapshot:
    return MemorySnapshot(frame, {RAM: bytes(values)})


def knowledge_for(layout: dict[int, VariableKind], **extra: object) -> Knowledge:
    k = Knowledge("g")
    for off, kind in layout.items():
        k.propose(RAM, off, kind=kind, name=kind.label, evidence=SURE)
    return k


# ---------------------------------------------------------- basic rules


def test_first_snapshot_only_starts_watching() -> None:
    k = knowledge_for({0: VariableKind.PROGRESS})
    d = EventDetector(k)
    assert d.observe(snap(0, 5)) == ()
    assert d.watching == ("v1",) and d.observed == 1


def test_progress_setback_recovery_and_flags() -> None:
    k = knowledge_for(
        {
            0: VariableKind.PROGRESS,
            1: VariableKind.LIVES,
            2: VariableKind.FLAG,
            3: VariableKind.HEALTH,
        }
    )
    d = EventDetector(k)
    d.observe(snap(0, 0, 3, 0, 10))
    assert d.observe(snap(1, 0, 3, 0, 10)) == ()
    events = d.observe(snap(2, 1, 2, 1, 7))
    kinds = [e.kind for e in events]
    assert kinds == [EventKind.PROGRESS, EventKind.SETBACK, EventKind.FLAG_ON, EventKind.SETBACK]
    assert events[0] == GameEvent(EventKind.PROGRESS, 2, "v1", "Progress went up to 1.", 0, 1)
    assert events[1].delta == -1 and "Lives dropped to 2" in events[1].detail
    only_health = d.observe(snap(3, 1, 2, 1, 9))
    assert [e.kind for e in only_health] == [EventKind.RECOVERY]
    assert d.observe(snap(4, 1, 2, 1, 9)) == ()
    assert [e.kind for e in d.observe(snap(5, 1, 2, 0, 9))] == [EventKind.FLAG_OFF]
    # progress going down on its own is not an event
    d.observe(snap(6, 3, 2, 0, 9))
    assert d.observe(snap(7, 2, 2, 0, 9)) == ()


def test_asked_values_report_every_move() -> None:
    """T-860: a value a person asked about reports UP and DOWN, whatever its kind."""
    k = knowledge_for({0: VariableKind.POSITION, 1: VariableKind.LIVES, 2: VariableKind.SPEED})
    d = EventDetector(k)
    assert d.asked == frozenset()
    d.observe(snap(0, 10, 3, 0))
    assert d.observe(snap(1, 12, 3, 1)) == ()  # positions and speeds are not tracked
    d.ask(["v1", "v2"])
    assert d.observe(snap(2, 12, 3, 1)) == ()  # asking starts the watch; no false move
    events = d.observe(snap(3, 15, 2, 2))
    assert events[0] == GameEvent(EventKind.UP, 3, "v1", "Position went up to 15.", 12, 15)
    assert [e.kind for e in events[1:]] == [EventKind.DOWN, EventKind.SETBACK]  # lives: both
    assert events[1].delta == -1 and events[2].variable_id == "v2"
    assert d.observe(snap(4, 15, 2, 3)) == ()  # the speed was not asked about
    assert [e.kind for e in d.observe(snap(5, 9, 2, 3))] == [EventKind.DOWN]
    d.ask([])  # no longer asked: the position is not tracked again
    assert d.observe(snap(6, 20, 2, 3)) == ()
    assert "v1" not in d.watching
    # A rejected or unknown value is never tracked, however often it is asked for.
    bad = k.propose(RAM, 3, kind=VariableKind.UNKNOWN, evidence=SURE)
    k.reject(bad.id)
    d.ask([bad.id, "nope"])
    assert d.observe(snap(7, 20, 2, 3, 1)) == () and d.observe(snap(8, 20, 2, 3, 2)) == ()


def test_reset_needs_two_values_returning_home_with_a_drop() -> None:
    k = knowledge_for({0: VariableKind.PROGRESS, 1: VariableKind.LIVES})
    d = EventDetector(k)
    d.observe(snap(0, 0, 3))
    d.observe(snap(1, 2, 1))
    events = d.observe(snap(2, 0, 3))
    assert events[-1].kind is EventKind.RESET and events[-1].variable_id is None
    assert "Progress" in events[-1].detail and "Lives" in events[-1].detail
    # only one value returning: no reset
    d.observe(snap(3, 1, 3))
    assert all(e.kind is not EventKind.RESET for e in d.observe(snap(4, 0, 3)))
    # both return but nothing dropped: a plain recovery, not a reset
    d2 = EventDetector(knowledge_for({0: VariableKind.HEALTH, 1: VariableKind.HEALTH}))
    d2.observe(snap(0, 5, 5))
    d2.observe(snap(1, 2, 2))
    kinds = {e.kind for e in d2.observe(snap(2, 5, 5))}
    assert kinds == {EventKind.RECOVERY}


def test_reset_needs_one_value_the_ai_is_fairly_sure_of() -> None:
    """T-838: LOW guesses going home together are noise (the real game's stack bytes)."""
    from retro_ai_lab.understanding.events import RESET_NEEDS, RESET_SURE
    from retro_ai_lab.understanding.knowledge import Confidence, Evidence

    assert RESET_NEEDS == 2 and RESET_SURE is Confidence.MEDIUM
    low = [Evidence("guess", "", 1.0)]
    k = Knowledge("g")
    for off, kind in enumerate((VariableKind.PROGRESS, VariableKind.LIVES, VariableKind.SCORE)):
        k.propose(RAM, off, kind=kind, name=kind.label, evidence=low)
    assert all(v.confidence is Confidence.LOW for v in k)
    d = EventDetector(k)
    d.observe(snap(0, 0, 3, 0))
    d.observe(snap(1, 2, 1, 7))
    assert all(e.kind is not EventKind.RESET for e in d.observe(snap(2, 0, 3, 0)))
    # One of them fairly sure is enough.
    k2 = Knowledge("g")
    k2.propose(RAM, 0, kind=VariableKind.PROGRESS, name="P", evidence=low)
    k2.propose(RAM, 1, kind=VariableKind.LIVES, name="L", evidence=[Evidence("seen", "", 3.0)])
    lives = k2.at(RAM, 1)
    assert lives is not None and lives.confidence is Confidence.MEDIUM
    d2 = EventDetector(k2)
    d2.observe(snap(0, 0, 3))
    d2.observe(snap(1, 2, 1))
    assert any(e.kind is EventKind.RESET for e in d2.observe(snap(2, 0, 3)))


def test_score_gain_fires_on_unusual_jumps_only() -> None:
    k = Knowledge("g")
    k.propose(RAM, 0, encoding=Encoding.U16LE, kind=VariableKind.SCORE, name="Score", evidence=SURE)
    d = EventDetector(k)
    score = 0
    d.observe(snap(0, 0, 0))
    seen: list[GameEvent] = []
    for i in range(1, 40):
        score += 2
        if i == 20:
            score += 50
        seen.extend(d.observe(snap(i, score & 0xFF, score >> 8)))
    assert [e.kind for e in seen] == [EventKind.SCORE_GAIN]
    assert seen[0].frame == 20 and seen[0].delta == 52


def test_only_confident_unrejected_tracked_variables_are_watched() -> None:
    k = Knowledge("g")
    k.propose(RAM, 0, kind=VariableKind.PROGRESS)  # Unknown confidence
    k.propose(RAM, 1, kind=VariableKind.PROGRESS, evidence=SURE)
    rejected = k.propose(RAM, 2, kind=VariableKind.PROGRESS, evidence=SURE)
    k.reject(rejected.id)
    k.propose(RAM, 3, kind=VariableKind.POSITION, evidence=SURE)  # wrong kind
    d = EventDetector(k)
    d.observe(snap(0, 0, 0, 0, 0))
    assert d.watching == ("v2",)
    events = d.observe(snap(1, 1, 1, 1, 1))
    assert [e.variable_id for e in events] == ["v2"]
    strict = EventDetector(k, at_least=Confidence.HIGH)
    strict.observe(snap(0, 0, 0, 0, 0))
    assert strict.watching == ()


def test_detector_follows_knowledge_edits_and_reset() -> None:
    k = knowledge_for({0: VariableKind.PROGRESS})
    d = EventDetector(k)
    d.observe(snap(0, 0, 0))
    k.update("v1", kind=VariableKind.LIVES)  # a person corrected it
    assert d.observe(snap(1, 1, 0)) == ()  # re-armed, no event on the edit frame
    assert d.observe(snap(2, 0, 0))[0].kind is EventKind.SETBACK
    d.reset()
    assert d.observed == 0 and d.watching == ()
    with pytest.raises(ConfigurationError):
        d.observe("nope")  # type: ignore[arg-type]
    with pytest.raises(ConfigurationError):
        EventDetector(k, gain_factor=1.0)
    with pytest.raises(ConfigurationError):
        k.propose(RAM, 5, encoding=Encoding.U16LE, kind=VariableKind.SCORE, evidence=SURE)
        EventDetector(k).observe(snap(0, 0))  # too short for the 16-bit one: skipped quietly
        raise ConfigurationError("reached")  # the observe above must not raise


# --------------------------------------------------------------- log


def test_event_log() -> None:
    log = EventLog(capacity=3)
    e = [GameEvent(EventKind.PROGRESS, i, "v1", "") for i in range(5)]
    log.extend(tuple(e[:2]))
    log.extend(tuple(e[2:]))
    assert len(log) == 3 and log.total == 5
    assert log.recent(2) == (e[3], e[4]) and log.recent(0) == ()
    assert log.count(EventKind.PROGRESS) == 3 and log.count(EventKind.RESET) == 0
    assert log.since(4) == (e[4],)
    assert list(log) == e[2:]
    log.clear()
    assert len(log) == 0 and log.total == 0
    with pytest.raises(ConfigurationError):
        EventLog(0)
    with pytest.raises(ConfigurationError):
        log.recent(-1)


# ---------------------------------------------- on the real toy racer


def test_toy_racer_story() -> None:
    k = Knowledge("toyracer")
    k.propose(
        RAM,
        LAYOUT.score,
        encoding=Encoding.U16LE,
        kind=VariableKind.SCORE,
        name="Score",
        evidence=SURE,
    )
    k.propose(RAM, LAYOUT.lives, kind=VariableKind.LIVES, name="Lives", evidence=SURE)
    k.propose(RAM, LAYOUT.checkpoint, kind=VariableKind.PROGRESS, name="Checkpoint", evidence=SURE)
    k.propose(RAM, LAYOUT.lap, kind=VariableKind.PROGRESS, name="Lap", evidence=SURE)
    k.propose(RAM, LAYOUT.game_over, kind=VariableKind.FLAG, name="Game over", evidence=SURE)
    game = ToyRacer(seed=2)
    d = EventDetector(k)
    log = EventLog()
    d.observe(MemorySnapshot(0, {RAM: game.ram}))
    for i in range(1, 900):
        game.step(["start"] if game.state.game_over else ["up"])
        log.extend(d.observe(MemorySnapshot(i, {RAM: game.ram})))
    assert log.count(EventKind.PROGRESS) >= 3, "checkpoints should register as progress"
    assert log.count(EventKind.SETBACK) >= 1, "crashes should register as setbacks"
    assert log.count(EventKind.SCORE_GAIN) >= 1, "checkpoint bonuses should stand out"
    assert log.count(EventKind.FLAG_ON) >= 1 and log.count(EventKind.RESET) >= 1
    flag_on = next(e for e in log if e.kind is EventKind.FLAG_ON)
    reset = next(e for e in log if e.kind is EventKind.RESET and e.frame > flag_on.frame)
    assert reset.frame > flag_on.frame, "the restart comes after the game over"


# ------------------------------------------------------------ T-841


class CountingSnapshot(MemorySnapshot):
    """Counts how many values the detector reads out of it."""

    reads = 0

    def get(self, region: str, offset: int = 0, length: int | None = None) -> bytes:
        CountingSnapshot.reads += 1
        return super().get(region, offset, length)


def test_stale_variables_cost_nothing_per_snapshot() -> None:
    k = knowledge_for({0: VariableKind.PROGRESS, 1: VariableKind.SCORE})
    for i in range(300):  # knowledge about memory this emulator no longer shares
        k.propose("ciram_nametables", i, kind=VariableKind.SCORE, evidence=SURE)
    d = EventDetector(k)
    d.observe(CountingSnapshot(0, {RAM: bytes([0, 0])}))
    CountingSnapshot.reads = 0
    for frame in range(1, 11):
        d.observe(CountingSnapshot(frame, {RAM: bytes([frame, 0])}))
    assert CountingSnapshot.reads == 20  # two live variables, ten snapshots
    assert d.watching == ("v1", "v2")


def test_a_new_variable_is_watched_from_the_next_snapshot() -> None:
    k = knowledge_for({0: VariableKind.PROGRESS})
    d = EventDetector(k)
    d.observe(snap(0, 0, 0))
    k.propose(RAM, 1, kind=VariableKind.LIVES, evidence=SURE)
    d.observe(snap(1, 0, 3))  # starts watching the new one
    assert d.watching == ("v1", "v2")
    assert [e.kind for e in d.observe(snap(2, 0, 2))] == [EventKind.SETBACK]
    # A snapshot with another set of regions is looked at afresh too.
    k.propose("wram", 0, kind=VariableKind.PROGRESS, evidence=SURE)
    d.observe(MemorySnapshot(3, {RAM: bytes([0, 2]), "wram": bytes([1])}))
    assert d.watching == ("v1", "v2", "v3")
    assert [
        e.kind for e in d.observe(MemorySnapshot(4, {RAM: bytes([0, 2]), "wram": bytes([2])}))
    ] == [EventKind.PROGRESS]


# ------------------------------------------------------------ T-845


def test_reset_events_name_the_values_they_blame() -> None:
    k = knowledge_for({0: VariableKind.PROGRESS, 1: VariableKind.LIVES})
    d = EventDetector(k)
    d.observe(snap(0, 0, 3))
    d.observe(snap(1, 2, 1))
    reset = d.observe(snap(2, 0, 3))[-1]
    assert reset.kind is EventKind.RESET and reset.variables == ("v1", "v2")
    assert GameEvent(EventKind.PROGRESS, 1, "v1", "up").variables == ()


def test_a_cycling_value_no_longer_counts_toward_a_reset() -> None:
    k = knowledge_for({0: VariableKind.PROGRESS, 1: VariableKind.LIVES, 2: VariableKind.SCORE})
    k.support("v1", Evidence(CYCLES, "cycles", 0.0))
    k.support("v2", Evidence(CYCLES, "cycles", 0.0))
    assert k.get("v1").confidence is Confidence.MEDIUM  # weight 0: as sure as before
    d = EventDetector(k)
    d.observe(snap(0, 0, 3, 0))
    d.observe(snap(1, 2, 1, 5))
    events = d.observe(snap(2, 0, 3, 5))  # the two cyclers go home: no reset
    assert all(e.kind is not EventKind.RESET for e in events)
    assert any(e.kind is EventKind.SETBACK for e in d.observe(snap(3, 0, 2, 5)))  # still watched
    # Through a saved and reloaded knowledge too.
    again = EventDetector(Knowledge.from_dict(k.to_dict()))
    again.observe(snap(0, 0, 3, 0))
    again.observe(snap(1, 2, 1, 5))
    assert all(e.kind is not EventKind.RESET for e in again.observe(snap(2, 0, 3, 5)))


def test_a_reset_needs_most_of_the_moved_values_to_go_home() -> None:
    """T-846: a few values back at their start while many others are not is no restart."""
    from retro_ai_lab.understanding.events import RESET_SHARE

    assert RESET_SHARE == 0.5
    layout = {i: VariableKind.PROGRESS for i in range(5)}
    layout[5] = VariableKind.LIVES
    d = EventDetector(knowledge_for(layout))
    d.observe(snap(0, 0, 0, 0, 0, 0, 3))
    d.observe(snap(1, 1, 2, 3, 4, 5, 1))  # all six moved
    assert d.moved == 6
    # Two go home (one of them sure, and a drop): 2 of 6 is not the game starting over.
    events = d.observe(snap(2, 0, 2, 3, 4, 5, 3))
    assert all(e.kind is not EventKind.RESET for e in events)
    # Three of six: that is half, and a restart.
    d.observe(snap(3, 1, 2, 3, 4, 5, 1))
    events = d.observe(snap(4, 0, 0, 3, 4, 5, 3))
    assert [e.kind for e in events if e.kind is EventKind.RESET]
    assert events[-1].variables == ("v1", "v2", "v6")


def test_flags_switching_back_are_not_a_restart() -> None:
    """T-867: a door closing is not the game starting over; a number must drop home."""
    kinds = {0: VariableKind.FLAG, 1: VariableKind.FLAG, 2: VariableKind.PROGRESS}
    d = EventDetector(knowledge_for(kinds))
    d.observe(snap(0, 0, 0, 0))
    d.observe(snap(1, 1, 1, 2))  # both flags on, progress away from home
    d.observe(snap(2, 1, 1, 3))
    # Both flags switch back while progress stays out: two of three moved, one sure, a
    # "drop" from 1 to 0, and still no restart.
    events = d.observe(snap(3, 0, 0, 3))
    assert {e.kind for e in events} == {EventKind.FLAG_OFF}
    # Progress dropping home alongside a flag is a restart, and the flag is named too.
    d.observe(snap(4, 1, 0, 3))
    events = d.observe(snap(5, 0, 0, 0))
    reset = [e for e in events if e.kind is EventKind.RESET]
    assert reset and set(reset[0].variables) == {"v1", "v3"}


def test_values_that_never_moved_do_not_count_against_a_reset() -> None:
    """T-846: a value that sat still all along is not one the restart has to bring home."""
    kinds = {0: VariableKind.PROGRESS, 1: VariableKind.LIVES, 2: VariableKind.SCORE}
    d = EventDetector(knowledge_for(kinds))
    d.observe(snap(0, 0, 3, 9))
    d.observe(snap(1, 2, 1, 9))  # the score never moves
    assert d.moved == 2
    assert any(e.kind is EventKind.RESET for e in d.observe(snap(2, 0, 3, 9)))
    d.reset()
    assert d.moved == 0


def test_only_the_best_few_of_a_kind_are_watched() -> None:
    """T-850: 40 LOW flags: the 16 strongest are watched, the rest wait for evidence."""
    k = Knowledge("g")
    for i in range(40):
        k.propose(RAM, i, kind=VariableKind.FLAG, evidence=[Evidence("t", "", 1.0)])
    d = EventDetector(k)
    d.observe(snap(0, *([0] * 40)))
    assert len(d.watching) == 16
    k.support("v40", Evidence("t", "", 2.0))  # the last one earns a seat
    d.observe(snap(1, *([0] * 40)))
    assert "v40" in d.watching and "v16" not in d.watching and len(d.watching) == 16


# ---------------------------------------------------------- T-1004: a jump back


def test_rebase_takes_the_values_without_judging_the_move() -> None:
    """The app put the game back to an earlier moment: nothing the game did."""
    k = knowledge_for({0: VariableKind.PROGRESS, 1: VariableKind.LIVES, 2: VariableKind.SCORE})
    d = EventDetector(k)
    assert d.observe(snap(0, 5, 3, 100)) == ()
    d.observe(snap(1, 6, 3, 110))  # the score's usual gain is ten
    (lost,) = d.observe(snap(2, 6, 2, 110))
    assert lost.kind is EventKind.SETBACK
    seen, moved = d.observed, d.moved
    # Back to before the life went: progress down, a life more, the score lower.
    assert d.rebase(snap(3, 5, 3, 100)) is None
    assert d.observed == seen and d.moved == moved
    assert d.observe(snap(4, 5, 3, 100)) == ()
    # What happens next is judged from the values it was put back to.
    again = d.observe(snap(5, 6, 2, 110))
    assert sorted((e.kind.value, e.before, e.after) for e in again) == [
        ("progress", 5, 6),
        ("setback", 3, 2),
    ]
    # The score's usual gain was kept: ten more is nothing, forty more is a jump.
    assert d.observe(snap(6, 6, 2, 120)) == ()
    assert [e.kind for e in d.observe(snap(7, 6, 2, 160))] == [EventKind.SCORE_GAIN]


def test_rebase_keeps_first_values_and_starts_watching_new_ones() -> None:
    k = knowledge_for({0: VariableKind.PROGRESS, 1: VariableKind.LIVES, 2: VariableKind.SCORE})
    d = EventDetector(k)
    d.observe(snap(0, 0, 3, 0))
    d.observe(snap(1, 4, 2, 50))
    # Put back to the very start: every value is at its first value again, and that is
    # not the game starting over.
    d.rebase(snap(2, 0, 3, 0))
    assert d.observe(snap(3, 0, 3, 0)) == ()
    # A real restart later is still measured against the same first values.
    d.observe(snap(4, 4, 2, 50))
    kinds = {e.kind for e in d.observe(snap(5, 0, 3, 0))}
    assert EventKind.RESET in kinds
    # A detector that had seen nothing yet starts watching at the jump.
    fresh = EventDetector(k)
    fresh.rebase(snap(0, 5, 3, 100))
    assert fresh.watching == ("v1", "v2", "v3") and fresh.observed == 0
    assert fresh.observe(snap(1, 5, 3, 100)) == ()
    (lost,) = fresh.observe(snap(2, 5, 2, 100))
    assert (lost.kind, lost.before, lost.after) == (EventKind.SETBACK, 3, 2)
    with pytest.raises(ConfigurationError):
        fresh.rebase("nope")  # type: ignore[arg-type]
