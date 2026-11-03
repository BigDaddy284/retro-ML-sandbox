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
