"""T-709: a longer look at progress-like values."""

from __future__ import annotations

import pytest

from retro_ai_lab.core import ConfigurationError
from retro_ai_lab.observation import MemorySnapshot
from retro_ai_lab.understanding import Evidence, Knowledge, LongLook, VariableKind
from retro_ai_lab.understanding.longlook import COUNTS_ROUNDS, SAME_PLACES, THIRD_VALUE

RAM = "ram"
SURE = [Evidence("test", "", 6.0)]
FLAG, POSITION, CHECKPOINT, LAP, SCATTERED, TOGGLER = range(6)


def knowledge() -> Knowledge:
    k = Knowledge("test")
    k.propose(RAM, FLAG, kind=VariableKind.FLAG, name="Flag", evidence=SURE)
    k.propose(RAM, POSITION, kind=VariableKind.POSITION, name="Position", evidence=SURE)
    k.propose(RAM, CHECKPOINT, kind=VariableKind.PROGRESS, name="Checkpoint", evidence=SURE)
    k.propose(RAM, LAP, kind=VariableKind.FLAG, name="Lap", evidence=SURE)
    k.propose(RAM, SCATTERED, kind=VariableKind.PROGRESS, name="Scattered", evidence=SURE)
    k.propose(RAM, TOGGLER, kind=VariableKind.FLAG, name="Toggler", evidence=SURE)
    return k


def frames(n: int) -> list[bytes]:
    """A track of 40 places with a checkpoint every 10; a lap per round."""
    rows = []
    scattered = 0
    for i in range(n):
        position = i % 40
        checkpoint = position // 10
        lap = i // 40
        flag = 0 if i < 50 else 1 if i < 100 else 2
        if i % 17 == 16:
            scattered += 1  # steps at ever different places
        toggler = (i // 7) % 2
        rows.append(bytes([flag, position, checkpoint, lap, scattered, toggler]))
    return rows


def look_at(k: Knowledge, rows: list[bytes]) -> LongLook:
    look = LongLook(k)
    for i, row in enumerate(rows):
        look.observe(MemorySnapshot(i, {RAM: row}))
    return look


def var(k: Knowledge, offset: int):
    v = k.at(RAM, offset)
    assert v is not None
    return v


def kinds(k: Knowledge, offset: int) -> set[str]:
    return {e.kind for e in var(k, offset).evidence}


def test_a_flag_that_reaches_a_third_value_counts() -> None:
    k = knowledge()
    look = look_at(k, frames(60))
    assert look.review() == ()  # only two values seen so far
    assert var(k, FLAG).kind is VariableKind.FLAG
    look_at(k, frames(120))
    look = look_at(k, frames(120))
    changed = look.review()
    assert var(k, FLAG).kind is VariableKind.PROGRESS and THIRD_VALUE in kinds(k, FLAG)
    assert var(k, FLAG) in changed
    assert var(k, TOGGLER).kind is VariableKind.FLAG and kinds(k, TOGGLER) == {"test"}
    assert look.review() == ()  # nothing new to say


def test_checkpoint_steps_at_the_same_places_of_the_position() -> None:
    k = knowledge()
    look = look_at(k, frames(200))
    look.review()
    assert SAME_PLACES in kinds(k, CHECKPOINT)
    assert any("Position" in e.detail for e in var(k, CHECKPOINT).evidence)
    assert COUNTS_ROUNDS not in kinds(k, CHECKPOINT)  # it drops when the position starts over
    assert SAME_PLACES not in kinds(k, SCATTERED) and COUNTS_ROUNDS not in kinds(k, SCATTERED)
    assert var(k, SCATTERED).kind is VariableKind.PROGRESS
    assert kinds(k, POSITION) == {"test"} and var(k, POSITION).kind is VariableKind.POSITION
    assert look.review() == ()
    # A checkpoint first taken for a position is set right by the same look.
    k = knowledge()
    k.update(var(k, CHECKPOINT).id, kind=VariableKind.POSITION)
    look_at(k, frames(200)).review()
    assert var(k, CHECKPOINT).kind is VariableKind.PROGRESS and SAME_PLACES in kinds(k, CHECKPOINT)


def test_lap_counts_the_rounds_and_stops_being_a_flag() -> None:
    k = knowledge()
    look = look_at(k, frames(200))  # five rounds
    look.review()
    lap = var(k, LAP)
    assert lap.kind is VariableKind.PROGRESS
    assert COUNTS_ROUNDS in kinds(k, LAP) and THIRD_VALUE in kinds(k, LAP)
    rounds = [e for e in lap.evidence if e.kind == COUNTS_ROUNDS]
    assert len(rounds) == 1 and "starts over" in rounds[0].detail
    assert look.review() == ()


def test_skips_rejected_low_confidence_and_uncovered_variables() -> None:
    k = knowledge()
    k.reject(var(k, LAP).id)
    k.propose("other", 0, kind=VariableKind.FLAG, name="Elsewhere", evidence=SURE)
    k.propose(RAM, 7, kind=VariableKind.FLAG, name="Beyond", evidence=SURE)  # past the rows
    maybe = k.propose(RAM, 8, kind=VariableKind.PROGRESS, name="Maybe")  # no evidence at all
    look = look_at(k, frames(200))
    assert var(k, LAP).id not in look.watching and maybe.id not in look.watching
    assert look.observed == 200
    look.review()
    assert var(k, LAP).kind is VariableKind.FLAG and kinds(k, LAP) == {"test", "user"}
    assert LongLook(Knowledge("empty")).review() == ()
    with pytest.raises(ConfigurationError):
        look.observe("not a snapshot")  # type: ignore[arg-type]
    look.reset()
    assert look.watching == () and look.observed == 0
