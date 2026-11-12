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
        rows.append(bytes([flag, position, checkpoint, lap, scattered, toggler, i % 4]))
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


def test_the_long_look_picks_up_knowledge_changes() -> None:
    k = knowledge()
    rows = frames(200)
    look = LongLook(k)
    for i, row in enumerate(rows[:100]):
        look.observe(MemorySnapshot(i, {RAM: row}))
    late = k.propose(RAM, 6, kind=VariableKind.PROGRESS, name="Late", evidence=SURE)
    k.reject(var(k, SCATTERED).id)
    for i, row in enumerate(rows[100:], start=100):
        look.observe(MemorySnapshot(i, {RAM: row}))
    assert late.id in look.watching
    assert look.observed == 200


# ------------------------------------------------------------ T-849


def test_values_that_cannot_be_checkpoint_counters_stop_being_paired() -> None:
    """A timer stepping every frame and a counter cycling 0..7 are dropped; the real
    checkpoint and lap keep their evidence."""
    from retro_ai_lab.understanding.longlook import FEW_VALUES, SETTLED, TOO_FAST

    k = knowledge()
    timer = k.propose(RAM, 7, kind=VariableKind.PROGRESS, name="Timer", evidence=SURE)
    cycler = k.propose(RAM, 8, kind=VariableKind.PROGRESS, name="Cycler", evidence=SURE)
    rows = [row + bytes([i & 0xFF, i % 8]) for i, row in enumerate(frames(200))]
    look = LongLook(k)
    for i, row in enumerate(rows):
        look.observe(MemorySnapshot(i, {RAM: row}))
        if i == FEW_VALUES:  # the timer has shown 17 faces by now
            assert timer.id not in look.tracking
        if i == SETTLED // 2:  # the cycler is not judged until it has settled
            assert cycler.id in look.tracking
    assert timer.id not in look.tracking and cycler.id not in look.tracking
    assert TOO_FAST == 8 and SETTLED == 64
    checkpoint, lap = var(k, CHECKPOINT), var(k, LAP)
    assert checkpoint.id in look.tracking and lap.id in look.tracking
    look.review()
    assert SAME_PLACES in kinds(k, CHECKPOINT) and COUNTS_ROUNDS in kinds(k, LAP)
    assert kinds(k, 7) == {"test"} and kinds(k, 8) == {"test"}


def test_a_pair_with_too_many_places_is_dropped() -> None:
    """A stepper that stepped at more driver values than can make FEW_PLACES places."""
    from retro_ai_lab.understanding.longlook import MAX_PLACE_VALUES

    assert MAX_PLACE_VALUES == 40
    k = Knowledge("test")
    driver = k.propose(RAM, 0, kind=VariableKind.POSITION, name="Pos", evidence=SURE)
    stepper = k.propose(RAM, 1, kind=VariableKind.PROGRESS, name="Count", evidence=SURE)
    look = LongLook(k)
    count = 0
    for i in range(1000):
        if i % 20 == 19:
            count = (count + 1) % 8  # slow enough, few faces, at a new place each time
        look.observe(MemorySnapshot(i, {RAM: bytes([(i * 7) % 251, count])}))
    assert stepper.id in look.tracking  # still a candidate against other drivers
    assert look._steppers[stepper.id].dead == {driver.id}
    assert look._steppers[stepper.id].places == {}
    look.review()
    assert kinds(k, 1) == {"test"}


def test_many_fast_values_are_all_unpaired_after_settling() -> None:
    """200 fast steppers and 200 drivers: the quadratic work ends once they settle."""
    from retro_ai_lab.understanding.longlook import SETTLED

    k = Knowledge("test")
    for i in range(200):
        k.propose(RAM, i, kind=VariableKind.PROGRESS, evidence=SURE)  # driver and stepper
        k.propose(RAM, 200 + i, kind=VariableKind.FLAG, evidence=SURE)  # stepper only
    look = LongLook(k)
    for i in range(SETTLED + 2):
        row = bytes([(i + j) % 8 for j in range(200)]) + bytes([(i + j) % 3 for j in range(200)])
        look.observe(MemorySnapshot(i, {RAM: row}))
    # The AI acts on 16 of each kind (T-850): all of them unpaired once settled.
    assert len(look.watching) == 32 and look.tracking == ()


def test_a_mover_is_neither_a_driver_nor_a_stepper() -> None:
    """T-911: something moving on its own is not a place progress happens at, nor progress."""
    from retro_ai_lab.understanding.knowledge import MOVER

    k = knowledge()
    mover = k.propose(
        RAM, 6, kind=VariableKind.POSITION, evidence=[Evidence(MOVER, "Moves both ways.", 6.0)]
    )
    look = look_at(k, frames(200))
    assert mover.id not in look._drivers and mover.id not in look._steppers
    assert var(k, POSITION).id in look._drivers and var(k, CHECKPOINT).id in look._steppers


# ---------------------------------------------------------- T-1004: a jump back


def test_rebase_counts_nothing_across_a_jump() -> None:
    """The app put the game back to an earlier moment: no step, no place, no starting over."""
    k = knowledge()
    rows = frames(36)  # most of the first round: position 35, checkpoint 3
    look, plain = LongLook(k), LongLook(k)
    for i, row in enumerate(rows):
        look.observe(MemorySnapshot(i, {RAM: row}))
        plain.observe(MemorySnapshot(i, {RAM: row}))

    def counted(ll: LongLook) -> dict[str, tuple[int, int, int, int]]:
        return {
            i: (s.ups, s.downs, s.seen, sum(len(at) for at in s.places.values()))
            for i, s in ll._steppers.items()
        }

    before = counted(look)
    rounds = {i: dict(s.rounds) for i, s in look._steppers.items()}
    seen = look.observed
    # Back to frame 4 of the same round: the position drops by most of its range and
    # the checkpoint by three, as at a lap's end.
    look.rebase(MemorySnapshot(36, {RAM: rows[4]}))
    assert counted(look) == before and look.observed == seen
    assert {i: dict(s.rounds) for i, s in look._steppers.items()} == rounds
    assert {i: s.last for i, s in look._steppers.items()} == {
        i: s.last for i, s in _after(k, rows[:5]).items()
    }
    # The one fed the same jump as a snapshot counted it as the game's own doing.
    plain.observe(MemorySnapshot(36, {RAM: rows[4]}))
    assert counted(plain) != before
    # The next real step is counted from where it was put back to.
    for i, row in enumerate(rows[5:12], start=37):
        look.observe(MemorySnapshot(i, {RAM: row}))
    checkpoint = k.at(RAM, CHECKPOINT)
    assert checkpoint is not None
    ups, downs, _, _ = counted(look)[checkpoint.id]
    assert (ups, downs) == (before[checkpoint.id][0] + 1, before[checkpoint.id][1])
    with pytest.raises(ConfigurationError):
        look.rebase("nope")  # type: ignore[arg-type]


def _after(k: Knowledge, rows: list[bytes]):  # type: ignore[no-untyped-def]
    ll = LongLook(k)
    for i, row in enumerate(rows):
        ll.observe(MemorySnapshot(i, {RAM: row}))
    return ll._steppers
