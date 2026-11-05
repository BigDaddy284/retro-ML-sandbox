"""T-302: variable candidates from change statistics."""

from __future__ import annotations

from collections.abc import Iterable

import pytest

from retro_ai_lab.core import ConfigurationError
from retro_ai_lab.emulator.fake import ToyRacer
from retro_ai_lab.emulator.fake.toy_racer import LAYOUT, RAM_SIZE
from retro_ai_lab.observation import ChangeTracker, MemorySnapshot
from retro_ai_lab.understanding import (
    Candidate,
    Confidence,
    Encoding,
    Knowledge,
    VariableKind,
    find_candidates,
    propose_candidates,
)
from retro_ai_lab.understanding.variables import (
    BOUNCE_UPS,
    LIVES_SAMPLES,
    MIN_TRANSITIONS,
    MOVER,
    flag_doubt,
    lives_bounce,
    lives_busy,
    lives_doubt,
    progress_doubt,
)


def track(rows: Iterable[bytes]) -> ChangeTracker:
    t = ChangeTracker()
    for i, row in enumerate(rows):
        t.observe(MemorySnapshot(i, {"ram": row}))
    return t


def by_offset(cands: tuple[Candidate, ...]) -> dict[int, Candidate]:
    return {c.offset: c for c in cands}


# ------------------------------------------------------ synthetic bytes


def test_nothing_before_enough_transitions() -> None:
    t = track(bytes([i]) for i in range(10))
    assert find_candidates(t) == ()
    assert find_candidates(t, min_transitions=5) != ()
    with pytest.raises(ConfigurationError):
        find_candidates(t, min_transitions=0)


def test_counter_constant_and_noise() -> None:
    import random

    rng = random.Random(1)
    rows = [bytes([i & 0xFF, 7, rng.randrange(256)]) for i in range(300)]
    cands = by_offset(find_candidates(track(rows)))
    assert set(cands) == {0}
    c = cands[0]
    assert c.kind is VariableKind.COUNTER and c.encoding is Encoding.U8
    assert {e.kind for e in c.evidence} == {"monotonic", "every-frame", "unit-steps"}
    assert c.weight == 4.0 and c.width == 1 and c.covers(0) and not c.covers(1)


def test_16bit_counter_pair_and_score_pair() -> None:
    rows = []
    score = 0
    for i in range(600):
        if i % 7 == 0:
            score += 10
        elif i % 3 == 0:
            score += 1
        rows.append(bytes([i & 0xFF, i >> 8, score & 0xFF, score >> 8]))
    cands = by_offset(find_candidates(track(rows)))
    assert set(cands) == {0, 2}
    assert cands[0].encoding is Encoding.U16LE and cands[0].kind is VariableKind.COUNTER
    assert cands[2].encoding is Encoding.U16LE and cands[2].kind is VariableKind.SCORE
    assert any(e.kind == "16-bit" for e in cands[2].evidence)


def test_lives_flag_progress_speed_and_unknown() -> None:
    rows = []
    for i in range(400):
        lives = 3 - (i // 150)  # 3, 2, 1
        flag = 1 if i >= 380 else 0
        level = i // 100  # 0..3
        speed = (i // 3) % 4  # 0..3 up and down, changes often
        junk = (i * 37) % 23 if i % 4 == 0 else 5  # many values, both ways, sometimes
        rows.append(bytes([lives, flag, level, speed, junk]))
    cands = by_offset(find_candidates(track(rows)))
    assert cands[0].kind is VariableKind.LIVES
    assert cands[1].kind is VariableKind.FLAG
    assert cands[2].kind is VariableKind.PROGRESS
    assert cands[3].kind is VariableKind.SPEED
    assert cands[4].kind is VariableKind.UNKNOWN


def test_position_like_byte() -> None:
    rows = []
    pos = 0
    for i in range(400):
        pos = (pos + (i % 5)) & 0xFF  # climbs by 0..4 and rolls over
        rows.append(bytes([pos]))
    (c,) = find_candidates(track(rows))
    assert c.kind is VariableKind.POSITION and c.encoding is Encoding.U8


def test_candidate_order_and_pair_takes_precedence_over_singles() -> None:
    rows = [bytes([i & 0xFF, i >> 8, 1]) for i in range(600)]
    cands = find_candidates(track(rows))
    assert [c.offset for c in cands] == [0]  # byte 2 constant, byte 1 absorbed into the pair


# ------------------------------------------------------------ proposing


def test_propose_adds_unknown_locations_only() -> None:
    k = Knowledge("g")
    k.propose("ram", 1, name="Known")
    rejected = k.propose("ram", 5)
    k.reject(rejected.id)
    cands = (
        Candidate("ram", 0, Encoding.U16LE, VariableKind.SCORE, ()),  # overlaps byte 1
        Candidate("ram", 3, Encoding.U8, VariableKind.LIVES, ()),
        Candidate("ram", 5, Encoding.U8, VariableKind.FLAG, ()),  # rejected: leave alone
        Candidate("sram", 1, Encoding.U8, VariableKind.FLAG, ()),  # other region
    )
    added = propose_candidates(k, cands)
    assert [(v.region, v.offset, v.kind) for v in added] == [
        ("ram", 3, VariableKind.LIVES),
        ("sram", 1, VariableKind.FLAG),
    ]
    assert propose_candidates(k, cands) == ()  # idempotent


# ------------------------------------------------- on the real toy game


def drive(frames: int, seed: int = 2) -> ChangeTracker:
    game = ToyRacer(seed=seed)
    t = ChangeTracker()
    t.observe(MemorySnapshot(0, {"ram": game.ram}))
    for i in range(1, frames):
        # Mostly accelerate, weave a little, never brake; crashes are welcome.
        pressed: list[str] = ["up"]
        if i % 90 in range(20, 24):
            pressed.append("left")
        if i % 130 in range(40, 44):
            pressed.append("right")
        if game.state.game_over:
            pressed = ["start"]
        game.step(pressed)
        t.observe(MemorySnapshot(i, {"ram": game.ram}))
    return t


def test_finds_the_toy_racers_variables_without_knowing_the_layout() -> None:
    t = drive(900)
    cands = by_offset(find_candidates(t))
    assert cands[LAYOUT.frame].kind is VariableKind.COUNTER
    assert cands[LAYOUT.frame].encoding is Encoding.U16LE
    assert cands[LAYOUT.score].kind is VariableKind.SCORE
    assert cands[LAYOUT.score].encoding is Encoding.U16LE
    assert cands[LAYOUT.timer].kind is VariableKind.COUNTER
    assert cands[LAYOUT.position].kind is VariableKind.POSITION
    assert cands[LAYOUT.speed].kind is VariableKind.SPEED
    assert cands[LAYOUT.checkpoint].kind is VariableKind.PROGRESS
    assert cands[LAYOUT.lap].kind is VariableKind.PROGRESS
    assert cands[LAYOUT.lives].kind in (VariableKind.LIVES, VariableKind.PROGRESS)
    for off in range(LAYOUT.noise, LAYOUT.noise + 4):
        assert off not in cands, "noise proposed as a variable"
    for off in range(LAYOUT.header, RAM_SIZE):
        assert off not in cands, "constant proposed as a variable"


def test_proposals_land_in_knowledge_at_modest_confidence() -> None:
    k = Knowledge("toyracer-2")
    added = propose_candidates(k, find_candidates(drive(900)))
    assert len(added) >= 6 and len(k) == len(added)
    for v in k:
        assert v.confidence in (Confidence.LOW, Confidence.MEDIUM), v
        assert v.evidence, "every proposal explains itself"
    assert k.best(VariableKind.SCORE) is not None
    assert k.best(VariableKind.SCORE).offset == LAYOUT.score  # type: ignore[union-attr]
    assert MIN_TRANSITIONS <= 900


def test_a_life_count_restored_by_restarts_does_not_bounce() -> None:
    """T-864: lives are lost one at a time and restored on a restart; a bouncer is not lives."""
    from retro_ai_lab.observation.diff import ByteStats

    assert BOUNCE_UPS == 4
    assert not lives_bounce(ByteStats(samples=600, changes=20, ups=0, downs=20))
    assert not lives_bounce(ByteStats(samples=600, changes=5, ups=3, downs=2))  # too few ups
    assert not lives_bounce(ByteStats(samples=600, changes=26, ups=6, downs=20))  # Toy Racer
    assert not lives_bounce(ByteStats(samples=600, changes=12, ups=4, downs=8))  # two lives
    assert lives_bounce(ByteStats(samples=600, changes=11, ups=4, downs=7))
    assert lives_bounce(ByteStats(samples=600, changes=217, ups=101, downs=116))  # the log


def test_a_life_count_changes_rarely() -> None:
    """T-865: a byte that drops nearly every step is a sub-counter, not a life count."""
    from retro_ai_lab.observation.diff import ByteStats

    assert LIVES_SAMPLES == 100
    assert not lives_busy(ByteStats(samples=60, changes=50, ups=10, downs=40))  # too early to say
    assert not lives_busy(ByteStats(samples=600, changes=26, ups=6, downs=20))  # Toy Racer
    assert lives_busy(ByteStats(samples=600, changes=496, ups=99, downs=397))  # the log
    assert lives_doubt(ByteStats(samples=600, changes=26, ups=6, downs=20)) is None
    assert lives_doubt(ByteStats(samples=600, changes=11, ups=4, downs=7)) == "bounces"
    assert lives_doubt(ByteStats(samples=600, changes=496, ups=99, downs=397)) == "busy"
    assert lives_doubt(ByteStats(samples=600, changes=217, ups=101, downs=116)) == "busy"
    # The classifier no longer proposes lives for a byte that ticks down 5, 4, 3, 2, 1, 5 ...
    rows = [bytes([5 - (i % 5), 3 - (i // 150)]) for i in range(450)]
    cands = by_offset(find_candidates(track(rows)))
    assert cands[1].kind is VariableKind.LIVES
    assert 0 not in cands or cands[0].kind is not VariableKind.LIVES


def test_a_flag_switches_rarely() -> None:
    """T-869: a two-value byte toggling every few steps is an animation bit, not a flag."""
    from retro_ai_lab.observation.diff import ByteStats

    assert flag_doubt(ByteStats(samples=600, changes=140, ups=70, downs=70)) == "busy"  # the log
    assert flag_doubt(ByteStats(samples=600, changes=4, ups=2, downs=2)) is None  # game over
    assert flag_doubt(ByteStats(samples=60, changes=20, ups=10, downs=10)) is None  # too early
    # The classifier proposes nothing for a bit that blinks every eight steps, whichever way
    # it leans, and still a flag for one that switches twice in 400 steps.
    rows = [
        bytes([(i // 8) % 2, 1 if i % 16 == 0 else 0, 1 if 200 <= i < 380 else 0])
        for i in range(400)
    ]
    cands = by_offset(find_candidates(track(rows)))
    assert 0 not in cands and 1 not in cands
    assert cands[2].kind is VariableKind.FLAG


def test_progress_changes_rarely_too() -> None:
    """T-866: a level, lap or checkpoint steps up rarely and only comes down on a restart."""
    from retro_ai_lab.observation.diff import ByteStats

    assert progress_doubt(ByteStats(samples=600, changes=36, ups=27, downs=9)) is None  # checkpoint
    assert progress_doubt(ByteStats(samples=600, changes=7, ups=4, downs=3)) is None  # lap
    assert (
        progress_doubt(ByteStats(samples=600, changes=124, ups=117, downs=7)) == "busy"
    )  # the log
    assert progress_doubt(ByteStats(samples=600, changes=12, ups=5, downs=7)) == "bounces"
    assert progress_doubt(ByteStats(samples=60, changes=50, ups=49, downs=1)) is None  # too early


# ------------------------------------------------------ T-911: things that move


def enemy_rows(n: int = 400, *, hop: int = 3, every: int = 1) -> list[bytes]:
    """A byte like an enemy's x: drifts both ways by up to ``hop`` every ``every`` frames."""
    import random

    rng = random.Random(911)
    rows = []
    x = 120
    for i in range(n):
        if i % every == 0:
            x = max(0, min(255, x + rng.randint(-hop, hop)))
        rows.append(bytes([x]))
    return rows


def test_something_moving_both_ways_in_small_steps_is_a_position() -> None:
    (c,) = find_candidates(track(enemy_rows()))
    assert c.kind is VariableKind.POSITION and c.encoding is Encoding.U8
    assert any(e.kind == MOVER for e in c.evidence)
    assert "moves" in c.evidence[0].detail.lower()


def test_a_mover_that_jumps_or_hardly_moves_is_not_a_position() -> None:
    (jumpy,) = find_candidates(track(enemy_rows(hop=40, every=2)))  # every frame would be noise
    assert jumpy.kind is VariableKind.UNKNOWN
    (slow,) = find_candidates(track(enemy_rows(every=20)))  # a rare stepper, as before T-911
    assert slow.kind is not VariableKind.POSITION


def test_a_mover_rolling_past_zero_is_still_a_position() -> None:
    rows = []
    x = 2
    for i in range(400):
        x = (x + (1 if (i // 50) % 2 == 0 else -1) * (i % 3)) & 0xFF  # drifts over the 0/255 edge
        rows.append(bytes([x]))
    (c,) = find_candidates(track(rows))
    assert c.kind is VariableKind.POSITION


def test_movers_do_not_steal_the_other_kinds() -> None:
    """The junk, speed, lives, flag and progress bytes of the kinds test are unchanged."""
    rows = []
    for i in range(400):
        lives = 3 - (i // 150)
        flag = 1 if i >= 380 else 0
        level = i // 100
        speed = (i // 3) % 4
        junk = (i * 37) % 23 if i % 4 == 0 else 5
        rows.append(bytes([lives, flag, level, speed, junk]))
    cands = by_offset(find_candidates(track(rows)))
    assert [cands[i].kind for i in range(5)] == [
        VariableKind.LIVES,
        VariableKind.FLAG,
        VariableKind.PROGRESS,
        VariableKind.SPEED,
        VariableKind.UNKNOWN,
    ]


# ------------------------------------------------------------ T-914: smooth movers


def goomba_rows(n: int = 400) -> list[bytes]:
    """An enemy's x on screen: walks left 2 or 3 every step and respawns at the right."""
    rows = []
    x = 200
    for i in range(n):
        x -= 2 + (i % 2)
        if x < 8:
            x = 240  # a new enemy takes the slot: one big hop
        rows.append(bytes([x]))
    return rows


def test_a_thing_moving_every_step_in_small_hops_is_not_noise() -> None:
    """The user's enemies changed every step, which the noise rule threw away first."""
    (c,) = find_candidates(track(goomba_rows()))
    assert c.kind is VariableKind.POSITION and any(e.kind == MOVER for e in c.evidence)


def test_random_every_step_is_still_noise_and_a_countdown_is_a_counter() -> None:
    import random

    rng = random.Random(914)
    rows = [bytes([rng.randrange(256), (255 - i) & 0xFF]) for i in range(400)]
    cands = by_offset(find_candidates(track(rows)))
    assert 0 not in cands  # random every frame: noise, as before
    assert cands[1].kind is VariableKind.COUNTER  # down by one every frame: a timer, not a mover


def test_a_slow_steady_countdown_is_not_a_mover_either() -> None:
    rows = [bytes([(200 - i // 6) & 0xFF]) for i in range(400)]  # one down every six steps
    (c,) = find_candidates(track(rows))
    assert c.kind is not VariableKind.POSITION
