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
    Evidence,
    Knowledge,
    Variable,
    VariableKind,
    find_candidates,
    propose_candidates,
    rethink_candidates,
)
from retro_ai_lab.understanding.knowledge import INPUT_LINKED, is_mover
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


def test_a_value_climbing_by_the_same_amount_every_step_is_a_counter() -> None:
    """T-921: the user's frame counter, read every four frames, sat in a position seat."""
    rows = [bytes([(i * 4) & 0xFF]) for i in range(400)]
    (c,) = find_candidates(track(rows))
    assert c.kind is VariableKind.COUNTER
    assert any(e.detail == "Always by exactly 4." for e in c.evidence)
    # a score that only ever adds ten, now and then, is still a score
    score = []
    total = 0
    for i in range(400):
        if i % 15 == 0:
            total += 10
        score.append(bytes([total & 0xFF]))
    (s,) = find_candidates(track(score))
    assert s.kind is VariableKind.SCORE


def test_a_counter_that_jumps_now_and_then_is_still_a_counter() -> None:
    """T-924: the user's frame counter, reset by the game now and then, kept its seat."""
    rows = [bytes([((i % 300) * 4) & 0xFF]) for i in range(1200)]  # 4, 8, ... then back to 0
    (c,) = find_candidates(track(rows))
    assert c.kind is VariableKind.COUNTER
    assert any(e.detail == "Nearly always by 4." for e in c.evidence)
    steady = [bytes([(i * 4) & 0xFF]) for i in range(400)]
    (s,) = find_candidates(track(steady))
    assert any(e.detail == "Always by exactly 4." for e in s.evidence)
    # a position that climbs in varied steps and is thrown back now and then is no counter
    rows = []
    pos = 0
    for i in range(1200):
        pos = 0 if i % 300 == 0 else (pos + (i % 5)) & 0xFF
        rows.append(bytes([pos]))
    (p,) = find_candidates(track(rows))
    assert p.kind is VariableKind.POSITION


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


def _planted(k: Knowledge, offset: int, kind: VariableKind, *evidence: Evidence) -> Variable:
    return k.propose("ram", offset, kind=kind, evidence=evidence)


def test_the_ai_changes_its_mind_about_its_own_guess() -> None:
    """T-916: a guess the AI made itself takes the classifier's new kind and evidence."""
    k = Knowledge("g")
    early = _planted(
        k,
        3,
        VariableKind.SPEED,
        Evidence("few-values", "Only 5 different values seen.", 1.0),
        Evidence("played", "Moved 40 times in a go of 100 steps.", 0.5),
        Evidence("busy", "Still changed 40 times in 100 steps.", 0.0),
    )
    same = _planted(k, 7, VariableKind.LIVES, Evidence("rare", "Changes rarely.", 0.5))
    fresh = (
        Evidence(MOVER, "Moves in small steps, often.", 1.5),
        Evidence("many-values", "40 different values seen.", 0.5),
    )
    cands = (
        Candidate("ram", 3, Encoding.U8, VariableKind.POSITION, fresh),
        Candidate(  # the same kind for the same reasons: nothing to do (T-922 reads the reasons)
            "ram", 7, Encoding.U8, VariableKind.LIVES, (Evidence("rare", "Changes rarely.", 0.5),)
        ),
        Candidate("ram", 9, Encoding.U8, VariableKind.FLAG, ()),  # nobody's guess yet
    )
    (changed,) = rethink_candidates(k, cands)
    assert changed.id == early.id and changed.kind is VariableKind.POSITION
    assert [e.kind for e in changed.evidence] == [MOVER, "many-values", "played"]
    assert not changed.rejected and k.get(same.id) == same and k.at("ram", 9) is None
    assert rethink_candidates(k, cands) == ()  # idempotent
    # a 16-bit guess is not the same place as an 8-bit candidate (a change of width is
    # out of scope)
    wide = k.propose("ram", 20, encoding=Encoding.U16LE, kind=VariableKind.SCORE)
    narrow = Candidate("ram", 20, Encoding.U8, VariableKind.LIVES, ())
    assert rethink_candidates(k, (narrow,)) == () and k.get(wide.id) == wide


def test_a_persons_word_and_the_links_verdict_are_never_rethought() -> None:
    k = Knowledge("g")
    sworn = _planted(k, 1, VariableKind.SPEED)
    k.confirm(sworn.id)
    named = k.propose("ram", 2, kind=VariableKind.SPEED, name="Hearts")
    refused = _planted(k, 3, VariableKind.SPEED)
    k.reject(refused.id)
    told = _planted(k, 4, VariableKind.SPEED, Evidence("told", "You said you lost a life.", 1.0))
    found = _planted(k, 5, VariableKind.SPEED, Evidence("restarts", "Went down by one.", 9.0))
    mine = _planted(k, 6, VariableKind.POSITION, Evidence(INPUT_LINKED, "Changes when Left.", 1.0))
    held = (sworn, named, refused, told, found, mine)
    cands = tuple(Candidate("ram", v.offset, Encoding.U8, VariableKind.LIVES, ()) for v in held)
    assert rethink_candidates(k, cands) == ()
    assert [k.get(v.id).kind for v in held] == [v.kind for v in held]
    # an input-linked guess of another kind is the classifier's own and may change
    linked = _planted(k, 8, VariableKind.SPEED, Evidence(INPUT_LINKED, "Changes when Left.", 1.0))
    (changed,) = rethink_candidates(
        k, (Candidate("ram", 8, Encoding.U8, VariableKind.POSITION, fresh_mover()),)
    )
    assert changed.id == linked.id and changed.kind is VariableKind.POSITION
    assert [e.kind for e in changed.evidence] == [MOVER, "many-values", INPUT_LINKED]


def fresh_mover() -> tuple[Evidence, ...]:
    return (
        Evidence(MOVER, "Moves in small steps, often.", 1.5),
        Evidence("many-values", "40 different values seen.", 0.5),
    )


def test_a_guess_the_ai_dropped_comes_back_as_another_kind() -> None:
    k = Knowledge("g")
    dropped = _planted(k, 3, VariableKind.LIVES, Evidence("busy", "Still busy.", 0.0))
    k.reject(dropped.id, by="the AI")
    again = Candidate("ram", 3, Encoding.U8, VariableKind.LIVES, ())
    assert rethink_candidates(k, (again,)) == () and k.get(dropped.id).rejected
    (back,) = rethink_candidates(
        k, (Candidate("ram", 3, Encoding.U8, VariableKind.POSITION, fresh_mover()),)
    )
    assert back.id == dropped.id and not back.rejected and back.kind is VariableKind.POSITION
    assert [e.kind for e in back.evidence] == [MOVER, "many-values"]  # the doubts are gone
    assert propose_candidates(k, (again,)) == ()  # still the same place, never duplicated


def test_a_guess_is_rethought_once_the_byte_has_changed_twice_as_often() -> None:
    """T-917: a byte on the edge of a rule must not flip kind at every review."""
    k = Knowledge("g")
    v = k.propose("ram", 3, kind=VariableKind.PROGRESS, judged=20)
    unknown = Candidate("ram", 3, Encoding.U8, VariableKind.UNKNOWN, (), changes=39)
    assert rethink_candidates(k, (unknown,)) == () and k.get(v.id).kind is VariableKind.PROGRESS
    enough = Candidate("ram", 3, Encoding.U8, VariableKind.UNKNOWN, (), changes=40)
    (changed,) = rethink_candidates(k, (enough,))
    assert changed.kind is VariableKind.UNKNOWN and changed.judged == 40
    back = Candidate("ram", 3, Encoding.U8, VariableKind.PROGRESS, (), changes=41)
    assert rethink_candidates(k, (back,)) == ()  # not until 80 changes
    # a planted guess, or one from a file before T-917, has no record and is rethought at once
    planted = k.propose("ram", 5, kind=VariableKind.SPEED)
    (changed,) = rethink_candidates(
        k, (Candidate("ram", 5, Encoding.U8, VariableKind.LIVES, (), changes=3),)
    )
    assert changed.id == planted.id and changed.kind is VariableKind.LIVES and changed.judged == 3


def test_candidates_and_proposals_carry_the_changes_judged_on() -> None:
    rows = enemy_rows()
    (c,) = find_candidates(track(rows[:40]))
    assert c.changes == track(rows[:40]).stats("ram", 0).changes > 0
    k = Knowledge("g")
    (v,) = propose_candidates(k, (c,))
    assert v.judged == c.changes
    wide = find_candidates(drive(900))
    pair = next(c for c in wide if c.encoding is Encoding.U16LE)
    assert pair.changes == drive(900).stats("ram", pair.offset).changes  # the lower byte's


def test_a_guess_is_rethought_when_its_evidence_changes() -> None:
    """T-922: the user's enemy sprites were positions by the climbing rule for good."""
    k = Knowledge("g")
    climbing = _planted(
        k,
        3,
        VariableKind.POSITION,
        Evidence("monotonic", "Keeps climbing and rolls over.", 1.0),
        Evidence("varied-steps", "Moves by small, varying amounts.", 1.0),
        Evidence("played", "Moved 40 times in a go of 100 steps.", 0.5),
    )
    k.update(climbing.id, judged=10)
    same = Candidate(
        "ram",
        3,
        Encoding.U8,
        VariableKind.POSITION,
        (
            Evidence("monotonic", "Keeps climbing and rolls over.", 1.0),
            Evidence("varied-steps", "Moves by small, varying amounts.", 1.0),
            Evidence("frequent", "Changes often.", 0.5),
        ),
        changes=30,
    )
    (changed,) = rethink_candidates(k, (same,))  # "frequent" is a new ground
    assert changed.kind is VariableKind.POSITION and changed.judged == 30
    assert [e.kind for e in changed.evidence] == ["monotonic", "varied-steps", "frequent", "played"]
    again = Candidate("ram", 3, Encoding.U8, VariableKind.POSITION, same.evidence[::-1], 70)
    assert rethink_candidates(k, (again,)) == ()  # the same grounds in another order
    mover = Candidate("ram", 3, Encoding.U8, VariableKind.POSITION, fresh_mover(), changes=59)
    assert rethink_candidates(k, (mover,)) == ()  # not until twice the changes (T-917)
    mover = Candidate("ram", 3, Encoding.U8, VariableKind.POSITION, fresh_mover(), changes=60)
    (moving,) = rethink_candidates(k, (mover,))
    assert moving.kind is VariableKind.POSITION and is_mover(moving)
    assert [e.kind for e in moving.evidence] == [MOVER, "many-values", "played"]
    # the real thing: a climbing look first, the whole walk later
    rows = enemy_rows()
    early = [bytes([(120 + i) & 0xFF]) for i in range(35)] + rows[:5]  # climbs, then wanders
    k2 = Knowledge("g")
    (first,) = propose_candidates(k2, find_candidates(track(early)))
    assert first.kind is VariableKind.POSITION and not is_mover(first)
    (later,) = rethink_candidates(k2, find_candidates(track(early + rows)))
    assert later.id == first.id and is_mover(later)


def test_a_walking_enemy_judged_early_becomes_a_mover_later() -> None:
    """The user's log: an enemy slot called "a speed" on its first values stayed so."""
    rows = enemy_rows()
    k = Knowledge("g")
    (early,) = propose_candidates(k, find_candidates(track(rows[:40])))
    assert early.kind is VariableKind.SPEED
    later = find_candidates(track(rows))
    assert propose_candidates(k, later) == ()  # the old rule: judged once
    (rethought,) = rethink_candidates(k, later)
    assert rethought.id == early.id and rethought.kind is VariableKind.POSITION
    assert any(e.kind == MOVER for e in rethought.evidence)


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


def visitor_rows(
    n: int = 2400, *, every: int = 200, alive: int = 12, span: int = 30
) -> list[bytes]:
    """An enemy slot filled now and then: spawns, walks left 2 or 3 a step, is cleared."""
    import random

    rng = random.Random(918)
    rows = []
    x = 0
    for i in range(n):
        phase = i % every
        if phase == 0:
            x = 200 + rng.randint(0, span)  # a new thing takes the slot: one big hop
        elif phase < alive:
            x -= 2 + (phase % 2)
        elif phase == alive:
            x = 0  # the slot is cleared: one big hop back
        rows.append(bytes([x & 0xFF]))
    return rows


def test_a_thing_on_screen_only_now_and_then_is_a_mover() -> None:
    """T-918: the user's enemies are on screen for seconds of each go, so rare overall."""
    for alive in (12, 30, 60):
        (c,) = find_candidates(track(visitor_rows(alive=alive)))
        st = track(visitor_rows(alive=alive)).stats("ram", 0)
        assert c.kind is VariableKind.POSITION and any(e.kind == MOVER for e in c.evidence), (
            alive,
            st.change_rate,
        )
    assert track(visitor_rows()).stats("ram", 0).change_rate < 0.1  # rare, as on the user's game


def test_a_brief_visitor_with_few_values_is_not_a_position() -> None:
    few = visitor_rows(alive=4, span=0)  # four steps from 200 each visit: ten values at most
    (c,) = find_candidates(track(few))
    assert c.kind is not VariableKind.POSITION and len(track(few).stats("ram", 0).distinct) <= 16
    (slow,) = find_candidates(track(enemy_rows(every=20)))  # a rare stepper, as before T-911
    assert slow.kind is not VariableKind.POSITION


def test_a_slow_steady_countdown_is_not_a_mover_either() -> None:
    rows = [bytes([(200 - i // 6) & 0xFF]) for i in range(400)]  # one down every six steps
    (c,) = find_candidates(track(rows))
    assert c.kind is not VariableKind.POSITION
