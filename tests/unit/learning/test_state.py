"""T-501: discrete states from believed variables."""

from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from retro_ai_lab.core import ConfigurationError
from retro_ai_lab.emulator.fake import ToyRacer
from retro_ai_lab.emulator.fake.toy_racer import LAYOUT
from retro_ai_lab.learning import State, StateEncoder, bucket
from retro_ai_lab.learning.features import DEFAULT_WIDTH
from retro_ai_lab.learning.state import (
    DEFAULT_MOST,
    FINE_KEY,
    FINE_STEP,
    KIND_RANK,
    MOST_CLOSER,
    MOST_DISTANCES,
    bucket_low,
    bucket_span,
    bucket_word,
    coarse_bucket,
    distance_id,
    fine_bucket,
    is_fine,
    split_distance,
)
from retro_ai_lab.observation import MemorySnapshot
from retro_ai_lab.understanding import (
    Confidence,
    Encoding,
    Evidence,
    Knowledge,
    Variable,
    VariableKind,
)

RAM = "ram"
SURE = [Evidence("test", "", 6.0)]
MAYBE = [Evidence("test", "", 1.0)]


def snap(ram: bytes, frame: int = 0) -> MemorySnapshot:
    return MemorySnapshot(frame, {RAM: ram})


# ------------------------------------------------------------ binning


def test_bucket_rule() -> None:
    assert [bucket(v, VariableKind.POSITION) for v in range(8)] == list(range(8))
    assert bucket(8, VariableKind.POSITION) == 8 and bucket(15, VariableKind.POSITION) == 8
    assert bucket(16, VariableKind.POSITION) == 9 and bucket(31, VariableKind.POSITION) == 9
    assert bucket(255, VariableKind.POSITION) == 12
    assert bucket(65535, VariableKind.SCORE) == 20
    assert bucket(-1, VariableKind.SPEED) == -2 and bucket(-20, VariableKind.SPEED) == -10
    assert bucket(0, VariableKind.FLAG) == 0 and bucket(200, VariableKind.FLAG) == 1
    assert bucket(5, VariableKind.LIVES, bins=4) == 4 and bucket(3, VariableKind.LIVES, bins=4) == 3


@given(value=st.integers(min_value=0, max_value=65535), bins=st.integers(min_value=2, max_value=64))
def test_bucket_is_monotonic_and_small(value: int, bins: int) -> None:
    b = bucket(value, VariableKind.SCORE, bins)
    assert 0 <= b <= bins + 16
    assert bucket(value + 1, VariableKind.SCORE, bins) >= b
    if value < bins:
        assert b == value


def test_fine_buckets_split_a_coarse_bucket_into_steps() -> None:
    """T-979: the user's AI died at one stretch of ground in its 128-wide top bucket."""
    assert FINE_STEP == 8 and FINE_KEY == 1 << 32
    pos = VariableKind.POSITION
    fine = [fine_bucket(v, pos) for v in range(128, 256)]
    assert len(set(fine)) == 16 and fine[0] == 12 * FINE_KEY and fine[-1] == 12 * FINE_KEY + 15
    assert fine == sorted(fine) and all(is_fine(b) for b in fine)
    assert fine_bucket(-130, pos) == -1 - fine_bucket(130, pos) and is_fine(fine_bucket(-130, pos))
    assert not is_fine(bucket(255, pos)) and not is_fine(bucket(-255, pos))
    assert bucket_word(fine_bucket(130, pos)) == "12 (128..135)"
    assert bucket_word(fine_bucket(255, pos)) == "12 (248..255)"
    assert bucket_word(fine_bucket(-130, pos)) == "-13 (-135..-128)"
    assert bucket_word(12) == "12" and bucket_word(-13) == "-13"
    assert bucket_span(12) == 128 and bucket_span(-13) == 128 and bucket_span(8) == 8
    assert bucket_span(7) == 1 and bucket_span(-8) == 1 and bucket_span(9) == 16
    assert bucket_span(5, bins=4) == 8  # bins 4: bucket 5 holds 8..15
    assert bucket_low(12) == 128 and bucket_low(7) == 7 and bucket_low(5, bins=4) == 8
    assert bucket_word(fine_bucket(65535, VariableKind.SCORE)) == "20 (65528..65535)"
    with pytest.raises(ConfigurationError):
        bucket_low(-13)
    # A fine bucket knows the coarse one it came from, and a state can go coarse (T-982).
    assert coarse_bucket(fine_bucket(130, pos)) == 12 and coarse_bucket(12) == 12
    assert coarse_bucket(fine_bucket(-130, pos)) == -13 and coarse_bucket(-13) == -13
    fine = State((("v1", fine_bucket(130, pos)), ("v2", 3), ("v3", fine_bucket(-130, pos))))
    assert fine.is_fine and not fine.coarse().is_fine
    assert fine.coarse() == State((("v1", 12), ("v2", 3), ("v3", -13)))
    with pytest.raises(ConfigurationError):
        bucket_span(fine_bucket(130, pos))
    # The fine bucket's range is the coarse bucket's: every value stays in its bucket.
    for v in range(0, 65536, 7):
        assert fine_bucket(v, pos) // FINE_KEY == bucket(v, pos)


def test_the_encoder_looks_closer_where_told() -> None:
    k, me, it, score = moving_world()
    enc = StateEncoder(k)
    apart = distance_id(it.id, me.id)
    pos = VariableKind.POSITION

    def parts(me_at: int, it_at: int) -> dict[str, int]:
        return dict(enc.encode(snap(bytes([0, me_at, it_at, 9]))).parts)

    before = parts(130, 20)
    assert before[me.id] == 12 and before[apart] == bucket(-110, pos) == -12
    assert enc.closer == () and "looking closer" not in enc.describe()
    # The death watch names my position in bucket 12 and the distance in its bucket;
    # a fine sight, an exact one and bucket 8 (already 8 wide) are left alone.
    taken = enc.look_closer(
        [(me.id, 12), (me.id, fine_bucket(130, pos)), (score.id, 3), (score.id, 8), (apart, -12)]
    )
    assert taken == ((me.id, 12), (apart, -12))
    assert enc.closer == ((me.id, 12), (apart, -12))
    assert enc.describe().endswith("; looking closer at 2 sights where it keeps dying")
    after = parts(130, 20)
    assert after[me.id] == fine_bucket(130, pos) and after[apart] == fine_bucket(-110, pos)
    assert after[score.id] == before[score.id]
    assert parts(135, 20)[me.id] == after[me.id] and parts(136, 20)[me.id] != after[me.id]
    assert parts(100, 20)[me.id] == 11  # outside the bucket: coarse as before
    assert parts(130, 135)[apart] == 5  # the distance outside its bucket too
    assert enc.look_closer([(me.id, 12)]) == ()  # already looked at
    enc.refresh()  # a refresh keeps the closer looks
    assert parts(130, 20)[me.id] == fine_bucket(130, pos)
    # At most MOST_CLOSER sights.
    assert MOST_CLOSER == 8
    more = enc.look_closer([(score.id, b) for b in (9, 10, 11, 12, 13, 14, 15)])
    assert len(more) == 6 and (score.id, 15) not in more
    assert len(enc.closer) == MOST_CLOSER
    s = enc.encode(snap(bytes([0, 130, 20, 9])))
    assert f"{me.id}=12 (128..135)" in s.describe()
    assert State.from_key(s.key) == s


# ------------------------------------------------------------ states


def test_state_key_round_trip_and_describe() -> None:
    k = Knowledge("g")
    lane = k.propose(RAM, 3, kind=VariableKind.POSITION, name="Lane", evidence=SURE)
    s = State(((lane.id, 2), ("v9", -3)))
    assert s.key == f"{lane.id}=2;v9=-3" and State.from_key(s.key) == s
    assert hash(s) == hash(State.from_key(s.key)) and {s: 1}[State.from_key(s.key)] == 1
    assert s.describe(k) == "Lane=2, v9=-3" and s.describe() == f"{lane.id}=2, v9=-3"
    wide = State(tuple((f"v{i}", i) for i in range(6)))
    assert wide.describe(most=4) == "v0=0, v1=1, v2=2, v3=3 and 2 more"  # T-889
    assert wide.describe(most=6) == wide.describe() and s.describe(k, most=2) == "Lane=2, v9=-3"
    with pytest.raises(ConfigurationError):
        wide.describe(most=0)
    assert State().key == "" and State.from_key("") == State()
    assert State().describe() == "nothing known yet"
    with pytest.raises(ConfigurationError):
        State.from_key("a=b")


# ------------------------------------------------------------ encoder


def test_encoder_filters_and_refreshes() -> None:
    k = Knowledge("g")
    enc = StateEncoder(k)
    assert enc.variables == () and enc.encode(snap(bytes(32))) == State()
    k.propose(RAM, 0, kind=VariableKind.SPEED, name="Guess")  # no evidence
    bad = k.propose(RAM, 1, kind=VariableKind.SPEED, evidence=SURE)
    k.reject(bad.id)
    k.propose(RAM, 2, kind=VariableKind.INPUT, evidence=SURE)
    k.propose(RAM, 3, kind=VariableKind.UNKNOWN, evidence=SURE)
    k.propose(RAM, 5, kind=VariableKind.COUNTER, evidence=SURE)  # frame counters: noise
    low = k.propose(RAM, 4, kind=VariableKind.LIVES, evidence=MAYBE)
    assert enc.encode(snap(bytes(32))) == State()  # not refreshed yet
    assert [v.id for v in enc.refresh()] == [low.id]
    assert enc.encode(snap(bytes([0, 0, 0, 0, 3]))) == State(((low.id, 3),))
    strict = StateEncoder(k, at_least=Confidence.MEDIUM)
    assert strict.variables == ()
    with pytest.raises(ConfigurationError):
        StateEncoder(k, bins=1)
    with pytest.raises(ConfigurationError):
        StateEncoder(k, bins=65)


def test_encoder_orders_by_address_and_skips_uncovered() -> None:
    k = Knowledge("g")
    score = k.propose(RAM, 8, encoding=Encoding.U16LE, kind=VariableKind.SCORE, evidence=SURE)
    flag = k.propose(RAM, 1, kind=VariableKind.FLAG, evidence=SURE)
    other = k.propose("vram", 0, kind=VariableKind.POSITION, evidence=SURE)
    enc = StateEncoder(k, bins=4)
    ram = bytearray(16)
    ram[1] = 7
    ram[8], ram[9] = 0x10, 0x00  # 16 -> bucket 4 + log2(16/4) = 6
    s = enc.encode(snap(bytes(ram)))
    assert s.parts == ((flag.id, 1), (score.id, 6))  # vram is not in the snapshot
    assert other.id not in s.key
    assert [v.id for v in enc.variables] == [flag.id, score.id, other.id]


def test_toy_racer_states_stay_small_but_tell_lanes_apart() -> None:
    k = Knowledge("toyracer")
    lane = k.propose(RAM, LAYOUT.lane, kind=VariableKind.POSITION, name="Lane", evidence=SURE)
    k.propose(RAM, LAYOUT.speed, kind=VariableKind.SPEED, name="Speed", evidence=SURE)
    k.propose(RAM, LAYOUT.lives, kind=VariableKind.LIVES, name="Lives", evidence=SURE)
    k.propose(RAM, LAYOUT.checkpoint, kind=VariableKind.PROGRESS, name="Checkpoint", evidence=SURE)
    enc = StateEncoder(k)
    game = ToyRacer(seed=2)
    seen: set[State] = set()
    lanes: set[int] = set()
    for i in range(900):
        held = ["start"] if game.state.game_over else ["up"]
        if i % 60 < 5:
            held.append("left" if (i // 60) % 2 else "right")
        game.step(held)
        s = enc.encode(snap(game.ram, i))
        seen.add(s)
        lanes.add(dict(s.parts)[lane.id])
    assert 4 <= len(seen) < 200
    assert lanes == {0, 1, 2}
    assert "Lane=" in next(iter(seen)).describe(k)


def test_encoder_uses_the_best_few_of_a_kind() -> None:
    """T-850: the state is built from at most 16 of a kind, so it stays learnable."""
    k = Knowledge("g")
    for i in range(30):
        k.propose(RAM, i, kind=VariableKind.FLAG, evidence=MAYBE)
    best = k.propose(RAM, 31, kind=VariableKind.FLAG, evidence=SURE)
    lives = k.propose(RAM, 30, kind=VariableKind.LIVES, evidence=MAYBE)
    enc = StateEncoder(k)
    ids = [v.id for v in enc.variables]
    assert len(ids) == 17 and best.id in ids and lives.id in ids
    assert [v.offset for v in enc.variables] == sorted(v.offset for v in enc.variables)


def test_the_state_takes_the_useful_kinds_first() -> None:
    """T-908: the user's slots filled with low-address flags before any position got one."""
    assert DEFAULT_MOST == DEFAULT_WIDTH and KIND_RANK[0] is VariableKind.POSITION
    k = Knowledge("g")
    flags = [k.propose(RAM, i, kind=VariableKind.FLAG, evidence=SURE) for i in range(30)]
    confirmed = k.confirm(flags[20].id)
    position = k.propose(RAM, 516, kind=VariableKind.POSITION, evidence=MAYBE)
    lives = k.propose(RAM, 600, kind=VariableKind.LIVES, evidence=MAYBE)
    score = k.propose(RAM, 700, kind=VariableKind.SCORE, evidence=MAYBE)
    enc = StateEncoder(k, most=8)
    assert enc.most == 8
    ids = [v.id for v in enc.variables]
    assert len(ids) == 8 and {position.id, lives.id, score.id} <= set(ids)
    assert confirmed.id in ids  # the best flag is the confirmed one
    assert [v.offset for v in enc.variables] == sorted(v.offset for v in enc.variables)
    assert sum(v.kind is VariableKind.FLAG for v in enc.variables) == 5
    # No cap: everything believed, as before.
    assert len(StateEncoder(k, most=None).variables) == 16 + 3  # 16 flags (T-850) + 3
    with pytest.raises(ConfigurationError):
        StateEncoder(k, most=0)


# ------------------------------------------------------------ T-912: distances


def moving_world() -> tuple[Knowledge, Variable, Variable, Variable]:
    """My position at 1, something moving at 2, a score at 3 (T-912)."""
    from retro_ai_lab.understanding.knowledge import MOVER

    k = Knowledge("g")
    me = k.propose(
        RAM, 1, kind=VariableKind.POSITION, evidence=[Evidence("monotonic", "Climbs.", 6.0)]
    )
    it = k.propose(RAM, 2, kind=VariableKind.POSITION, evidence=[Evidence(MOVER, "Moves.", 6.0)])
    score = k.propose(RAM, 3, kind=VariableKind.SCORE, evidence=SURE)
    return k, me, it, score


def test_the_distance_from_each_moving_thing_is_part_of_the_state() -> None:
    k, me, it, score = moving_world()
    enc = StateEncoder(k)
    assert enc.pairs == ((it, me),)
    apart = distance_id(it.id, me.id)

    def parts(me_at: int, it_at: int) -> dict[str, int]:
        return dict(enc.encode(snap(bytes([0, me_at, it_at, 9]))).parts)

    close = parts(100, 105)
    assert close[apart] == 5 and close[me.id] == bucket(100, VariableKind.POSITION)
    assert parts(100, 80)[apart] == bucket(-20, VariableKind.POSITION) == -10  # behind me
    assert parts(100, 230)[apart] == bucket(130, VariableKind.POSITION)  # far ahead
    # Touching, a jump away and across the screen are three different situations;
    # under the doubling rule alone 64 and 127 were one.
    assert len({parts(64, x)[apart] for x in (66, 80, 127)}) == 3
    assert list(close) == [me.id, it.id, score.id, apart]  # values by address, then distances
    assert State.from_key(enc.encode(snap(bytes([0, 100, 105, 9]))).key) == enc.encode(
        snap(bytes([0, 100, 105, 9]))
    )


def test_distance_parts_are_named_in_words() -> None:
    k, me, it, _ = moving_world()
    k.correct(me.id, name="Mario")
    s = StateEncoder(k).encode(snap(bytes([0, 100, 105, 9])))
    assert f"{it.display_name} from Mario=5" in s.describe(k)
    assert f"{it.id} from {me.id}=5" in s.describe()  # without the knowledge, the ids
    assert split_distance(distance_id("v2", "v1")) == ("v2", "v1")
    assert split_distance("v2") is None and split_distance("~v1") is None
    with pytest.raises(ConfigurationError):
        distance_id("a~b", "c")


def test_no_distances_without_both_a_mover_and_a_position_of_my_own() -> None:
    from retro_ai_lab.understanding.knowledge import INPUT_LINKED, MOVER

    k = Knowledge("g")
    k.propose(RAM, 1, kind=VariableKind.POSITION, evidence=[Evidence("monotonic", "", 6.0)])
    assert StateEncoder(k).pairs == ()  # nothing moves
    k2 = Knowledge("g")
    k2.propose(RAM, 1, kind=VariableKind.POSITION, evidence=[Evidence(MOVER, "", 6.0)])
    k2.propose(RAM, 2, kind=VariableKind.POSITION, evidence=[Evidence(MOVER, "", 6.0)])
    assert StateEncoder(k2).pairs == ()  # two things move, none is mine
    k3 = Knowledge("g")
    mine = k3.propose(
        RAM,
        1,
        kind=VariableKind.POSITION,
        evidence=[Evidence(MOVER, "", 6.0), Evidence(INPUT_LINKED, "", 1.0)],
    )
    wide = k3.propose(
        RAM,
        2,
        encoding=Encoding.U16LE,
        kind=VariableKind.POSITION,
        evidence=[Evidence(MOVER, "", 6.0)],
    )
    elsewhere = k3.propose(
        "vram", 0, kind=VariableKind.POSITION, evidence=[Evidence(MOVER, "", 6.0)]
    )
    near = k3.propose(RAM, 4, kind=VariableKind.POSITION, evidence=[Evidence(MOVER, "", 6.0)])
    enc = StateEncoder(k3)
    assert enc.pairs == ((near, mine),)  # not the 16-bit one, not the one in another region
    assert wide.id in {v.id for v in enc.variables} and elsewhere.id in {
        v.id for v in enc.variables
    }
    s = enc.encode(snap(bytes([0, 10, 0, 0, 0])))  # the snapshot has no mover value at 4? it has
    assert distance_id(near.id, mine.id) in dict(s.parts)
    s2 = enc.encode(MemorySnapshot(0, {RAM: bytes([0, 10, 0, 0])}))  # too short for ram[4]
    assert distance_id(near.id, mine.id) not in dict(s2.parts)


def test_seats_are_kept_for_distances_when_anything_moves() -> None:
    from retro_ai_lab.understanding.knowledge import MOVER

    k = Knowledge("g")
    me = k.propose(RAM, 0, kind=VariableKind.POSITION, evidence=[Evidence("monotonic", "", 6.0)])
    movers = [
        k.propose(
            RAM, 1 + i, kind=VariableKind.POSITION, evidence=[Evidence(MOVER, "", 6.0 - i * 0.1)]
        )
        for i in range(12)
    ]
    for i in range(20):
        k.propose(RAM, 20 + i, kind=VariableKind.FLAG, evidence=SURE)
    enc = StateEncoder(k, most=12)
    assert len(enc.variables) == 12 - MOST_DISTANCES
    assert me in enc.variables  # positions are seated first
    assert len(enc.pairs) <= MOST_DISTANCES and all(p is me for _, p in enc.pairs)
    assert len(enc.encode(snap(bytes(40))).parts) == len(enc.variables) + len(enc.pairs) <= 12
    big = StateEncoder(k, most=None)
    assert len(big.pairs) == MOST_DISTANCES and [m for m, _ in big.pairs] == movers[:MOST_DISTANCES]
    assert StateEncoder(k, most=1).variables  # the seats never go below one


def test_moving_things_get_distance_seats_without_a_position_seat() -> None:
    """T-923: the user's enemy sprites ranked below six plain positions and were unseen."""
    from retro_ai_lab.learning.state import DISTANCE_SEP
    from retro_ai_lab.understanding.knowledge import INPUT_LINKED, MOVER, is_mover

    k = Knowledge("g")
    me = k.propose(RAM, 0, kind=VariableKind.POSITION, evidence=[Evidence(INPUT_LINKED, "", 1.5)])
    plain = [
        k.propose(RAM, 1 + i, kind=VariableKind.POSITION, evidence=[Evidence("monotonic", "", 6.0)])
        for i in range(8)
    ]
    movers = [
        k.propose(RAM, 10 + i, kind=VariableKind.POSITION, evidence=[Evidence(MOVER, "", 2.0)])
        for i in range(3)
    ]
    for i in range(20):
        k.propose(RAM, 20 + i, kind=VariableKind.FLAG, evidence=SURE)
    enc = StateEncoder(k, most=14)  # 6 seats after the 8 kept for distances: 3 positions, 3 flags
    seated = [v for v in enc.variables if v.kind is VariableKind.POSITION]
    assert seated and all(v in plain for v in seated)  # the movers won no seat
    assert me not in enc.variables and not any(is_mover(v) for v in enc.variables)
    assert sorted(m.id for m, _ in enc.pairs) == sorted(m.id for m in movers)
    assert all(p is me for _, p in enc.pairs)
    state = enc.encode(snap(bytes([5] + [0] * 9 + [20, 40, 60] + [0] * 30)))
    distances = [p for p in state.parts if DISTANCE_SEP in p[0]]
    assert len(distances) == 3 and len(state.parts) == len(enc.variables) + 3
    assert "(3 moving)" in enc.describe() and "3 distances to moving things" in enc.describe()
    # a seated mover is counted once
    k.update(movers[0].id, evidence=(Evidence(MOVER, "", 9.0),))
    enc.refresh()
    assert movers[0].id in {v.id for v in enc.variables} and "(3 moving)" in enc.describe()


# ------------------------------------------------------------ T-913: what it sees


def test_the_encoder_says_what_the_brain_sees_in_words() -> None:
    k, _me, _it, _score = moving_world()
    assert StateEncoder(k).describe() == (
        "3 values: 2 positions (1 moving), 1 distance to a moving thing from Position at ram[1], "
        "1 score"
    )
    assert StateEncoder(Knowledge("empty")).describe() == "nothing yet"
    k2 = Knowledge("g")
    k2.propose(RAM, 0, kind=VariableKind.LIVES, evidence=SURE)
    for i in range(3):
        k2.propose(RAM, 1 + i, kind=VariableKind.FLAG, evidence=SURE)
    k2.propose(RAM, 5, kind=VariableKind.PROGRESS, evidence=SURE)
    k2.propose(RAM, 6, kind=VariableKind.PROGRESS, evidence=SURE)
    k2.propose(RAM, 7, kind=VariableKind.COUNTER, evidence=SURE)  # ignored kinds are not seen
    assert StateEncoder(k2).describe() == "6 values: 1 life count, 2 progress values, 3 switches"


def test_describe_names_the_position_distances_come_from() -> None:
    """T-929: the user's log could not say whether the AI knew its own position."""
    from retro_ai_lab.understanding.knowledge import INPUT_LINKED, MOVER

    k = Knowledge("g")
    pos = VariableKind.POSITION
    k.propose(RAM, 0, kind=pos, evidence=[Evidence("monotonic", "Climbs.", 6.0)])
    k.propose(RAM, 1, kind=pos, evidence=[Evidence(INPUT_LINKED, "Left/Right", 1.5)])
    k.propose(RAM, 3, kind=pos, evidence=[Evidence(MOVER, "Moves.", 1.5)])
    assert "1 distance to a moving thing from Position at ram[1] which the stick moves" in (
        StateEncoder(k).describe()
    )


def test_copies_of_my_position_count_once_in_the_pairs() -> None:
    """T-931: the user's eight distance seats went to one enemy measured from four copies."""
    from retro_ai_lab.understanding.knowledge import INPUT_LINKED, MOVER

    k = Knowledge("g")
    pos = VariableKind.POSITION
    stick = Evidence(INPUT_LINKED, "Changes when A, Left or Right is held.", 2.0)
    copies = [k.propose(RAM, 10 + 4 * i, kind=pos, evidence=[stick]) for i in range(4)]
    jump = Evidence(INPUT_LINKED, "Changes when A is held.", 1.0)
    y = k.propose(RAM, 40, kind=pos, evidence=[jump])
    movers = [
        k.propose(RAM, 100 + i, kind=pos, evidence=[Evidence(MOVER, "Moves.", 1.5)])
        for i in range(3)
    ]
    enc = StateEncoder(k)
    assert enc.pairs == tuple((m, copies[0]) for m in movers) + tuple((m, y) for m in movers)
    for i in range(5):  # eight movers: eight seats, all measured from the first copy
        movers.append(k.propose(RAM, 103 + i, kind=pos, evidence=[Evidence(MOVER, "Moves.", 1.5)]))
    enc.refresh()
    assert len(enc.pairs) == 8 and all(p is copies[0] for _, p in enc.pairs)
    assert sorted(m.id for m, _ in enc.pairs) == sorted(m.id for m in movers)


# ------------------------------------------------------------ T-915: my position


def test_distances_are_measured_from_the_position_the_stick_moves() -> None:
    """The user's game had ten positions; the scrolling one outranked Mario's own."""
    from retro_ai_lab.understanding.knowledge import INPUT_LINKED, MOVER

    k = Knowledge("g")
    pos = VariableKind.POSITION
    scroll = k.propose(RAM, 0, kind=pos, evidence=[Evidence("monotonic", "Climbs.", 6.0)])
    me = k.propose(RAM, 1, kind=pos, evidence=[Evidence(INPUT_LINKED, "Left/Right", 1.5)])
    me_too = k.propose(RAM, 2, kind=pos, evidence=[Evidence(INPUT_LINKED, "Up/Down", 1.0)])
    it = k.propose(RAM, 3, kind=pos, evidence=[Evidence(MOVER, "Moves.", 1.5)])
    enc = StateEncoder(k)
    assert enc.pairs == ((it, me), (it, me_too))  # the scrolling position is nobody's
    assert scroll in enc.variables
    k.confirm(scroll.id)  # a person saying "this is my position" makes it mine
    k.correct(scroll.id, name="Where I am")
    enc.refresh()
    assert [p.id for _, p in enc.pairs] == [scroll.id, me.id, me_too.id]
    k2 = Knowledge("g")
    only = k2.propose(RAM, 0, kind=pos, evidence=[Evidence("monotonic", "Climbs.", 6.0)])
    thing = k2.propose(RAM, 3, kind=pos, evidence=[Evidence(MOVER, "Moves.", 1.5)])
    assert StateEncoder(k2).pairs == ((thing, only),)  # nothing linked yet: as before
