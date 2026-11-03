"""T-501: discrete states from believed variables."""

from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from retro_ai_lab.core import ConfigurationError
from retro_ai_lab.emulator.fake import ToyRacer
from retro_ai_lab.emulator.fake.toy_racer import LAYOUT
from retro_ai_lab.learning import State, StateEncoder, bucket
from retro_ai_lab.observation import MemorySnapshot
from retro_ai_lab.understanding import Confidence, Encoding, Evidence, Knowledge, VariableKind

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


# ------------------------------------------------------------ states


def test_state_key_round_trip_and_describe() -> None:
    k = Knowledge("g")
    lane = k.propose(RAM, 3, kind=VariableKind.POSITION, name="Lane", evidence=SURE)
    s = State(((lane.id, 2), ("v9", -3)))
    assert s.key == f"{lane.id}=2;v9=-3" and State.from_key(s.key) == s
    assert hash(s) == hash(State.from_key(s.key)) and {s: 1}[State.from_key(s.key)] == 1
    assert s.describe(k) == "Lane=2, v9=-3" and s.describe() == f"{lane.id}=2, v9=-3"
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
