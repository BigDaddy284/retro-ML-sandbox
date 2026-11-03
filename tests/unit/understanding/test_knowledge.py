"""T-301: the knowledge model (variables, evidence, confidence)."""

from __future__ import annotations

from dataclasses import replace

import pytest
from hypothesis import given
from hypothesis import strategies as st

from retro_ai_lab.core import ConfigurationError
from retro_ai_lab.observation import MemorySnapshot
from retro_ai_lab.understanding import (
    Confidence,
    Encoding,
    Evidence,
    Knowledge,
    Relationship,
    Variable,
    VariableKind,
    confidence_from_evidence,
)
from retro_ai_lab.understanding.knowledge import THRESHOLDS, decode

# ---------------------------------------------------------- confidence


def test_confidence_is_ordered_and_labelled() -> None:
    assert Confidence.UNKNOWN < Confidence.LOW < Confidence.MEDIUM < Confidence.HIGH
    assert Confidence.HIGH < Confidence.CONFIRMED
    assert Confidence.MEDIUM <= Confidence.MEDIUM
    assert Confidence.CONFIRMED.label == "Confirmed"


@pytest.mark.parametrize(
    ("weights", "expected"),
    [
        ([], Confidence.UNKNOWN),
        ([0.5], Confidence.UNKNOWN),
        ([1.0], Confidence.LOW),
        ([1.0, 2.0], Confidence.MEDIUM),
        ([3.0, 3.0], Confidence.HIGH),
        ([100.0], Confidence.HIGH),  # never CONFIRMED automatically
    ],
)
def test_confidence_from_evidence(weights: list[float], expected: Confidence) -> None:
    assert confidence_from_evidence(weights) is expected


def test_confidence_rejects_negative_weight() -> None:
    with pytest.raises(ConfigurationError):
        confidence_from_evidence([1.0, -0.1])
    assert THRESHOLDS[0][1] is Confidence.HIGH


# ------------------------------------------------------------ decoding


@pytest.mark.parametrize(
    ("raw", "enc", "value"),
    [
        (b"\x7f", Encoding.U8, 127),
        (b"\xff", Encoding.U8, 255),
        (b"\xff", Encoding.S8, -1),
        (b"\x80", Encoding.S8, -128),
        (b"\x34\x12", Encoding.U16LE, 0x1234),
        (b"\x12\x34", Encoding.U16BE, 0x1234),
        (b"\x42", Encoding.BCD, 42),
        (b"\x00", Encoding.FLAG, 0),
        (b"\x09", Encoding.FLAG, 1),
    ],
)
def test_decode(raw: bytes, enc: Encoding, value: int) -> None:
    assert decode(raw, enc) == value
    assert enc.width == len(raw)


def test_decode_rejects_bad_input() -> None:
    with pytest.raises(ConfigurationError):
        decode(b"\x00\x00", Encoding.U8)
    with pytest.raises(ConfigurationError):
        decode(b"\x00", Encoding.U16LE)
    with pytest.raises(ConfigurationError, match="BCD"):
        decode(b"\x1a", Encoding.BCD)


# ----------------------------------------------------------- variables


def test_variable_defaults_and_confidence() -> None:
    v = Variable("v1", "ram", 8)
    assert v.width == 1 and v.kind is VariableKind.UNKNOWN
    assert v.confidence is Confidence.UNKNOWN
    assert v.display_name == "Unknown at ram[8]"
    v2 = v.with_evidence(Evidence("monotonic", "only ever went up", 2.0), Evidence("x", "", 1.0))
    assert v2.confidence is Confidence.MEDIUM and v.confidence is Confidence.UNKNOWN
    named = Variable("v1", "ram", 8, name="Score", kind=VariableKind.SCORE)
    assert named.display_name == "Score"


def test_variable_decodes_from_snapshot() -> None:
    snap = MemorySnapshot(0, {"ram": b"\x00\x34\x12\x05"})
    assert Variable("a", "ram", 1, Encoding.U16LE).decode(snap) == 0x1234
    assert Variable("b", "ram", 3).decode(snap) == 5
    with pytest.raises(ConfigurationError):
        Variable("c", "ram", 3, Encoding.U16LE).decode(snap)  # runs off the end


@pytest.mark.parametrize(
    "kwargs",
    [
        {"id": "", "region": "ram", "offset": 0},
        {"id": "v", "region": "", "offset": 0},
        {"id": "v", "region": "ram", "offset": -1},
        {"id": "v", "region": "ram", "offset": True},
        {"id": "v", "region": "ram", "offset": 0, "confirmed": True, "rejected": True},
    ],
)
def test_variable_rejects_bad_values(kwargs: dict[str, object]) -> None:
    with pytest.raises(ConfigurationError):
        Variable(**kwargs)  # type: ignore[arg-type]


def test_evidence_validation_and_round_trip() -> None:
    e = Evidence("input-linked", "changes when Up is held", 1.5, frame=12)
    assert Evidence.from_dict(e.to_dict()) == e
    with pytest.raises(ConfigurationError):
        Evidence("", "x")
    with pytest.raises(ConfigurationError):
        Evidence("k", "x", -1)
    with pytest.raises(ConfigurationError):
        Evidence.from_dict({"detail": "no kind"})


def test_variable_round_trip_and_bad_dict() -> None:
    v = Variable(
        "v9",
        "ram",
        2,
        Encoding.U16BE,
        VariableKind.SCORE,
        "Score",
        (Evidence("a", "b", 2.0),),
        confirmed=True,
        notes="hi",
    )
    assert Variable.from_dict(v.to_dict()) == v
    with pytest.raises(ConfigurationError):
        Variable.from_dict({"id": "x", "region": "ram", "offset": 0, "encoding": "u99"})
    with pytest.raises(ConfigurationError):
        Variable.from_dict({"region": "ram", "offset": 0})


# ----------------------------------------------------------- knowledge


@pytest.fixture
def knowledge() -> Knowledge:
    return Knowledge("toyracer-0", title="Toy Racer")


def test_knowledge_starts_empty(knowledge: Knowledge) -> None:
    assert len(knowledge) == 0 and list(knowledge) == []
    assert knowledge.at("ram", 0) is None
    assert knowledge.best(VariableKind.SCORE) is None
    with pytest.raises(ConfigurationError):
        knowledge.get("v1")
    with pytest.raises(ConfigurationError):
        Knowledge("")


def test_propose_add_and_lookup(knowledge: Knowledge) -> None:
    v = knowledge.propose("ram", 8, encoding=Encoding.U16LE, kind=VariableKind.SCORE)
    assert v.id == "v1" and "v1" in knowledge
    assert knowledge.at("ram", 8) == v
    assert knowledge.propose("ram", 4).id == "v2"
    with pytest.raises(ConfigurationError, match="already"):
        knowledge.add(Variable("v1", "ram", 20))
    with pytest.raises(ConfigurationError, match="already"):
        knowledge.propose("ram", 8)  # same location
    knowledge.add(Variable("custom", "ram", 30))
    assert knowledge.new_id() == "v3"


def test_update_support_and_queries(knowledge: Knowledge) -> None:
    v = knowledge.propose("ram", 8, kind=VariableKind.SCORE)
    knowledge.support("v1", Evidence("monotonic", "went up", 2.0))
    assert knowledge.get("v1").confidence is Confidence.LOW
    knowledge.support("v1", Evidence("checkpoint", "jumps at checkpoints", 2.0))
    assert knowledge.get("v1").confidence is Confidence.MEDIUM
    other = knowledge.propose("ram", 9, kind=VariableKind.SCORE, evidence=[Evidence("x", "", 1.0)])
    assert knowledge.by_kind(VariableKind.SCORE) == (knowledge.get("v1"), other)
    assert knowledge.best(VariableKind.SCORE) == knowledge.get("v1")
    assert knowledge.confident() == (knowledge.get("v1"),)
    assert knowledge.confident(Confidence.LOW) == (knowledge.get("v1"), other)
    assert set(knowledge.needs_help()) == {knowledge.get("v1"), other}
    renamed = knowledge.update("v1", name="Points")
    assert renamed.name == "Points" and renamed.id == v.id
    with pytest.raises(ConfigurationError):
        knowledge.update("v1", id="zzz")


def test_confirm_correct_reject(knowledge: Knowledge) -> None:
    knowledge.propose("ram", 8, kind=VariableKind.COUNTER, evidence=[Evidence("a", "", 1.0)])
    confirmed = knowledge.confirm("v1", by="Boss")
    assert confirmed.confidence is Confidence.CONFIRMED and confirmed.confirmed
    assert confirmed.evidence[-1].kind == "user" and "Boss" in confirmed.evidence[-1].detail
    assert knowledge.needs_help() == ()
    fixed = knowledge.correct("v1", name="Score", kind=VariableKind.SCORE)
    assert fixed.name == "Score" and fixed.kind is VariableKind.SCORE and fixed.confirmed
    with pytest.raises(ConfigurationError):
        knowledge.correct("v1")
    rejected = knowledge.reject("v1")
    assert rejected.rejected and not rejected.confirmed
    assert rejected.confidence is Confidence.UNKNOWN
    assert knowledge.by_kind(VariableKind.SCORE) == ()
    # a rejected location can be proposed again with a new meaning
    again = knowledge.propose("ram", 8, kind=VariableKind.LIVES)
    assert again.id == "v2"
    knowledge.remove("v1")
    assert "v1" not in knowledge
    with pytest.raises(ConfigurationError):
        knowledge.remove("v1")


def test_knowledge_round_trip(knowledge: Knowledge) -> None:
    knowledge.propose("ram", 8, kind=VariableKind.SCORE, evidence=[Evidence("a", "b", 2.0)])
    knowledge.confirm("v1")
    knowledge.propose("ram", 3, name="Lane")
    data = knowledge.to_dict()
    copy = Knowledge.from_dict(data)
    assert copy.game_id == "toyracer-0" and copy.title == "Toy Racer"
    assert [v.to_dict() for v in copy] == [v.to_dict() for v in knowledge]
    assert copy.new_id() == "v3"
    assert copy.get("v1").confidence is Confidence.CONFIRMED


def test_knowledge_from_bad_dict() -> None:
    with pytest.raises(ConfigurationError):
        Knowledge.from_dict({"title": "x"})
    with pytest.raises(ConfigurationError, match="newer"):
        Knowledge.from_dict({"game_id": "g", "schema": 99})
    with pytest.raises(ConfigurationError):
        Knowledge.from_dict({"game_id": "g", "variables": [{"id": "v"}]})
    with pytest.raises(ConfigurationError):
        Knowledge.from_dict({"game_id": "g", "variables": "nope"})


@given(
    st.lists(
        st.tuples(st.integers(0, 31), st.sampled_from(list(VariableKind)), st.floats(0, 5)),
        max_size=12,
        unique_by=lambda t: t[0],
    )
)
def test_round_trip_preserves_everything(items: list[tuple[int, VariableKind, float]]) -> None:
    k = Knowledge("g")
    for off, kind, w in items:
        k.propose("ram", off, kind=kind, evidence=[Evidence("e", "d", w)])
    copy = Knowledge.from_dict(k.to_dict())
    assert copy.to_dict() == k.to_dict()
    for v in k:
        assert copy.get(v.id).confidence is v.confidence


# ------------------------------------------------------------ T-841


def test_version_counts_every_change(knowledge: Knowledge) -> None:
    assert knowledge.version == 0
    v = knowledge.propose("ram", 0, kind=VariableKind.SCORE)
    assert knowledge.version == 1
    knowledge.get(v.id), knowledge.at("ram", 0), list(knowledge), knowledge.to_dict()
    assert knowledge.version == 1  # reads leave it alone
    knowledge.support(v.id, Evidence("monotonic", "up", 2.0))
    knowledge.update(v.id, name="Score")
    knowledge.confirm(v.id)
    knowledge.correct(v.id, name="Points")  # an update and a confirm
    knowledge.reject(v.id)
    assert knowledge.version == 7
    rel = Relationship("progress:v1", "score_gain:v1", 2, 2, 2, 5)
    knowledge.set_relationship(rel)
    knowledge.confirm_relationship(rel.id)
    knowledge.reject_relationship(rel.id)
    knowledge.remove_relationship(rel.id)
    assert knowledge.version == 7  # relationships are rewritten every event: not counted
    knowledge.remove(v.id)
    assert knowledge.version == 8
    loaded = Knowledge.from_dict(knowledge.to_dict())
    assert isinstance(loaded.version, int)


def test_confidence_follows_new_evidence(knowledge: Knowledge) -> None:
    v = knowledge.propose("ram", 0, kind=VariableKind.SCORE)
    assert v.confidence is Confidence.UNKNOWN
    assert knowledge.support(v.id, Evidence("monotonic", "up", 2.0)).confidence is Confidence.LOW
    assert knowledge.support(v.id, Evidence("x", "", 2.0)).confidence is Confidence.MEDIUM
    assert knowledge.confirm(v.id).confidence is Confidence.CONFIRMED
    assert knowledge.reject(v.id).confidence is Confidence.UNKNOWN
    again = replace(knowledge.get(v.id), rejected=False)
    assert again.confidence is Confidence.MEDIUM
    assert "confidence" not in again.to_dict()
    assert Variable.from_dict(again.to_dict()).confidence is Confidence.MEDIUM


def test_wake_button_round_trips(knowledge: Knowledge) -> None:
    assert knowledge.wake_button is None
    assert Knowledge.from_dict(knowledge.to_dict()).wake_button is None
    knowledge.wake_button = "start"
    data = knowledge.to_dict()
    assert data["wake_button"] == "start"
    assert Knowledge.from_dict(data).wake_button == "start"
    with pytest.raises(ConfigurationError):
        Knowledge.from_dict({**data, "wake_button": 3})
    with pytest.raises(ConfigurationError):
        Knowledge.from_dict({**data, "wake_button": ""})
