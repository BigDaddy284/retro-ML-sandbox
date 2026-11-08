"""T-301: the knowledge model (variables, evidence, confidence)."""

from __future__ import annotations

from dataclasses import replace

import pytest
from hypothesis import given, settings
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
from retro_ai_lab.understanding.knowledge import THRESHOLDS, belief_rank, decode

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
        judged=37,
    )
    assert Variable.from_dict(v.to_dict()) == v
    assert v.to_dict()["judged"] == 37
    old = {k: val for k, val in v.to_dict().items() if k != "judged"}  # a file from before T-917
    assert Variable.from_dict(old).judged == 0


def test_forget_judged() -> None:
    """T-926: the judged counts are a run's, and go when a game connects."""
    k = Knowledge("g")
    a = k.propose("ram", 0, kind=VariableKind.SCORE, judged=37)
    b = k.propose("ram", 1, kind=VariableKind.LIVES, judged=4)
    c = k.propose("ram", 2, kind=VariableKind.FLAG)
    version = k.version
    assert k.forget_judged() == 2 and k.version == version + 1
    assert all(k.get(v.id).judged == 0 for v in (a, b, c))
    assert k.get(a.id).kind is VariableKind.SCORE and k.get(b.id).kind is VariableKind.LIVES
    assert k.forget_judged() == 0 and k.version == version + 1  # nothing to forget: no change
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


def test_the_knowledge_can_be_read_while_another_thread_rewrites_it() -> None:
    """T-847: the finder rewrote relationships on the play clock's worker while a handler
    on the GUI thread walked them: "dictionary changed size during iteration"."""
    import sys
    import threading

    from retro_ai_lab.understanding.knowledge import Relationship

    k = Knowledge("g")
    for i in range(400):
        k.propose("ram", i, kind=VariableKind.FLAG, name=f"f{i}", evidence=[Evidence("t", "", 3.0)])

    def rel(i: int, j: int) -> Relationship:
        return Relationship(f"flag_on:v{i}", f"flag_on:v{j}", 3, 5, 5, 10)

    stop = threading.Event()
    failures: list[BaseException] = []

    def writer() -> None:
        n = 0
        try:
            while not stop.is_set():
                n += 1
                for i in range(1, 60):
                    k.set_relationship(rel(i, (i + n) % 400 + 1))
                for i in range(1, 60):
                    k.remove_relationship(rel(i, (i + n) % 400 + 1).id)
                k.support("v1", Evidence("t", "", 0.1))
        except BaseException as exc:
            failures.append(exc)

    interval = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)  # switch threads as often as possible
    t = threading.Thread(target=writer, daemon=True)
    t.start()
    try:
        for _ in range(300):
            k.confident_relationships()
            k.relationships()
            list(k)
            k.to_dict()
            k.confident()
    finally:
        stop.set()
        t.join(timeout=10)
        sys.setswitchinterval(interval)
    assert not t.is_alive() and failures == []


def test_believed_counts_moving_things_apart() -> None:
    """T-928: sixteen plain positions pushed every enemy sprite out of the believed set."""
    from retro_ai_lab.understanding.knowledge import MOVER, is_mover

    k = Knowledge("g")
    plain = [
        k.propose("ram", i, kind=VariableKind.POSITION, evidence=[Evidence("monotonic", "", 2.5)])
        for i in range(20)
    ]
    movers = [
        k.propose("ram", 100 + i, kind=VariableKind.POSITION, evidence=[Evidence(MOVER, "", 2.0)])
        for i in range(3)
    ]
    chosen = k.believed()
    assert [v.id for v in chosen if not is_mover(v)] == [v.id for v in plain[:16]]
    assert [v.id for v in chosen if is_mover(v)] == [v.id for v in movers]
    two = k.believed(per_kind=2)
    assert sum(1 for v in two if is_mover(v)) == 2 and sum(1 for v in two if not is_mover(v)) == 2
    assert not any(is_mover(v) for v in k.offered())
    assert [v.id for v in k.offered() if v.kind is VariableKind.POSITION] == [plain[0].id]


def test_believed_keeps_the_best_few_of_each_kind() -> None:
    """T-850: the AI acts on its strongest 16 guesses of a kind, confirmed ones first."""
    from retro_ai_lab.understanding.knowledge import BELIEVED_PER_KIND

    assert BELIEVED_PER_KIND == 16
    k = Knowledge("g")
    weak = [Evidence("t", "", 1.0)]
    for i in range(40):
        k.propose("ram", i, kind=VariableKind.FLAG, evidence=weak)
    strong = k.propose("ram", 50, kind=VariableKind.FLAG, evidence=[Evidence("t", "", 2.0)])
    sure = k.propose("ram", 60, kind=VariableKind.FLAG, evidence=[Evidence("t", "", 3.0)])
    confirmed = k.propose("ram", 70, kind=VariableKind.FLAG, evidence=weak)
    k.confirm(confirmed.id)
    lives = k.propose("ram", 80, kind=VariableKind.LIVES, evidence=weak)
    rejected = k.propose("ram", 90, kind=VariableKind.LIVES, evidence=weak)
    k.reject(rejected.id)
    elsewhere = k.propose("vram", 0, kind=VariableKind.LIVES, evidence=[Evidence("t", "", 3.0)])
    chosen = k.believed()
    flags = [v for v in chosen if v.kind is VariableKind.FLAG]
    assert len(flags) == 16
    assert {confirmed.id, sure.id, strong.id} <= {v.id for v in flags}
    assert [v.offset for v in flags if v.offset < 40] == list(range(13))  # then by address
    assert [v.id for v in chosen if v.kind is VariableKind.LIVES] == [lives.id, elsewhere.id]
    assert [v.id for v in chosen] == sorted((v.id for v in chosen), key=lambda i: int(i[1:]))
    # Only the regions the emulator shares compete for the seats.
    assert elsewhere.id not in {v.id for v in k.believed(regions=("ram",))}
    assert len(k.believed(per_kind=2)) == 4
    assert k.believed(Confidence.MEDIUM) == (sure, k.get(confirmed.id), elsewhere)
    with pytest.raises(ConfigurationError):
        k.believed(per_kind=0)


def test_offered_keeps_one_seat_per_role_and_names_them() -> None:
    """T-858: rewards and steps are offered for the best guess of each role, in plain words."""
    from retro_ai_lab.understanding.knowledge import OFFERED_MANY, plain_names

    assert OFFERED_MANY == 3
    k = Knowledge("g")
    weak = [Evidence("t", "", 1.0)]
    strong = [Evidence("t", "", 3.0)]
    scores = [k.propose("ram", i, kind=VariableKind.SCORE, evidence=weak) for i in range(5)]
    best_score = k.propose("ram", 10, kind=VariableKind.SCORE, evidence=strong)
    told = k.propose("ram", 11, kind=VariableKind.LIVES, evidence=weak)
    k.confirm(told.id)
    k.correct(told.id, name="Hearts")
    k.propose("ram", 12, kind=VariableKind.LIVES, evidence=strong)  # loses to the confirmed one
    progress = [
        k.propose("ram", 20 + i, kind=VariableKind.PROGRESS, evidence=strong) for i in range(4)
    ]
    flags = [k.propose("ram", 30 + i, kind=VariableKind.FLAG, evidence=weak) for i in range(2)]
    offered = k.offered()
    assert [v.id for v in offered] == [
        best_score.id,
        told.id,
        *(v.id for v in progress[:3]),
        *(v.id for v in flags),
    ]
    assert scores[0].id not in {v.id for v in offered}
    assert k.offered(Confidence.MEDIUM) == (best_score, k.get(told.id), *progress[:3])
    names = plain_names(offered)
    assert names[best_score.id] == "Your score" and names[told.id] == "Hearts"
    assert [names[v.id] for v in progress[:3]] == [
        "Progress",
        "Progress (2nd guess)",
        "Progress (3rd guess)",
    ]
    assert [names[v.id] for v in flags] == ["Something", "Something (2nd guess)"]
    assert plain_names(()) == {}


def test_belief_rank_orders_as_believed_does() -> None:
    """T-908: the state encoder fills the brain's slots in the order believed() seats."""
    k = Knowledge("g")
    weak = k.propose("ram", 0, kind=VariableKind.FLAG, evidence=[Evidence("t", "", 1.0)])
    strong = k.propose("ram", 1, kind=VariableKind.FLAG, evidence=[Evidence("t", "", 6.0)])
    sure = k.propose("ram", 2, kind=VariableKind.FLAG, evidence=[Evidence("t", "", 1.0)])
    sure = k.confirm(sure.id)
    ranked = sorted((weak, strong, sure), key=belief_rank)
    assert [v.id for v in ranked] == [sure.id, strong.id, weak.id]
    assert [v.id for v in k.believed(per_kind=2)] == [strong.id, sure.id]  # knowledge order


def test_a_mover_is_called_something_moving_until_the_stick_moves_it() -> None:
    """T-911: a position found by its movement is not "Your position" until linked to a button."""
    from retro_ai_lab.understanding.knowledge import INPUT_LINKED, MOVER, is_mover, plain_names

    k = Knowledge("g")
    mover = k.propose(
        "ram", 1, kind=VariableKind.POSITION, evidence=[Evidence(MOVER, "Moves both ways.", 1.5)]
    )
    mine = k.propose(
        "ram", 2, kind=VariableKind.POSITION, evidence=[Evidence("monotonic", "Climbs.", 2.5)]
    )
    assert is_mover(mover) and not is_mover(mine)
    names = plain_names((mine, mover))
    assert names[mine.id] == "Your position" and names[mover.id] == "Something moving"
    linked = k.update(
        mover.id, evidence=(*mover.evidence, Evidence(INPUT_LINKED, "Changes when Left is held."))
    )
    assert not is_mover(linked)
    assert plain_names((linked,))[linked.id] == "Your position"
    assert plain_names((mover, linked))[linked.id] == "Your position"


@pytest.mark.parametrize(
    ("raw", "enc", "value"),
    [
        (b"\x58\x02", Encoding.UINT_LE, 600),
        (b"\x58\x02\x00", Encoding.UINT_LE, 600),
        (b"\x00\x02\x58", Encoding.UINT_BE, 600),
        (b"\x06\x00", Encoding.BCD_BE, 600),
        (b"\x00\x06", Encoding.BCD_LE, 600),
        (b"\x00\x06\x00", Encoding.BCD_BE, 600),
        (b"\x00\x00\x00\x06\x00\x00", Encoding.DIGITS_BE, 600),
        (b"\x00\x00\x06\x00\x00\x00", Encoding.DIGITS_LE, 600),
        (b"\x09", Encoding.DIGITS_BE, 9),
        (b"\x99\x99\x99\x99", Encoding.BCD_BE, 99_999_999),
    ],
)
def test_sized_encodings_decode_both_ways_round(raw: bytes, enc: Encoding, value: int) -> None:
    """T-941: a number over several bytes reads as one value, whichever way it is held."""
    assert decode(raw, enc) == value
    assert enc.sized and enc.width == 0


def test_a_sized_variable_needs_a_length() -> None:
    """T-941: a digit that is not a digit, a length out of range, and the plain words."""
    with pytest.raises(ConfigurationError, match="digit"):
        decode(b"\x00\x0a", Encoding.DIGITS_BE)
    with pytest.raises(ConfigurationError, match="BCD"):
        decode(b"\x1a\x00", Encoding.BCD_LE)
    with pytest.raises(ConfigurationError, match="1 to 8"):
        decode(b"\x00" * 9, Encoding.DIGITS_BE)
    with pytest.raises(ConfigurationError, match="1 to 4"):
        decode(b"\x00" * 5, Encoding.UINT_LE)
    with pytest.raises(ConfigurationError, match="takes 1 to 8"):
        Variable("v1", "ram", 0, Encoding.DIGITS_BE)
    with pytest.raises(ConfigurationError, match="takes 1 to 4"):
        Variable("v1", "ram", 0, Encoding.UINT_LE, length=5)
    with pytest.raises(ConfigurationError, match="takes 1 byte"):
        Variable("v1", "ram", 0, Encoding.U8, length=2)
    score = Variable("v1", "ram", 10, Encoding.DIGITS_BE, VariableKind.SCORE, "Score", length=6)
    assert score.width == 6 and score.address == "ram[10..15]"
    assert score.held_as == "6 digits, highest first"
    snapshot = MemorySnapshot(1, {"ram": bytes([0] * 10 + [0, 0, 1, 2, 5, 0, 7])})
    assert score.decode(snapshot) == 1250
    again = Variable.from_dict(score.to_dict())
    assert again == score and again.length == 6
    plain = Variable.from_dict({"id": "v2", "region": "ram", "offset": 3})
    assert plain.length == 0 and plain.width == 1 and plain.address == "ram[3]"
    assert plain.held_as == "1 byte"
    assert Variable("v3", "ram", 0, Encoding.U16LE, length=2).width == 2
    assert Encoding.U16BE.describe(2) == "2 bytes, high first"
    assert Encoding.BCD_LE.describe(3) == "packed decimal, 3 bytes, low first"
    assert Encoding.FLAG.describe(1) == "1 byte (on or off)"


# ------------------------------------------------- T-948 confident relationships version


def test_the_confident_relationships_version_moves_only_when_the_suggestions_would_see_it() -> None:
    k = Knowledge("g")
    v0 = k.confident_relationships_version
    k.set_relationship(Relationship("a", "b", 1, 1, 1, 5))  # UNKNOWN: not read by anyone
    assert k.confident_relationships_version == v0
    k.set_relationship(Relationship("a", "b", 2, 2, 2, 5))  # LOW: appears
    assert k.confident_relationships_version == v0 + 1
    k.set_relationship(Relationship("a", "b", 3, 3, 4, 5))  # still LOW: counts moved, level not
    assert k.confident_relationships_version == v0 + 1
    k.set_relationship(Relationship("a", "b", 6, 6, 6, 5))  # MEDIUM: level changed
    assert k.confident_relationships_version == v0 + 2
    k.set_relationship(Relationship("a", "b", 6, 60, 6, 5))  # rarely follows: UNKNOWN, gone
    assert k.confident_relationships_version == v0 + 3
    k.set_relationship(Relationship("a", "b", 12, 12, 12, 5))  # HIGH: back
    assert k.confident_relationships_version == v0 + 4
    k.confirm_relationship("a->b")  # CONFIRMED
    assert k.confident_relationships_version == v0 + 5
    k.set_relationship(Relationship("a", "b", 13, 13, 13, 5))  # verdict kept: still CONFIRMED
    assert k.confident_relationships_version == v0 + 5
    k.reject_relationship("a->b")  # gone from the confident ones
    assert k.confident_relationships_version == v0 + 6
    k.set_relationship(Relationship("a", "b", 14, 14, 14, 5))  # still rejected
    assert k.confident_relationships_version == v0 + 6
    k.remove_relationship("a->b")  # removing a rejected pair changes nothing visible
    assert k.confident_relationships_version == v0 + 6
    k.set_relationship(Relationship("c", "d", 2, 2, 2, 5))
    k.remove_relationship("c->d")  # removing a confident pair does
    assert k.confident_relationships_version == v0 + 8
    assert k.version == 0  # the variables' version is a different number


# --------------------------------------------- T-952 confident relationships index


def _old_confident(k: Knowledge, at_least: Confidence) -> tuple[Relationship, ...]:
    """The walk `confident_relationships` made before T-952."""
    return tuple(r for r in k.relationships() if not r.rejected and at_least <= r.confidence)


def _relationship_ops() -> st.SearchStrategy[tuple[str, ...]]:
    keys = st.sampled_from(["a", "b", "c", "d", "e"])
    pair = st.tuples(keys, keys).filter(lambda p: p[0] != p[1])
    write = st.tuples(
        st.just("set"), pair, st.integers(0, 15), st.integers(0, 15), st.integers(0, 15)
    )
    verdict = st.tuples(st.sampled_from(["confirm", "reject", "remove"]), pair)
    return st.one_of(write, verdict)  # type: ignore[return-value]


@given(st.lists(_relationship_ops(), min_size=1, max_size=60))
@settings(max_examples=100, deadline=None)
def test_the_confident_index_matches_the_old_walk_after_every_write(ops: list[tuple]) -> None:
    """T-952: the index gives the walk's relationships, in the walk's order, every time."""
    k = Knowledge("g")
    for op in ops:
        if op[0] == "set":
            _, (cause, effect), support, extra_cause, extra_effect = op
            k.set_relationship(
                Relationship(
                    cause, effect, support, support + extra_cause, support + extra_effect, 5
                )
            )
        else:
            rel_id = f"{op[1][0]}->{op[1][1]}"
            if rel_id not in {r.id for r in k.relationships()}:
                continue
            getattr(k, f"{op[0]}_relationship")(rel_id)
        for level in (Confidence.LOW, Confidence.MEDIUM, Confidence.HIGH, Confidence.CONFIRMED):
            assert k.confident_relationships(level) == _old_confident(k, level)


# ------------------------------------------------------- T-953 location index for at()


def _old_at(k: Knowledge, region: str, offset: int) -> Variable | None:
    """The walk `at` made before T-953."""
    for v in k:
        if v.region == region and v.offset == offset:
            return v
    return None


_GRID = [(r, o) for r in ("ram", "vram") for o in range(4)]


def _variable_ops() -> st.SearchStrategy[tuple]:
    where = st.sampled_from(_GRID)
    add = st.tuples(st.just("add"), where, st.booleans())  # rejected at birth or not
    move = st.tuples(st.just("move"), st.integers(0, 9), where)  # the nth variable moves
    verdict = st.tuples(st.sampled_from(["reject", "remove"]), st.integers(0, 9))
    return st.one_of(add, move, verdict)


@given(st.lists(_variable_ops(), min_size=1, max_size=40))
@settings(max_examples=100, deadline=None)
def test_at_reads_the_index_and_answers_like_the_old_walk(ops: list[tuple]) -> None:
    """T-953: adds, moves, rejections and removals leave `at` answering as the walk did."""
    k = Knowledge("g")
    for op in ops:
        ids = [v.id for v in k]
        if op[0] == "add":
            _, (region, offset), rejected = op
            clash = _old_at(k, region, offset)
            if clash is not None and not clash.rejected:
                with pytest.raises(ConfigurationError):
                    k.propose(region, offset)
                continue
            v = k.propose(region, offset)
            if rejected:
                k.reject(v.id)
        elif not ids:
            continue
        elif op[0] == "move":
            _, nth, (region, offset) = op
            k.update(ids[nth % len(ids)], region=region, offset=offset)
        else:
            getattr(k, op[0])(ids[op[1] % len(ids)])
        for region, offset in _GRID:
            assert k.at(region, offset) == _old_at(k, region, offset)
