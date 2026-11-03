"""T-305: relationships between events."""

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
    Evidence,
    GameEvent,
    Knowledge,
    Relationship,
    RelationshipFinder,
    VariableKind,
    describe_key,
    event_key,
)

RAM = "ram"
SURE = [Evidence("test", "", 3.0)]


def ev(kind: EventKind, frame: int, var: str | None = "v1") -> GameEvent:
    return GameEvent(kind, frame, var, "")


# ---------------------------------------------------------------- model


def test_relationship_rates_confidence_and_validation() -> None:
    r = Relationship("progress:v1", "score_gain:v2", 6, 8, 6, 5)
    assert r.id == "progress:v1->score_gain:v2"
    assert r.follow_rate == 0.75 and r.precede_rate == 1.0
    assert r.confidence is Confidence.MEDIUM
    assert Relationship("a", "b", 2, 3, 3, 1).confidence is Confidence.LOW
    assert Relationship("a", "b", 12, 12, 13, 1).confidence is Confidence.HIGH
    assert Relationship("a", "b", 1, 1, 1, 1).confidence is Confidence.UNKNOWN
    assert Relationship("a", "b", 2, 10, 2, 1).confidence is Confidence.UNKNOWN  # rarely follows
    assert Relationship("a", "b", 0, 0, 0, 1).follow_rate == 0.0
    assert Relationship("a", "b", 1, 1, 1, 1, confirmed=True).confidence is Confidence.CONFIRMED
    assert Relationship("a", "b", 12, 12, 12, 1, rejected=True).confidence is Confidence.UNKNOWN
    for bad in (
        dict(cause="a", effect="a", support=0, cause_count=0, effect_count=0, window=1),
        dict(cause="", effect="b", support=0, cause_count=0, effect_count=0, window=1),
        dict(cause="a", effect="b", support=3, cause_count=2, effect_count=5, window=1),
        dict(cause="a", effect="b", support=-1, cause_count=2, effect_count=5, window=1),
        dict(
            cause="a",
            effect="b",
            support=1,
            cause_count=2,
            effect_count=5,
            window=1,
            confirmed=True,
            rejected=True,
        ),
    ):
        with pytest.raises(ConfigurationError):
            Relationship(**bad)  # type: ignore[arg-type]
    assert Relationship.from_dict(r.to_dict()) == r
    with pytest.raises(ConfigurationError):
        Relationship.from_dict({"cause": "a"})


def test_knowledge_keeps_relationships_and_user_verdicts() -> None:
    k = Knowledge("g")
    r = k.set_relationship(Relationship("a", "b", 2, 2, 2, 3))
    assert k.relationships() == (r,) and k.relationship(r.id) == r
    k.confirm_relationship(r.id)
    updated = k.set_relationship(Relationship("a", "b", 5, 6, 5, 3))
    assert updated.support == 5 and updated.confirmed  # counts refreshed, verdict kept
    k.reject_relationship(r.id)
    assert k.relationship(r.id).rejected and not k.relationship(r.id).confirmed
    assert k.confident_relationships() == ()
    k.confirm_relationship(r.id)
    assert k.confident_relationships(Confidence.CONFIRMED) == (k.relationship(r.id),)
    data = k.to_dict()
    assert data["schema"] == 2 and len(data["relationships"]) == 1
    copy = Knowledge.from_dict(data)
    assert copy.relationship(r.id) == k.relationship(r.id)
    assert Knowledge.from_dict({"game_id": "old", "schema": 1}).relationships() == ()
    k.remove_relationship(r.id)
    with pytest.raises(ConfigurationError):
        k.relationship(r.id)
    with pytest.raises(ConfigurationError):
        k.remove_relationship(r.id)


def test_keys_and_descriptions() -> None:
    k = Knowledge("g")
    k.propose(RAM, 0, kind=VariableKind.PROGRESS, name="Checkpoint")
    assert event_key(ev(EventKind.PROGRESS, 1)) == "progress:v1"
    assert event_key(ev(EventKind.RESET, 1, None)) == "reset:*"
    assert describe_key("progress:v1", k) == "Progress of Checkpoint"
    assert describe_key("progress:v9", k) == "Progress of v9"
    assert describe_key("reset:*") == "Reset"
    assert describe_key("weird:v1") == "weird of v1"


# --------------------------------------------------------------- finder


def test_finder_counts_pairs_inside_the_window_only() -> None:
    k = Knowledge("g")
    f = RelationshipFinder(k, window=3)
    assert f.observe([ev(EventKind.PROGRESS, 0)]) == ()
    (rel,) = f.observe([ev(EventKind.SCORE_GAIN, 2, "v2")])
    assert rel.id == "progress:v1->score_gain:v2" and rel.support == 1
    assert f.observe([ev(EventKind.SCORE_GAIN, 10, "v2")]) == ()  # too late to pair
    f.observe([ev(EventKind.PROGRESS, 20)])
    (rel,) = f.observe([ev(EventKind.SCORE_GAIN, 23, "v2")])
    assert (rel.support, rel.cause_count, rel.effect_count) == (2, 2, 3)
    assert rel.follow_rate == 1.0 and rel.precede_rate == pytest.approx(2 / 3)
    assert f.seen == 5
    f.reset()
    assert f.seen == 0


def test_finder_same_key_and_same_frame_rules() -> None:
    k = Knowledge("g")
    f = RelationshipFinder(k, window=5)
    f.observe([ev(EventKind.PROGRESS, 0), ev(EventKind.PROGRESS, 0)])  # same key: no pair
    assert k.relationships() == ()
    f.observe([ev(EventKind.SETBACK, 1, "v2"), ev(EventKind.FLAG_ON, 1, "v3")])
    ids = {r.id for r in k.relationships()}
    assert ids == {
        "progress:v1->setback:v2",
        "progress:v1->flag_on:v3",
        "setback:v2->flag_on:v3",
        "flag_on:v3->setback:v2",  # same frame: both ways
    }
    # a second cause occurrence inside the window does not double count one effect
    f.observe([ev(EventKind.PROGRESS, 2)])
    f.observe([ev(EventKind.PROGRESS, 3)])
    f.observe([ev(EventKind.SETBACK, 4, "v2")])
    assert k.relationship("progress:v1->setback:v2").support == 2
    with pytest.raises(ConfigurationError):
        f.observe(["nope"])  # type: ignore[list-item]
    with pytest.raises(ConfigurationError):
        RelationshipFinder(k, window=-1)


def test_rates_refresh_when_only_counts_change() -> None:
    k = Knowledge("g")
    f = RelationshipFinder(k, window=2)
    f.observe([ev(EventKind.PROGRESS, 0)])
    f.observe([ev(EventKind.SCORE_GAIN, 1, "v2")])
    for frame in range(10, 20):
        f.observe([ev(EventKind.PROGRESS, frame * 10)])  # progress without any gain
    rel = k.relationship("progress:v1->score_gain:v2")
    assert rel.cause_count == 11 and rel.follow_rate == pytest.approx(1 / 11)
    assert rel.confidence is Confidence.UNKNOWN


# ------------------------------------------------ on the real toy racer


def test_toy_racer_relationships() -> None:
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
    checkpoint = k.at(RAM, LAYOUT.checkpoint)
    lives = k.at(RAM, LAYOUT.lives)
    flag = k.at(RAM, LAYOUT.game_over)
    score = k.at(RAM, LAYOUT.score)
    assert checkpoint and lives and flag and score
    game = ToyRacer(seed=2)
    d = EventDetector(k)
    f = RelationshipFinder(k, window=2)
    d.observe(MemorySnapshot(0, {RAM: game.ram}))
    for i in range(1, 2400):
        game.step(["start"] if game.state.game_over else ["up"])
        f.observe(d.observe(MemorySnapshot(i, {RAM: game.ram})))
    bonus = k.relationship(f"progress:{checkpoint.id}->score_gain:{score.id}")
    assert bonus.support >= 3 and bonus.follow_rate >= 0.6, bonus
    assert bonus.confidence >= Confidence.LOW
    game_over = k.relationship(f"setback:{lives.id}->flag_on:{flag.id}")
    assert game_over.support >= 1 and game_over.precede_rate == 1.0, game_over
    restart = k.relationship(f"flag_on:{flag.id}->reset:*")
    assert restart.precede_rate == 1.0
    text = describe_key(bonus.cause, k) + " leads to " + describe_key(bonus.effect, k)
    assert text == "Progress of Checkpoint leads to Score gain of Score"
