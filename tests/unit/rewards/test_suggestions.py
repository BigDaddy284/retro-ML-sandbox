"""T-402: behaviours and suggested values derived from the knowledge."""

from __future__ import annotations

from retro_ai_lab.emulator.fake import ToyRacer
from retro_ai_lab.emulator.fake.toy_racer import LAYOUT
from retro_ai_lab.observation import MemorySnapshot
from retro_ai_lab.rewards import (
    Behaviour,
    Category,
    RewardProfile,
    Trigger,
    apply_suggestions,
    suggest_behaviours,
)
from retro_ai_lab.understanding import (
    Confidence,
    Encoding,
    EventDetector,
    EventKind,
    Evidence,
    Knowledge,
    Relationship,
    VariableKind,
)

RAM = "ram"
SURE = [Evidence("test", "", 6.0)]  # High
MAYBE = [Evidence("test", "", 1.0)]  # Low
HALF = [Evidence("test", "", 3.0)]  # Medium


def by_key(behaviours: tuple[Behaviour, ...]) -> dict[str, Behaviour]:
    return {b.id: b for b in behaviours}


def test_each_variable_kind_gives_the_expected_behaviours() -> None:
    k = Knowledge("g")
    k.propose(RAM, 0, kind=VariableKind.PROGRESS, name="Lap", evidence=SURE)
    k.propose(RAM, 1, kind=VariableKind.SCORE, name="Score", evidence=SURE)
    k.propose(RAM, 2, kind=VariableKind.LIVES, name="Lives", evidence=SURE)
    k.propose(RAM, 3, kind=VariableKind.HEALTH, name="Health", evidence=SURE)
    k.propose(RAM, 4, kind=VariableKind.FLAG, name="Door", evidence=SURE)
    k.propose(RAM, 5, kind=VariableKind.POSITION, name="X", evidence=SURE)
    k.propose(RAM, 6, kind=VariableKind.UNKNOWN, evidence=SURE)
    ids = {v.name: v.id for v in k}
    got = by_key(suggest_behaviours(k))
    expected = {
        f"progress:{ids['Lap']}": (Category.PROGRESSION, 10.0, "Lap goes up"),
        f"score_gain:{ids['Score']}": (Category.COLLECTION, 5.0, "Score jumps up"),
        f"setback:{ids['Lives']}": (Category.SURVIVAL, -10.0, "Losing Lives"),
        f"recovery:{ids['Lives']}": (Category.SURVIVAL, 5.0, "Gaining Lives"),
        f"setback:{ids['Health']}": (Category.SURVIVAL, -3.0, "Health drops"),
        f"recovery:{ids['Health']}": (Category.SURVIVAL, 2.0, "Health recovers"),
        f"flag_on:{ids['Door']}": (Category.QUESTS, 2.0, "Door switches on"),
        f"flag_off:{ids['Door']}": (Category.QUESTS, 0.0, "Door switches off"),
        "reset:*": (Category.SURVIVAL, -5.0, "Starting over"),
    }
    assert set(got) == set(expected)
    for key, (category, value, name) in expected.items():
        b = got[key]
        assert (b.category, b.suggested, b.value, b.name) == (category, value, value, name), key
        assert b.enabled and not b.custom
        kind_value, variable_id = key.split(":")
        assert b.trigger.kind.value == kind_value
        assert b.trigger.variable_id == (None if variable_id == "*" else variable_id)


def test_filters_out_uncertain_and_rejected_variables() -> None:
    k = Knowledge("g")
    k.propose(RAM, 0, kind=VariableKind.PROGRESS, name="Lap")  # no evidence: Unknown
    bad = k.propose(RAM, 1, kind=VariableKind.SCORE, name="Score", evidence=SURE)
    k.reject(bad.id)
    low = k.propose(RAM, 2, kind=VariableKind.LIVES, name="Lives", evidence=MAYBE)
    assert set(by_key(suggest_behaviours(k))) == {
        f"setback:{low.id}",
        f"recovery:{low.id}",
        "reset:*",
    }
    assert set(by_key(suggest_behaviours(k, at_least=Confidence.MEDIUM))) == {"reset:*"}
    assert set(by_key(suggest_behaviours(Knowledge("empty")))) == {"reset:*"}


def test_confidence_scales_the_suggested_value() -> None:
    k = Knowledge("g")
    low = k.propose(RAM, 0, kind=VariableKind.PROGRESS, name="A", evidence=MAYBE)
    mid = k.propose(RAM, 1, kind=VariableKind.PROGRESS, name="B", evidence=HALF)
    high = k.propose(RAM, 2, kind=VariableKind.PROGRESS, name="C", evidence=SURE)
    sure = k.propose(RAM, 3, kind=VariableKind.PROGRESS, name="D", evidence=MAYBE)
    k.confirm(sure.id)
    got = by_key(suggest_behaviours(k))
    assert got[f"progress:{low.id}"].suggested == 5.0
    assert got[f"progress:{mid.id}"].suggested == 7.5
    assert got[f"progress:{high.id}"].suggested == 10.0
    assert got[f"progress:{sure.id}"].suggested == 10.0
    assert "not sure yet" in got[f"progress:{low.id}"].notes
    assert "not sure yet" not in got[f"progress:{high.id}"].notes


def test_relationships_boost_causes_of_progress_and_spot_game_over() -> None:
    k = Knowledge("g")
    lap = k.propose(RAM, 0, kind=VariableKind.PROGRESS, name="Lap", evidence=SURE)
    coin = k.propose(RAM, 1, kind=VariableKind.SCORE, name="Coins", evidence=SURE)
    flag = k.propose(RAM, 2, kind=VariableKind.FLAG, name="Mystery flag", evidence=SURE)
    weak = k.propose(RAM, 3, kind=VariableKind.FLAG, name="Other flag", evidence=SURE)
    k.set_relationship(Relationship(f"score_gain:{coin.id}", f"progress:{lap.id}", 8, 8, 9, 5))
    k.set_relationship(Relationship(f"flag_on:{flag.id}", "reset:*", 4, 4, 4, 5))
    # Too weak to count (support 1).
    k.set_relationship(Relationship(f"flag_on:{weak.id}", "reset:*", 1, 1, 4, 5))
    got = by_key(suggest_behaviours(k))
    assert got[f"score_gain:{coin.id}"].suggested == 7.5
    assert "leads to progress" in got[f"score_gain:{coin.id}"].notes
    over = got[f"flag_on:{flag.id}"]
    assert (over.category, over.suggested) == (Category.SURVIVAL, -15.0)
    assert over.name == "Mystery flag switches on (game over?)"
    assert got[f"flag_on:{weak.id}"].suggested == 2.0
    # A rejected relationship is ignored.
    k.reject_relationship(f"score_gain:{coin.id}->progress:{lap.id}")
    assert by_key(suggest_behaviours(k))[f"score_gain:{coin.id}"].suggested == 5.0


def test_apply_adds_updates_and_is_idempotent() -> None:
    k = Knowledge("g")
    lap = k.propose(RAM, 0, kind=VariableKind.PROGRESS, name="Lap", evidence=MAYBE)
    p = RewardProfile("g")
    first = apply_suggestions(p, k)
    assert first.added == (f"progress:{lap.id}", "reset:*")
    assert first.updated == () and first.removed == ()
    assert not first.is_empty
    second = apply_suggestions(p, k)
    assert second.is_empty and (second.added, second.updated, second.removed) == ((), (), ())
    # More evidence: the value follows the new suggestion.
    k.update(lap.id, evidence=tuple(SURE))
    third = apply_suggestions(p, k)
    assert third.updated == (f"progress:{lap.id}",)
    assert p.get(f"progress:{lap.id}").value == 10.0
    assert p.get(f"progress:{lap.id}").suggested == 10.0
    # A rename flows through without counting as a value change.
    k.update(lap.id, name="Round")
    fourth = apply_suggestions(p, k)
    assert fourth.updated == (f"progress:{lap.id}",)
    assert p.get(f"progress:{lap.id}").name == "Round goes up"
    assert apply_suggestions(p, k).is_empty


def test_apply_keeps_a_persons_value_and_their_own_behaviours() -> None:
    k = Knowledge("g")
    lap = k.propose(RAM, 0, kind=VariableKind.PROGRESS, name="Lap", evidence=MAYBE)
    p = RewardProfile("g")
    mine = Behaviour("mine", "Mine", Category.EXPLORATION, Trigger(EventKind.PROGRESS), 1.0, 1.0)
    p.add(mine)
    apply_suggestions(p, k)
    p.set_value(f"progress:{lap.id}", 42.0)
    k.update(lap.id, evidence=tuple(SURE))
    report = apply_suggestions(p, k)
    assert report.updated == (f"progress:{lap.id}",)
    b = p.get(f"progress:{lap.id}")
    assert (b.value, b.suggested, b.custom) == (42.0, 10.0, True)
    assert p.get("mine") == mine
    # Rejecting the variable: a person's behaviour stays (disabled), automatic ones go.
    k.reject(lap.id)
    report = apply_suggestions(p, k)
    assert report.removed == ()
    assert report.updated == (f"progress:{lap.id}",)
    assert not p.get(f"progress:{lap.id}").enabled
    p.reset_to_suggested(f"progress:{lap.id}")
    report = apply_suggestions(p, k)
    assert report.removed == (f"progress:{lap.id}",)
    assert f"progress:{lap.id}" not in p
    assert "mine" in p and p.get("mine") == mine


def test_toy_racer_profile_rewards_checkpoints_and_punishes_crashes() -> None:
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
    p = RewardProfile("toyracer")
    apply_suggestions(p, k)
    assert {b.name for b in p} >= {
        "Checkpoint goes up",
        "Lap goes up",
        "Score jumps up",
        "Losing Lives",
        "Gaining Lives",
        "Game over switches on",
        "Starting over",
    }
    game = ToyRacer(seed=2)
    d = EventDetector(k)
    d.observe(MemorySnapshot(0, {RAM: game.ram}))
    earned: dict[EventKind, float] = {}
    for i in range(1, 900):
        game.step(["start"] if game.state.game_over else ["up"])
        for e in d.observe(MemorySnapshot(i, {RAM: game.ram})):
            earned[e.kind] = earned.get(e.kind, 0.0) + p.reward_for(e)
    assert earned[EventKind.PROGRESS] > 0
    assert earned[EventKind.SETBACK] < 0
    assert earned[EventKind.RESET] < 0


def test_suggestions_cover_the_best_few_of_a_kind() -> None:
    """T-850: 100 believed scores suggest 16 behaviours, not 100."""
    k = Knowledge("g")
    for i in range(100):
        k.propose(RAM, i, kind=VariableKind.SCORE, evidence=[Evidence("t", "", 1.0)])
    best = k.propose(RAM, 100, kind=VariableKind.SCORE, evidence=SURE)
    suggested = by_key(suggest_behaviours(k))
    assert len([key for key in suggested if key.startswith("score_gain:")]) == 16
    assert f"score_gain:{best.id}" in suggested
