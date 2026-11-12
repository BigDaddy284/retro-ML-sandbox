"""T-402: behaviours and suggested values derived from the knowledge."""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from retro_ai_lab.emulator.fake import ToyRacer
from retro_ai_lab.emulator.fake.toy_racer import LAYOUT
from retro_ai_lab.observation import MemorySnapshot
from retro_ai_lab.rewards import (
    Behaviour,
    Category,
    RewardProfile,
    Trigger,
    apply_suggestions,
    rule_key,
    suggest_behaviours,
)
from retro_ai_lab.rewards.suggestions import RelationshipWatch
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
    mid = k.propose(RAM, 1, kind=VariableKind.PROGRESS, name="B", evidence=HALF)
    high = k.propose(RAM, 2, kind=VariableKind.PROGRESS, name="C", evidence=SURE)
    sure = k.propose(RAM, 3, kind=VariableKind.PROGRESS, name="D", evidence=MAYBE)
    k.confirm(sure.id)
    got = by_key(suggest_behaviours(k))
    alone = Knowledge("h")  # only three progress guesses are offered at once (T-858)
    low = alone.propose(RAM, 0, kind=VariableKind.PROGRESS, name="A", evidence=MAYBE)
    assert by_key(suggest_behaviours(alone))[f"progress:{low.id}"].suggested == 5.0
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


def test_only_the_best_guess_of_each_role_is_rewarded_in_plain_words() -> None:
    """T-858: 100 believed scores suggest one behaviour, the best, named "Your score"."""
    k = Knowledge("g")
    for i in range(100):
        k.propose(RAM, i, kind=VariableKind.SCORE, evidence=[Evidence("t", "", 1.0)])
    best = k.propose(RAM, 100, kind=VariableKind.SCORE, evidence=SURE)
    suggested = by_key(suggest_behaviours(k))
    assert [key for key in suggested if key.startswith("score_gain:")] == [f"score_gain:{best.id}"]
    assert suggested[f"score_gain:{best.id}"].name == "Your score jumps up"
    # A confirmed guess beats a stronger-looking unconfirmed one, and a person's name is kept.
    told = k.propose(RAM, 101, kind=VariableKind.SCORE, evidence=[Evidence("t", "", 1.0)])
    k.confirm(told.id)
    k.correct(told.id, name="Coins")
    suggested = by_key(suggest_behaviours(k))
    assert [key for key in suggested if key.startswith("score_gain:")] == [f"score_gain:{told.id}"]
    assert suggested[f"score_gain:{told.id}"].name == "Coins jumps up"
    # Lives read as "a life"; two unnamed progress guesses say which guess they are.
    lives = k.propose(RAM, 102, kind=VariableKind.LIVES, evidence=SURE)
    first = k.propose(RAM, 103, kind=VariableKind.PROGRESS, evidence=SURE)
    second = k.propose(RAM, 104, kind=VariableKind.PROGRESS, evidence=HALF)
    suggested = by_key(suggest_behaviours(k))
    assert suggested[f"setback:{lives.id}"].name == "Losing a life"
    assert suggested[f"recovery:{lives.id}"].name == "Gaining a life"
    assert suggested[f"progress:{first.id}"].name == "Progress goes up"
    assert suggested[f"progress:{second.id}"].name == "Progress (2nd guess) goes up"


def test_a_dismissed_rule_is_not_suggested_again() -> None:
    """T-943: the user removed rows and "it auto readds it": a removed rule stays removed,
    whichever guess of that kind the AI believes next."""
    k = Knowledge("g")
    first = k.propose(RAM, 0, kind=VariableKind.LIVES, name="Lives", evidence=MAYBE)
    p = RewardProfile("g")
    apply_suggestions(p, k)
    assert f"setback:{first.id}" in p and f"recovery:{first.id}" in p
    p.dismiss(rule_key(EventKind.SETBACK, VariableKind.LIVES))
    report = apply_suggestions(p, k)
    assert report.removed == (f"setback:{first.id}",) and f"recovery:{first.id}" in p
    # A surer lives guess takes the seat: still no "Losing" row.
    second = k.propose(RAM, 1, kind=VariableKind.LIVES, name="Hearts", evidence=SURE)
    report = apply_suggestions(p, k)
    assert f"setback:{second.id}" not in p and f"recovery:{second.id}" in p
    assert f"setback:{first.id}" not in report.added
    # The reset row can go the same way, and bringing back undoes it all.
    p.dismiss("reset:*")
    apply_suggestions(p, k)
    assert "reset:*" not in p
    assert set(p.bring_back()) == {"setback:<lives>", "reset:*"} and p.dismissed == frozenset()
    report = apply_suggestions(p, k)
    assert set(report.added) == {f"setback:{second.id}", "reset:*"}
    assert rule_key(EventKind.PROGRESS, VariableKind.PROGRESS) == "progress:<progress>"


# ------------------------------------------------------------- T-948 relationship watch


@st.composite
def _relationship_edits(draw: st.DrawFn) -> list[tuple[str, str, int, int, int]]:
    """Writes, confirms, rejects and removals on a few pairs, with watch reads between."""
    edits = []
    for _ in range(draw(st.integers(1, 40))):
        op = draw(st.sampled_from(["set", "set", "set", "confirm", "reject", "remove", "read"]))
        pair = draw(st.sampled_from(["a->b", "a->c", "b->c"]))
        edits.append(
            (op, pair, draw(st.integers(0, 12)), draw(st.integers(0, 12)), draw(st.integers(0, 12)))
        )
    return edits


def _old_rule(k: Knowledge) -> frozenset[tuple[str, Confidence]]:
    return frozenset((r.id, r.confidence) for r in k.confident_relationships())


@settings(max_examples=80, deadline=None)
@given(edits=_relationship_edits())
def test_the_watch_reports_every_change_the_old_walk_found_and_no_more_than_the_edits_made(
    edits: list[tuple[str, str, int, int, int]],
) -> None:
    k = Knowledge("g")
    watch = RelationshipWatch()
    assert watch.changed(k)  # a knowledge it has not seen
    last_seen = _old_rule(k)
    moved = False  # did any edit since the last read change what the suggestions read?
    for op, pair, support, cause_count, effect_count in edits:
        before = _old_rule(k)
        cause, _, effect = pair.partition("->")
        if op == "set":
            cause_count = max(cause_count, support)
            effect_count = max(effect_count, support)
            k.set_relationship(Relationship(cause, effect, support, cause_count, effect_count, 5))
        elif pair in {r.id for r in k.relationships()}:
            if op == "confirm":
                k.confirm_relationship(pair)
            elif op == "reject":
                k.reject_relationship(pair)
            elif op == "remove":
                k.remove_relationship(pair)
        if op != "read":
            moved = moved or _old_rule(k) != before
            continue
        now = _old_rule(k)
        changed = watch.changed(k)
        if now != last_seen:
            assert changed  # the old walk would have said so
        assert changed == moved  # and nothing says so unless an edit moved the confident ones
        last_seen = now
        moved = False
    assert not watch.changed(k) or moved
    watch.reset()
    assert watch.changed(k)
    assert watch.changed(Knowledge("g"))  # a fresh knowledge object counts as a change


def test_a_switch_row_says_it_counts_once_a_go() -> None:
    """T-994: a switch pays the first time it goes that way in a go, and its row says so."""
    k = Knowledge("g")
    door = k.propose(RAM, 0, kind=VariableKind.FLAG, name="Door", evidence=SURE)
    unsure = k.propose(RAM, 1, kind=VariableKind.FLAG, name="Lamp", evidence=MAYBE)
    lives = k.propose(RAM, 2, kind=VariableKind.LIVES, name="Lives", evidence=SURE)
    lap = k.propose(RAM, 3, kind=VariableKind.PROGRESS, name="Lap", evidence=SURE)
    got = by_key(suggest_behaviours(k))
    once = "Counts the first time in a go."
    assert got[f"flag_on:{door.id}"].notes == once
    assert got[f"flag_off:{door.id}"].notes == once
    # After what else there is to say about the row.
    assert got[f"flag_on:{unsure.id}"].notes == (
        "The AI is not sure yet; the value is smaller until it is. " + once
    )
    for key in (f"setback:{lives.id}", f"recovery:{lives.id}", f"progress:{lap.id}", "reset:*"):
        assert once not in got[key].notes, key
