"""Suggested behaviours and reward values from the game knowledge.

DESIGN.md §5.4: the software recommends reward values from how important
things look, and the recommendations stay editable. Importance here comes
from three signals the understanding engine already produces:

* what kind of thing a variable is (progress matters more than a coin);
* how sure the AI is about it (shaky beliefs get smaller values);
* what it leads to (a believed relationship whose effect is progress makes
  the cause worth more; a flag that leads to a reset looks like game over).

Behaviour ids are the trigger key (``"<event kind>:<variable id>"``), so a
suggestion made twice is the same behaviour, and behaviours a person makes
themselves (any other id) are never touched.
"""

from __future__ import annotations

from dataclasses import dataclass

from retro_ai_lab.rewards.model import Behaviour, Category, RewardProfile, Trigger, clamp
from retro_ai_lab.understanding.events import EventKind
from retro_ai_lab.understanding.knowledge import (
    Confidence,
    Knowledge,
    Variable,
    VariableKind,
    plain_names,
)

# (event kind, variable kind) -> (category, base value, name template)
RULES: dict[tuple[EventKind, VariableKind], tuple[Category, float, str]] = {
    (EventKind.PROGRESS, VariableKind.PROGRESS): (Category.PROGRESSION, 10.0, "{name} goes up"),
    (EventKind.SCORE_GAIN, VariableKind.SCORE): (Category.COLLECTION, 5.0, "{name} jumps up"),
    (EventKind.SETBACK, VariableKind.LIVES): (Category.SURVIVAL, -10.0, "Losing {name}"),
    (EventKind.RECOVERY, VariableKind.LIVES): (Category.SURVIVAL, 5.0, "Gaining {name}"),
    (EventKind.SETBACK, VariableKind.HEALTH): (Category.SURVIVAL, -3.0, "{name} drops"),
    (EventKind.RECOVERY, VariableKind.HEALTH): (Category.SURVIVAL, 2.0, "{name} recovers"),
    (EventKind.FLAG_ON, VariableKind.FLAG): (Category.QUESTS, 2.0, "{name} switches on"),
    (EventKind.FLAG_OFF, VariableKind.FLAG): (Category.QUESTS, 0.0, "{name} switches off"),
}
RESET_KEY = "reset:*"
RESET_RULE = (Category.SURVIVAL, -5.0, "Starting over")
GAME_OVER_RULE = (Category.SURVIVAL, -15.0, "{name} switches on (game over?)")

CONFIDENCE_SCALE: dict[Confidence, float] = {
    Confidence.LOW: 0.5,
    Confidence.MEDIUM: 0.75,
    Confidence.HIGH: 1.0,
    Confidence.CONFIRMED: 1.0,
}
PROGRESS_BOOST = 1.5
UNSURE_NOTE = "The AI is not sure yet; the value is smaller until it is."
PROGRESS_NOTE = "This leads to progress, so it is worth more."
GAME_OVER_NOTE = "This is followed by everything starting over, so it looks like game over."


def _key(kind: EventKind, variable_id: str | None) -> str:
    return Trigger(kind, variable_id).key


def _causes(knowledge: Knowledge) -> tuple[set[str], set[str]]:
    """Trigger keys that lead to progress, and flags that lead to a reset."""
    progress: set[str] = set()
    game_over: set[str] = set()
    progress_keys = {
        _key(EventKind.PROGRESS, v.id)
        for v in knowledge.believed()
        if v.kind is VariableKind.PROGRESS
    }
    for rel in knowledge.confident_relationships(Confidence.LOW):
        if rel.effect in progress_keys:
            progress.add(rel.cause)
        if rel.effect == RESET_KEY and rel.cause.startswith(EventKind.FLAG_ON.value + ":"):
            game_over.add(rel.cause)
    return progress, game_over


def suggest_behaviours(
    knowledge: Knowledge, *, at_least: Confidence = Confidence.LOW
) -> tuple[Behaviour, ...]:
    """Behaviours the knowledge supports, each with a suggested value."""
    leads_to_progress, game_over_flags = _causes(knowledge)
    out: list[Behaviour] = []
    offered = knowledge.offered(at_least)  # the best guess of each role (T-858)
    names = plain_names(offered)
    for var in offered:
        for (event_kind, var_kind), rule in RULES.items():
            if var_kind is not var.kind:
                continue
            out.append(
                _build(var, names[var.id], event_kind, rule, leads_to_progress, game_over_flags)
            )
    out.append(
        Behaviour(
            RESET_KEY,
            RESET_RULE[2],
            RESET_RULE[0],
            Trigger(EventKind.RESET),
            suggested=RESET_RULE[1],
            value=RESET_RULE[1],
            notes="Everything went back to the start, like after a game over.",
        )
    )
    return tuple(out)


def _build(
    var: Variable,
    name: str,
    event_kind: EventKind,
    rule: tuple[Category, float, str],
    leads_to_progress: set[str],
    game_over_flags: set[str],
) -> Behaviour:
    key = _key(event_kind, var.id)
    category, base, template = rule
    notes: list[str] = []
    if key in game_over_flags:
        category, base, template = GAME_OVER_RULE
        notes.append(GAME_OVER_NOTE)
    elif key in leads_to_progress:
        base *= PROGRESS_BOOST
        notes.append(PROGRESS_NOTE)
    scale = CONFIDENCE_SCALE[var.confidence]
    if scale < 1.0:
        notes.append(UNSURE_NOTE)
    value = clamp(round(base * scale, 2))
    return Behaviour(
        key,
        template.format(name=name),
        category,
        Trigger(event_kind, var.id),
        suggested=value,
        value=value,
        notes=" ".join(notes),
    )


class RelationshipWatch:
    """Says whether the relationships the suggestions read have changed (T-842).

    The finder reports changed relationships on most steps of a real game, but
    the suggestions only read which ones are at least LOW and how sure each is,
    and that moves rarely. Rebuilding 2,400 behaviours and 800 objectives on
    every step cost the app more than the emulator's step itself.
    """

    def __init__(self) -> None:
        self._seen: frozenset[tuple[str, Confidence]] | None = None

    def changed(self, knowledge: Knowledge) -> bool:
        """True (once) when the confident relationships differ from the last call."""
        now = frozenset((r.id, r.confidence) for r in knowledge.confident_relationships())
        if now == self._seen:
            return False
        self._seen = now
        return True

    def reset(self) -> None:
        self._seen = None


@dataclass(frozen=True, slots=True)
class SuggestionReport:
    added: tuple[str, ...]
    updated: tuple[str, ...]
    removed: tuple[str, ...]

    @property
    def is_empty(self) -> bool:
        return not (self.added or self.updated or self.removed)


def is_automatic(behaviour_id: str) -> bool:
    """Automatic behaviours are named after their trigger key."""
    kind_value, sep, _ = behaviour_id.partition(":")
    if not sep:
        return False
    try:
        EventKind(kind_value)
    except ValueError:
        return False
    return True


def apply_suggestions(
    profile: RewardProfile, knowledge: Knowledge, *, at_least: Confidence = Confidence.LOW
) -> SuggestionReport:
    """Bring the profile in line with the knowledge without losing a person's choices.

    New behaviours are added; changed suggestions are re-suggested (a value
    a person set stays); names, categories and notes are refreshed. An
    automatic behaviour whose variable is gone or rejected is removed, or
    only disabled if a person set its value.
    """
    wanted = {b.id: b for b in suggest_behaviours(knowledge, at_least=at_least)}
    added: list[str] = []
    updated: list[str] = []
    removed: list[str] = []
    for bid, fresh in wanted.items():
        current = profile.find(bid)
        if current is None:
            profile.add(fresh)
            profile.reset_to_suggested(bid)  # apply the profile's personality weights
            added.append(bid)
            continue
        changed = False
        if current.suggested != fresh.suggested:
            profile.suggest(bid, fresh.suggested)
            changed = True
        current = profile.get(bid)
        if (current.name, current.category, current.notes) != (
            fresh.name,
            fresh.category,
            fresh.notes,
        ):
            profile.update(bid, name=fresh.name, category=fresh.category, notes=fresh.notes)
            changed = True
        if changed:
            updated.append(bid)
    for bid in profile.ids():
        if bid in wanted or not is_automatic(bid):
            continue
        current = profile.get(bid)
        if current.custom:
            if current.enabled:
                profile.set_enabled(bid, False)
                updated.append(bid)
        else:
            profile.remove(bid)
            removed.append(bid)
    return SuggestionReport(tuple(added), tuple(updated), tuple(removed))


__all__ = [
    "CONFIDENCE_SCALE",
    "PROGRESS_BOOST",
    "RESET_KEY",
    "RULES",
    "RelationshipWatch",
    "SuggestionReport",
    "apply_suggestions",
    "is_automatic",
    "suggest_behaviours",
]
