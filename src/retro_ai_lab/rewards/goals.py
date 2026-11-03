"""Goals and objectives (DESIGN.md §4).

A :class:`Goal` is the big thing a child wants the AI to do ("Win a racing
championship"), broken into :class:`Objective` steps. Each objective has a
plain description, a detection method (a :class:`Trigger` on a game event),
a reward for completing it, an optional penalty for a setback before it is
done, a completion status, a confidence level, an on/off setting and the
objectives it depends on. Children can edit all of it.

``suggest_goal`` is the AI's first go at breaking a goal down from what it
believes about the game (§4.1 "where it has sufficient information"): one
objective per believed progress variable, ordered by the relationships it
has noticed. ``merge_suggestions`` adds new ones later without touching a
person's edits.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field, replace
from typing import Any

from retro_ai_lab.core import ConfigurationError
from retro_ai_lab.rewards.model import MAX_VALUE, Trigger
from retro_ai_lab.understanding.events import EventKind, GameEvent
from retro_ai_lab.understanding.knowledge import Confidence, Knowledge, VariableKind

SCHEMA_VERSION = 1
DEFAULT_REWARD = 25.0  # finishing a step of the goal matters more than a point

GOAL_IDEAS: tuple[str, ...] = (
    "Become the champion",
    "Finish the adventure",
    "Complete every mission",
    "Win the championship",
    "Discover everything in the game",
    "Get as far as you can",
)


def _check_value(value: float, what: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ConfigurationError(f"{what} must be a number.") from exc
    if not math.isfinite(number) or not -MAX_VALUE <= number <= MAX_VALUE:
        raise ConfigurationError(f"{what} must be between -{MAX_VALUE:g} and {MAX_VALUE:g}.")
    return number


@dataclass(frozen=True, slots=True)
class Objective:
    id: str
    description: str
    trigger: Trigger | None = None  # how completion is detected; None = only by hand
    reward: float = DEFAULT_REWARD
    penalty: float = 0.0  # applied (as a negative number) when the setback happens first
    setback_trigger: Trigger | None = None
    completed: bool = False
    completed_frame: int | None = None
    confidence: Confidence = Confidence.UNKNOWN  # how sure the detection method is
    enabled: bool = True
    depends_on: tuple[str, ...] = ()
    notes: str = ""
    suggested: bool = False  # made by the AI (a person's edits clear this)

    def __post_init__(self) -> None:
        if not self.id or not isinstance(self.id, str):
            raise ConfigurationError("Objective needs an id.")
        if not self.description or not self.description.strip():
            raise ConfigurationError("Objective needs a description.")
        if self.trigger is not None and not isinstance(self.trigger, Trigger):
            raise ConfigurationError("Objective trigger must be a Trigger.")
        if self.setback_trigger is not None and not isinstance(self.setback_trigger, Trigger):
            raise ConfigurationError("Objective setback trigger must be a Trigger.")
        if not isinstance(self.confidence, Confidence):
            raise ConfigurationError("Objective confidence must be a Confidence.")
        object.__setattr__(self, "reward", _check_value(self.reward, "Reward"))
        penalty = _check_value(self.penalty, "Penalty")
        if penalty > 0:
            raise ConfigurationError("Penalty must be zero or negative.")
        object.__setattr__(self, "penalty", penalty)
        deps = tuple(dict.fromkeys(self.depends_on))
        if self.id in deps:
            raise ConfigurationError("An objective can't depend on itself.")
        object.__setattr__(self, "depends_on", deps)

    def replace(self, **changes: Any) -> Objective:
        return replace(self, **changes)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "description": self.description,
            "trigger": self.trigger.to_dict() if self.trigger else None,
            "reward": self.reward,
            "penalty": self.penalty,
            "setback_trigger": self.setback_trigger.to_dict() if self.setback_trigger else None,
            "completed": self.completed,
            "completed_frame": self.completed_frame,
            "confidence": self.confidence.value,
            "enabled": self.enabled,
            "depends_on": list(self.depends_on),
            "notes": self.notes,
            "suggested": self.suggested,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Objective:
        try:
            trigger = data.get("trigger")
            setback = data.get("setback_trigger")
            return cls(
                data["id"],
                data["description"],
                Trigger.from_dict(trigger) if trigger else None,
                reward=data.get("reward", DEFAULT_REWARD),
                penalty=data.get("penalty", 0.0),
                setback_trigger=Trigger.from_dict(setback) if setback else None,
                completed=bool(data.get("completed", False)),
                completed_frame=data.get("completed_frame"),
                confidence=Confidence(data.get("confidence", Confidence.UNKNOWN.value)),
                enabled=bool(data.get("enabled", True)),
                depends_on=tuple(data.get("depends_on", ())),
                notes=str(data.get("notes", "")),
                suggested=bool(data.get("suggested", False)),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ConfigurationError(f"Bad objective: {exc}") from exc


@dataclass(frozen=True, slots=True)
class GoalOutcome:
    """What one event did to the goal."""

    reward: float
    completed: tuple[Objective, ...]
    penalised: tuple[Objective, ...]

    @property
    def is_empty(self) -> bool:
        return not (self.completed or self.penalised)


@dataclass(slots=True)
class Goal:
    game_id: str
    title: str = ""
    notes: str = ""
    _items: dict[str, Objective] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.game_id or not isinstance(self.game_id, str):
            raise ConfigurationError("Goal needs a game id.")

    # ------------------------------------------------------------ container

    def __len__(self) -> int:
        return len(self._items)

    def __iter__(self) -> Iterator[Objective]:
        return iter(tuple(self._items.values()))

    def __contains__(self, objective_id: object) -> bool:
        return objective_id in self._items

    def ids(self) -> tuple[str, ...]:
        return tuple(self._items)

    def get(self, objective_id: str) -> Objective:
        try:
            return self._items[objective_id]
        except KeyError:
            raise ConfigurationError(f"No objective {objective_id!r}.") from None

    def find(self, objective_id: str) -> Objective | None:
        return self._items.get(objective_id)

    def new_id(self) -> str:
        n = 1
        while f"obj{n}" in self._items:
            n += 1
        return f"obj{n}"

    def add(self, objective: Objective) -> Objective:
        if not isinstance(objective, Objective):
            raise ConfigurationError("Only an Objective can be added.")
        if objective.id in self._items:
            raise ConfigurationError(f"Objective {objective.id!r} is already in the goal.")
        self._check_dependencies(objective.id, objective.depends_on)
        self._items[objective.id] = objective
        return objective

    def remove(self, objective_id: str) -> None:
        self.get(objective_id)
        del self._items[objective_id]
        for oid, o in list(self._items.items()):
            if objective_id in o.depends_on:
                self._items[oid] = o.replace(
                    depends_on=tuple(d for d in o.depends_on if d != objective_id)
                )

    def update(self, objective_id: str, **changes: Any) -> Objective:
        """A person's edit (description, triggers, reward, penalty, enabled, notes, confidence)."""
        allowed = {
            "description",
            "trigger",
            "reward",
            "penalty",
            "setback_trigger",
            "enabled",
            "notes",
            "confidence",
        }
        unknown = set(changes) - allowed
        if unknown:
            raise ConfigurationError(f"Can't update {sorted(unknown)}.")
        current = self.get(objective_id)
        updated = current.replace(**changes, suggested=False)
        self._items[objective_id] = updated
        return updated

    def reword(self, objective_id: str, description: str) -> Objective:
        """Change the wording only, keeping whether the AI suggested it."""
        current = self.get(objective_id)
        updated = current.replace(description=description)
        self._items[objective_id] = updated
        return updated

    def move(self, objective_id: str, position: int) -> None:
        """Put an objective at ``position`` in the list order."""
        self.get(objective_id)
        ids = [i for i in self._items if i != objective_id]
        if not 0 <= position <= len(ids):
            raise ConfigurationError("position is out of range.")
        ids.insert(position, objective_id)
        self._items = {i: self._items[i] for i in ids}

    def set_dependencies(self, objective_id: str, depends_on: Iterable[str]) -> Objective:
        deps = tuple(dict.fromkeys(depends_on))
        self._check_dependencies(objective_id, deps)
        updated = self.get(objective_id).replace(depends_on=deps)
        self._items[objective_id] = updated
        return updated

    def _check_dependencies(self, objective_id: str, deps: tuple[str, ...]) -> None:
        for d in deps:
            if d == objective_id:
                raise ConfigurationError("An objective can't depend on itself.")
            if d not in self._items:
                raise ConfigurationError(f"Unknown objective {d!r} in dependencies.")
        # Would objective_id be reachable from any of its new dependencies? Then it's a cycle.
        seen: set[str] = set()
        stack = list(deps)
        while stack:
            current = stack.pop()
            if current == objective_id:
                raise ConfigurationError("That would make the objectives depend on each other.")
            if current in seen:
                continue
            seen.add(current)
            node = self._items.get(current)
            if node is not None:
                stack.extend(node.depends_on)

    # ------------------------------------------------------------ progress

    def completed_ids(self) -> frozenset[str]:
        return frozenset(i for i, o in self._items.items() if o.completed)

    def is_ready(self, objective_id: str) -> bool:
        """Enabled, not done, and every dependency done."""
        o = self.get(objective_id)
        done = self.completed_ids()
        return o.enabled and not o.completed and all(d in done for d in o.depends_on)

    def ready(self) -> tuple[Objective, ...]:
        return tuple(o for o in self._items.values() if self.is_ready(o.id))

    def next(self) -> Objective | None:
        """The first objective the AI should be working on."""
        for o in self.ready():
            return o
        return None

    def progress(self) -> tuple[int, int]:
        """(done, total) over enabled objectives."""
        enabled = [o for o in self._items.values() if o.enabled]
        return sum(1 for o in enabled if o.completed), len(enabled)

    @property
    def is_complete(self) -> bool:
        done, total = self.progress()
        return total > 0 and done == total

    def complete(self, objective_id: str, frame: int | None = None) -> Objective:
        updated = self.get(objective_id).replace(completed=True, completed_frame=frame)
        self._items[objective_id] = updated
        return updated

    def uncomplete(self, objective_id: str) -> Objective:
        updated = self.get(objective_id).replace(completed=False, completed_frame=None)
        self._items[objective_id] = updated
        return updated

    def reset_progress(self) -> None:
        for oid in self.ids():
            self.uncomplete(oid)

    def observe(self, event: GameEvent) -> GoalOutcome:
        """Apply one game event: complete ready objectives, penalise setbacks."""
        reward = 0.0
        completed: list[Objective] = []
        penalised: list[Objective] = []
        for oid in self.ids():
            o = self._items[oid]
            if not o.enabled or o.completed:
                continue
            if o.setback_trigger is not None and o.penalty < 0 and o.setback_trigger.matches(event):
                reward += o.penalty
                penalised.append(o)
            if o.trigger is not None and self.is_ready(oid) and o.trigger.matches(event):
                done = self.complete(oid, event.frame)
                reward += done.reward
                completed.append(done)
        return GoalOutcome(reward, tuple(completed), tuple(penalised))

    # ------------------------------------------------------------ persistence

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA_VERSION,
            "game_id": self.game_id,
            "title": self.title,
            "notes": self.notes,
            "objectives": [o.to_dict() for o in self._items.values()],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Goal:
        try:
            schema = int(data.get("schema", SCHEMA_VERSION))
            if schema > SCHEMA_VERSION:
                raise ConfigurationError(
                    f"This goal file is from a newer version (schema {schema})."
                )
            goal = cls(
                data["game_id"], title=str(data.get("title", "")), notes=str(data.get("notes", ""))
            )
            items = [Objective.from_dict(item) for item in data.get("objectives", [])]
            # Dependencies may point forward; add without them, then set them.
            for o in items:
                goal.add(o.replace(depends_on=()))
            for o in items:
                goal.set_dependencies(o.id, o.depends_on)
        except (KeyError, TypeError, AttributeError) as exc:
            raise ConfigurationError(f"Bad goal data: {exc}") from exc
        return goal


# ------------------------------------------------------------ suggestions


def suggest_goal(
    knowledge: Knowledge, *, title: str = "", at_least: Confidence = Confidence.LOW
) -> Goal:
    """The AI's breakdown of a goal from the progress it believes it can see."""
    goal = Goal(knowledge.game_id, title=title or GOAL_IDEAS[-1])
    merge_suggestions(goal, knowledge, at_least=at_least)
    return goal


def merge_suggestions(
    goal: Goal, knowledge: Knowledge, *, at_least: Confidence = Confidence.LOW
) -> tuple[str, ...]:
    """Add suggested objectives the goal lacks and retire stale ones. Returns new ids.

    Objectives the child wrote are never touched. A suggested one is removed when
    its variable is known but no longer believed to track progress (the AI changed
    its mind, so any "completion" it saw was a misreading); its wording follows
    the variable's name.
    """
    progress = [
        v
        for v in knowledge
        if v.kind is VariableKind.PROGRESS and not v.rejected and v.confidence >= at_least
    ]
    keys = {Trigger(EventKind.PROGRESS, v.id).key: v for v in progress}
    for o in list(goal):
        if not o.suggested or o.trigger is None or o.trigger.kind is not EventKind.PROGRESS:
            continue
        if o.trigger.key not in keys:
            vid = o.trigger.variable_id
            if vid is not None and vid in knowledge:
                goal.remove(o.id)
            continue
        wanted = f"Get further in {keys[o.trigger.key].display_name}"
        if o.description != wanted:
            goal.reword(o.id, wanted)
    # Order by relationships: a cause comes before its effect.
    order = {k: i for i, k in enumerate(keys)}
    for rel in knowledge.confident_relationships(Confidence.LOW):
        if rel.cause in order and rel.effect in order and order[rel.cause] > order[rel.effect]:
            order[rel.cause], order[rel.effect] = order[rel.effect], order[rel.cause]
    ordered = sorted(keys, key=lambda k: order[k])
    existing = {o.trigger.key for o in goal if o.trigger is not None}
    added: list[str] = []
    previous: str | None = None
    for key in ordered:
        var = keys[key]
        if key in existing:
            previous = next(o.id for o in goal if o.trigger is not None and o.trigger.key == key)
            continue
        oid = f"goal:{key}"
        deps = (previous,) if previous is not None and previous in goal else ()
        goal.add(
            Objective(
                oid,
                f"Get further in {var.display_name}",
                Trigger(EventKind.PROGRESS, var.id),
                confidence=var.confidence,
                depends_on=deps,
                notes="Suggested by the AI from what it has seen so far.",
                suggested=True,
            )
        )
        added.append(oid)
        previous = oid
    return tuple(added)


__all__ = [
    "DEFAULT_REWARD",
    "GOAL_IDEAS",
    "SCHEMA_VERSION",
    "Goal",
    "GoalOutcome",
    "Objective",
    "merge_suggestions",
    "suggest_goal",
]
