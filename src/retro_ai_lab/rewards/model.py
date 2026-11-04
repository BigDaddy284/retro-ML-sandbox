"""The reward model: behaviours a child can reward or penalise.

DESIGN.md §5.1 asks that every behaviour the AI can be judged on has a
``+``, a ``-``, a current value, an enable/disable toggle and a suggested
value, and that children never see the mathematics. So the model is
deliberately plain: a :class:`Behaviour` is a named thing that happens in
the game (a :class:`Trigger` on an event kind and, optionally, one
variable) filed under a :class:`Category` (§5.2), and a
:class:`RewardProfile` is the set of behaviours for one game with the
controls above. ``reward_for`` turns a detected event into a number the
learner (Phase 5) can use.

Suggested values (§5.4) are kept apart from the current value: a
suggestion moves the current value only while nobody has touched it, so a
child's choice is never silently overwritten.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any

from retro_ai_lab.core import ConfigurationError
from retro_ai_lab.understanding.events import EventKind, GameEvent

SCHEMA_VERSION = 1
STEP = 1.0  # what one click of + or - does
MAX_VALUE = 100.0  # values live in [-MAX_VALUE, MAX_VALUE]


class Category(Enum):
    """The groups behaviours are shown under, in DESIGN.md §5.2 order."""

    EXPLORATION = "exploration"
    COLLECTION = "collection"
    COMBAT = "combat"
    SURVIVAL = "survival"
    PROGRESSION = "progression"
    QUESTS = "quests"
    EFFICIENCY = "efficiency"
    COMPLETION = "completion"
    CONTROLS = "controls"  # how the buttons are used (T-874; added to the design's eight)

    @property
    def label(self) -> str:
        return "Quests and Story" if self is Category.QUESTS else self.value.capitalize()


def _check_number(value: float, what: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ConfigurationError(f"{what} must be a number.") from exc
    if not math.isfinite(number):
        raise ConfigurationError(f"{what} must be a finite number.")
    return number


def _check_in_range(value: float, what: str) -> float:
    number = _check_number(value, what)
    if not -MAX_VALUE <= number <= MAX_VALUE:
        raise ConfigurationError(f"{what} must be between -{MAX_VALUE:g} and {MAX_VALUE:g}.")
    return number


def clamp(value: float) -> float:
    return max(-MAX_VALUE, min(MAX_VALUE, value))


@dataclass(frozen=True, slots=True)
class Trigger:
    """What has to happen in the game for a behaviour to count.

    ``variable_id`` of ``None`` means "any variable of that kind".
    """

    kind: EventKind
    variable_id: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.kind, EventKind):
            raise ConfigurationError("Trigger kind must be an EventKind.")

    @property
    def key(self) -> str:
        return f"{self.kind.value}:{self.variable_id or '*'}"

    def matches(self, event: GameEvent) -> bool:
        if event.kind is not self.kind:
            return False
        return self.variable_id is None or event.variable_id == self.variable_id

    def describe(self) -> str:
        if self.variable_id is None:
            return self.kind.label
        return f"{self.kind.label} of {self.variable_id}"

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind.value, "variable_id": self.variable_id}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Trigger:
        try:
            return cls(EventKind(data["kind"]), data.get("variable_id"))
        except (KeyError, TypeError, ValueError) as exc:
            raise ConfigurationError(f"Bad trigger: {exc}") from exc


@dataclass(frozen=True, slots=True)
class Behaviour:
    """One thing the AI can be rewarded (or penalised) for."""

    id: str
    name: str
    category: Category
    trigger: Trigger
    suggested: float = 0.0
    value: float = 0.0
    enabled: bool = True
    custom: bool = False  # a person set the value; suggestions stop moving it
    notes: str = ""

    def __post_init__(self) -> None:
        if not self.id or not isinstance(self.id, str):
            raise ConfigurationError("Behaviour needs an id.")
        if not self.name or not isinstance(self.name, str):
            raise ConfigurationError("Behaviour needs a name.")
        if not isinstance(self.category, Category):
            raise ConfigurationError("Behaviour category must be a Category.")
        if not isinstance(self.trigger, Trigger):
            raise ConfigurationError("Behaviour trigger must be a Trigger.")
        object.__setattr__(self, "suggested", _check_in_range(self.suggested, "Suggested value"))
        object.__setattr__(self, "value", _check_in_range(self.value, "Reward value"))

    @property
    def display_name(self) -> str:
        return self.name

    @property
    def is_penalty(self) -> bool:
        return self.value < 0

    def replace(self, **changes: Any) -> Behaviour:
        return replace(self, **changes)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "category": self.category.value,
            "trigger": self.trigger.to_dict(),
            "suggested": self.suggested,
            "value": self.value,
            "enabled": self.enabled,
            "custom": self.custom,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Behaviour:
        try:
            return cls(
                data["id"],
                data["name"],
                Category(data["category"]),
                Trigger.from_dict(data["trigger"]),
                suggested=data.get("suggested", 0.0),
                value=data.get("value", data.get("suggested", 0.0)),
                enabled=bool(data.get("enabled", True)),
                custom=bool(data.get("custom", False)),
                notes=str(data.get("notes", "")),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ConfigurationError(f"Bad behaviour: {exc}") from exc


@dataclass(slots=True)
class RewardProfile:
    """Every behaviour for one game, with the child-facing controls."""

    game_id: str
    personality: str = "custom"
    _items: dict[str, Behaviour] = field(default_factory=dict, init=False, repr=False)
    _weights: dict[Category, float] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.game_id or not isinstance(self.game_id, str):
            raise ConfigurationError("RewardProfile needs a game id.")
        self.personality = self._check_personality(self.personality)
        self._weights = self.check_weights({})

    def __setattr__(self, name: str, value: Any) -> None:
        if name == "personality":
            value = self._check_personality(value)
        object.__setattr__(self, name, value)

    @staticmethod
    def _check_personality(value: Any) -> str:
        if not value or not isinstance(value, str):
            raise ConfigurationError("Personality needs a name.")
        return value

    # ------------------------------------------------------------ container

    def __len__(self) -> int:
        return len(self._items)

    def __iter__(self) -> Iterator[Behaviour]:
        return iter(tuple(self._items.values()))

    def __contains__(self, behaviour_id: object) -> bool:
        return behaviour_id in self._items

    def ids(self) -> tuple[str, ...]:
        return tuple(self._items)

    def get(self, behaviour_id: str) -> Behaviour:
        try:
            return self._items[behaviour_id]
        except KeyError:
            raise ConfigurationError(f"No behaviour {behaviour_id!r}.") from None

    def find(self, behaviour_id: str) -> Behaviour | None:
        return self._items.get(behaviour_id)

    def add(self, behaviour: Behaviour) -> Behaviour:
        if not isinstance(behaviour, Behaviour):
            raise ConfigurationError("Only a Behaviour can be added.")
        if behaviour.id in self._items:
            raise ConfigurationError(f"Behaviour {behaviour.id!r} is already in the profile.")
        self._items[behaviour.id] = behaviour
        return behaviour

    def remove(self, behaviour_id: str) -> None:
        self.get(behaviour_id)
        del self._items[behaviour_id]

    def by_category(self) -> dict[Category, tuple[Behaviour, ...]]:
        groups: dict[Category, list[Behaviour]] = {c: [] for c in Category}
        for b in self._items.values():
            groups[b.category].append(b)
        return {c: tuple(v) for c, v in groups.items()}

    def enabled_count(self) -> int:
        return sum(1 for b in self._items.values() if b.enabled)

    def active_count(self) -> int:
        """Enabled rows worth something: a row at 0 points changes nothing (T-874)."""
        return sum(1 for b in self._items.values() if b.enabled and b.value != 0.0)

    # ------------------------------------------------------------ weights

    @property
    def weights(self) -> dict[Category, float]:
        """Multiplier per category (a personality); untouched values follow them."""
        return dict(self._weights)

    @staticmethod
    def check_weights(weights: Mapping[Category, float]) -> dict[Category, float]:
        out = {c: 1.0 for c in Category}
        for category, weight in weights.items():
            if not isinstance(category, Category):
                raise ConfigurationError("Weights must be keyed by Category.")
            number = _check_number(weight, f"{category.label} weight")
            if number <= 0:
                raise ConfigurationError(f"{category.label} weight must be positive.")
            out[category] = number
        return out

    def set_weights(self, weights: Mapping[Category, float]) -> None:
        """Replace the weights (missing categories go back to 1.0) and rescale untouched values."""
        self._weights = self.check_weights(weights)
        for bid, b in list(self._items.items()):
            if not b.custom:
                self._items[bid] = b.replace(value=self._weighted(b))

    def _weighted(self, behaviour: Behaviour, suggested: float | None = None) -> float:
        base = behaviour.suggested if suggested is None else suggested
        return clamp(base * self._weights[behaviour.category])

    # ------------------------------------------------------------ controls

    def _update(self, behaviour_id: str, **changes: Any) -> Behaviour:
        updated = self.get(behaviour_id).replace(**changes)
        self._items[behaviour_id] = updated
        return updated

    def increase(self, behaviour_id: str, step: float = STEP) -> Behaviour:
        """The ``+`` button."""
        return self.set_value(behaviour_id, self.get(behaviour_id).value + self._step(step))

    def decrease(self, behaviour_id: str, step: float = STEP) -> Behaviour:
        """The ``-`` button."""
        return self.set_value(behaviour_id, self.get(behaviour_id).value - self._step(step))

    @staticmethod
    def _step(step: float) -> float:
        number = _check_number(step, "Step")
        if number <= 0:
            raise ConfigurationError("Step must be positive.")
        return number

    def set_value(self, behaviour_id: str, value: float) -> Behaviour:
        """Set the current value (clamped into range) and mark it as a person's choice."""
        number = clamp(_check_number(value, "Reward value"))
        return self._update(behaviour_id, value=number, custom=True)

    def set_enabled(self, behaviour_id: str, enabled: bool) -> Behaviour:
        return self._update(behaviour_id, enabled=bool(enabled))

    def update(self, behaviour_id: str, **changes: Any) -> Behaviour:
        """Change descriptive fields (name, category, notes) without touching the controls."""
        allowed = {"name", "category", "notes"}
        unknown = set(changes) - allowed
        if unknown:
            raise ConfigurationError(f"Can't update {sorted(unknown)}; use the controls.")
        return self._update(behaviour_id, **changes)

    def suggest(self, behaviour_id: str, suggested: float) -> Behaviour:
        """Record a new suggested value; follow it unless a person has set the value."""
        number = clamp(_check_number(suggested, "Suggested value"))
        current = self.get(behaviour_id)
        if current.custom:
            return self._update(behaviour_id, suggested=number)
        return self._update(behaviour_id, suggested=number, value=self._weighted(current, number))

    def reset_to_suggested(self, behaviour_id: str | None = None) -> Behaviour | None:
        """Put one behaviour (or all of them) back on its suggested value."""
        if behaviour_id is None:
            for bid in self.ids():
                self.reset_to_suggested(bid)
            return None
        current = self.get(behaviour_id)
        return self._update(behaviour_id, value=self._weighted(current), custom=False)

    # ------------------------------------------------------------ rewards

    def matching(self, event: GameEvent) -> tuple[Behaviour, ...]:
        """Enabled behaviours whose trigger fires on ``event``."""
        return tuple(b for b in self._items.values() if b.enabled and b.trigger.matches(event))

    def reward_for(self, event: GameEvent) -> float:
        return float(sum(b.value for b in self.matching(event)))

    def reward_for_events(self, events: Iterable[GameEvent]) -> float:
        return float(sum(self.reward_for(e) for e in events))

    # ------------------------------------------------------------ persistence

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA_VERSION,
            "game_id": self.game_id,
            "personality": self.personality,
            "weights": {c.value: w for c, w in self._weights.items()},
            "behaviours": [b.to_dict() for b in self._items.values()],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RewardProfile:
        try:
            schema = int(data.get("schema", SCHEMA_VERSION))
            if schema > SCHEMA_VERSION:
                raise ConfigurationError(
                    f"This rewards file is from a newer version (schema {schema})."
                )
            profile = cls(data["game_id"], personality=data.get("personality", "custom"))
            profile._weights = cls.check_weights(
                {Category(k): v for k, v in data.get("weights", {}).items()}
            )
            for item in data.get("behaviours", []):
                profile.add(Behaviour.from_dict(item))
        except (KeyError, TypeError, AttributeError, ValueError) as exc:
            raise ConfigurationError(f"Bad rewards data: {exc}") from exc
        return profile


__all__ = [
    "MAX_VALUE",
    "SCHEMA_VERSION",
    "STEP",
    "Behaviour",
    "Category",
    "RewardProfile",
    "Trigger",
    "clamp",
]
