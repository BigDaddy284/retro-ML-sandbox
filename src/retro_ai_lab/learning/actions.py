"""The learner's menu of controller actions (DESIGN.md Phase 5).

Built from whatever buttons the connected emulator reports: "do nothing",
every single button, and every direction + one other button combo (so a
racer can steer while accelerating and a platformer can run and jump).
Opposite directions are never pressed together, and pause/menu buttons
are left out by default so the AI can't pause the game on itself.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from random import Random
from typing import Any

from retro_ai_lab.core import Button, ConfigurationError

DIRECTIONS: tuple[str, ...] = ("up", "down", "left", "right")
OPPOSITES: frozenset[frozenset[str]] = frozenset(
    {frozenset({"up", "down"}), frozenset({"left", "right"})}
)
DEFAULT_EXCLUDE: tuple[str, ...] = ("start", "select")
NOTHING = "do nothing"


@dataclass(frozen=True, slots=True)
class Action:
    index: int
    buttons: frozenset[str]

    @property
    def label(self) -> str:
        if not self.buttons:
            return NOTHING
        return " + ".join(sorted(self.buttons, key=_order))


def _order(name: str) -> tuple[int, str]:
    return (DIRECTIONS.index(name) if name in DIRECTIONS else len(DIRECTIONS), name)


class ActionSpace:
    def __init__(self, actions: Iterable[frozenset[str] | Iterable[str]]) -> None:
        items: list[Action] = []
        seen: set[frozenset[str]] = set()
        for raw in actions:
            buttons = frozenset(raw)
            if not all(isinstance(b, str) and b for b in buttons):
                raise ConfigurationError("Action buttons must be non-empty names.")
            if buttons in seen:
                raise ConfigurationError(f"Duplicate action {sorted(buttons)}.")
            seen.add(buttons)
            items.append(Action(len(items), buttons))
        if not items:
            raise ConfigurationError("An action space needs at least one action.")
        self._actions = tuple(items)
        self._by_buttons = {a.buttons: a for a in items}

    @classmethod
    def from_buttons(
        cls,
        buttons: Sequence[Button] | Sequence[str],
        *,
        combos: bool = True,
        exclude: Iterable[str] = DEFAULT_EXCLUDE,
    ) -> ActionSpace:
        names = [b.name if isinstance(b, Button) else b for b in buttons]
        if len(set(names)) != len(names):
            raise ConfigurationError("Buttons must be unique.")
        left_out = set(exclude)
        usable = [n for n in names if n not in left_out]
        directions = [n for n in usable if n in DIRECTIONS]
        others = [n for n in usable if n not in DIRECTIONS]
        menu: list[frozenset[str]] = [frozenset()]
        menu.extend(frozenset({n}) for n in usable)
        if combos:
            menu.extend(frozenset({d, o}) for d in directions for o in others)
        return cls(menu)

    # ------------------------------------------------------------ container

    def __len__(self) -> int:
        return len(self._actions)

    def __iter__(self) -> Iterator[Action]:
        return iter(self._actions)

    def __getitem__(self, index: int) -> Action:
        if not isinstance(index, int) or isinstance(index, bool) or not 0 <= index < len(self):
            raise ConfigurationError(f"No action {index!r}.")
        return self._actions[index]

    @property
    def labels(self) -> tuple[str, ...]:
        return tuple(a.label for a in self._actions)

    def buttons_for(self, index: int) -> frozenset[str]:
        return self[index].buttons

    def index_of(self, buttons: Iterable[str]) -> int:
        key = frozenset(buttons)
        try:
            return self._by_buttons[key].index
        except KeyError:
            raise ConfigurationError(f"No action for {sorted(key)}.") from None

    def matches(self, buttons: Iterable[str]) -> Action:
        """The exact action for a button set, else the largest action it contains."""
        key = frozenset(buttons)
        exact = self._by_buttons.get(key)
        if exact is not None:
            return exact
        best = self._actions[0]
        for a in self._actions:
            if a.buttons <= key and len(a.buttons) > len(best.buttons):
                best = a
        return best

    def sample(self, rng: Random) -> int:
        return rng.randrange(len(self._actions))

    # ------------------------------------------------------------ persistence

    def to_dict(self) -> dict[str, Any]:
        return {"actions": [sorted(a.buttons) for a in self._actions]}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ActionSpace:
        try:
            return cls(frozenset(item) for item in data["actions"])
        except (KeyError, TypeError) as exc:
            raise ConfigurationError(f"Bad action space: {exc}") from exc

    def __eq__(self, other: object) -> bool:
        return isinstance(other, ActionSpace) and self._actions == other._actions

    def __hash__(self) -> int:
        return hash(self._actions)


__all__ = ["DEFAULT_EXCLUDE", "DIRECTIONS", "NOTHING", "Action", "ActionSpace"]
