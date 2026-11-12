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
NEAR_SHARE = 0.5  # of the moves it tries, how many are one button from its own (T-998)


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
        self._near: dict[int, tuple[int, ...]] = {}  # the moves one button away, as asked

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

    def near(self, index: int) -> tuple[int, ...]:
        """The moves one button away from this one that keep a button it holds.

        One button more, or one of two let go. Letting go of everything is
        stopping, not a small change, and from holding nothing every button is
        one away, so neither counts: controls with no combinations have no
        near moves at all.
        """
        buttons = self[index].buttons
        found = self._near.get(index)
        if found is None:
            found = self._near[index] = tuple(
                a.index
                for a in self._actions
                if len(a.buttons ^ buttons) == 1 and a.buttons & buttons
            )
        return found

    def sample_near(self, rng: Random, index: int) -> int:
        """A move to try around ``index``, the move that would otherwise be made (T-998).

        ``NEAR_SHARE`` of the time one of the moves one button away from it, each
        as likely as the next; otherwise any move at all, so nothing is out of
        reach and a brain that knows nothing yet still tries everything. Getting
        over a gap is "what I was doing, plus one button", held: a pick from
        every combination hardly ever keeps the run-up. Where no move is near
        it is the plain pick, with the same single draw.
        """
        near = self.near(index)
        if near and rng.random() < NEAR_SHARE:
            return rng.choice(near)
        return self.sample(rng)

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
