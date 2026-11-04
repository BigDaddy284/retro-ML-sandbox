"""Rewards a person adds themselves (T-860): any value going up or down."""

from __future__ import annotations

from retro_ai_lab.core import ConfigurationError
from retro_ai_lab.rewards.model import Behaviour, Category, Trigger
from retro_ai_lab.understanding.events import EventKind
from retro_ai_lab.understanding.knowledge import Variable

MINE = "mine"  # the id prefix; not an event kind, so suggestions leave the row alone
OWN_NOTE = "You added this."


def own_id(variable_id: str, goes_up: bool) -> str:
    return f"{MINE}:{'up' if goes_up else 'down'}:{variable_id}"


def is_own(behaviour_id: str) -> bool:
    return behaviour_id.startswith(MINE + ":")


def own_behaviour(var: Variable, name: str, goes_up: bool, value: float) -> Behaviour:
    """A row for ``name`` going up (or down), worth ``value`` each step it does."""
    if var.rejected:
        raise ConfigurationError(f"{var.display_name} was rejected; put it back first.")
    if not name.strip():
        raise ConfigurationError("The reward needs a name.")
    return Behaviour(
        own_id(var.id, goes_up),
        f"{name.strip()} goes {'up' if goes_up else 'down'}",
        Category.PROGRESSION,
        Trigger(EventKind.UP if goes_up else EventKind.DOWN, var.id),
        suggested=value,
        value=value,
        custom=True,
        notes=OWN_NOTE,
    )


__all__ = ["MINE", "OWN_NOTE", "is_own", "own_behaviour", "own_id"]
