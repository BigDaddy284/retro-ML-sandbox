"""Exploring follows results (T-932).

Both learners explore less as they learn (T-875's schedule), whatever the goes
are doing. The user's 20:03 log showed the cost: exploring reached its 5% floor
after 14 goes and the AI then lost lives faster with every dozen goes, locked
into the habit it had when exploring stopped. A :class:`ResultsWatch` keeps the
rewards of the last goes and, when the latest window is worse than the one
before, lifts exploring back to ``Settings.rebound`` and holds it there while the
goes stay worse. Once they recover, the lift eases off over a window of goes and
the schedule takes over again.

Exploring comes in runs (T-983). Rolling for a random move at every step makes a
try last one step, four frames: a twitch that never holds a button long enough
to show what holding it does. :class:`Runs` holds a random move for a run of
steps instead, and starts runs less often so the share of explored steps is
still what the schedule says.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from random import Random
from typing import Any

from retro_ai_lab.core import ConfigurationError

log = logging.getLogger(__name__)

RESULTS_WINDOW = 10  # goes per window; the latest window is judged against the one before
NOTICEABLE = 0.1  # a drop smaller than this share of the rewards' spread is noise
MOST_RUN = 100  # the longest run a setting may ask for


class ResultsWatch:
    """The last goes' rewards, and the lift on exploring they call for."""

    def __init__(self, rebound: float, window: int = RESULTS_WINDOW) -> None:
        if window < 1:
            raise ConfigurationError("window must be at least 1.")
        self._rebound = float(rebound)
        self._window = window
        self._rewards: list[float] = []
        self._hold = 0  # goes of easing left before the lift is gone (window = the full lift)

    @property
    def lift(self) -> float:
        """The floor exploring is held above right now (0 when the schedule rules)."""
        return self._rebound * self._hold / self._window

    @property
    def rewards(self) -> tuple[float, ...]:
        return tuple(self._rewards)

    def set_rebound(self, rebound: float) -> None:
        self._rebound = float(rebound)

    def slipping(self) -> tuple[float, float] | None:
        """(latest average, earlier average) when the latest window is worse; else None."""
        w = self._window
        if len(self._rewards) < 2 * w:
            return None
        latest = sum(self._rewards[-w:]) / w
        earlier = sum(self._rewards[-2 * w : -w]) / w
        spread = max(self._rewards) - min(self._rewards)
        if spread <= 0.0 or earlier - latest <= spread * NOTICEABLE:
            return None
        return latest, earlier

    def note(self, reward: float, *, goes: int) -> float:
        """Record a go's reward; returns the lift exploring must stay above from now on."""
        self._rewards.append(float(reward))
        del self._rewards[: -2 * self._window]
        if self._rebound <= 0.0:
            self._hold = 0
            return self.lift
        worse = self.slipping()
        if worse is not None:
            latest, earlier = worse
            if self._hold < self._window:
                log.info(
                    "Exploring back up to %.0f%% after go %d: the last %d goes averaged "
                    "%+.1f points, down from %+.1f.",
                    self._rebound * 100,
                    goes,
                    self._window,
                    latest,
                    earlier,
                )
            self._hold = self._window
        elif self._hold > 0:
            self._hold -= 1
        return self.lift

    def to_dict(self) -> dict[str, Any]:
        return {"rewards": list(self._rewards), "hold": self._hold}

    def load(self, data: dict[str, Any] | None) -> None:
        if not data:
            return
        rewards = data.get("rewards", [])
        if not isinstance(rewards, list):
            raise ConfigurationError("Saved results must be a list of numbers.")
        self._rewards = [float(r) for r in rewards][-2 * self._window :]
        hold = int(data.get("hold", 0))
        if not 0 <= hold <= self._window:
            raise ConfigurationError(f"Saved hold must be between 0 and {self._window}.")
        self._hold = hold


class Runs:
    """A random move held for a run of steps (T-983).

    At a free step a run starts with a chance chosen so that, with runs of 1 to
    ``most`` steps (each length as likely as the next), the share of explored
    steps is the ``epsilon`` asked for: ``e / (m(1 - e) + e)`` with ``m`` the
    mean run. ``most`` of 1 is one roll and one draw a step, as it was before.
    """

    def __init__(self, most: int = 1) -> None:
        self._most = _run_length(most)
        self._left = 0  # steps of the run still to come
        self._action = 0

    @property
    def most(self) -> int:
        return self._most

    @property
    def left(self) -> int:
        """Steps the run in hand still has to go (0 between runs)."""
        return self._left

    def set_most(self, most: int) -> None:
        self._most = _run_length(most)
        self.drop()

    def drop(self) -> None:
        """Forget the run in hand (a go ended; its move belonged to that go)."""
        self._left = 0
        self._action = 0

    def chance(self, epsilon: float) -> float:
        """The chance a run starts at a free step, for this share of explored steps."""
        if self._most == 1:
            return epsilon
        mean = (1 + self._most) / 2
        return epsilon / (mean * (1.0 - epsilon) + epsilon)

    def next(self, rng: Random, epsilon: float, sample: Callable[[Random], int]) -> int | None:
        """The move to explore with at this step, or None to do what looks best."""
        if self._left > 0:
            self._left -= 1
            return self._action
        if rng.random() >= self.chance(epsilon):
            return None
        action = sample(rng)
        if self._most > 1:
            self._left = rng.randint(1, self._most) - 1
            self._action = action
        return action

    def to_dict(self) -> dict[str, int]:
        return {"left": self._left, "action": self._action}

    def load(self, data: dict[str, Any] | None, *, actions: int) -> None:
        if not data:
            return
        left, action = int(data.get("left", 0)), int(data.get("action", 0))
        if not 0 <= left < self._most:
            raise ConfigurationError(f"A saved run must have 0 to {self._most - 1} steps left.")
        if not 0 <= action < actions:
            raise ConfigurationError("A saved run holds a move these controls don't have.")
        self._left, self._action = left, action if left else 0


def _run_length(most: object) -> int:
    if isinstance(most, bool) or not isinstance(most, int) or not 1 <= most <= MOST_RUN:
        raise ConfigurationError(f"A run must be a whole number of steps from 1 to {MOST_RUN}.")
    return most


__all__ = ["MOST_RUN", "NOTICEABLE", "RESULTS_WINDOW", "ResultsWatch", "Runs"]
