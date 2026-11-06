"""Exploring follows results (T-932).

Both learners explore less as they learn (T-875's schedule), whatever the goes
are doing. The user's 20:03 log showed the cost: exploring reached its 5% floor
after 14 goes and the AI then lost lives faster with every dozen goes, locked
into the habit it had when exploring stopped. A :class:`ResultsWatch` keeps the
rewards of the last goes and, when the latest window is worse than the one
before, lifts exploring back to ``Settings.rebound`` and holds it there while the
goes stay worse. Once they recover, the lift eases off over a window of goes and
the schedule takes over again.
"""

from __future__ import annotations

import logging
from typing import Any

from retro_ai_lab.core import ConfigurationError

log = logging.getLogger(__name__)

RESULTS_WINDOW = 10  # goes per window; the latest window is judged against the one before
NOTICEABLE = 0.1  # a drop smaller than this share of the rewards' spread is noise


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


__all__ = ["NOTICEABLE", "RESULTS_WINDOW", "ResultsWatch"]
