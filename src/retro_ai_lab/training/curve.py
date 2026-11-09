"""The learning curve: how the AI's goes are going, for graphs and plain words.

DESIGN.md §10 asks the Learning panel for "learning progress" and the Progress
panel for "training attempts, success rate, performance trends". Both read
one :class:`LearningCurve` built from the history of goes.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from retro_ai_lab.core import ConfigurationError
from retro_ai_lab.training.records import EpisodeRecord

DEFAULT_WINDOW = 10
TOO_EARLY = "too early"
IMPROVING = "improving"
SLIPPING = "slipping"
STEADY = "steady"
NOTICEABLE = 0.1  # a change smaller than this share of the rewards' spread is noise


@dataclass(frozen=True, slots=True)
class CurvePoint:
    number: int
    reward: float
    steps: int
    average: float  # rolling mean over the last ``window`` goes, this one included


class LearningCurve:
    def __init__(self, goes: Sequence[EpisodeRecord], window: int = DEFAULT_WINDOW) -> None:
        if window < 1:
            raise ConfigurationError("window must be at least 1.")
        self._window = window
        self._goes = tuple(goes)
        points: list[CurvePoint] = []
        for i, go in enumerate(self._goes):
            recent = self._goes[max(0, i + 1 - window) : i + 1]
            average = sum(g.result.reward for g in recent) / len(recent)
            points.append(CurvePoint(go.number, go.result.reward, go.result.steps, average))
        self._points = tuple(points)

    # ------------------------------------------------------------ queries

    @property
    def window(self) -> int:
        return self._window

    @property
    def points(self) -> tuple[CurvePoint, ...]:
        return self._points

    @property
    def rewards(self) -> tuple[float, ...]:
        return tuple(p.reward for p in self._points)

    @property
    def attempts(self) -> int:
        return len(self._goes)

    @property
    def successes(self) -> int:
        """Goes in which at least one objective was completed."""
        return sum(1 for g in self._goes if g.result.succeeded)

    @property
    def success_rate(self) -> float:
        return self.successes / self.attempts if self._goes else 0.0

    @property
    def best(self) -> CurvePoint | None:
        return max(self._points, key=lambda p: p.reward) if self._points else None

    @property
    def longest(self) -> CurvePoint | None:
        return max(self._points, key=lambda p: p.steps) if self._points else None

    @property
    def latest_average(self) -> float | None:
        return self._points[-1].average if self._points else None

    @property
    def earlier_average(self) -> float | None:
        """The rolling average of the window before the latest one, if there was one."""
        if len(self._points) < 2 * self._window:
            return None
        return self._points[-1 - self._window].average

    @property
    def trend(self) -> str:
        latest, earlier = self.latest_average, self.earlier_average
        if latest is None or earlier is None:
            return TOO_EARLY
        rewards = self.rewards
        spread = max(rewards) - min(rewards)
        change = latest - earlier
        if abs(change) <= spread * NOTICEABLE:
            return STEADY
        return IMPROVING if change > 0 else SLIPPING

    def describe(self) -> str:
        n = self.attempts
        if n == 0:
            return "No goes yet."
        goes = "go" if n == 1 else "goes"
        best = self.best
        assert best is not None
        parts = [f"{n} {goes} so far; the best scored {best.reward:+g} points (go {best.number})."]
        if self.successes:
            parts.append(f"Reached a goal in {self.successes} of them ({self.success_rate:.0%}).")
        latest, earlier = self.latest_average, self.earlier_average
        assert latest is not None
        trend = self.trend
        w = self._window
        if trend == TOO_EARLY:
            parts.append(f"The last {min(n, w)} averaged {latest:+.1f} points.")
        elif trend == IMPROVING:
            parts.append(
                f"Getting better: the last {w} averaged {latest:+.1f} points, "
                f"up from {earlier:+.1f}."
            )
        elif trend == SLIPPING:
            parts.append(
                f"Slipping: the last {w} averaged {latest:+.1f} points, down from {earlier:+.1f}."
            )
        else:
            parts.append(f"Holding steady at about {latest:+.1f} points a go.")
        return " ".join(parts)


def describe_survival(goes: Sequence[EpisodeRecord], window: int = DEFAULT_WINDOW) -> str:
    """Lives lost per go and the steps before the first loss, last window against the one before.

    T-934: the user's logs needed the "Losing Lives" earners of 160 go lines counted by hand
    to say whether the AI was surviving longer.
    """
    if window < 1:
        raise ConfigurationError("window must be at least 1.")
    if len(goes) < 2 * window:
        return f"Survival: too early to compare, {len(goes)} of {2 * window} goes played."
    latest, earlier = goes[-window:], goes[-2 * window : -window]

    def per_go(part: Sequence[EpisodeRecord]) -> float:
        return sum(g.result.setbacks for g in part) / len(part)

    def first_loss(part: Sequence[EpisodeRecord]) -> str:
        steps = [g.result.first_setback_step for g in part if g.result.first_setback_step]
        return f"{sum(steps) / len(steps):.0f} steps" if steps else "never"

    def points(part: Sequence[EpisodeRecord]) -> float:
        return sum(g.result.reward for g in part) / len(part)

    return (
        f"Survival over the last {window} goes: lives lost per go {per_go(latest):.1f} "
        f"(was {per_go(earlier):.1f} over the {window} before); the first life lost after "
        f"{first_loss(latest)} on average (was {first_loss(earlier)}). Points per go "
        f"{points(latest):+,.1f} (was {points(earlier):+,.1f}){describe_cost(latest, earlier)}."
    )


def describe_cost(latest: Sequence[EpisodeRecord], earlier: Sequence[EpisodeRecord]) -> str:
    """ "; what moved most is x position goes down, -84.0 a go (was -26.0)": the earner whose
    average over the latest goes differs most from its average over the earlier ones (T-966).
    Empty when no go in either window earned anything."""
    if not latest or not earlier:
        raise ConfigurationError("describe_cost needs goes in both windows.")

    def per_go(part: Sequence[EpisodeRecord]) -> dict[str, tuple[str, float]]:
        totals: dict[str, tuple[str, float]] = {}
        for g in part:
            for e in g.result.earned:
                name, amount = totals.get(e.id, (e.name, 0.0))
                totals[e.id] = (name, amount + e.amount)
        return {i: (n, a / len(part)) for i, (n, a) in totals.items()}

    now, before = per_go(latest), per_go(earlier)
    if not now and not before:
        return ""
    names = {i: n for i, (n, _) in (*before.items(), *now.items())}
    moved = max(names, key=lambda i: abs(now.get(i, ("", 0.0))[1] - before.get(i, ("", 0.0))[1]))
    was, is_now = before.get(moved, ("", 0.0))[1], now.get(moved, ("", 0.0))[1]
    return f"; what moved most is {names[moved]}, {is_now:+,.1f} a go (was {was:+,.1f})"


__all__ = [
    "DEFAULT_WINDOW",
    "IMPROVING",
    "SLIPPING",
    "STEADY",
    "TOO_EARLY",
    "CurvePoint",
    "LearningCurve",
    "describe_cost",
    "describe_survival",
]
