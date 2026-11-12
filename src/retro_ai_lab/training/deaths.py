"""What the AI saw just before it died (T-968).

The user's log showed the points per go climbing while the deaths stayed at
1.2 a go: the AI got further but died as often. Whether it can even see what
kills it decides what to fix, so the runner hands this watch every step's
state and, at a death, the state ``LOOKAHEAD_STEPS`` before it (when the
loss was still avoidable). Every window of goes, :func:`DeathWatch.describe`
names the sights (a state part in a bucket) that were there before most
deaths and seldom otherwise: "Before the 12 deaths in the last 10 goes the AI
saw Position at ram[520] from x position=1 in 9 of them (15% of all steps)".
When nothing stands out the line says so: the AI may not see what kills it.
Nothing about any game is read; the parts are whatever the brain sees.
"""

from __future__ import annotations

from collections import Counter, deque

from retro_ai_lab.core import ConfigurationError
from retro_ai_lab.learning.state import State, bucket_word, part_name
from retro_ai_lab.understanding.knowledge import Knowledge

DEATHS_AT_LEAST = 3  # fewer deaths in a window say nothing
STANDS_OUT_SHARE = 0.5  # a sight must be there before at least half the deaths
STANDS_OUT_LIFT = 0.25  # and that much more often than over all steps
MOST_SIGHTS = 2  # sights named per line

Sight = tuple[str, int]  # a state part's id and the bucket it was in


class DeathWatch:
    """Tallies the sights of every step and of the steps before deaths, by go."""

    def __init__(self, window: int = 10) -> None:
        if window < 1:
            raise ConfigurationError("window must be at least 1.")
        self._window = window
        self._steps: deque[Counter[Sight]] = deque(maxlen=window)
        self._deaths: deque[list[tuple[Sight, ...]]] = deque(maxlen=window)
        self._steps.append(Counter())
        self._deaths.append([])

    @property
    def window(self) -> int:
        return self._window

    def step(self, state: State) -> None:
        """One step of play saw ``state``."""
        self._steps[-1].update(state.parts)

    def death(self, before: State) -> None:
        """A life was lost; ``before`` is the state a few steps earlier."""
        self._deaths[-1].append(before.parts)

    def end_go(self) -> None:
        """The go is over: the next tallies are its own."""
        self._steps.append(Counter())
        self._deaths.append([])

    def deaths(self) -> int:
        """Deaths seen over the goes in the window."""
        return sum(len(d) for d in self._deaths)

    def _lifted(self) -> list[tuple[float, int, float, Sight]]:
        """(lift, deaths with it, share of steps, sight) for the sights that stand out,
        by lift; empty with fewer than ``DEATHS_AT_LEAST`` deaths."""
        deaths = [parts for go in self._deaths for parts in go]
        if len(deaths) < DEATHS_AT_LEAST:
            return []
        steps_seen: Counter[Sight] = Counter()
        for c in self._steps:
            steps_seen.update(c)
        steps = max(1, max(steps_seen.values(), default=0))  # steps in the window
        at_deaths: Counter[Sight] = Counter()
        for parts in deaths:
            at_deaths.update(parts)
        lifted = []
        for sight, n in at_deaths.items():
            share = n / len(deaths)
            usual = steps_seen.get(sight, 0) / steps
            if share >= STANDS_OUT_SHARE and share - usual >= STANDS_OUT_LIFT:
                lifted.append((share - usual, n, usual, sight))
        lifted.sort(key=lambda x: (-x[0], -x[1], x[3]))
        return lifted

    def standouts(self) -> tuple[Sight, ...]:
        """The sights :meth:`describe` names, in its order (T-979); none when it names none."""
        return tuple(sight for _, _, _, sight in self._lifted()[:MOST_SIGHTS])

    def describe(self, knowledge: Knowledge | None = None) -> str:
        """The sights that stood out before the window's deaths, or that none did;
        "" with fewer than ``DEATHS_AT_LEAST`` deaths."""
        deaths = sum(len(go) for go in self._deaths)
        if deaths < DEATHS_AT_LEAST:
            return ""
        lifted = self._lifted()
        goes = len(self._deaths)
        head = f"Before the {deaths} deaths in the last {goes} goes"
        if not lifted:
            return (
                f"{head} nothing the AI sees stood out: the same sights as any other "
                "step. It may not see what kills it."
            )
        words = [
            f"{part_name(sight[0], knowledge)}={bucket_word(sight[1])} in {n} of them "
            f"({usual:.0%} of all steps)"
            for _, n, usual, sight in lifted[:MOST_SIGHTS]
        ]
        return f"{head} the AI saw {' and '.join(words)}."


__all__ = [
    "DEATHS_AT_LEAST",
    "MOST_SIGHTS",
    "STANDS_OUT_LIFT",
    "STANDS_OUT_SHARE",
    "DeathWatch",
    "Sight",
]
