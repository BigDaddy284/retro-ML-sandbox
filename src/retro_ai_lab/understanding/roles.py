"""Working out what each kind of thing on the screen is, from what it does to the player (T-975).

The AI already knows from memory when a life is lost and when the score goes
up (the events the understanding engine finds, and the death the person's
own reward names). It also knows which thing on the screen is itself
(T-973/T-974) and follows every thing (T-972). Setting the two side by side
says what a kind of thing is, with nobody telling it:

* the player touched a thing of kind X and a life went soon after: X hurts;
  a kind that hurts and moves on its own is an **enemy**, one that stays put
  is a **hazard** (spikes, a flame);
* the player touched it, it vanished, and the score went up: an **item**;
* the player touched it and nothing happened: it is harmless, which also
  counts against the first two.

Every touch is evidence, and the answer carries the same confidence levels
as the memory guesses: one touch is a low guess, two that agree a medium
one, four a high one; a role needs at least half its kind's touches to
agree. Kinds the player never touched stay unknown.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import StrEnum

from retro_ai_lab.core import ConfigurationError
from retro_ai_lab.observation.sprites import SpriteTag
from retro_ai_lab.observation.tracking import Track
from retro_ai_lab.understanding.knowledge import Confidence

TOUCH_GAP = 2  # pixels between two boxes that still count as touching
SOON = 30  # frames after a touch in which a life lost or score is put down to it
HALF = 0.5  # share of a kind's touches that must agree on its role


class Happening(StrEnum):
    """What memory said happened (from the events the engine already finds)."""

    LIFE_LOST = "life lost"
    SCORE_UP = "score up"


@dataclass(slots=True)
class KindEvidence:
    """What touching one kind of thing has done so far."""

    touches: int = 0
    hurt: int = 0  # a life went soon after
    collected: int = 0  # it vanished and the score went up
    moves: bool = False  # seen moving through the level on its own

    @property
    def harmless(self) -> int:
        return self.touches - self.hurt - self.collected


@dataclass(frozen=True, slots=True)
class Role:
    kind: str
    tag: SpriteTag
    confidence: Confidence
    touches: int

    def describe(self) -> str:
        words = {
            SpriteTag.ENEMY: "an enemy",
            SpriteTag.HAZARD: "something that hurts",
            SpriteTag.ITEM: "something to pick up",
            SpriteTag.OTHER: "harmless",
        }[self.tag]
        return f"{self.kind} is {words} ({self.confidence.label.lower()}, {self.touches} touches)."


@dataclass(slots=True)
class _Pending:
    kind: str
    track: int
    frame: int
    gone: bool = False


def _touching(a: Track, b: Track, gap: int = TOUCH_GAP) -> bool:
    return (
        a.x - gap < b.x + b.width
        and b.x - gap < a.x + a.width
        and a.y - gap < b.y + b.height
        and b.y - gap < a.y + a.height
    )


def _confidence(n: int) -> Confidence:
    if n >= 4:
        return Confidence.HIGH
    if n >= 2:
        return Confidence.MEDIUM
    return Confidence.LOW


@dataclass(slots=True)
class RoleLearner:
    """Collects touches and happenings frame by frame; :meth:`roles` gives the answers."""

    soon: int = SOON
    evidence: dict[str, KindEvidence] = field(default_factory=dict)
    _pending: list[_Pending] = field(default_factory=list)
    _touching: set[int] = field(default_factory=set)  # tracks touching the player last frame

    def __post_init__(self) -> None:
        if not isinstance(self.soon, int) or self.soon < 1:
            raise ConfigurationError("soon must be at least 1 frame.")

    def observe(
        self,
        frame: int,
        tracks: Iterable[Track],
        player: int | None,
        happenings: Iterable[Happening] = (),
    ) -> None:
        """One frame: the tracks on screen, the player's track id, what memory said happened."""
        tracks = list(tracks)
        by_id = {t.id: t for t in tracks}
        happened = frozenset(happenings)
        for t in tracks:
            if t.kind is not None and t.moves and t.id != player:
                self._of(t.kind).moves = True
        # Things touched earlier that are gone now (picked up, or defeated).
        for p in self._pending:
            if p.track not in by_id or by_id[p.track].missing > 0:
                p.gone = True
        # What memory said settles the touches still waiting. A life lost goes to the
        # latest touch: a game freezes the player when it is hit, and the life
        # counter drops a few seconds later, after nothing else was touched.
        if Happening.LIFE_LOST in happened and self._pending:
            p = self._pending.pop()
            self._of(p.kind).hurt += 1
            self._pending.clear()  # a life lost is put down to one touch only
        elif Happening.SCORE_UP in happened:
            for p in [p for p in self._pending if p.gone]:
                self._of(p.kind).collected += 1
                self._pending.remove(p)
                break
        # Touches that waited too long were harmless.
        self._pending = [p for p in self._pending if frame - p.frame <= self.soon]
        # New touches this frame (a touch lasting several frames counts once).
        me = by_id.get(player) if player is not None else None
        now: set[int] = set()
        if me is not None and me.missing == 0:
            for t in tracks:
                if t.id == player or t.kind is None or t.missing > 0:
                    continue
                if _touching(me, t):
                    now.add(t.id)
                    if t.id not in self._touching:
                        self._of(t.kind).touches += 1
                        self._pending.append(_Pending(t.kind, t.id, frame))
        self._touching = now

    @property
    def waiting(self) -> bool:
        """A touch is waiting to be settled (a life lost now would be put down to it)."""
        return bool(self._pending)

    def _of(self, kind: str) -> KindEvidence:
        ev = self.evidence.get(kind)
        if ev is None:
            ev = self.evidence[kind] = KindEvidence()
        return ev

    def roles(self) -> tuple[Role, ...]:
        """What each touched kind is, as far as the touches say."""
        out: list[Role] = []
        for kind, ev in sorted(self.evidence.items()):
            if ev.touches == 0:
                continue
            if ev.hurt and ev.hurt >= HALF * ev.touches:
                tag = SpriteTag.ENEMY if ev.moves else SpriteTag.HAZARD
                n = ev.hurt
            elif ev.collected and ev.collected >= HALF * ev.touches:
                tag, n = SpriteTag.ITEM, ev.collected
            else:
                tag, n = SpriteTag.OTHER, ev.harmless
            out.append(Role(kind, tag, _confidence(n), ev.touches))
        return tuple(out)


__all__ = ["Happening", "KindEvidence", "Role", "RoleLearner"]
