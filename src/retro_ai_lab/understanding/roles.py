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

A touch is kept with the way it was made (T-1051): from above, from the
side or from below. An enemy that is come down on harmlessly and costs a
life from the side is one kind met two ways, and it hurts when it does so
in any one way. And a life lost with no touch waiting is put down to the
thing that came closest in the looks before: a look comes only every few
frames, and what a game shows of a hit need not show the two touching.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import StrEnum

from retro_ai_lab.core import ConfigurationError
from retro_ai_lab.observation.signature import Box
from retro_ai_lab.observation.sprites import SpriteTag
from retro_ai_lab.observation.tracking import Track
from retro_ai_lab.understanding.knowledge import Confidence

TOUCH_GAP = 2  # pixels between two boxes that still count as touching
SOON = 30  # frames after a touch in which a life lost or score is put down to it
HALF = 0.5  # share of a kind's touches that must agree on its role
REACH = 32  # pixels off within which a thing is noted as close: one look's travel (T-1051)
ALSO = 3  # other touches that were waiting, named with a life lost


class Happening(StrEnum):
    """What memory said happened (from the events the engine already finds)."""

    LIFE_LOST = "life lost"
    SCORE_UP = "score up"


class Way(StrEnum):
    """How the player met a thing (T-1051): where it was against the thing then."""

    ABOVE = "from above"
    SIDE = "from the side"
    BELOW = "from below"


def way_of(me: Box, thing: Box) -> Way:
    """The way the player in ``me`` meets ``thing``: from above when its feet are no
    lower than a little into the thing's top, from below when its head is no higher
    than a little into the thing's bottom, and from the side otherwise. A little is
    half the height of the lower of the two: beside a thing much taller than itself
    the player is beside it, wherever along it."""
    _, y, _, h = me
    _, ty, _, th = thing
    into = min(h, th) / 2
    if y + h <= ty + into:
        return Way.ABOVE
    if y >= ty + th - into:
        return Way.BELOW
    return Way.SIDE


@dataclass(slots=True)
class WayEvidence:
    """What touching a kind of thing one way has done so far (T-1051)."""

    touches: int = 0
    hurt: int = 0
    collected: int = 0


@dataclass(slots=True)
class KindEvidence:
    """What touching one kind of thing has done so far."""

    touches: int = 0
    hurt: int = 0  # a life went soon after
    collected: int = 0  # it vanished and the score went up
    moves: bool = False  # seen moving through the level on its own
    # The same by the way of the touch (T-1051), and the lives put down to it as the
    # closest thing, with no touch seen (they are among ``touches`` and ``hurt``).
    ways: dict[Way, WayEvidence] = field(default_factory=dict)
    unseen: int = 0

    @property
    def harmless(self) -> int:
        return self.touches - self.hurt - self.collected

    def way(self, way: Way) -> WayEvidence:
        ev = self.ways.get(way)
        if ev is None:
            ev = self.ways[way] = WayEvidence()
        return ev

    def hurts(self) -> tuple[Way, ...]:
        """The ways in which a life went after at least half its touches."""
        return tuple(
            way
            for way in Way
            if (ev := self.ways.get(way)) is not None and ev.hurt and ev.hurt >= HALF * ev.touches
        )

    def describe(self) -> str:
        """After how many touches of each way a life went; nothing for a kind no life
        was ever put down to."""
        if not self.hurt:
            return ""
        said = [
            f"{ev.hurt} of {ev.touches} {way.value}"
            for way in Way
            if (ev := self.ways.get(way)) is not None and ev.touches
        ]
        closest = (
            f" ({self.unseen} of them with no touch seen: it was the closest thing)"
            if self.unseen
            else ""
        )
        return f"A life went after {', '.join(said)}{closest}." if said else ""


@dataclass(frozen=True, slots=True)
class Blame:
    """What a life lost was put down to (T-1051): the kind, the way it was met, how
    many looks before, and how far off it was (nought for a touch). ``seen`` is
    whether the two were seen touching; ``also`` the other touches waiting then."""

    kind: str
    way: Way
    ago: int
    gap: int
    seen: bool
    also: tuple[tuple[str, Way, int], ...] = ()


@dataclass(frozen=True, slots=True)
class Role:
    kind: str
    tag: SpriteTag
    confidence: Confidence
    touches: int
    ways: tuple[Way, ...] = ()  # the ways it hurts (T-1051); none where it does not
    detail: str = ""  # after how many touches of each way a life went

    def describe(self) -> str:
        words = {
            SpriteTag.ENEMY: "an enemy",
            SpriteTag.HAZARD: "something that hurts",
            SpriteTag.ITEM: "something to pick up",
            SpriteTag.OTHER: "harmless",
        }[self.tag]
        said = f"{self.kind} is {words} ({self.confidence.label.lower()}, {self.touches} touches)."
        return f"{said} {self.detail}" if self.detail else said


@dataclass(slots=True)
class _Pending:
    kind: str
    track: int
    frame: int
    gone: bool = False
    way: Way = Way.SIDE


@dataclass(frozen=True, slots=True)
class _Near:
    """A thing seen close to the player and not touching it (T-1051)."""

    kind: str
    frame: int
    gap: int
    way: Way


def _gap(a: Box, b: Track) -> int:
    """Pixels between the two boxes, the larger of across and up-and-down; nought
    when they overlap."""
    ax, ay, aw, ah = a
    return max(b.x - (ax + aw), ax - (b.x + b.width), b.y - (ay + ah), ay - (b.y + b.height), 0)


def _touching(a: Box, b: Track, gap: int = TOUCH_GAP) -> bool:
    ax, ay, aw, ah = a
    return (
        ax - gap < b.x + b.width
        and b.x - gap < ax + aw
        and ay - gap < b.y + b.height
        and b.y - gap < ay + ah
    )


def within(t: Track, box: Box, gap: int = TOUCH_GAP) -> bool:
    """The middle of ``t`` lies in ``box``: it is the thing the box was put round."""
    x, y, w, h = box
    cx, cy = t.centre
    return x - gap <= cx < x + w + gap and y - gap <= cy < y + h + gap


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
    _near: list[_Near] = field(default_factory=list)  # close and not touching, still waiting
    _frame: int = 0  # the last frame observed
    last_blame: Blame | None = None  # what the last life told of was put down to (T-1051)

    def __post_init__(self) -> None:
        if not isinstance(self.soon, int) or self.soon < 1:
            raise ConfigurationError("soon must be at least 1 frame.")

    def observe(
        self,
        frame: int,
        tracks: Iterable[Track],
        player: int | None,
        happenings: Iterable[Happening] = (),
        *,
        me: Box | None = None,
    ) -> None:
        """One frame: the tracks on screen, the player's track id, what memory said happened.

        ``me`` is the box the player was seen in when it is known by its looks rather
        than as a track (T-987): touches are measured from the box, and the tracks
        whose middle lies in it are the player, not something it touched. A caller
        that gives boxes gives None when the player is out of sight.
        """
        tracks = list(tracks)
        self._frame = frame
        by_id = {t.id: t for t in tracks}
        happened = frozenset(happenings)
        if me is not None:
            mine = {t.id for t in tracks if t.missing == 0 and within(t, me)}
        else:
            mine = {player} if player is not None else set()
            seen = by_id.get(player) if player is not None else None
            if seen is not None and seen.missing == 0:
                me = (seen.x, seen.y, seen.width, seen.height)
        for t in tracks:
            if t.kind is not None and t.moves and t.id not in mine:
                self._of(t.kind).moves = True
        # Things touched earlier that are gone now (picked up, or defeated).
        for p in self._pending:
            if p.track not in by_id or by_id[p.track].missing > 0:
                p.gone = True
        # What memory said settles the touches still waiting. A life lost goes to the
        # latest touch: a game freezes the player when it is hit, and the life
        # counter drops a few seconds later, after nothing else was touched.
        if Happening.LIFE_LOST in happened:
            self.life_lost()
        elif Happening.SCORE_UP in happened:
            for p in [p for p in self._pending if p.gone]:
                ev = self._of(p.kind)
                ev.collected += 1
                ev.way(p.way).collected += 1
                self._pending.remove(p)
                break
        # Touches that waited too long were harmless, and what was close is forgotten.
        self._pending = [p for p in self._pending if frame - p.frame <= self.soon]
        self._near = [n for n in self._near if frame - n.frame <= self.soon]
        # New touches this frame (a touch lasting several frames counts once).
        now: set[int] = set()
        if me is not None:
            for t in tracks:
                if t.id in mine or t.kind is None or t.missing > 0:
                    continue
                way = way_of(me, (t.x, t.y, t.width, t.height))
                if _touching(me, t):
                    now.add(t.id)
                    if t.id not in self._touching:
                        ev = self._of(t.kind)
                        ev.touches += 1
                        ev.way(way).touches += 1
                        self._pending.append(_Pending(t.kind, t.id, frame, way=way))
                elif (gap := _gap(me, t)) <= REACH:
                    self._near.append(_Near(t.kind, frame, gap, way))
        self._touching = now

    @property
    def waiting(self) -> bool:
        """A touch is waiting to be settled (a life lost now would be put down to it)."""
        return bool(self._pending)

    @property
    def close(self) -> bool:
        """With no touch waiting, a thing seen close by is (T-1051): a life lost now
        would be put down to the closest."""
        return not self._pending and bool(self._near)

    def life_lost(self) -> None:
        """A life went: it is put down to the latest touch still waiting, and to that
        one only (:meth:`observe` does this for a life lost it is told of; a caller
        with no new frame to show, at the end of a go, calls it directly). With no
        touch waiting it is put down to the thing that came closest in the looks
        before, the latest of equals, as a touch that was not seen (T-1051). What it
        was put down to is in :attr:`last_blame`: None where nothing was waiting."""
        self.last_blame = None
        if self._pending:
            p = self._pending.pop()
            ev = self._of(p.kind)
            ev.hurt += 1
            ev.way(p.way).hurt += 1
            also = tuple((o.kind, o.way, self._frame - o.frame) for o in self._pending[::-1])
            self.last_blame = Blame(p.kind, p.way, self._frame - p.frame, 0, True, also[:ALSO])
        elif self._near:
            n = min(self._near, key=lambda n: (n.gap, -n.frame))
            ev = self._of(n.kind)
            ev.touches += 1
            ev.hurt += 1
            ev.unseen += 1
            by_way = ev.way(n.way)
            by_way.touches += 1
            by_way.hurt += 1
            self.last_blame = Blame(n.kind, n.way, self._frame - n.frame, n.gap, False)
        self._pending.clear()
        self._near.clear()

    def let_go(self) -> None:
        """Something else explains what happened (a fall): the touches still waiting
        were harmless (T-988), and what was close is forgotten."""
        self._pending.clear()
        self._near.clear()

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
            ways = ev.hurts()
            by_all = bool(ev.hurt) and ev.hurt >= HALF * ev.touches
            if ways or by_all:
                # In any one way (T-1051), or by all its touches together as before.
                tag = SpriteTag.ENEMY if ev.moves else SpriteTag.HAZARD
                n = ev.hurt if by_all else sum(ev.ways[way].hurt for way in ways)
            elif ev.collected and ev.collected >= HALF * ev.touches:
                tag, n = SpriteTag.ITEM, ev.collected
            else:
                tag, n = SpriteTag.OTHER, ev.harmless
            out.append(Role(kind, tag, _confidence(n), ev.touches, ways, ev.describe()))
        return tuple(out)


__all__ = [
    "REACH",
    "Blame",
    "Happening",
    "KindEvidence",
    "Role",
    "RoleLearner",
    "Way",
    "WayEvidence",
    "way_of",
    "within",
]
