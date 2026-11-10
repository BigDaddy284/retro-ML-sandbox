"""Working out which thing on the screen is the player (T-973).

Nobody tells the AI which sprite it controls. It finds out the way a child
does: press a button and watch what moves. From one starting spot (a save
state) it plays the same frames several times: once holding nothing, then
holding each of its buttons in turn. Each run's frames go through an
:class:`~retro_ai_lab.observation.tracking.ObjectTracker`, and each thing is
matched across runs by where it was at the start (every run starts from the
same picture).

A thing answers to a button when, with that button held, it moved through
the level and nothing in the hold-nothing run went the same way (a walker
that walks whatever is pressed does the same in both runs; a jump that
lands again still differed on the way). Things are matched across button
runs by where they started. The player is the thing that answered to the
most buttons. One button only is a weak answer (the whole track scrolling
past when the racer speeds up, for instance); the player normally answers
to two or more (left and right, up and down, jump). A player standing on
the ground is part of the ground's patch on a still screen, so it is cut
out by its motion and may be first seen a frame late, without a kind.

Running the experiment against an emulator (save states, holding buttons,
capturing frames) is the caller's job: :func:`find_player` takes a
``run(buttons, frames)`` function, so it is tested on pictures alone.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass

from retro_ai_lab.core import ConfigurationError, Frame
from retro_ai_lab.observation.objects import ObjectCatalogue
from retro_ai_lab.observation.tracking import ObjectTracker
from retro_ai_lab.understanding.knowledge import Confidence

DEFAULT_FRAMES = 16  # frames each run lasts
MOVED = (
    4  # pixels apart, at the frame they differ most, for a run to count as "this button moved it"
)
SAME_START = 2  # pixels apart at the start for two runs' things to be the same thing

Run = Callable[[frozenset[str], int], Sequence[Frame]]


@dataclass(frozen=True, slots=True)
class PlayerFinding:
    """Which thing answered to the buttons, and how sure the AI is."""

    kind: str | None  # the catalogue's kind, when the player was seen whole
    start: tuple[int, int]  # its centre on the starting screen
    buttons: frozenset[str]  # the buttons that moved it
    tried: int  # buttons tried
    confidence: Confidence

    def describe(self) -> str:
        names = ", ".join(sorted(self.buttons))
        return (
            f"The thing at {self.start} moves when {names} is held "
            f"({len(self.buttons)} of {self.tried} buttons): it is you."
        )


@dataclass(frozen=True, slots=True)
class _Outcome:
    path: dict[int, tuple[int, int]]  # frame index -> position in first-screen coordinates
    kind: str | None

    @property
    def start(self) -> tuple[int, int]:
        return self.path[min(self.path)]

    @property
    def moved(self) -> bool:
        xs = [p[0] for p in self.path.values()]
        ys = [p[1] for p in self.path.values()]
        return max(xs) - min(xs) >= MOVED or max(ys) - min(ys) >= MOVED


def _outcomes(frames: Sequence[Frame], catalogue: ObjectCatalogue) -> list[_Outcome]:
    """Each thing followed through a run, placed in the first screen's coordinates.

    The tracker counts the camera from the first frame, so a level position is
    a position on the first screen; a thing cut out of the scenery by its motion
    (a hero standing on the ground) first shows up a frame or two late.
    """
    if not frames:
        raise ConfigurationError("A run must give back at least one frame.")
    tracker = ObjectTracker(catalogue)
    paths: dict[int, dict[int, tuple[int, int]]] = {}
    kinds: dict[int, str | None] = {}
    for i, frame in enumerate(frames):
        for t in tracker.update(frame):
            if t.missing == 0:
                _, lx, ly = t.history[-1]
                paths.setdefault(t.id, {})[i] = (lx, ly)
                if t.kind is not None:
                    kinds[t.id] = t.kind
    return [_Outcome(path, kinds.get(tid)) for tid, path in paths.items()]


def _apart(a: dict[int, tuple[int, int]], b: dict[int, tuple[int, int]]) -> float | None:
    """How far apart two paths are at the frame they differ most; None if never together."""
    shared = a.keys() & b.keys()
    if not shared:
        return None
    worst = 0.0
    for i in shared:
        dx, dy = a[i][0] - b[i][0], a[i][1] - b[i][1]
        worst = max(worst, float((dx * dx + dy * dy) ** 0.5))
    return worst


def _does_the_same(o: _Outcome, base: list[_Outcome]) -> bool:
    """Something in the hold-nothing run went the same way: the button changed nothing."""
    for b in base:
        apart = _apart(o.path, b.path)
        if apart is not None and apart < MOVED:
            return True
    return False


def find_player(
    run: Run,
    buttons: Iterable[str],
    *,
    frames: int = DEFAULT_FRAMES,
    catalogue: ObjectCatalogue | None = None,
) -> PlayerFinding | None:
    """Play from the same start with each button held; the thing that answers is the player.

    ``run(held, n)`` must put the game back at the same start, hold ``held``
    for ``n`` frames and return the frames it saw (one per frame or every few).
    Returns None when nothing on the screen answered to any button.
    """
    chosen = sorted(set(buttons))
    if not chosen:
        raise ConfigurationError("Try at least one button.")
    if not isinstance(frames, int) or frames < 2:
        raise ConfigurationError("frames must be at least 2.")
    catalogue = catalogue if catalogue is not None else ObjectCatalogue()
    base = _outcomes(run(frozenset(), frames), catalogue)
    # Candidates: things that moved with a button held and did not do the same without.
    groups: list[_Group] = []
    for button in chosen:
        for o in _outcomes(run(frozenset({button}), frames), catalogue):
            if not o.moved or _does_the_same(o, base):
                continue
            group = next((g for g in groups if _same(o.start, g.start, SAME_START + MOVED)), None)
            if group is None:
                groups.append(_Group(o.start, {button}, o.kind))
            else:
                group.buttons.add(button)
                group.kind = group.kind if group.kind is not None else o.kind
    if not groups:
        return None
    # Most buttons first; among equals the lowest on the screen, then the leftmost.
    best = max(groups, key=lambda g: (len(g.buttons), g.start[1], -g.start[0]))
    n = len(best.buttons)
    if n >= 3:
        confidence = Confidence.HIGH
    elif n == 2:
        confidence = Confidence.MEDIUM
    else:
        confidence = Confidence.LOW
    return PlayerFinding(best.kind, best.start, frozenset(best.buttons), len(chosen), confidence)


@dataclass(slots=True)
class _Group:
    """One thing as seen across the button runs (matched by where it started)."""

    start: tuple[int, int]
    buttons: set[str]
    kind: str | None


def _same(a: tuple[int, int], b: tuple[int, int], within: int) -> bool:
    return abs(a[0] - b[0]) <= within and abs(a[1] - b[1]) <= within


__all__ = ["PlayerFinding", "Run", "find_player"]
