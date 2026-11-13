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

The finding also says what the player looks like (T-987): the colours it is
drawn in, learnt from the screens on which it was seen moving
(:class:`~retro_ai_lab.observation.signature.ColourSignature`), so that it
can be picked out of any later screen without having to be followed there.

Running the experiment against an emulator (save states, holding buttons,
capturing frames) is the caller's job: :func:`find_player` takes a
``run(buttons, frames)`` function, so it is tested on pictures alone.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from retro_ai_lab.core import ConfigurationError, Frame
from retro_ai_lab.observation.objects import ObjectCatalogue
from retro_ai_lab.observation.signature import Box, ColourSignature, Sample
from retro_ai_lab.observation.tracking import ObjectTracker
from retro_ai_lab.understanding.knowledge import Confidence

if TYPE_CHECKING:  # the services package imports this one; no cycle at run time
    from retro_ai_lab.services.connection import ConnectionService

log = logging.getLogger(__name__)

DEFAULT_FRAMES = 16  # screens each run looks at
DEFAULT_EVERY = 2  # emulator frames between two screens of a run
PLAYER_SLOT = "player"  # where the experiment keeps the spot it plays from
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
    colours: ColourSignature | None = None  # what it looks like, to find it on any screen (T-987)

    def describe(self) -> str:
        names = ", ".join(sorted(self.buttons))
        return (
            f"The thing at {self.start} moves when {names} is held "
            f"({len(self.buttons)} of {self.tried} buttons): it is you."
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "start": list(self.start),
            "buttons": sorted(self.buttons),
            "tried": self.tried,
            "confidence": self.confidence.name,
            "colours": self.colours.to_dict() if self.colours is not None else None,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> PlayerFinding:
        try:
            kind = data["kind"]
            x, y = data["start"]
            buttons = data["buttons"]
            tried = data["tried"]
            confidence = Confidence[data["confidence"]]
        except (KeyError, TypeError, ValueError) as exc:
            raise ConfigurationError(f"Bad saved player: {exc}") from exc
        if kind is not None and not isinstance(kind, str):
            raise ConfigurationError("A saved player's kind must be text or null.")
        if not all(isinstance(v, int) and not isinstance(v, bool) for v in (x, y, tried)):
            raise ConfigurationError("A saved player's start and tried must be whole numbers.")
        if not isinstance(buttons, list) or not all(isinstance(b, str) for b in buttons):
            raise ConfigurationError("A saved player's buttons must be a list of names.")
        # A finding saved before T-987 has no colours; the AI looks again at the next Start.
        saved = data.get("colours")
        if saved is not None and not isinstance(saved, Mapping):
            raise ConfigurationError("A saved player's colours must be a mapping or null.")
        colours = ColourSignature.from_dict(saved) if saved is not None else None
        return cls(kind, (x, y), frozenset(buttons), tried, confidence, colours)


@dataclass(frozen=True, slots=True)
class _Outcome:
    path: dict[int, tuple[int, int]]  # frame index -> position in first-screen coordinates
    kind: str | None
    edges: dict[int, tuple[int, int, int, int]]  # frame index -> left, top, right, bottom
    screen: dict[int, tuple[int, int]]  # frame index -> top-left on the screen itself

    def box(self, i: int) -> Box:
        """Where it was on the screen itself at frame ``i``: x, y, width, height."""
        left, top, right, bottom = self.edges[i]
        return (*self.screen[i], right - left, bottom - top)

    @property
    def moved_on_screen(self) -> bool:
        """It moved on the screen itself. A score, a timer or a life counter stays put
        on the screen while the camera follows the player, so in the level it seems to
        move with the buttons; the player moves on the screen in some run (T-977)."""
        xs = [p[0] for p in self.screen.values()]
        ys = [p[1] for p in self.screen.values()]
        return max(xs) - min(xs) >= MOVED or max(ys) - min(ys) >= MOVED

    @property
    def start(self) -> tuple[int, int]:
        return self.path[min(self.path)]

    @property
    def moved(self) -> bool:
        """It went somewhere: both its edges moved, across or up and down (T-977).

        A bar that grows (a score, a timer) or a row of lives that shrinks keeps one
        edge where it is; on the practice racer the score bar answered to every
        button once kinds stopped changing with every new length.
        """

        def spread(i: int) -> int:
            values = [e[i] for e in self.edges.values()]
            return max(values) - min(values)

        across = spread(0) >= MOVED and spread(2) >= MOVED
        updown = spread(1) >= MOVED and spread(3) >= MOVED
        return across or updown


def _outcomes(frames: Sequence[Frame], catalogue: ObjectCatalogue) -> list[_Outcome]:
    """Each thing followed through a run, placed in the first screen's coordinates.

    The tracker counts the camera from the first frame, so a level position is
    a position on the first screen; a thing cut out of the scenery by its motion
    (a hero standing on the ground) first shows up a frame or two late.
    """
    if not frames:
        raise ConfigurationError("A run must give back at least one frame.")
    # Not filled out by their colours (T-1016): a thing half off the screen's edge would
    # be a patch that grows or shrinks there, and its path is what the runs are compared by.
    tracker = ObjectTracker(catalogue, whole=False)
    paths: dict[int, dict[int, tuple[int, int]]] = {}
    edges: dict[int, dict[int, tuple[int, int, int, int]]] = {}
    screen: dict[int, dict[int, tuple[int, int]]] = {}
    kinds: dict[int, str | None] = {}
    for i, frame in enumerate(frames):
        for t in tracker.update(frame):
            if t.missing == 0:
                _, lx, ly = t.history[-1]
                paths.setdefault(t.id, {})[i] = (lx, ly)
                left, top = lx - t.width // 2, ly - t.height // 2
                edges.setdefault(t.id, {})[i] = (left, top, left + t.width, top + t.height)
                screen.setdefault(t.id, {})[i] = (t.x, t.y)
                if t.kind is not None:
                    kinds[t.id] = t.kind
    return [_Outcome(path, kinds.get(tid), edges[tid], screen[tid]) for tid, path in paths.items()]


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
        screens = run(frozenset({button}), frames)
        for o in _outcomes(screens, catalogue):
            if not o.moved or _does_the_same(o, base):
                continue
            group = next((g for g in groups if _same(o.start, g.start, SAME_START + MOVED)), None)
            if group is None:
                group = _Group(o.start, {button}, o.kind, o.moved_on_screen)
                groups.append(group)
            else:
                group.buttons.add(button)
                group.kind = group.kind if group.kind is not None else o.kind
                group.on_screen = group.on_screen or o.moved_on_screen
            # Where it was seen moving, screen by screen: what its colours are learnt from.
            group.seen.extend((screens[i], o.box(i)) for i in sorted(o.path))
    # A thing that moved on the screen itself beats one that only seemed to move
    # because the camera did (a score, a timer); a game whose camera always centres
    # the player has none such, and then the level's view decides (T-977).
    if any(g.on_screen for g in groups):
        groups = [g for g in groups if g.on_screen]
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
    return PlayerFinding(
        best.kind,
        best.start,
        frozenset(best.buttons),
        len(chosen),
        confidence,
        ColourSignature.learn(best.seen),
    )


@dataclass(slots=True)
class _Group:
    """One thing as seen across the button runs (matched by where it started)."""

    start: tuple[int, int]
    buttons: set[str]
    kind: str | None
    on_screen: bool = False  # moved on the screen itself in some run
    seen: list[Sample] = field(default_factory=list)  # screens and the box it moved in


def _same(a: tuple[int, int], b: tuple[int, int], within: int) -> bool:
    return abs(a[0] - b[0]) <= within and abs(a[1] - b[1]) <= within


def emulator_run(connection: ConnectionService, slot: str, *, every: int = DEFAULT_EVERY) -> Run:
    """A ``run`` for :func:`find_player` that plays the emulator from save slot ``slot``.

    Each run loads the slot, takes a screenshot, then holds the buttons for
    ``every`` frames per further screenshot (one bundled step each), and lets
    go of the pad at the end.
    """
    if not isinstance(every, int) or every < 1:
        raise ConfigurationError("every must be at least 1.")
    known = {b.name: b for b in connection.call(lambda a: a.buttons())}

    def run(held: frozenset[str], frames: int) -> list[Frame]:
        unknown = sorted(held - known.keys())
        if unknown:
            raise ConfigurationError(f"Unknown buttons for this game: {unknown}.")
        pressed = tuple(known[n] for n in sorted(held))
        connection.call(lambda a: a.load_state(slot))
        out = [connection.call(lambda a: a.capture_frame())]
        for _ in range(frames - 1):
            result = connection.call(lambda a: a.step(every, pressed=pressed, screenshot=True))
            if result.frame is None:
                raise ConfigurationError("The emulator gave no screenshot back.")
            out.append(result.frame)
        connection.call(lambda a: a.set_input(()))
        return out

    return run


def find_player_in_game(
    connection: ConnectionService,
    buttons: Iterable[str],
    *,
    frames: int = DEFAULT_FRAMES,
    every: int = DEFAULT_EVERY,
    catalogue: ObjectCatalogue | None = None,
) -> PlayerFinding | None:
    """Run :func:`find_player` on the connected game from where it is now, and put it back.

    Needs screenshots, save states and input; without them the answer is None.
    """
    caps = connection.call(lambda a: a.capabilities())
    if not (caps.screen_capture and caps.save_states and caps.input and caps.frame_advance):
        log.info("This emulator can't show the AI its screen from a save state.")
        return None
    connection.call(lambda a: a.save_state(PLAYER_SLOT))
    try:
        return find_player(
            emulator_run(connection, PLAYER_SLOT, every=every),
            buttons,
            frames=frames,
            catalogue=catalogue,
        )
    finally:
        connection.call(lambda a: a.load_state(PLAYER_SLOT))


__all__ = [
    "DEFAULT_EVERY",
    "DEFAULT_FRAMES",
    "PLAYER_SLOT",
    "PlayerFinding",
    "Run",
    "emulator_run",
    "find_player",
    "find_player_in_game",
]
