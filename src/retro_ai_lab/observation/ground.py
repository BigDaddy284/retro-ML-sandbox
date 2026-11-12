"""How much new ground a go has covered, measured off the screen (T-993).

Nothing in a game's memory has to be understood for this. On a game whose
screen scrolls, somewhere new is what the camera has newly shown: how far it
has travelled since the go began (the tracker measures that between every two
looks, T-992). On a screen that has never scrolled it is where the player
newly stands, counted from where it was first seen. :class:`Ground` keeps the
furthest that place has been to the right, to the left, down and up, and
counts each step of the player's own size past it once.

No direction is named as the way on and no game is known: a level that runs to
the left, or up, pays the same. Once the screen has been seen to scroll, where
the player stands on it no longer counts, for good: a jump and a fall into a
pit move the player and not the camera, and as first written (the camera's
travel plus the place on the screen) a fall paid its two or three steps down
at once, about what the life it cost took away a second later. On a screen
that never scrolls a first jump and a first fall still pay their steps, once a
go.

After a cut to another place the camera's travel is no longer known. The
caller says so with :meth:`Ground.back`, and the place is taken to be where the
go began: the furthest reached is kept, so ground walked again pays nothing,
and a new room pays only once it is as far from its door as the go's furthest.
That under-pays a new room and never pays the same ground twice.
"""

from __future__ import annotations

from retro_ai_lab.core import ConfigurationError

MOST_A_LOOK = 4  # steps paid for one look at most: a misread jump is not a fortune
SMALLEST = 4  # pixels: the step is the player's own larger side, kept within these
LARGEST = 32

RIGHT, LEFT, DOWN, UP = range(4)


def _pair(value: object, what: str) -> tuple[int, int]:
    if (
        not isinstance(value, tuple)
        or len(value) != 2
        or any(isinstance(v, bool) or not isinstance(v, int) for v in value)
    ):
        raise ConfigurationError(f"{what} is (across, down) in whole pixels.")
    return value


class Ground:
    """The new ground of one go. Call :meth:`begin` when a go starts, :meth:`look`
    at every look and :meth:`back` when the screen jumped to another place."""

    def __init__(self, *, most: int = MOST_A_LOOK) -> None:
        if isinstance(most, bool) or not isinstance(most, int) or most < 1:
            raise ConfigurationError("most must be a whole number, 1 or more.")
        self._most = most
        self._unit: int | None = None  # pixels in a step; kept from go to go
        self._scrolls = False  # the camera has been seen to move; kept from go to go
        self._camera = (0, 0)  # the camera's travel since the go began or it was put back
        self._first: tuple[int, int] | None = None  # where on the screen it was first seen
        self._offset = (0, 0)  # where it stands on the screen, from there
        self._reach = [0, 0, 0, 0]  # pixels: right, left, down, up
        self._paid = [0, 0, 0, 0]  # steps of each already counted

    @property
    def unit(self) -> int | None:
        """Pixels in one step: the player's larger side when first seen; None before."""
        return self._unit

    @property
    def scrolls(self) -> bool:
        """The screen has been seen to scroll: only the camera's travel counts."""
        return self._scrolls

    @property
    def place(self) -> tuple[int, int]:
        """Where it is, in pixels across and down from where the go began: the camera's
        travel on a screen that scrolls, where it stands on one that never has."""
        return self._camera if self._scrolls else self._offset

    @property
    def reach(self) -> tuple[int, int, int, int]:
        """The furthest it has been this go, in pixels: right, left, down, up."""
        return (self._reach[RIGHT], self._reach[LEFT], self._reach[DOWN], self._reach[UP])

    @property
    def steps(self) -> int:
        """Steps of new ground counted this go."""
        return sum(self._paid)

    def begin(self) -> None:
        """A go starts: nothing reached, nothing counted. The step's size is kept, and
        so is whether the screen scrolls."""
        self.back()
        self._reach = [0, 0, 0, 0]
        self._paid = [0, 0, 0, 0]

    def back(self) -> None:
        """It is back where the go began (the screen jumped to another place). What
        was reached stays reached."""
        self._camera = (0, 0)
        self._first = None
        self._offset = (0, 0)

    def look(
        self,
        moved: tuple[int, int],
        me: tuple[int, int] | None = None,
        size: int | None = None,
    ) -> int:
        """One look: the camera ``moved`` since the last one, and the player's middle
        is at ``me`` on the screen (None: not seen) and ``size`` pixels on its larger
        side. Returns the steps of new ground, ``most`` at most."""
        dx, dy = _pair(moved, "The camera's move")
        if me is not None:
            _pair(me, "Where the player is")
        if size is not None and (isinstance(size, bool) or not isinstance(size, int) or size < 1):
            raise ConfigurationError("The player's size is a whole number of pixels, 1 or more.")
        self._camera = (self._camera[0] + dx, self._camera[1] + dy)
        if dx or dy:
            self._scrolls = True
        if me is not None:
            if self._first is None:
                self._first = me
            self._offset = (me[0] - self._first[0], me[1] - self._first[1])
            if self._unit is None and size is not None:
                self._unit = max(SMALLEST, min(LARGEST, size))
        x, y = self.place
        for way, far in ((RIGHT, x), (LEFT, -x), (DOWN, y), (UP, -y)):
            self._reach[way] = max(self._reach[way], far)
        if self._unit is None:
            return 0  # no step to count in yet; what is reached is counted when there is
        new = 0
        for way in (RIGHT, LEFT, DOWN, UP):
            steps = self._reach[way] // self._unit
            new += steps - self._paid[way]
            self._paid[way] = steps  # past the most, the rest is not owed
        return min(new, self._most)


__all__ = ["DOWN", "LARGEST", "LEFT", "MOST_A_LOOK", "RIGHT", "SMALLEST", "UP", "Ground"]
