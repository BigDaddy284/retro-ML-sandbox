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

After a cut to another place the camera's travel is no longer known. Where the
caller only knows that much it says so with :meth:`Ground.back`, and the place
is taken to be the door of the place it was in: the furthest reached is kept,
so ground walked again pays nothing.

A go can pass through more places than one: a level's end leads to the next, a
door to a room. Counted from the door of the place before, a new level paid
nothing until it was as long as the last (T-993 said so: "that under-pays a new
room"), and on a real game that was every level after the first. So the ground
is kept place by place (T-1038). :meth:`Ground.new_place` says it is somewhere
it has not been this go: at that place's door, with nothing reached there, so
its first step pays. :meth:`Ground.go_to` says it is back at the door of a place
it has been in: what it reached there stays reached. Which place a screen shows
is the caller's to tell (:mod:`retro_ai_lab.observation.places`); this only
counts.

The furthest reached and the steps counted are kept twice (T-1028): for the go,
which is what pays, and for the story of the go, which goes back with the game
when it is put back to a saved moment. A go that was put back walks ground it
has been paid for, and is paid nothing; the same moves played straight from the
start would be paid there. :attr:`Ground.again` says how many steps that is at
each look, for whoever compares the go with one played straight.

Whether there is new ground to be had at all is another matter (T-1058). Where
the camera has not moved for a good while (:attr:`Ground.still`), going on gets
nowhere new: a game of one screen once its floor is walked, a room that stands
still in a game that scrolls. Whoever asks for new ground as proof of getting
past something asks in vain there, and can know it from this.

Getting past something is another matter again (T-1062). A search after a life
lost takes a try for the way past when it stays alive and gets somewhere new, and
after the game has put the player back at a place's door there is nowhere new
within reach: the go has been beyond. So a third count is kept, for the life the
go is on. :meth:`Ground.new_life` starts it over when the caller takes the game
to have put the player back, and :attr:`Ground.fresh` says at every look how many
steps the life had not covered. Until the first put-back it is the go's own
count. What the life before had reached when it went is kept too: a moment
saved in that life (:attr:`Kept.life`) can still be gone back to, by a try or by
the go, and that life is then on again as far as it had got. Nothing of this is
paid for: what pays is the go's new ground.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from retro_ai_lab.core import ConfigurationError

MOST_A_LOOK = 4  # steps paid for one look at most: a misread jump is not a fortune
# Looks in a row with the camera not seen to move before the screen is taken to stand
# still (T-1058): 20 seconds of play at a look every 8 frames.
STILL_LOOKS = 150
SMALLEST = 4  # pixels: the step is the player's own larger side, kept within these
LARGEST = 32

RIGHT, LEFT, DOWN, UP = range(4)
NOWHERE = (0, 0, 0, 0)  # nothing reached, nothing counted

_Four = tuple[int, int, int, int]


def _pair(value: object, what: str) -> tuple[int, int]:
    if (
        not isinstance(value, tuple)
        or len(value) != 2
        or any(isinstance(v, bool) or not isinstance(v, int) for v in value)
    ):
        raise ConfigurationError(f"{what} is (across, down) in whole pixels.")
    return value


@dataclass(frozen=True, slots=True)
class Kept:
    """Where a :class:`Ground` was at one moment (T-1000), to be put back there when
    the game is: the camera's travel, where on the screen the player was first seen,
    and where it stood from there; and how far the story of the go had got by then
    and what it had counted (T-1028), in pixels and steps: right, left, down, up; and
    which of the go's lives it was on (T-1062)."""

    camera: tuple[int, int]
    first: tuple[int, int] | None
    offset: tuple[int, int]
    reach: tuple[int, int, int, int] = (0, 0, 0, 0)
    counted: tuple[int, int, int, int] = (0, 0, 0, 0)
    # The place it was in, and how far the story had got in every place the go had
    # been in by then, in the order of their numbers (T-1038).
    place: int = 0
    elsewhere: tuple[tuple[_Four, _Four], ...] = ()
    # Which of the go's lives it was on (T-1062): 0 until the game first put it back.
    life: int = 0


class _Place:
    """What the go, the story of the go and the life it is on reached and counted in
    one place."""

    __slots__ = (
        "life_paid",
        "life_reach",
        "paid",
        "reach",
        "story_paid",
        "story_reach",
        "was_paid",
        "was_reach",
    )

    def __init__(self) -> None:
        self.reach = [0, 0, 0, 0]  # pixels: right, left, down, up
        self.paid = [0, 0, 0, 0]  # steps of each already counted
        # The same two for the story of the go, which goes back with the game (T-1028).
        self.story_reach = [0, 0, 0, 0]
        self.story_paid = [0, 0, 0, 0]
        # And for the life it is on (T-1062): what it has reached since the game last
        # put it back. With what the life before had reached when it went: a saved
        # moment of that life can still be gone back to.
        self.life_reach = [0, 0, 0, 0]
        self.life_paid = [0, 0, 0, 0]
        self.was_reach = [0, 0, 0, 0]
        self.was_paid = [0, 0, 0, 0]

    def copy(self) -> _Place:
        other = _Place()
        other.reach, other.paid = list(self.reach), list(self.paid)
        other.story_reach, other.story_paid = list(self.story_reach), list(self.story_paid)
        other.life_reach, other.life_paid = list(self.life_reach), list(self.life_paid)
        other.was_reach, other.was_paid = list(self.was_reach), list(self.was_paid)
        return other

    def new_life(self) -> None:
        """The life it is on starts over; what it had reached is the life before's."""
        self.was_reach, self.was_paid = self.life_reach, self.life_paid
        self.life_reach, self.life_paid = [0, 0, 0, 0], [0, 0, 0, 0]

    def life_before(self) -> None:
        """The life before is on again, as far as it had got."""
        self.life_reach, self.life_paid = list(self.was_reach), list(self.was_paid)

    def story(self) -> tuple[_Four, _Four]:
        reach, counted = self.story_reach, self.story_paid
        return (
            (reach[RIGHT], reach[LEFT], reach[DOWN], reach[UP]),
            (counted[RIGHT], counted[LEFT], counted[DOWN], counted[UP]),
        )

    def story_back(self, reach: Sequence[int], counted: Sequence[int]) -> None:
        """The story goes back to what it had here; never ahead of the go's own (a
        place kept in another go would be)."""
        self.story_reach = [min(far, mine) for far, mine in zip(reach, self.reach, strict=True)]
        self.story_paid = [min(had, mine) for had, mine in zip(counted, self.paid, strict=True)]


class Ground:
    """The new ground of one go. Call :meth:`begin` when a go starts, :meth:`look`
    at every look, :meth:`back` when the screen jumped and it is only known to be
    somewhere else, and :meth:`new_place` or :meth:`go_to` when it is known where."""

    def __init__(self, *, most: int = MOST_A_LOOK) -> None:
        if isinstance(most, bool) or not isinstance(most, int) or most < 1:
            raise ConfigurationError("most must be a whole number, 1 or more.")
        self._most = most
        self._unit: int | None = None  # pixels in a step; kept from go to go
        self._scrolls = False  # the camera has been seen to move; kept from go to go
        self._quiet = 0  # looks since it was last seen to move; kept from go to go (T-1058)
        self._camera = (0, 0)  # the camera's travel since the go began or it was put back
        self._first: tuple[int, int] | None = None  # where on the screen it was first seen
        self._offset = (0, 0)  # where it stands on the screen, from there
        # The places the go has been in, in the order it came to them, and the one
        # it is in (T-1038).
        self._places = [_Place()]
        self._here = self._places[0]
        self._again = 0
        self._fresh = 0
        self._life = 0  # lives the game has put it back after, this go (T-1062)

    @property
    def unit(self) -> int | None:
        """Pixels in one step: the player's larger side when first seen; None before."""
        return self._unit

    @property
    def scrolls(self) -> bool:
        """The screen has been seen to scroll: only the camera's travel counts."""
        return self._scrolls

    @property
    def still(self) -> bool:
        """The camera has not been seen to move for ``STILL_LOOKS`` looks running, this
        go's and the goes' before it (T-1058): a game of one screen, or a room that
        stands still in a game that scrolls. There is no new ground to be got to
        there by going anywhere. One move of the camera ends it."""
        return self._quiet >= STILL_LOOKS

    @property
    def place(self) -> tuple[int, int]:
        """Where it is, in pixels across and down from where the go began: the camera's
        travel on a screen that scrolls, where it stands on one that never has."""
        return self._camera if self._scrolls else self._offset

    @property
    def reach(self) -> tuple[int, int, int, int]:
        """The furthest it has been this go in the place it is in, in pixels: right,
        left, down, up."""
        reach = self._here.reach
        return (reach[RIGHT], reach[LEFT], reach[DOWN], reach[UP])

    @property
    def steps(self) -> int:
        """Steps of new ground counted this go, in every place."""
        return sum(sum(place.paid) for place in self._places)

    @property
    def places(self) -> int:
        """How many places the go has been in (T-1038): one until it is told of more."""
        return len(self._places)

    @property
    def place_number(self) -> int:
        """Which of them it is in: 0 is where the go began."""
        return self._places.index(self._here)

    @property
    def again(self) -> int:
        """Steps of the last look that the go had counted before it was put back and
        its story had not (T-1028): a go played straight would be paid them here.
        Nothing while the go has never been put back."""
        return self._again

    @property
    def fresh(self) -> int:
        """Steps of the last look that the life it is on had not covered (T-1062)."""
        return self._fresh

    def new_life(self) -> None:
        """The game put it back after a life lost (T-1062): what the life it is on has
        reached starts over, in every place, and what it had reached is kept as the
        life before's. Said with :meth:`back` or :meth:`go_to`, which say where the
        game put it."""
        for place in self._places:
            place.new_life()
        self._life += 1

    def begin(self) -> None:
        """A go starts: nothing reached, nothing counted. The step's size is kept, and
        so is whether the screen scrolls."""
        self._places = [_Place()]
        self._here = self._places[0]
        self.back()
        self._again = 0
        self._fresh = 0
        self._life = 0

    def back(self) -> None:
        """It is back at the door of the place it is in: where the go began, or where
        it came into this place (the screen jumped, and no more is known). What was
        reached stays reached."""
        self._camera = (0, 0)
        self._first = None
        self._offset = (0, 0)

    def new_place(self) -> int:
        """It is somewhere it has not been this go (T-1038): at that place's door, with
        nothing reached there. Returns the place's number."""
        self._here = _Place()
        self._places.append(self._here)
        self.back()
        return len(self._places) - 1

    def go_to(self, number: int) -> None:
        """It is at the door of place ``number``, which it has been in this go (T-1038):
        what it reached there stays reached."""
        if (
            isinstance(number, bool)
            or not isinstance(number, int)
            or not 0 <= number < len(self._places)
        ):
            raise ConfigurationError(f"This go has no place numbered {number!r}.")
        self._here = self._places[number]
        self.back()

    def keep(self) -> Kept:
        """Where it is now, to be put back to (T-1000)."""
        reach, counted = self._here.story()
        return Kept(
            self._camera,
            self._first,
            self._offset,
            reach,
            counted,
            self.place_number,
            tuple(place.story() for place in self._places),
            self._life,
        )

    def put_back(self, kept: Kept) -> None:
        """The game was put back to the moment ``kept`` was taken: that is the place
        again. What was reached stays reached and what was counted stays counted, so
        ground walked again pays nothing and the first step past the furthest does.
        The story's reach and count go back to what they were then (T-1028), in
        every place: nothing in one the go came to later. It is in the place it was
        in then (T-1038); one this go has not got (kept in another go) is made, with
        nothing reached. A moment of an earlier life than the one it is on (T-1062)
        puts the life before back on, with what that had reached when it went, in
        every place: the game is where that life was. Only the life before is kept;
        a moment of the life it is on, or of a later one, changes nothing of that."""
        if not isinstance(kept, Kept):
            raise ConfigurationError("Only a place this kept can be put back.")
        if isinstance(kept.place, bool) or not isinstance(kept.place, int) or kept.place < 0:
            raise ConfigurationError("A kept place is numbered 0 or more.")
        if isinstance(kept.life, bool) or not isinstance(kept.life, int) or kept.life < 0:
            raise ConfigurationError("A kept life is numbered 0 or more.")
        while len(self._places) <= kept.place:
            self._places.append(_Place())
        self._here = self._places[kept.place]
        self._camera, self._first, self._offset = kept.camera, kept.first, kept.offset
        for number, place in enumerate(self._places):
            if place is self._here:
                place.story_back(kept.reach, kept.counted)
            elif number < len(kept.elsewhere):
                place.story_back(*kept.elsewhere[number])
            else:
                place.story_back(NOWHERE, NOWHERE)
        if kept.life < self._life:
            for place in self._places:
                place.life_before()
            self._life = kept.life
        self._again = 0

    def copy(self) -> Ground:
        """Another count in the same state, for a try that must not change this one."""
        other = Ground(most=self._most)
        other._unit, other._scrolls, other._quiet = self._unit, self._scrolls, self._quiet
        other._camera, other._first, other._offset = self._camera, self._first, self._offset
        other._places = [place.copy() for place in self._places]
        other._here = other._places[self.place_number]
        other._again = self._again
        other._fresh = self._fresh
        other._life = self._life
        return other

    def look(
        self,
        moved: tuple[int, int],
        me: tuple[int, int] | None = None,
        size: int | None = None,
    ) -> int:
        """One look: the camera ``moved`` since the last one, and the player's middle
        is at ``me`` on the screen (None: not seen) and ``size`` pixels on its larger
        side. Returns the steps of new ground, ``most`` at most. :attr:`again` then
        says how many more its story covered, and :attr:`fresh` how many the life it
        is on had not (T-1062)."""
        dx, dy = _pair(moved, "The camera's move")
        if me is not None:
            _pair(me, "Where the player is")
        if size is not None and (isinstance(size, bool) or not isinstance(size, int) or size < 1):
            raise ConfigurationError("The player's size is a whole number of pixels, 1 or more.")
        self._camera = (self._camera[0] + dx, self._camera[1] + dy)
        if dx or dy:
            self._scrolls = True
            self._quiet = 0
        else:
            self._quiet += 1
        if me is not None:
            if self._first is None:
                self._first = me
            self._offset = (me[0] - self._first[0], me[1] - self._first[1])
            if self._unit is None and size is not None:
                self._unit = max(SMALLEST, min(LARGEST, size))
        x, y = self.place
        here = self._here
        for way, far in ((RIGHT, x), (LEFT, -x), (DOWN, y), (UP, -y)):
            here.reach[way] = max(here.reach[way], far)
            here.story_reach[way] = max(here.story_reach[way], far)
            here.life_reach[way] = max(here.life_reach[way], far)
        if self._unit is None:
            return 0  # no step to count in yet; what is reached is counted when there is
        new = told = fresh = 0
        for way in (RIGHT, LEFT, DOWN, UP):
            steps = here.life_reach[way] // self._unit
            fresh += steps - here.life_paid[way]
            here.life_paid[way] = steps
            steps = here.reach[way] // self._unit
            new += steps - here.paid[way]
            here.paid[way] = steps  # past the most, the rest is not owed
            steps = here.story_reach[way] // self._unit
            told += steps - here.story_paid[way]
            here.story_paid[way] = steps
        new = min(new, self._most)
        self._fresh = min(fresh, self._most)
        # The story is never behind the go's own count, so this is never below zero.
        self._again = min(told, self._most) - new
        return new


__all__ = [
    "DOWN",
    "LARGEST",
    "LEFT",
    "MOST_A_LOOK",
    "RIGHT",
    "SMALLEST",
    "UP",
    "Ground",
    "Kept",
]
