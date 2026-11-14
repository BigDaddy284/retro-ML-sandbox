"""The best way played so far, to begin the next go with (T-1014, T-1015).

A go that loses a life goes back and finds a way past (T-1007), three times a
go at most. Nothing of that helped the next go: it met the same places and
spent its three searches on them again, so however long a go might last it
never got past the first few. A person who has found the way through the
start of a level does not look for it again each time: they play what they
know and practise where it ends.

So the runner keeps the moves of the go that was paid the most, up to the step
at which it had been paid the most, and the next go, which starts from the same
saved spot, begins by playing them. What a go is paid is the person's rewards
added up (T-1015; it was the new ground counted off the screen, and on a game
whose point is not getting far the AI replayed a way to nowhere): a way is only
remembered for what the person pays for, and with nothing paid above zero there
is no way. Played again from the same save a
game does the same again, so the go arrives at the end of what it knows alive
and with its searches unused. The moves go through the ordinary step each
time: they are practice for the learner too.

This module is the bookkeeping and nothing else: no emulator, no game and no
button is named here.

- The *way* is what is remembered: moves from the start, and what the go had
  been paid after each.
- The *story* is the go being played: the same, so far.

Only moves the AI made itself can be played again. A step somebody else drove,
or a button pressed to get a stuck game going, ends the story where it is.

A way is worth what it is paid now (T-1026). What a go is paid changes under it:
the AI drops a guess a reward went by, a goal is reached once and for all. A way
remembered with such pay in it kept the old figure, and no later go could match
it: on one practice level nine goes in a row got further than the way and were
turned down. So while a go plays the way it began with, the route notes what the
go has been paid after each step of it, and a go that played all of it and was
paid less for it than the way was remembered for makes the way again: that go's
figures, up to the step at which it had been paid the most. A way paid as much
at its end as it was remembered for, or more, is left as it is: played again a
way is often paid more (the go that found it went back, and ground walked again
is not paid twice), and that says nothing against it.
"""

from __future__ import annotations

import math
from collections.abc import Hashable

from retro_ai_lab.core import ConfigurationError


def _whole(value: object, what: str, least: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < least:
        raise ConfigurationError(f"{what} must be a whole number, {least} or more.")
    return value


def _amount(value: object, what: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ConfigurationError(f"{what} must be a number.")
    return float(value)


class Route:
    """The remembered way, and the story of the go being played."""

    def __init__(self) -> None:
        self._start: Hashable | None = None  # where the remembered way starts from
        self._moves: list[int] = []
        self._paid: list[float] = []  # what the go had been paid after each move of the way
        self._story: list[int] = []
        self._story_paid: list[float] = []
        self._live = False  # the story can still be added to
        self._playing = 0  # steps of the way this go began with
        # What this go had been paid after each step of the way, noted for as long as
        # it has played nothing but the way (T-1026).
        self._again: list[float] = []
        self._on_way = False
        self._shortened = 0

    # ------------------------------------------------------------ what is remembered

    @property
    def moves(self) -> tuple[int, ...]:
        """The remembered way: the moves to play from the start."""
        return tuple(self._moves)

    @property
    def paid(self) -> float:
        """What the go the remembered way came from had been paid at its end."""
        return self._paid[-1] if self._paid else 0.0

    @property
    def counted(self) -> float:
        """What the story has been paid so far: steps gone back over are not in it."""
        return self._story_paid[-1] if self._story_paid else 0.0

    @property
    def story(self) -> tuple[int, ...]:
        """The moves of the go being played that could be played again."""
        return tuple(self._story)

    @property
    def playing(self) -> int:
        """How many steps of the remembered way this go began with."""
        return self._playing

    @property
    def shortened(self) -> int:
        """How many steps the way lost when the last go ended, because played again
        they were paid nothing more (T-1026). 0 from the next go's beginning."""
        return self._shortened

    def forget(self) -> None:
        """Drop the remembered way: it no longer leads anywhere."""
        self._moves.clear()
        self._paid.clear()
        self._start = None

    # ------------------------------------------------------------ a go

    def begin(self, start: Hashable | None) -> tuple[int, ...]:
        """A go begins from ``start``, the saved spot it was put back to. Returns the
        moves to play first.

        ``None`` is a go that was not put back to a saved spot: nothing can be
        played again from wherever the game happens to be, and nothing of such a
        go is remembered. A way remembered from another start is forgotten.
        """
        self._story.clear()
        self._story_paid.clear()
        self._live = start is not None
        self._playing = 0
        self._again.clear()
        self._on_way = True
        self._shortened = 0
        if start is None:
            return ()
        if start != self._start:
            self.forget()
            self._start = start
        self._playing = len(self._moves)
        return tuple(self._moves)

    def did(self, move: int | None, paid: float) -> None:
        """One step of the go was played with ``move``; ``paid`` is what the story has
        been paid so far, this step included (:attr:`counted` plus this step's pay).

        ``None`` is a step that cannot be played again: the story ends before it.
        """
        paid = _amount(paid, "What the go has been paid")
        if not self._live:
            return
        if move is None:
            self._live = False
            return
        self._story.append(_whole(move, "A move"))
        self._story_paid.append(paid)
        at = len(self._story) - 1
        if (
            self._on_way
            and at == len(self._again)  # no step has been gone back over
            and at < len(self._moves)
            and self._moves[at] == move
        ):
            self._again.append(paid)
        else:
            self._on_way = False  # it has played something else: no longer the way's pay

    def went_back(self, steps: int) -> None:
        """The game was put back ``steps`` steps: they are no longer the story."""
        _whole(steps, "The steps gone back")
        if not self._live:
            return
        if steps > len(self._story):
            raise ConfigurationError("The story is not that long.")
        keep = len(self._story) - steps
        del self._story[keep:]
        del self._story_paid[keep:]

    def lost(self, lead: int) -> None:
        """A life was lost and the loss stands: the story ends ``lead`` steps before
        it, so that playing it again arrives short of the place, alive."""
        _whole(lead, "The lead")
        if not self._live:
            return
        keep = max(0, len(self._story) - lead)
        del self._story[keep:]
        del self._story_paid[keep:]
        self._live = False

    def failed(self, lead: int) -> bool:
        """A life was lost on the step just played (the last one of the story), and the
        remembered way does not hold there: it was being played, or it ends within
        ``lead`` steps before the loss and no way past was found. The way is cut
        ``lead`` steps before the loss. True when something was cut.

        Nothing is cut when the way ends earlier than that, when the go had left the
        way before there (it went back and played something else), or when the story
        is over and no longer says where the go is.
        """
        _whole(lead, "The lead")
        if not self._live:
            return False
        keep = max(0, len(self._story) - lead)
        if keep >= len(self._moves) or self._story[:keep] != self._moves[:keep]:
            return False
        del self._moves[keep:]
        del self._paid[keep:]
        return True

    def end(self) -> bool:
        """The go is over. Its story, up to the step at which it had been paid the most,
        becomes the remembered way if that is more than the way was paid. True when
        it did.

        A story that began by playing the whole way is measured against what the way
        was paid in this go, not in the go it came from: a reward paid once only (a
        goal reached for the first time) would otherwise leave a way no later go can
        match.

        Before that, a way this go played whole and was paid less for than it was
        remembered for is made again from this go (T-1026, :meth:`_paid_now`).
        """
        story, paid = self._story, self._story_paid
        self._live = False
        again, self._again = self._again, []  # a second end() has nothing to go by
        self._paid_now(again)
        if not paid or self._start is None:
            return False
        best = max(paid)
        upto = paid.index(best) + 1
        had = len(self._moves)
        if had and story[:had] == self._moves:
            if upto <= had or best <= paid[had - 1]:
                return False
        elif best <= self.paid:
            return False
        self._moves = story[:upto]
        self._paid = paid[:upto]
        self._shortened = 0  # whatever was dropped above, this is another way
        return True

    def _paid_now(self, again: list[float]) -> None:
        """The go is over, and ``again`` is what it had been paid after each step of the
        way while it played nothing else. If that is the whole way the go began with
        (a way cut for a lost life in this go ends where the loss says, and is left)
        and the go was paid less at its end than the way was remembered for, the way
        becomes what this go says of it: its moves up to the first step at which the
        go had been paid the most, with this go's figures. With nothing paid above
        zero there is no way.

        A way paid as much at its end, or more, is left as it is remembered.
        """
        was = len(self._moves)
        if not was or len(again) != was or was != self._playing:
            return
        if again[-1] >= self._paid[-1]:
            return
        most = max(again)
        keep = again.index(most) + 1 if most > 0 else 0
        self._shortened = was - keep
        self._moves = self._moves[:keep]
        self._paid = again[:keep]


__all__ = ["Route"]
