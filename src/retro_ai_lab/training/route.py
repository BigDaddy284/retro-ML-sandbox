"""The furthest way played so far, to begin the next go with (T-1014).

A go that loses a life goes back and finds a way past (T-1007), three times a
go at most. Nothing of that helped the next go: it met the same places and
spent its three searches on them again, so however long a go might last it
never got past the first few. A person who has found the way through the
start of a level does not look for it again each time: they play what they
know and practise where it ends.

So the runner keeps the moves of the go that counted the most new ground, up
to the step it last got somewhere new, and the next go, which starts from the
same saved spot, begins by playing them. Played again from the same save a
game does the same again, so the go arrives at the end of what it knows alive
and with its searches unused. The moves go through the ordinary step each
time: they are practice for the learner too.

This module is the bookkeeping and nothing else: no emulator, no game and no
button is named here.

- The *way* is what is remembered: moves from the start, and the new ground
  counted after each.
- The *story* is the go being played: the same, so far.

Only moves the AI made itself can be played again. A step somebody else drove,
or a button pressed to get a stuck game going, ends the story where it is.
"""

from __future__ import annotations

from collections.abc import Hashable

from retro_ai_lab.core import ConfigurationError


def _whole(value: object, what: str, least: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < least:
        raise ConfigurationError(f"{what} must be a whole number, {least} or more.")
    return value


class Route:
    """The remembered way, and the story of the go being played."""

    def __init__(self) -> None:
        self._start: Hashable | None = None  # where the remembered way starts from
        self._moves: list[int] = []
        self._ground: list[int] = []  # new ground counted after each move of the way
        self._story: list[int] = []
        self._story_ground: list[int] = []
        self._live = False  # the story can still be added to
        self._playing = 0  # steps of the way this go began with

    # ------------------------------------------------------------ what is remembered

    @property
    def moves(self) -> tuple[int, ...]:
        """The remembered way: the moves to play from the start."""
        return tuple(self._moves)

    @property
    def ground(self) -> int:
        """The new ground the remembered way counted, in the watch's steps."""
        return self._ground[-1] if self._ground else 0

    @property
    def story(self) -> tuple[int, ...]:
        """The moves of the go being played that could be played again."""
        return tuple(self._story)

    @property
    def playing(self) -> int:
        """How many steps of the remembered way this go began with."""
        return self._playing

    def forget(self) -> None:
        """Drop the remembered way: it no longer leads anywhere."""
        self._moves.clear()
        self._ground.clear()
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
        self._story_ground.clear()
        self._live = start is not None
        self._playing = 0
        if start is None:
            return ()
        if start != self._start:
            self.forget()
            self._start = start
        self._playing = len(self._moves)
        return tuple(self._moves)

    def did(self, move: int | None, ground: int) -> None:
        """One step of the go was played with ``move``; ``ground`` is the new ground
        the go has counted so far.

        ``None`` is a step that cannot be played again: the story ends before it.
        """
        _whole(ground, "The ground counted")
        if not self._live:
            return
        if move is None:
            self._live = False
            return
        self._story.append(_whole(move, "A move"))
        self._story_ground.append(ground)

    def went_back(self, steps: int) -> None:
        """The game was put back ``steps`` steps: they are no longer the story."""
        _whole(steps, "The steps gone back")
        if not self._live:
            return
        if steps > len(self._story):
            raise ConfigurationError("The story is not that long.")
        keep = len(self._story) - steps
        del self._story[keep:]
        del self._story_ground[keep:]

    def lost(self, lead: int) -> None:
        """A life was lost and the loss stands: the story ends ``lead`` steps before
        it, so that playing it again arrives short of the place, alive."""
        _whole(lead, "The lead")
        if not self._live:
            return
        keep = max(0, len(self._story) - lead)
        del self._story[keep:]
        del self._story_ground[keep:]
        self._live = False

    def failed(self, lead: int) -> bool:
        """A life was lost while the remembered way was being played: it does not
        hold there. The way is cut ``lead`` steps before the step of the loss (the
        last one of the story). True when something was cut."""
        _whole(lead, "The lead")
        keep = max(0, len(self._story) - lead)
        if keep >= len(self._moves):
            return False
        del self._moves[keep:]
        del self._ground[keep:]
        return True

    def end(self) -> bool:
        """The go is over. Its story, up to the step it last counted new ground, becomes
        the remembered way if it counted more than that did. True when it did."""
        story, counted = self._story, self._story_ground
        self._live = False
        if not counted or self._start is None:
            return False
        best = max(counted)
        if best <= self.ground:
            return False
        upto = counted.index(best) + 1
        self._moves = story[:upto]
        self._ground = counted[:upto]
        return True


__all__ = ["Route"]
