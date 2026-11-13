"""Trying the small changes to what it did until one gets past (T-1006).

When the AI loses a life, the go has a save from a little earlier and the list
of moves made since (:mod:`retro_ai_lab.training.moments`). :func:`find_way`
plays that list again from the save with one small change at a time
(:mod:`retro_ai_lab.training.ways`), off the record
(:mod:`retro_ai_lab.training.trial`), until one stays alive and reaches ground
the go had not been on.

A try is played blind first: no screenshot, one read of the life values a step.
Most tries lose the life again, and that is all there is to know about them.
One that stays alive is played once more with a screen watch, because alive is
not enough (standing still stays alive): it is the way only if the watch counts
new ground. Every play, blind or looked at, counts against the budget, so a
search always ends.

The prototype found a way in 9 to 142 tries where there was one, all of them a
move one button from the one made, held 4 to 16 steps.

What it did is not always worth changing (T-1012). A learner that wandered
about before the life went (back a little, a jump too early, a button that
does nothing) leaves a list no single small change gets across with: a third
of the searches on the practice game found nothing, nearly all of them that
way. So a search that is given a plain move, the one that has paid best,
tries the plain way first: that move held from the save, and where that loses
the life, the same small changes to it. Measured on 179 searches it finds a
way in 86% against 70% and sooner (median try 19 and 43 against 42 and 54);
the two together find one in 97%. The plain way has half the tries at most,
and what it did has the rest: it is what gets past where an earlier jump has
to stay in, and where the plain way only walks into a wall.
"""

from __future__ import annotations

import hashlib
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace

from retro_ai_lab.core import ConfigurationError
from retro_ai_lab.learning.actions import ActionSpace
from retro_ai_lab.services.connection import ConnectionService
from retro_ai_lab.training.trial import Outcome, play
from retro_ai_lab.training.vision import DEFAULT_LOOK_EVERY, ScreenWatch
from retro_ai_lab.training.ways import Change, changes
from retro_ai_lab.understanding.knowledge import Variable

BUDGET = 120  # plays a search may make
CARRY_ON = 20  # steps a try runs on past where the life went
PLAIN_SHARE = 0.5  # of the budget: the most the plain way may use (T-1012)
# Steps past its end a way must be able to live through (T-1019): as far before a life
# lost as the app looks for its cause (``moments.LEAD``).
OUTLAST = 30
END_SLOT = "back_end"  # where the end of a try is saved while it is asked about (T-1019)
CUT_BY = 4  # steps: a way that ends badly is tried this much shorter each time
CUT_WITHIN = 40  # steps from its end: no further back is a place to stop it looked for
# The room the AI has at the end of a try (T-1019): how many of its moves, held on,
# keep the life.
NONE, TIGHT, ROOMY = 0, 1, 2


@dataclass(frozen=True, slots=True)
class Way:
    """A change that got past."""

    moves: tuple[int, ...]  # what to play from the save: the changed trace and on
    change: Change
    tries: int  # plays the search had made when it was found, its own two included
    ground: int  # steps of new ground the watch counted along it
    # The plain way (T-1012): the move held all the way, when the change is to that and
    # not to what it did, and the step holding it lost the life at. With no life lost
    # there is nothing to change: the change is then that move, held from the start.
    # At a standstill (T-1018) no life is lost either, and the change is to the plain
    # way where holding it stood still too.
    plain: int | None = None
    lost_at: int | None = None
    # Steps left off the end of the try it came from (T-1019): after them the life
    # went whatever was held, and after these moves it does not.
    cut: int = 0


@dataclass(frozen=True, slots=True)
class Search:
    """What a search came to."""

    way: Way | None
    tries: int  # plays made, blind and looked at
    # How long it took (T-1008). Not part of what was found: two searches that
    # find the same in the same tries are the same search.
    seconds: float = field(default=0.0, compare=False)
    out_of_time: bool = False  # it stopped because its time was up, not its tries
    # Tries that got past and were turned down all the same (T-1019): the life went
    # just after them whatever was held, and no part of them that it could live on
    # from got past.
    doomed: int = 0


def _whole(value: object, what: str, least: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < least:
        raise ConfigurationError(f"{what} must be a whole number, {least} or more.")
    return value


def find_way(
    connection: ConnectionService,
    slot: str,
    trace: Sequence[int],
    space: ActionSpace,
    *,
    frames_per_step: int,
    lives: Sequence[Variable],
    watch: Callable[[], ScreenWatch],
    budget: int = BUDGET,
    carry_on: int = CARRY_ON,
    look_every: int = DEFAULT_LOOK_EVERY,
    seconds: float | None = None,
    clock: Callable[[], float] = time.monotonic,
    plain: int | None = None,
    stood: bool = False,
    enough: Callable[[Outcome], bool] | None = None,
    outlast: int = 0,
) -> Search:
    """Look for a small change to ``trace``, played from the save in ``slot``, that
    stays alive and reaches new ground.

    ``trace`` is what the AI did from the save to the step the life went.
    ``lives`` are the values whose going down is a life lost; ``watch`` gives a
    fresh screen watch for one try each time it is called (the go's
    :meth:`ScreenWatch.trial`). The game is left wherever the last try ended:
    putting it back is the caller's, and so is an emulator error.

    On a real emulator a try costs real time. With ``seconds``, no play is started
    once that long has passed on ``clock`` since the search began (T-1008); a play
    in flight is finished.

    With ``plain``, a move, the plain way is tried first (T-1012): that move held
    for as long as the trace and the carry-on. Alive and on new ground, it is the
    way. Where it loses a life, the small changes are tried on the plain moves up
    to that step, with at most ``PLAIN_SHARE`` of the budget; after that, and at
    once where the plain way stays alive and gets nowhere new, the changes to
    ``trace``. Moves played once are not played again and cost no try.

    With ``stood`` the search is for a way on from where the AI stood still with no
    life lost (T-1018), and ``trace`` is what it did up to shortly after it came to
    the spot. Where the plain way stays alive and gets nowhere new (a wall), the
    small changes to the plain way are tried before the changes to ``trace``, and
    both from the start forwards: the earliest that gets on has the least standing
    in it. At a
    wall nearly every change stays alive, so the blind play weeds out next to
    nothing and every try would cost a play under the watch, some twenty times a
    blind one. So the blind play of a change also takes the screen it ends on, and a
    change that ends on the very screen a play that got nowhere ended on (holding
    the plain move, or an earlier change that was looked at) has got nowhere
    either: it is not looked at.

    ``enough`` is asked of every try that stayed alive and counted new ground, and
    False says it is no way all the same. After a life lost nobody asks: past the
    place is enough. At a standstill the caller asks that the try ended out of the
    spot it stood about: a hop of two steps and standing again is no way on.

    With ``outlast`` above 0 a way must be one the AI can live on from (T-1019). A
    try is alive while its life has not gone, and in many games the life goes a
    while after the fall or the touch that costs it: on the practice game more
    than half the ways found ended inside that while, mostly because the try,
    past the place, carried on into the next one. So a try that got past is asked
    a question at its end: each move the controls have, held for ``outlast`` steps
    from there ("do nothing" first), does it keep the life? At least half do
    (``ROOMY``), and the try is the way. Fewer (``TIGHT``: one wrong move and the
    life goes) or none (``NONE``), and the try has usually carried on into the next
    place: the way is then the try up to the last place, going back from its end
    ``CUT_BY`` steps at a time and at most ``CUT_WITHIN``, that leaves the AI room,
    if the try has got past by then under the watch. Where no part does both, a
    try that can be lived on from at all is the way as it is, and one that cannot
    is cut to its longest part that can, if that got past. ``Way.cut`` is what was left
    off. No such part, and the try is turned down (``Search.doomed``) and the
    search goes on. The end of a try is kept in the save ``END_SLOT`` while it is
    asked about.
    A question is one try of the budget, however many moves it takes.
    """
    if not slot or not isinstance(slot, str):
        raise ConfigurationError("A search needs the name of the save to start from.")
    if not trace:
        raise ConfigurationError("There is nothing it did to change.")
    if not lives:
        raise ConfigurationError(
            "A search needs a life or health value: without one nothing is seen to be lost."
        )
    if not callable(watch):
        raise ConfigurationError("watch must give a screen watch for a try when called.")
    _whole(budget, "The budget of tries", 1)
    _whole(carry_on, "carry_on", 0)
    _whole(frames_per_step, "frames_per_step", 1)
    _whole(look_every, "look_every", 1)
    if seconds is not None and (
        isinstance(seconds, bool) or not isinstance(seconds, (int, float)) or not seconds > 0
    ):
        raise ConfigurationError("The time a search may take must be a number above 0.")
    if not callable(clock):
        raise ConfigurationError("clock must give the time when called.")
    if not isinstance(stood, bool):
        raise ConfigurationError("stood must be true or false.")
    if enough is not None and not callable(enough):
        raise ConfigurationError("enough must say of a try whether it is a way, when called.")
    for move in trace:
        space[move]  # refuses a move the controls don't have, before anything is played
    if plain is not None:
        space[plain]
    _whole(outlast, "outlast", 0)
    began = clock()
    tries, late, doomed = 0, False, 0
    # The moves the question is put with: "do nothing" first, where there is one.
    every = sorted(range(len(space)), key=lambda move: bool(space.buttons_for(move)))

    def taken() -> float:
        return max(0.0, clock() - began)

    def may_play() -> bool:
        """Whether another play may start; says so when it is the time that is up."""
        nonlocal late
        if tries >= budget:
            return False
        late = seconds is not None and taken() >= seconds
        return not late

    played: set[tuple[int, ...]] = set()
    # The screens that plays which got nowhere new ended on (``stood``), as digests.
    nowhere: set[bytes] = set()

    def got_on(seen: Outcome) -> bool:
        """Alive and on ground the go had not been on, and ``enough`` where it is asked."""
        return seen.past and (enough is None or bool(enough(seen)))

    def ended_on(outcome: Outcome) -> bytes | None:
        if outcome.screen is None:
            return None
        return hashlib.blake2b(outcome.screen, digest_size=16).digest()

    def room() -> int | None:
        """From where the game now is, each move held for ``outlast`` steps: ``ROOMY``
        where at least half of them keep the life, ``TIGHT`` where fewer do, ``NONE``
        where none does. None where the tries or the time ran out before it was asked."""
        nonlocal tries
        if not may_play():
            return None
        tries += 1
        connection.call(lambda a: a.save_state(END_SLOT))
        enough_keep = (len(every) + 1) // 2
        kept = 0
        for asked, move in enumerate(every, start=1):
            after = play(
                connection,
                END_SLOT,
                (move,) * outlast,
                space,
                frames_per_step=frames_per_step,
                lives=lives,
            )
            kept += not after.lost
            if kept >= enough_keep:
                return ROOMY
            if kept and kept + len(every) - asked < enough_keep:
                return TIGHT  # some keep it, and too few are left to make it half
        return TIGHT if kept else NONE

    def room_after(moves: tuple[int, ...]) -> int | None:
        """Play ``moves`` blind and ask at their end."""
        nonlocal tries
        if not may_play():
            return None
        tries += 1
        there = play(connection, slot, moves, space, frames_per_step=frames_per_step, lives=lives)
        return NONE if there.lost else room()

    def settled(
        moves: tuple[int, ...], seen: Outcome, least: int
    ) -> tuple[tuple[int, ...], int] | None:
        """A try that got past, with the game at its end: the moves of it that are the
        way and the new ground along them, or None where it is no way (turned down, or
        the search ran out before it was known). ``least`` is the fewest moves a part
        of it may have and still be this try."""
        nonlocal tries, doomed

        def part(upto: int) -> tuple[tuple[int, ...], int] | None:
            """That many of its moves, if they still get past under the watch."""
            nonlocal tries
            tries += 1
            cut = play(
                connection,
                slot,
                moves[:upto],
                space,
                frames_per_step=frames_per_step,
                lives=lives,
                watch=watch(),
                look_every=look_every,
            )
            return (moves[:upto], cut.ground) if got_on(cut) else None

        if not outlast:
            return moves, seen.ground
        whole = room()
        if whole is None:
            return None  # the tries or the time ran out: not known, so not a way
        if whole == ROOMY:
            return moves, seen.ground
        # At its end the AI is in a tight spot, or past saving: usually the try, past
        # the place, has carried on into the next one. Back from its end, the first
        # place that leaves it room is where the way stops, if the try has got past
        # by then. (Not the first place it can be lived on from at all: that is the
        # very edge of the next danger.)
        tight = len(moves) if whole == TIGHT else 0  # its longest part that can be lived on from
        for upto in range(len(moves) - CUT_BY, max(least, len(moves) - CUT_WITHIN) - 1, -CUT_BY):
            has = room_after(moves[:upto])
            if has is None:
                return None
            if has == ROOMY:
                if not may_play():
                    return None
                kept = part(upto)
                if kept is not None:
                    return kept
                break  # room, and not past yet: further back it is no further on
            if has == TIGHT and not tight:
                tight = upto
        # No part leaves it room and has got past: as it is where it can be lived on
        # from, else its longest part that can be, if that got past.
        if whole == TIGHT:
            return moves, seen.ground
        if tight:
            if not may_play():
                return None
            kept = part(tight)
            if kept is not None:
                return kept
        doomed += 1
        return None

    def found(way: Way | None) -> Search:
        return Search(way, tries, taken(), late and way is None, doomed)

    def seek(base: Sequence[int], limit: int) -> Way | None:
        """The first small change to ``base`` that gets past, as a way (the plain way's
        part is the caller's to add), while the tries stay under ``limit``."""
        nonlocal tries
        # At a standstill the earliest change that gets on is tried first: it is the
        # one with the least standing in it (T-1018). And a hold is played out whole:
        # the trace ends where it came to stand, with no room in it for a long one.
        for change in changes(base, space, early=stood, whole=stood):
            moves = change.apply(base, carry_on, whole=stood)
            if moves in played:
                continue  # the other part of the search has been here
            if tries >= limit or not may_play():
                break
            tries += 1
            blind = play(
                connection,
                slot,
                moves,
                space,
                frames_per_step=frames_per_step,
                lives=lives,
                end=stood,
            )
            if blind.lost or ended_on(blind) in nowhere:
                played.add(moves)  # lost, or where getting nowhere ended up (T-1018)
                continue
            if tries >= limit or not may_play():
                break  # alive and not looked at yet: not settled, so not marked as played
            played.add(moves)
            tries += 1
            seen = play(
                connection,
                slot,
                moves,
                space,
                frames_per_step=frames_per_step,
                lives=lives,
                watch=watch(),
                look_every=look_every,
            )
            if got_on(seen):
                kept = settled(moves, seen, change.at + 1)
                if kept is not None:
                    return Way(kept[0], change, tries, kept[1], cut=len(moves) - len(kept[0]))
                continue  # it got past and cannot be lived on from: not "nowhere"
            end = ended_on(blind)
            if end is not None:
                nowhere.add(end)
        return None

    share = int(budget * PLAIN_SHARE)
    if plain is not None and share >= 1 and may_play():
        held = (plain,) * (len(trace) + carry_on)
        tries += 1
        first = play(
            connection,
            slot,
            held,
            space,
            frames_per_step=frames_per_step,
            lives=lives,
            end=stood,
        )
        if first.lost:
            base = (plain,) * first.steps
            got = seek(base, share)
            if got is not None:
                return found(replace(got, plain=plain, lost_at=first.steps))
        elif tries < share and may_play():
            tries += 1
            seen = play(
                connection,
                slot,
                held,
                space,
                frames_per_step=frames_per_step,
                lives=lives,
                watch=watch(),
                look_every=look_every,
            )
            past = got_on(seen)
            kept = settled(held, seen, 1) if past else None
            if kept is not None:
                moves, ground = kept
                change = Change(0, plain, len(trace))
                cut = len(held) - len(moves)
                return found(Way(moves, change, tries, ground, plain=plain, cut=cut))
            if stood and not past:
                # It stood still and holding the plain move stands still too (a wall):
                # the small changes to the plain way, a jump at some step of it. The
                # screen this ended on is what getting nowhere looks like.
                played.add(held)
                end = ended_on(first)
                if end is not None:
                    nowhere.add(end)
                got = seek((plain,) * len(trace), share)
                if got is not None:
                    return found(replace(got, plain=plain))
    return found(None if late else seek(trace, budget))


__all__ = [
    "BUDGET",
    "CARRY_ON",
    "CUT_BY",
    "CUT_WITHIN",
    "END_SLOT",
    "NONE",
    "OUTLAST",
    "PLAIN_SHARE",
    "ROOMY",
    "TIGHT",
    "Search",
    "Way",
    "find_way",
]
