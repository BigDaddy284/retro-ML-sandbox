"""T-1031: a search tries its changes before the last chance only."""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from retro_ai_lab.core import ConfigurationError
from retro_ai_lab.emulator.fake import toy_hopper
from retro_ai_lab.training import search
from retro_ai_lab.training.chance import last_chance
from retro_ai_lab.training.search import BUDGET, CARRY_ON, PLAIN_SHARE, Search, find_way
from retro_ai_lab.training.trial import PAST, Outcome
from retro_ai_lab.training.ways import changes
from tests.unit.training.test_search import Script, look_for
from tests.unit.training.test_trial import FRAMES, LIVES, SPOT, Hopper, a_go_that_fell_in

GAP_ENDS = 240  # the far edge of the first gap on this level
DEAD = Outcome(steps=10, lost=True, ground=0)


def lost_at(step: int) -> Outcome:
    return Outcome(steps=step, lost=True, ground=0)


def test_no_change_is_tried_after_the_last_chance(
    hopper: Hopper, monkeypatch: pytest.MonkeyPatch
) -> None:
    right, left = hopper.move("right"), hopper.move("left")
    trace = [right] * 6 + [left] * 4

    def tried(plain_lost: int = 12, **kw: object) -> list[tuple[int, ...]]:
        script = Script([lost_at(plain_lost)] + [DEAD] * 200)
        monkeypatch.setattr(search, "play", script)
        out = look_for(hopper, trace, object, **kw)
        assert out == Search(None, len(script.calls))
        assert not any(looked for _, looked in script.calls)
        return [moves for moves, _ in script.calls]

    def applied(base: Sequence[int], upto: int | None) -> list[tuple[int, ...]]:
        return [c.apply(base, CARRY_ON) for c in changes(base, hopper.space, upto=upto)]

    # Six of its ten steps come before the last chance: the changes to what it did are
    # placed at steps 5, 3 and 1, and at none of the four after.
    plays = tried(live=6)
    assert plays == applied(trace, 6)
    assert {next(i for i, m in enumerate(moves) if m != trace[i]) for moves in plays} == {1, 3, 5}
    assert len(plays) < len(applied(trace, None))
    # With the plain way first: it loses the life at step 12, so its last chance is
    # taken to lie as far before that as the trace's does before its end, 4 steps.
    plays = tried(live=6, plain=right)
    plain = (right,) * 12
    assert plays[0] == (right,) * (10 + CARRY_ON)
    assert plays[1:] == applied(plain, 8) + applied(trace, 6)
    first = [next(i for i, m in enumerate(moves) if m != right) for moves in plays[1:]]
    assert set(first[: len(applied(plain, 8))]) == {1, 3, 5, 7}
    # Even the first step too late: there is nothing of what it did to change. The
    # plain way is still played, and changed where it got further than the trace did:
    # here two steps further, so there is one place, its second step.
    assert tried(live=0) == []
    assert tried(live=0, plain=right) == [(right,) * (10 + CARRY_ON), *applied(plain, 2)]
    assert {moves.index(next(m for m in moves if m != right)) for moves in applied(plain, 2)} == {1}
    # A plain way that got no further has no step before its last chance: played once.
    assert tried(10, live=0, plain=right) == [(right,) * (10 + CARRY_ON)]
    held = (left,) * (10 + CARRY_ON)
    assert tried(8, live=2, plain=left) == [held, *applied(trace, 2)]  # 8 - (10 - 2) = 0
    assert tried(5, live=2, plain=left) == [held, *applied(trace, 2)]  # and less than none
    # The budget and the plain way's share of it hold as ever.
    script = Script([lost_at(12)] + [DEAD] * 200)
    monkeypatch.setattr(search, "play", script)
    out = look_for(hopper, trace, object, live=6, plain=right, budget=6)
    assert out == Search(None, 6)
    assert [moves for moves, _ in script.calls] == (
        [(right,) * (10 + CARRY_ON), *applied(plain, 8)[:2], *applied(trace, 6)[:3]]
    )
    assert PLAIN_SHARE == 0.5


def test_it_gets_across_with_a_long_pause(hopper: Hopper) -> None:
    hopper.game.dying_frames = 200
    watch, mark, died = a_go_that_fell_in(hopper)
    right = hopper.move("right")
    trace = [right] * died
    chance = last_chance(
        hopper.connection, SPOT, trace, hopper.space, frames_per_step=FRAMES, lives=[LIVES]
    )
    assert chance is not None and (chance.live, died) == (21, 74)
    for plain in (None, right):
        found = look_for(hopper, trace, lambda: watch.trial(mark), live=chance.live, plain=plain)
        way = found.way
        assert way is not None and way.ground >= PAST
        # The change is before the last chance, a move a button from the one made, and
        # found at once: the first place tried is the last at which it still matters.
        assert way.change.at == 20 and way.change.move in hopper.space.near(right)
        assert way.tries == found.tries <= (4 if plain is None else 5)
        assert (way.plain, way.lost_at) == ((None, None) if plain is None else (right, died))
        # The game agrees: played from the save it is alive, with its lives, past the gap.
        assert not hopper.play(list(way.moves)).lost
        assert hopper.game.state.x > GAP_ENDS
        assert hopper.game.state.lives == toy_hopper.START_LIVES
    # Without it the same search spends its first tries on the pause and the fall,
    # 53 steps where nothing can be changed.
    slow = look_for(hopper, trace, lambda: watch.trial(mark))
    assert slow.way is not None and slow.tries > 50 > 10 * found.tries
    # Gone back only as far as the AI went before, 40 steps from the counter, the save
    # is in the fall already: with or without it, nothing is found.
    buttons = {b.name: b for b in hopper.adapter.buttons()}
    hopper.adapter.load_state(SPOT)
    hopper.adapter.step(FRAMES * (died - 40), pressed=[buttons["right"]])
    hopper.adapter.save_state("late")
    late = find_way(
        hopper.connection,
        "late",
        [right] * 40,
        hopper.space,
        frames_per_step=FRAMES,
        lives=[LIVES],
        watch=lambda: watch.trial(mark),
    )
    listed = len(list(changes([right] * 40, hopper.space)))
    assert late == Search(None, listed) and 100 < listed <= BUDGET
    with_it = find_way(
        hopper.connection,
        "late",
        [right] * 40,
        hopper.space,
        frames_per_step=FRAMES,
        lives=[LIVES],
        watch=lambda: watch.trial(mark),
        live=0,
    )
    assert with_it == Search(None, 0)  # and it costs no try to know


def test_without_it_the_search_is_what_it_was(
    hopper: Hopper, monkeypatch: pytest.MonkeyPatch
) -> None:
    watch, mark, died = a_go_that_fell_in(hopper)
    right = hopper.move("right")
    trace = [right] * died
    real = search.play
    calls: list[tuple[tuple[int, ...], bool]] = []

    def recorded(connection: object, slot: str, moves: Sequence[int], *args: object, **kw: object):  # type: ignore[no-untyped-def]
        calls.append((tuple(moves), kw.get("watch") is not None))
        return real(connection, slot, moves, *args, **kw)  # type: ignore[arg-type]

    monkeypatch.setattr(search, "play", recorded)

    def run(**kw: object) -> tuple[Search, list[tuple[tuple[int, ...], bool]]]:
        calls.clear()
        return look_for(hopper, trace, lambda: watch.trial(mark), **kw), list(calls)

    for plain in (None, right):
        was = run(plain=plain)
        assert was[0].way is not None and len(was[1]) == was[0].tries
        # None is the search as it was, and so is a last chance at the very end.
        assert run(plain=plain, live=None) == was
        assert run(plain=plain, live=died) == was
    # Nonsense is refused before anything is played.
    calls.clear()
    for bad in (-1, True, 2.5, "6", died + 1):
        with pytest.raises(ConfigurationError, match="live must be a whole number from 0 to"):
            look_for(hopper, trace, lambda: watch.trial(mark), live=bad)
    # It is for a life lost: at a standstill none is, and there is no last chance.
    with pytest.raises(ConfigurationError, match="standstill"):
        look_for(hopper, trace, lambda: watch.trial(mark), live=5, stood=True)
    assert calls == []
