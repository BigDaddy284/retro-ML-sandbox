"""T-1012: the search tries the plain way first, then small changes to what it did."""

from __future__ import annotations

import logging
from collections.abc import Sequence

import pytest

from retro_ai_lab.core import ConfigurationError
from retro_ai_lab.emulator.fake import toy_hopper
from retro_ai_lab.training import episode, search
from retro_ai_lab.training.search import BUDGET, CARRY_ON, PLAIN_SHARE, Search, Way
from retro_ai_lab.training.trial import PAST, Outcome
from retro_ai_lab.training.vision import DEFAULT_LOOK_EVERY, Mark, ScreenWatch
from retro_ai_lab.training.ways import Change, changes
from retro_ai_lab.understanding.player import find_player_in_game
from tests.unit.training import test_going_back as going_back
from tests.unit.training import test_trial as trial
from tests.unit.training.test_going_back import DIED_AT, Lab, steps
from tests.unit.training.test_search import ACROSS, ALIVE, DEAD, NOWHERE, Script, look_for
from tests.unit.training.test_search_at_a_loss import searching
from tests.unit.training.test_trial import FRAMES, SPOT, Hopper, a_go_that_fell_in

hopper = trial.hopper  # the practice game behind a connection (pytest fixture)
lab = going_back.lab  # the practice game behind the real services (pytest fixture)

GAP_ENDS = 240  # the far edge of the first gap on this level


def lost_at(step: int) -> Outcome:
    return Outcome(steps=step, lost=True, ground=0)


def scripted(
    hopper: Hopper, monkeypatch: pytest.MonkeyPatch, outcomes: Sequence[Outcome | Exception]
) -> tuple[Script, int, int]:
    """The plays scripted, and the two moves of these tests: back and on."""
    script = Script(outcomes)
    monkeypatch.setattr(search, "play", script)
    return script, hopper.move("left"), hopper.move("right")


# ------------------------------------------------------------ the rule, with scripted tries


def test_a_plain_way_that_gets_on_is_the_way(
    hopper: Hopper, monkeypatch: pytest.MonkeyPatch
) -> None:
    script, back, on = scripted(hopper, monkeypatch, [ALIVE, ACROSS])
    watches: list[object] = []

    def lend() -> object:
        watches.append(object())
        return watches[-1]

    found = look_for(hopper, [back] * 10, lend, carry_on=4, plain=on)
    # "On" held for as long as what it did and the carry-on: alive, and on new ground.
    held = (on,) * 14
    assert found == Search(Way(held, Change(0, on, 10), tries=2, ground=3, plain=on), 2)
    assert found.way is not None and found.way.lost_at is None
    assert found.way.moves == found.way.change.apply([on] * 10, 4)
    # Played blind, then once under the screen, and nothing else.
    assert script.calls == [(held, False), (held, True)] and len(watches) == 1


def test_where_the_plain_way_loses_the_life_its_changes_are_tried(
    hopper: Hopper, monkeypatch: pytest.MonkeyPatch
) -> None:
    script, back, on = scripted(
        hopper, monkeypatch, [lost_at(6), DEAD, ALIVE, NOWHERE, ALIVE, ACROSS]
    )
    trace = [back] * 10
    found = look_for(hopper, trace, object, carry_on=4, plain=on)
    # It lost the life at its sixth step: the changes are to six steps of "on", and
    # not one of them is a change to what it did.
    base = (on,) * 6
    listed = list(changes(base, hopper.space))
    way = found.way
    assert way == Way(listed[2].apply(base, 4), listed[2], tries=6, ground=3, plain=on, lost_at=6)
    assert way is not None and back not in way.moves and len(way.moves) == 6 + 4
    assert found.tries == 6
    assert script.calls == [
        ((on,) * 14, False),
        (listed[0].apply(base, 4), False),
        (listed[1].apply(base, 4), False),
        (listed[1].apply(base, 4), True),  # alive, and nowhere new: not the way
        (listed[2].apply(base, 4), False),
        (listed[2].apply(base, 4), True),
    ]
    # The life lost in the carry-on, past the end of what it did: the changes are to
    # all of those steps.
    script, back, on = scripted(hopper, monkeypatch, [lost_at(12), ALIVE, ACROSS])
    found = look_for(hopper, trace, object, carry_on=4, plain=on)
    first = next(changes((on,) * 12, hopper.space))
    assert found.way == Way(
        first.apply((on,) * 12, 4), first, tries=3, ground=3, plain=on, lost_at=12
    )


def test_the_plain_way_has_half_the_tries_and_nothing_is_played_twice(
    hopper: Hopper, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert PLAIN_SHARE == 0.5
    # What it did was the plain way: "on", ten steps, the life gone on the tenth.
    script, back, on = scripted(hopper, monkeypatch, [lost_at(10)] + [DEAD] * 500)
    trace = [on] * 10
    listed = list(changes(trace, hopper.space))
    out = look_for(hopper, trace, object, carry_on=4, plain=on, budget=8)
    assert out == Search(None, 8) and not out.out_of_time
    # Half of the eight for the plain way: itself and three changes. The other half
    # goes on down the same list from where that stopped, not from the top again.
    moves = [played for played, _ in script.calls]
    assert moves == [(on,) * 14] + [c.apply(trace, 4) for c in listed[:7]]
    assert len(set(moves)) == 8
    # With tries to spare the list runs out once, not twice.
    script, back, on = scripted(hopper, monkeypatch, [lost_at(10)] + [DEAD] * 500)
    out = look_for(hopper, trace, object, carry_on=4, plain=on, budget=10_000)
    assert out == Search(None, 1 + len(listed)) and 0 < len(listed) < 500
    # A wander is another list: the plain way's half, then changes to the wander.
    wander = [back] * 5 + [on] * 5
    script, back, on = scripted(hopper, monkeypatch, [lost_at(10)] + [DEAD] * 500)
    out = look_for(hopper, wander, object, carry_on=4, plain=on, budget=8)
    theirs = list(changes(wander, hopper.space))
    moves = [played for played, _ in script.calls]
    assert moves == (
        [(on,) * 14]
        + [c.apply(trace, 4) for c in listed[:3]]
        + [c.apply(wander, 4) for c in theirs[:4]]
    )
    # A change that stayed alive when the plain way's half ran out was never looked
    # at: it is not taken for played, and what it did may come to the same moves.
    script, back, on = scripted(hopper, monkeypatch, [lost_at(10), ALIVE, ALIVE, ACROSS])
    out = look_for(hopper, trace, object, carry_on=4, plain=on, budget=4)
    assert out.way == Way(listed[0].apply(trace, 4), listed[0], tries=4, ground=3)
    assert [seen for _, seen in script.calls] == [False, False, False, True]
    # A budget of one leaves the plain way no try at all.
    script, back, on = scripted(hopper, monkeypatch, [DEAD])
    out = look_for(hopper, trace, object, carry_on=4, plain=on, budget=1)
    assert out == Search(None, 1) and script.calls == [(listed[0].apply(trace, 4), False)]


def test_at_a_wall_it_goes_straight_on_to_what_it_did(
    hopper: Hopper, monkeypatch: pytest.MonkeyPatch
) -> None:
    script, _, on = scripted(hopper, monkeypatch, [ALIVE, NOWHERE, DEAD, ALIVE, ACROSS])
    trace = [on] * 6 + [hopper.move("a", "right")] * 4  # it had jumped something
    found = look_for(hopper, trace, object, carry_on=4, plain=on)
    listed = list(changes(trace, hopper.space))
    # Holding "on" stays alive and gets nowhere new: two tries, and no change to it
    # is tried. The way is a change to what it did, and says so.
    assert found.way == Way(listed[1].apply(trace, 4), listed[1], tries=5, ground=3)
    assert found.way is not None and found.way.plain is None and found.way.lost_at is None
    assert script.calls == [
        ((on,) * 14, False),
        ((on,) * 14, True),
        (listed[0].apply(trace, 4), False),
        (listed[1].apply(trace, 4), False),
        (listed[1].apply(trace, 4), True),
    ]
    # The plain way lost under the screen though it stayed alive blind: not the way.
    _, _, on = scripted(
        hopper, monkeypatch, [ALIVE, Outcome(steps=9, lost=True, ground=4), ALIVE, ACROSS]
    )
    found = look_for(hopper, trace, object, carry_on=4, plain=on)
    assert found.way == Way(listed[0].apply(trace, 4), listed[0], tries=4, ground=3)


def test_the_time_limit_holds_over_both_parts(
    hopper: Hopper, monkeypatch: pytest.MonkeyPatch
) -> None:
    now = [0.0]

    def timed(outcomes: Sequence[Outcome]) -> Script:
        script, _, _ = scripted(hopper, monkeypatch, outcomes)

        def played(*args: object, **kw: object) -> Outcome:
            now[0] += 1.0
            return script(*args, **kw)  # type: ignore[arg-type]

        monkeypatch.setattr(search, "play", played)
        now[0] = 0.0
        return script

    back, on = hopper.move("left"), hopper.move("right")
    trace = [back] * 10
    # Up while the plain way's changes are tried: what it did is not begun.
    script = timed([lost_at(6)] + [DEAD] * 50)
    out = look_for(hopper, trace, object, plain=on, seconds=3, clock=lambda: now[0])
    assert out == Search(None, 3, out_of_time=True) and len(script.calls) == 3
    assert all(back not in moves for moves, _ in script.calls)
    # Up after the plain way's first play, which stayed alive: it is not looked at.
    script = timed([ALIVE, ACROSS])
    out = look_for(hopper, trace, object, plain=on, seconds=1, clock=lambda: now[0])
    assert out == Search(None, 1, out_of_time=True)
    assert [seen for _, seen in script.calls] == [False]
    # Up before anything: nothing is played.
    script = timed([ALIVE, ACROSS])
    now[0] = 0.0
    ticks = iter([0.0, 5.0, 5.0, 5.0])
    out = look_for(hopper, trace, object, plain=on, seconds=1, clock=lambda: next(ticks))
    assert out.way is None and out.tries == 0 and out.out_of_time and script.calls == []
    # Up during what it did, after the plain way's half: the same answer.
    script = timed([lost_at(10)] + [DEAD] * 50)
    out = look_for(hopper, [on] * 10, object, plain=on, budget=6, seconds=5, clock=lambda: now[0])
    assert out == Search(None, 5, out_of_time=True)


def test_nonsense_is_refused_before_anything_is_played(
    hopper: Hopper, monkeypatch: pytest.MonkeyPatch
) -> None:
    script, back, on = scripted(hopper, monkeypatch, [])
    for wrong in (len(hopper.space), -1, True, 1.0, "right"):
        with pytest.raises(ConfigurationError):
            look_for(hopper, [back] * 10, object, plain=wrong)
    # A move the controls do not have in what it did, with a plain move that is fine.
    with pytest.raises(ConfigurationError):
        look_for(hopper, [back, len(hopper.space)], object, plain=on)
    with pytest.raises(ConfigurationError):
        look_for(hopper, [], object, plain=on)
    assert script.calls == []


# ------------------------------------------------------------ on the practice game itself


def a_go(hopper: Hopper, moves: Sequence[int]) -> tuple[ScreenWatch, Mark, int]:
    """The watch of a go that made ``moves`` from the save and lost a life, the mark
    of its start, and the step the life went at."""
    names = [b.name for b in hopper.adapter.buttons() if b.name not in ("start", "select")]
    hopper.adapter.load_state(SPOT)
    player = find_player_in_game(hopper.connection, names)
    assert player is not None and player.colours is not None
    watch = ScreenWatch(player)
    watch.begin_go()
    mark = watch.mark()
    buttons = {b.name: b for b in hopper.adapter.buttons()}
    hopper.adapter.load_state(SPOT)
    for step, move in enumerate(moves):
        pressed = [buttons[name] for name in sorted(hopper.space.buttons_for(move))]
        result = hopper.adapter.step(
            FRAMES, pressed=pressed, screenshot=step % DEFAULT_LOOK_EVERY == 0
        )
        if result.frame is not None:
            watch.look(result.frame)
        if hopper.game.state.lives < toy_hopper.START_LIVES:
            return watch, mark, step + 1
    raise AssertionError("it never lost a life")


def test_the_plain_way_across_the_first_gap(hopper: Hopper) -> None:
    watch, mark, died = a_go_that_fell_in(hopper)
    right, jump = hopper.move("right"), hopper.move("a", "right")
    trace = [right] * died
    before = (watch.looks, watch.ground.place, watch.ground.reach, watch.ground.steps, watch.me)
    as_it_was = look_for(hopper, trace, lambda: watch.trial(mark))
    found = look_for(hopper, trace, lambda: watch.trial(mark), plain=right)
    way, old = found.way, as_it_was.way
    assert way is not None and old is not None
    # What it did was the plain way, so the same change is found, one play later (the
    # plain way's own), and it says which it is and where holding "right" lost the life.
    assert (way.plain, way.lost_at) == (right, died) and (old.plain, old.lost_at) == (None, None)
    assert (way.change, way.moves, way.ground) == (old.change, old.moves, old.ground)
    assert way.change.move == jump and way.tries == found.tries == old.tries + 1
    assert found.tries <= BUDGET * PLAIN_SHARE and way.ground >= PAST
    # The game agrees: played from the save it is alive, with its lives, past the gap.
    assert not hopper.play(list(way.moves)).lost
    assert hopper.game.state.x > GAP_ENDS
    assert hopper.game.state.lives == toy_hopper.START_LIVES
    # The same search twice is the same, and nothing was done to the go's own watch.
    assert look_for(hopper, trace, lambda: watch.trial(mark), plain=right) == found
    assert before == (
        watch.looks,
        watch.ground.place,
        watch.ground.reach,
        watch.ground.steps,
        watch.me,
    )


def test_a_wander_before_the_gap_is_got_past(hopper: Hopper) -> None:
    """What the brain was doing in most of the searches that found nothing: on a
    little, back a little, a long jump well short of the gap, and on into it."""
    right, left, jump = hopper.move("right"), hopper.move("left"), hopper.move("a", "right")
    wander = [right] * 18 + [left] * 11 + [jump] * 12 + [right] * 80
    watch, mark, died = a_go(hopper, wander)
    trace = wander[:died]
    assert trace == [right] * 18 + [left] * 11 + [jump] * 12 + [right] * 17
    # The step before the life is counted lost, it is down in the first gap.
    col, wide = hopper.game.gaps[0]
    assert not hopper.play(trace[:-1]).lost
    assert toy_hopper.TILE * col < hopper.game.state.x < toy_hopper.TILE * (col + wide)
    # No small change to that gets across, with every try a search has.
    as_it_was = look_for(hopper, trace, lambda: watch.trial(mark))
    assert as_it_was == Search(None, BUDGET)
    # Holding "right" from the same save loses the life sooner, and one change to
    # that is the way, well inside the plain way's half of the tries.
    found = look_for(hopper, trace, lambda: watch.trial(mark), plain=right)
    way = found.way
    assert way is not None and way.plain == right
    assert way.lost_at is not None and way.lost_at < died
    assert way.change.move == jump and set(way.moves) == {right, jump}
    assert len(way.moves) == way.lost_at + CARRY_ON
    assert found.tries == way.tries <= BUDGET * PLAIN_SHARE / 2
    assert not hopper.play(list(way.moves)).lost
    assert hopper.game.state.x > GAP_ENDS
    assert hopper.game.state.lives == toy_hopper.START_LIVES
    # The way reaches ground the wander never did.
    assert way.ground >= PAST


# ------------------------------------------------------------ the runner


def test_the_runner_gives_its_best_paid_button(
    lab: Lab, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    real = episode.find_way
    given: list[int | None] = []

    def spy(*args: object, **kw: object) -> Search:
        plain = kw.get("plain")
        assert plain is None or isinstance(plain, int)
        given.append(plain)
        return real(*args, **kw)  # type: ignore[arg-type]

    monkeypatch.setattr(episode, "find_way", spy)
    walk, jump, back = lab.move("right"), lab.move("a", "right"), lab.move("left")
    # A go that walks into the gap: "right" is the button that was paid.
    r, _ = searching(lab)
    r.begin()
    with caplog.at_level(logging.INFO, logger="retro_ai_lab.training.episode"):
        steps(r, DIED_AT)
    assert r.paid_buttons == ("right",) and given == [walk]
    assert (r.ways_sought, r.ways_found) == (1, 1)
    said = [rec.getMessage() for rec in caplog.records if rec.getMessage().startswith("Lost a")]
    assert len(said) == 1 and said[0].endswith(
        ": holding right instead of what it did, with right + a for 4 steps, starting "
        "17 steps before that would lose the life."
    )
    r.finish()
    # A go that wanders and then walks in: the way is the plain one, and the player
    # is put across by "right" and one jump, with nothing of the wander in it.
    r, _ = searching(lab)
    lab.agent.script.extend([walk] * 18 + [back] * 11 + [jump] * 12)
    r.begin()
    while r.ways_sought == 0:
        assert r.step()
    assert given == [walk, walk] and r.ways_found == 1
    asked, played = lab.agent.asked, []
    while lab.agent.asked == asked:
        assert r.step()
        played.append(r.last_action)
    played.pop()
    assert set(played) == {walk, jump} and played.count(jump) in (4, 8, 16)
    assert lab.game.state.x > GAP_ENDS and lab.game.state.lives == toy_hopper.START_LIVES
    r.finish()


def test_with_nothing_paid_yet_there_is_no_plain_way(
    lab: Lab, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    real = episode.find_way
    given: list[int | None] = []

    def spy(*args: object, **kw: object) -> Search:
        given.append(kw.get("plain"))  # type: ignore[arg-type]
        return real(*args, **kw)  # type: ignore[arg-type]

    monkeypatch.setattr(episode, "find_way", spy)
    r, _ = searching(lab)
    monkeypatch.setattr(type(r), "paid_buttons", property(lambda self: ()))
    r.begin()
    with caplog.at_level(logging.INFO, logger="retro_ai_lab.training.episode"):
        steps(r, DIED_AT)
    # The search is as it was before this task, and its line reads as it did.
    assert given == [None] and r.ways_found == 1
    said = [rec.getMessage() for rec in caplog.records if rec.getMessage().startswith("Lost a")]
    assert len(said) == 1 and "found a way past on try 18," in said[0]
    assert said[0].endswith(" s: right + a for 4 steps, starting 17 steps before the life went.")
    r.finish()
