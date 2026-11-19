"""T-1060: a part of the way. Where even two changes do not get past, the change that
got furthest is kept.

A search that goes deeper (T-1059) finds a way of two changes and no more. A place
that wants three (three gaps close together) found nothing, search after search, and
what each search had seen was thrown away: which change got further. Such a search
now hands that change back, with its moves up to before its own loss was settled.
The runner goes back and plays it; the life goes further on, and the search there
looks for the rest.
"""

from __future__ import annotations

import dataclasses
import logging
from typing import Any, ClassVar

import pytest

from retro_ai_lab.emulator.fake.toy_hopper import START_LIVES, TILE
from retro_ai_lab.training import EpisodeRunner, episode
from retro_ai_lab.training.episode import AHEAD_EVERY, NOW_SLOT, STAND_AFTER, EpisodeResult
from retro_ai_lab.training.search import CARRY_ON, TOP, Search, Way
from retro_ai_lab.training.trial import PAST, Outcome
from retro_ai_lab.training.ways import Change, changes
from tests.unit.training import test_going_back as going_back
from tests.unit.training import test_trial as trial
from tests.unit.training.test_going_back import DIED_AT, Lab, steps
from tests.unit.training.test_last_chance_in_a_go import going
from tests.unit.training.test_looking_ahead import THING, Near, looking
from tests.unit.training.test_search import ACROSS, ALIVE, DEAD, look_for
from tests.unit.training.test_search_at_a_loss import searching
from tests.unit.training.test_search_carries_on import spot_of
from tests.unit.training.test_search_goes_deeper import (
    DID,
    LOG,
    PAUSE,
    Asked,
    TwoGaps,
    blind,
    looked,
    lost_at,
    no_watch,
    no_watch_but_once,
    nothing,
    put_in,
    said,
    scripted,
    second_spot,
    seconds,
)
from tests.unit.training.test_standing_still import standing
from tests.unit.training.test_trial import FRAMES, Hopper, a_go_that_fell_in

hopper = trial.hopper  # the practice game behind a connection (pytest fixture)
lab = going_back.lab  # the practice game behind the real services (pytest fixture)

LIVE = 6  # of the ten steps it did, the loss was settled at this one: 4 steps before
CUT = 13  # a try that lost its life at step 17 is cut as long before that


def lender() -> tuple[Any, list[object]]:
    """A watch for every look asked for, and the ones lent."""
    lent: list[object] = []

    def watch() -> object:
        lent.append(object())
        return lent[-1]

    return watch, lent


# -------------------------------------------------------------- what a search hands back


def test_a_deeper_search_for_nothing_hands_back_the_change_that_got_further(
    hopper: Hopper, monkeypatch: pytest.MonkeyPatch
) -> None:
    script, did, _ = scripted(hopper, monkeypatch, [])
    listed = list(changes(did, hopper.space, upto=LIVE))
    first = listed[0]
    assert (first.at, first.hold) == (5, 4)
    tried = first.apply(did, CARRY_ON)
    stone = tried[:17]  # what that try did up to the step its life went at
    after = seconds(stone, first, hopper.space, upto=CUT)
    script.outcomes[:] = [lost_at(17), lost_at(17, PAST)] + [DEAD] * 200
    found = look_for(hopper, did, no_watch_but_once(), deeper=True, live=LIVE, budget=400)
    part = found.part
    assert found.way is None and part is not None
    # Its moves end as long before its own loss as the first loss was settled before
    # its own: the whole change is in them, and nothing after that step.
    assert part == Way(tuple(tried[:CUT]), first, tries=found.tries, ground=PAST)
    assert (part.plain, part.lost_at, part.first) == (None, None, None)
    assert first.at + first.hold <= len(part.moves) == 17 - (DID - LIVE)
    # No play is made for it: the search is what it was.
    assert found.tries == len(listed) + 1 + len(after) and found.lapped
    assert script.calls[:2] == [blind(tried), looked(tried)]
    assert script.calls[2:] == [
        *[blind(change.apply(stone, CARRY_ON)) for change in after],
        *[blind(change.apply(did, CARRY_ON)) for change in listed[1:]],
    ]


@pytest.mark.parametrize(
    ("grounds", "kept"),
    [((3, 5), 1), ((5, 3), 0), ((3, 3), 0)],
    ids=["the second got further", "the first got further", "as far: the earlier one"],
)
def test_of_two_that_got_further_it_is_the_one_with_most_new_ground(
    hopper: Hopper, monkeypatch: pytest.MonkeyPatch, grounds: tuple[int, int], kept: int
) -> None:
    script, did, _ = scripted(hopper, monkeypatch, [])
    listed = list(changes(did, hopper.space, upto=LIVE))
    stones = [change.apply(did, CARRY_ON)[:17] for change in listed[:2]]
    afters = [seconds(stones[n], listed[n], hopper.space, upto=CUT) for n in range(2)]
    script.outcomes[:] = [
        lost_at(17),
        lost_at(17, grounds[0]),
        *[DEAD] * len(afters[0]),
        lost_at(17),
        lost_at(17, grounds[1]),
        *[DEAD] * 200,
    ]
    watch, lent = lender()
    found = look_for(hopper, did, watch, deeper=True, live=LIVE, budget=400)
    assert found.way is None and len(lent) == 2
    assert found.part == Way(
        tuple(stones[kept][:CUT]), listed[kept], tries=found.tries, ground=grounds[kept]
    )
    assert found.tries == len(listed) + 2 + len(afters[0]) + len(afters[1])


def test_a_search_that_finds_a_way_has_no_part(
    hopper: Hopper, monkeypatch: pytest.MonkeyPatch
) -> None:
    script, did, _ = scripted(hopper, monkeypatch, [])
    listed = list(changes(did, hopper.space, upto=LIVE))
    first = listed[0]
    after = seconds(first.apply(did, CARRY_ON)[:17], first, hopper.space, upto=CUT)
    # A step on with no second change that gets past, and then a single change that does.
    script.outcomes[:] = [lost_at(17), lost_at(17, PAST), *[DEAD] * len(after), ALIVE, ACROSS]
    watch, lent = lender()
    found = look_for(hopper, did, watch, deeper=True, live=LIVE, budget=400)
    assert found.way is not None and found.way.change == listed[1] and found.way.first is None
    assert found.part is None and len(lent) == 2


def test_no_part_without_going_deeper(hopper: Hopper, monkeypatch: pytest.MonkeyPatch) -> None:
    _, did, _ = scripted(hopper, monkeypatch, [lost_at(17)] + [DEAD] * 200)
    found = look_for(hopper, did, no_watch, live=LIVE)
    assert found.way is None and found.lapped and found.part is None


def test_no_part_where_the_screen_stands_still(
    hopper: Hopper, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, did, _ = scripted(hopper, monkeypatch, [lost_at(17)] + [DEAD] * 200)
    found = look_for(hopper, did, no_watch, deeper=True, still=True, live=LIVE)
    assert found.way is None and found.lapped and found.part is None


def test_no_part_without_the_step_the_loss_was_settled_at(
    hopper: Hopper, monkeypatch: pytest.MonkeyPatch
) -> None:
    script, did, listed = scripted(hopper, monkeypatch, [])
    script.outcomes[:] = [lost_at(17), lost_at(17, PAST)] + [DEAD] * 200
    # A step on, looked at and gone on from; but how long before its loss the moves
    # still count is not known.
    found = look_for(hopper, did, no_watch_but_once(), deeper=True, budget=400)
    assert found.way is None and found.lapped and found.tries > len(listed) + 1
    assert found.part is None


@pytest.mark.parametrize(
    "seen",
    [lost_at(17, PAST - 1), Outcome(steps=30, lost=False, ground=5)],
    ids=["not on new ground", "alive when looked at"],
)
def test_what_is_no_step_on_is_no_part(
    hopper: Hopper, monkeypatch: pytest.MonkeyPatch, seen: Outcome
) -> None:
    _, did, _ = scripted(hopper, monkeypatch, [lost_at(17), seen] + [DEAD] * 200)
    found = look_for(hopper, did, no_watch_but_once(), deeper=True, live=LIVE, budget=400)
    assert found.way is None and found.part is None


def test_a_look_that_lost_the_life_before_the_change_was_over_is_no_part(
    hopper: Hopper, monkeypatch: pytest.MonkeyPatch
) -> None:
    script, did, _ = scripted(hopper, monkeypatch, [])
    listed = list(changes(did, hopper.space, upto=LIVE))
    first = listed[0]
    # Played again under the watch the life went at step 12: four steps before that
    # is step 8, and the change is not over until step 9.
    script.outcomes[:] = [lost_at(17), lost_at(12, PAST)] + [DEAD] * 200
    found = look_for(hopper, did, no_watch_but_once(), deeper=True, live=LIVE, budget=400)
    assert first.at + first.hold == 9
    assert found.way is None and found.part is None
    # One step later it would have been: the cut may lie at the end of the change.
    script.outcomes[:] = [lost_at(17), lost_at(13, PAST)] + [DEAD] * 200
    found = look_for(hopper, did, no_watch_but_once(), deeper=True, live=LIVE, budget=400)
    assert found.part is not None and len(found.part.moves) == 9


def test_a_search_cut_short_keeps_what_it_had_seen_and_says_where_it_stopped(
    hopper: Hopper, monkeypatch: pytest.MonkeyPatch
) -> None:
    script, did, _ = scripted(hopper, monkeypatch, [])
    listed = list(changes(did, hopper.space, upto=LIVE))
    first = listed[0]
    tried = first.apply(did, CARRY_ON)
    stone = tried[:17]
    after = seconds(stone, first, hopper.space, upto=CUT)
    script.outcomes[:] = [lost_at(17), lost_at(17, PAST), DEAD, DEAD]
    cut = look_for(hopper, did, no_watch_but_once(), deeper=True, live=LIVE, budget=4)
    assert cut == Search(None, 4) and not cut.lapped
    assert cut.part == Way(tuple(tried[:CUT]), first, tries=4, ground=PAST)
    assert cut.stopped_at == (TOP[0], spot_of(first, did, LIVE))
    assert cut.deeper_at == (TOP[0], second_spot(after[2], first, stone[:CUT]))
    # Out of tries before the look: nothing has been seen to get further.
    script.outcomes[:] = [lost_at(17)]
    sooner = look_for(hopper, did, no_watch, deeper=True, live=LIVE, budget=1)
    assert sooner == Search(None, 1) and sooner.part is None


def test_a_part_from_the_plain_way_says_so(hopper: Hopper, monkeypatch: pytest.MonkeyPatch) -> None:
    script, _, _ = scripted(hopper, monkeypatch, [])
    back, on = hopper.move("left"), hopper.move("right")
    did = [back] * 12
    # What it did was settled at step 9 of 12: three steps. The plain way loses the
    # life at step 10, and a change to it loses its own at step 16.
    base = (on,) * 10
    first = next(iter(changes(base, hopper.space, upto=7)))
    tried = first.apply(base, CARRY_ON)
    script.outcomes[:] = [lost_at(10), lost_at(16), lost_at(16, PAST)] + [DEAD] * 800
    found = look_for(hopper, did, no_watch_but_once(), deeper=True, plain=on, live=9, budget=800)
    assert found.way is None
    assert found.part == Way(
        tuple(tried[:13]), first, tries=found.tries, ground=PAST, plain=on, lost_at=10
    )


@pytest.mark.parametrize(
    ("grounds", "of_the_plain_way"),
    [((3, 5), False), ((3, 3), True)],
    ids=["what it did got further", "as far: the plain way's came first"],
)
def test_the_two_lists_are_one_search(
    hopper: Hopper,
    monkeypatch: pytest.MonkeyPatch,
    grounds: tuple[int, int],
    of_the_plain_way: bool,
) -> None:
    script, _, _ = scripted(hopper, monkeypatch, [])
    back, on = hopper.move("left"), hopper.move("right")
    did = [back] * 12
    base = (on,) * 10
    theirs = list(changes(base, hopper.space, upto=7))
    stone = theirs[0].apply(base, CARRY_ON)[:16]
    after = seconds(stone, theirs[0], hopper.space, upto=13)
    mine = next(iter(changes(did, hopper.space, upto=9)))
    script.outcomes[:] = [
        lost_at(10),  # the plain way, held
        lost_at(16),
        lost_at(16, grounds[0]),  # its first change: a step on
        *[DEAD] * (len(after) + len(theirs) - 1),
        lost_at(18),
        lost_at(18, grounds[1]),  # the first change to what it did: another
        *[DEAD] * 800,
    ]
    watch, lent = lender()
    found = look_for(hopper, did, watch, deeper=True, plain=on, live=9, budget=800)
    part = found.part
    assert found.way is None and part is not None and len(lent) == 2
    if of_the_plain_way:
        assert (part.change, part.plain, part.lost_at) == (theirs[0], on, 10)
        assert part.moves == tuple(stone[:13])
    else:
        assert (part.change, part.plain, part.lost_at) == (mine, None, None)
        assert part.moves == tuple(mine.apply(did, CARRY_ON)[:15]) and part.ground == 5


# --------------------------------------------------- on a game with three gaps in a row


class ThreeGaps(TwoGaps):
    """Three gaps close together: no two changes to walking right get past."""

    GAPS: ClassVar[tuple[tuple[int, int], ...]] = ((14, 3), (19, 3), (24, 3))


class FourGaps(TwoGaps):
    GAPS: ClassVar[tuple[tuple[int, int], ...]] = ((14, 3), (19, 3), (24, 3), (29, 3))


def far_side(level: type[TwoGaps]) -> int:
    """Where the ground begins again behind the last gap."""
    first, wide = level.GAPS[-1]
    return (first + wide) * TILE


def test_three_gaps_find_no_way_of_two_and_a_part(hopper: Hopper) -> None:
    put_in(hopper, ThreeGaps(0), PAUSE)
    watch, mark, died = a_go_that_fell_in(hopper)
    right, jump = hopper.move("right"), hopper.move("a", "right")
    did = [right] * died
    live = died - PAUSE // FRAMES

    def look() -> object:
        return watch.trial(mark)

    found = look_for(hopper, did, look, live=live, budget=2000, deeper=True)
    part = found.part
    assert found.way is None and found.lapped and part is not None
    assert part.change.move == jump and part.ground >= PAST
    assert part.change.at + part.change.hold <= len(part.moves)
    # The game agrees. Its moves alone lose no life and are over the first gap, at the
    # second; walked on from, the life goes later than it did.
    alone = hopper.play(list(part.moves))
    over_the_first, over_the_second = (14 + 3) * TILE, (19 + 3) * TILE
    assert not alone.lost and hopper.game.state.lives == START_LIVES
    assert over_the_first < hopper.game.state.x < over_the_second
    on = hopper.play([*part.moves, *[right] * 120])
    assert on.lost and on.steps > died + part.change.hold
    # Without going deeper nothing is seen to get further.
    single = look_for(hopper, did, look, live=live, budget=2000)
    assert single.way is None and single.part is None


# ------------------------------------------------------------------------ the runner


def with_a_part(part: Way, lapped: bool = True) -> Search:
    return Search(None, 5, stopped_at=((0, 2), (1, 4)), lapped=lapped, part=part)


def a_part(lab: Lab, plain: bool = False) -> Way:
    """A jump over the practice level's first gap, kept up to two steps after it."""
    walk, jump = lab.move("right"), lab.move("a", "right")
    moves = (walk,) * 19 + (jump,) * 4 + (walk,) * 2
    if plain:
        return Way(moves, Change(19, jump, 4), tries=5, ground=3, plain=walk, lost_at=30)
    return Way(moves, Change(19, jump, 4), tries=5, ground=3)


KEPT = "One change got further before it lost the life"


def test_after_a_life_lost_a_part_is_played_as_a_way_is(
    lab: Lab, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    part = a_part(lab)
    asked = Asked(monkeypatch, [with_a_part(part), nothing(False)])
    r, moments = searching(lab, ways_per_go=3, give_up_after=None, max_steps=400)
    r.begin()
    steps(r, DIED_AT - 1)
    start = moments.kept[0]
    with caplog.at_level(logging.INFO, logger=LOG):
        assert r.step()  # the life goes, and inside the same step it goes back
    assert (r.ways_sought, r.ways_found, r.parts_kept, r.times_back) == (1, 0, 1, 1)
    # The game is at the saved moment, the life it lost there again, and the moments
    # are kept as after a way found; the game as it was at the loss is not gone to.
    assert lab.loaded[-1] == start.slot and NOW_SLOT not in lab.loaded
    assert lab.game.state.lives == START_LIVES
    assert moments.steps == 0 and moments.kept == (start,)
    # Its moves are played without asking the learner.
    before, learned = lab.agent.asked, len(lab.agent.learned)
    played = []
    while lab.agent.asked == before:
        assert r.step()
        played.append(r.last_action)
    played.pop()  # the step the learner was asked again
    assert tuple(played) == part.moves
    taught = [action for _, action, *_ in lab.agent.learned[learned:]]
    assert len(taught) >= 19 and taught == list(part.moves)[: len(taught)]  # and learned from
    # The loss stood for the count, as with a way.
    result = r.finish()
    assert (result.setbacks, result.first_setback_step) == (1, DIED_AT)
    assert (result.ways_sought, result.ways_found, result.parts_kept) == (1, 0, 1)
    (line,) = said(caplog, KEPT)
    assert line == (
        f"Lost a life at step {DIED_AT}. Went back {DIED_AT} steps and made 5 tries in 0.0 s: "
        f"none got past. One change got further before it lost the life (right + a for 4 "
        f"steps, starting {DIED_AT - 19} steps before the life went), so the AI keeps it: it "
        "plays that for 25 steps and carries on. The life goes further on, and the search "
        "there looks for the rest of the way."
    )
    assert asked.deeper == [False]


def test_a_part_is_played_to_its_end_though_the_go_s_steps_run_out(
    lab: Lab, monkeypatch: pytest.MonkeyPatch
) -> None:
    part = a_part(lab)
    Asked(monkeypatch, [with_a_part(part)])
    r, _ = searching(lab, ways_per_go=3, give_up_after=None, max_steps=DIED_AT + 3)
    r.begin()
    while r.step():
        pass
    # As a way just found is (T-1018): a go that ended part of the way through it
    # would have gone back over its ground for nothing.
    result = r.finish()
    assert result.ended_by == "max_steps" and result.steps == DIED_AT + len(part.moves)
    assert result.parts_kept == 1


def test_the_log_says_a_part_of_the_plain_way(
    lab: Lab, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    Asked(monkeypatch, [with_a_part(a_part(lab, plain=True))])
    r, _ = searching(lab)
    with caplog.at_level(logging.INFO, logger=LOG):
        r.begin()
        steps(r, DIED_AT)
    (line,) = said(caplog, KEPT)
    assert line.endswith(
        "none got past. One change got further before it lost the life (holding right "
        "instead of what it did, with right + a for 4 steps, starting 11 steps before that "
        "would lose the life), so the AI keeps it: it plays that for 25 steps and carries "
        "on. The life goes further on, and the search there looks for the rest of the way."
    )


def test_the_place_is_remembered_as_searched_for_nothing(
    lab: Lab, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    asked = Asked(monkeypatch, [with_a_part(a_part(lab)), nothing(False)])
    r, _ = searching(lab, ways_per_go=3, give_up_after=None, max_steps=400)
    r.begin()
    steps(r, DIED_AT)
    r.finish()
    # The next go loses the life at the same place: the search there carries on where
    # the last one stopped and, that one having gone through its list, goes deeper.
    lab.game.reset()
    with caplog.at_level(logging.INFO, logger=LOG):
        r.begin()
        steps(r, DIED_AT)
    assert asked.deeper == [False, True]
    assert len(said(caplog, "about here once before and found none")) == 1
    assert r.parts_kept == 0  # counted go by go


def test_without_a_part_a_search_for_nothing_is_what_it_was(
    lab: Lab, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    Asked(monkeypatch, [nothing(True)])
    r, _ = searching(lab, ways_per_go=3, give_up_after=None, max_steps=400)
    with caplog.at_level(logging.INFO, logger=LOG):
        r.begin()
        steps(r, DIED_AT)
    assert (r.ways_sought, r.ways_found, r.parts_kept, r.times_back) == (1, 0, 0, 0)
    assert lab.loaded[-1] == NOW_SLOT and lab.game.state.lives == START_LIVES - 1
    assert said(caplog, KEPT) == []
    (line,) = said(caplog, "none got past")
    assert line.endswith("made 5 tries in 0.0 s: none got past.")


def lose_lives(r: EpisodeRunner, searches: int) -> None:
    for _ in range(600):
        if r.ways_sought == searches:
            return
        assert r.step()
    raise AssertionError(f"it never made search {searches}")


def test_a_part_kept_is_neither_a_search_for_nothing_nor_a_way_found(
    lab: Lab, monkeypatch: pytest.MonkeyPatch
) -> None:
    walk = lab.move("right")
    short = Way((walk,) * 5, Change(1, walk, 4), tries=5, ground=3)  # it walks in again
    Asked(monkeypatch, [nothing(False), with_a_part(short), nothing(False)])
    r, _ = searching(lab, ways_per_go=3, give_up_after=2, max_steps=600)
    r.begin()
    lose_lives(r, 1)
    assert not r.gave_up
    # The second search keeps a part. Counted as one for nothing it would be the
    # second in a row, and going back would stop here.
    lose_lives(r, 2)
    assert r.parts_kept == 1 and not r.gave_up
    # Counted as a way found, the next one for nothing would be the first again.
    lose_lives(r, 3)
    assert r.gave_up


def test_a_part_is_not_taken_for_a_life_seen_coming(
    lab: Lab, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    Asked(monkeypatch, [with_a_part(a_part(lab))])
    r, _ = looking(lab, ways_per_go=2)
    near = Near(monkeypatch, r.watch)
    r.begin()
    steps(r, 5)
    at = lab.game.state.x
    with caplog.at_level(logging.INFO, logger=LOG):
        near.now = THING
        assert r.step()
    # The life is still to be had: the go carries on from where it was.
    assert (r.looked_ahead, r.ways_ahead, r.ways_ahead_found) == (1, 1, 0)
    assert (r.parts_kept, r.times_back) == (0, 0) and lab.loaded[-1] == NOW_SLOT
    assert lab.game.state.x >= at and said(caplog, KEPT) == []
    (line,) = said(caplog, "none got past")
    assert line.endswith("none got past, so it carries on.")
    assert AHEAD_EVERY > 1


def test_never_at_a_standstill(lab: Lab, monkeypatch: pytest.MonkeyPatch) -> None:
    Asked(monkeypatch, [with_a_part(a_part(lab))])
    r, _ = standing(lab, ways_per_go=3)
    r.begin()
    stand, jump = lab.move(), lab.move("a")
    for n in range(STAND_AFTER + 40):
        if r.ways_stood:
            break
        lab.agent.script.clear()
        lab.agent.script.append(jump if n % 8 < 3 else stand)
        r.step()
    assert r.ways_stood == 1 and (r.parts_kept, r.times_back) == (0, 0)
    assert lab.loaded[-1] == NOW_SLOT


def test_the_go_s_summary_says_how_often_and_an_older_result_reads_back() -> None:
    def result(sought: int, found: int, parts: int) -> EpisodeResult:
        return EpisodeResult(
            10, 40, 1.0, "max_steps", 3, ways_sought=sought, ways_found=found, parts_kept=parts
        )

    kept = " (and once kept the change that got furthest)"
    assert result(2, 0, 1).describe_ways() == ", went back 2 times and found no way past" + kept
    assert result(3, 1, 2).describe_ways() == (
        ", went back 3 times and found a way past once "
        "(and 2 times kept the change that got furthest)"
    )
    assert result(2, 1, 0).describe_ways() == ", went back 2 times and found a way past once"
    data = result(3, 1, 2).to_dict()
    assert data["parts_kept"] == 2 and EpisodeResult.from_dict(data) == result(3, 1, 2)
    del data["parts_kept"]
    assert EpisodeResult.from_dict(data) == result(3, 1, 0)
    assert EpisodeResult(10, 40, 1.0, "max_steps", 3).parts_kept == 0


# --------------------------------------------- the stand-ins through the real runner


class Held:
    """The real search, with the parts it hands back taken away where told to."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch, off: bool) -> None:
        self.off = off
        self.parts = 0
        self.real = episode.find_way
        monkeypatch.setattr(episode, "find_way", self)

    def __call__(self, *args: Any, **kw: Any) -> Search:
        found = self.real(*args, **kw)
        self.parts += found.part is not None
        return dataclasses.replace(found, part=None) if self.off else found


def goes(lab: Lab, level: type[TwoGaps], most: int) -> tuple[list[int], list[int], EpisodeRunner]:
    """Goes of three lives on ``level``, as the app runs them (the settle check on, a
    death of 54 steps), until one is beyond the last gap or ``most`` are played: how
    far each got, the parts each kept, and the runner."""
    game = level(0)
    lab.adapter._game = game
    lab.game = game
    r, _ = going(lab, PAUSE, last_chance=True, ways_per_go=3, max_steps=900)
    far, kept = [], []
    for _ in range(most):
        lab.game.reset()
        r.begin()
        best = 0
        while best <= far_side(level) + TILE and r.step():
            best = max(best, lab.game.state.x)
        far.append(best)
        kept.append(r.parts_kept)
        if best > far_side(level):
            break
        r.finish()
    return far, kept, r


@pytest.mark.parametrize(
    ("level", "in_go"), [(ThreeGaps, 2), (FourGaps, 3)], ids=["three gaps", "four gaps"]
)
def test_through_the_runner_gaps_in_a_row_are_got_past_a_part_at_a_time(
    lab: Lab, monkeypatch: pytest.MonkeyPatch, level: type[TwoGaps], in_go: int
) -> None:
    held = Held(monkeypatch, off=False)
    far, kept, r = goes(lab, level, most=4)
    assert len(far) == in_go and far[-1] > far_side(level)
    # Each go before got further than the first gap by a part kept, and no further
    # than the last.
    over_the_first = (14 + 3) * TILE
    assert all(over_the_first < best < far_side(level) for best in far[:-1])
    assert all(parts >= 1 for parts in kept) and held.parts == sum(kept)
    # The go that got there found the rest as a way, with the lives it had.
    assert r.ways_found == 1 and lab.game.state.lives >= 1


@pytest.mark.parametrize("level", [ThreeGaps, FourGaps], ids=["three gaps", "four gaps"])
def test_held_off_the_same_goes_never_get_past_the_second_gap(
    lab: Lab, monkeypatch: pytest.MonkeyPatch, level: type[TwoGaps]
) -> None:
    held = Held(monkeypatch, off=True)
    far, kept, r = goes(lab, level, most=4)
    # The searches saw changes that got further, and they were thrown away.
    assert held.parts >= 4 and kept == [0, 0, 0, 0]
    assert len(far) == 4 and max(far) < 19 * TILE + 3 * TILE
    assert r.ways_found == 0
