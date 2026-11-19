"""T-1059: where no single change gets past, the search goes on from one that got further.

Some places want two changes: a jump onto something and another off it, or one gap
close behind another. Every single change loses the life there, so a search made
the same tries go after go. Once the list of single changes has been gone through,
a change that loses the life later than it could have by waiting, and on new
ground, is a step on the way: the search goes on from it with a second change.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any, ClassVar

import pytest

from retro_ai_lab.core import ConfigurationError
from retro_ai_lab.emulator.fake import toy_hopper
from retro_ai_lab.emulator.fake.toy_hopper import PALETTES, START_LIVES, TILE, ToyHopper
from retro_ai_lab.learning.actions import ActionSpace
from retro_ai_lab.observation.ground import Ground
from retro_ai_lab.training import EpisodeRunner, episode, search
from retro_ai_lab.training.episode import AHEAD_EVERY, STAND_AFTER
from retro_ai_lab.training.search import CARRY_ON, TOP, Search, Spot, Way
from retro_ai_lab.training.trial import PAST, Outcome
from retro_ai_lab.training.ways import HOLDS, Change, changes
from tests.unit.training import test_going_back as going_back
from tests.unit.training import test_trial as trial
from tests.unit.training.test_going_back import DIED_AT, Lab, steps
from tests.unit.training.test_last_chance_in_a_go import going
from tests.unit.training.test_looking_ahead import THING, Near, looking
from tests.unit.training.test_search import ACROSS, ALIVE, DEAD, Script, look_for
from tests.unit.training.test_search_at_a_loss import searching
from tests.unit.training.test_search_carries_on import spot_of
from tests.unit.training.test_standing_still import standing
from tests.unit.training.test_trial import FRAMES, SPOT, Hopper, a_go_that_fell_in

hopper = trial.hopper  # the practice game behind a connection (pytest fixture)
lab = going_back.lab  # the practice game behind the real services (pytest fixture)

LOG = "retro_ai_lab.training.episode"
DID = 10  # steps of what it did, in the scripted searches: the life went at the last


def lost_at(step: int, ground: int = 0) -> Outcome:
    return Outcome(steps=step, lost=True, ground=ground)


def no_watch() -> object:
    raise AssertionError("nothing is looked at here")


def scripted(
    hopper: Hopper, monkeypatch: pytest.MonkeyPatch, outcomes: Sequence[Outcome]
) -> tuple[Script, list[int], list[Change]]:
    """The plays scripted, what it did (ten steps to the right) and the single changes
    to that, in the order a search tries them."""
    script = Script(outcomes)
    monkeypatch.setattr(search, "play", script)
    did = [hopper.move("right")] * DID
    return script, did, list(changes(did, hopper.space))


def seconds(
    stone: Sequence[int], first: Change, space: ActionSpace, upto: int | None = None
) -> list[Change]:
    """The second changes a search goes on with from a step on, by the rule: the small
    changes to the moves that try made, placed at or after the end of its change, the
    first change's move before the others."""
    after = first.at + first.hold
    every = [c for c in changes(stone, space, upto=upto) if c.at >= after]
    return [c for c in every if c.move == first.move] + [c for c in every if c.move != first.move]


def second_spot(change: Change, first: Change, stone: Sequence[int]) -> Spot:
    """Where a second change is in its list, as a search says it: the holds of the
    first change's move come before the holds of the others."""
    other = len(HOLDS) if change.move != first.move else 0
    return other + HOLDS.index(change.hold), len(stone) - 1 - change.at


def blind(moves: Sequence[int]) -> tuple[tuple[int, ...], bool]:
    return tuple(moves), False


def looked(moves: Sequence[int]) -> tuple[tuple[int, ...], bool]:
    return tuple(moves), True


# ------------------------------------------------- a list gone through to its end


def test_a_list_gone_through_to_its_end_says_so(
    hopper: Hopper, monkeypatch: pytest.MonkeyPatch
) -> None:
    script, did, listed = scripted(hopper, monkeypatch, [DEAD] * 18)
    assert len(listed) == 18
    whole = look_for(hopper, did, no_watch)
    assert whole == Search(None, 18) and whole.lapped and whole.deeper_at == TOP
    # One try short of the end is not the end.
    script.outcomes[:] = [DEAD] * 18
    short = look_for(hopper, did, no_watch, budget=17)
    assert short == Search(None, 17) and not short.lapped
    # A way found is no list gone through for nothing.
    script.outcomes[:] = [*[DEAD] * 17, ALIVE, ACROSS]
    found = look_for(hopper, did, object)
    assert found.way is not None and not found.lapped


def test_begun_further_down_it_is_the_end_of_the_list_that_counts(
    hopper: Hopper, monkeypatch: pytest.MonkeyPatch
) -> None:
    script, did, listed = scripted(hopper, monkeypatch, [])
    k = next(n for n, change in enumerate(listed) if change.hold == HOLDS[1])
    begin = (TOP[0], spot_of(listed[k], did))
    rest = len(listed) - k  # from there to the end of the list
    # With the searches before it, every change has been tried once when the end is
    # reached: there is no need to come all the way round.
    script.outcomes[:] = [DEAD] * rest
    at_the_end = look_for(hopper, did, no_watch, begin_at=begin, budget=rest)
    assert at_the_end.tries == rest and at_the_end.lapped
    assert at_the_end.stopped_at == (TOP[0], spot_of(listed[0], did))
    script.outcomes[:] = [DEAD] * rest
    before_it = look_for(hopper, did, no_watch, begin_at=begin, budget=rest - 1)
    assert before_it.tries == rest - 1 and not before_it.lapped
    script.outcomes[:] = [DEAD] * 18
    round_again = look_for(hopper, did, no_watch, begin_at=begin)
    assert round_again.tries == 18 and round_again.lapped


def test_it_is_the_list_of_changes_to_what_it_did_that_counts(
    hopper: Hopper, monkeypatch: pytest.MonkeyPatch
) -> None:
    script, did, listed = scripted(hopper, monkeypatch, [])
    back = [hopper.move("left")] * DID
    on = hopper.move("right")
    theirs = len(list(changes(back, hopper.space)))
    # The plain way loses the life at step 6, and its changes have half the tries.
    script.outcomes[:] = [lost_at(6)] + [DEAD] * 400
    both = look_for(hopper, back, no_watch, plain=on, budget=400)
    assert both.way is None and both.lapped
    script.outcomes[:] = [lost_at(6)] + [DEAD] * 400
    cut = look_for(hopper, back, no_watch, plain=on, budget=both.tries - 1)
    assert cut.tries == both.tries - 1 and not cut.lapped
    assert both.tries > theirs  # the plain way's list came first
    # Nothing to change (the loss was settled before the first step) is no list.
    script.outcomes[:] = []
    assert not look_for(hopper, did, no_watch, live=0).lapped
    assert listed


class Ticking(Script):
    """Scripted plays that take a second each, and the clock that says so."""

    now = 0.0

    def __call__(
        self, connection: object, slot: str, moves: Sequence[int], space: object, **kw: object
    ) -> Outcome:
        self.now += 1.0
        return super().__call__(connection, slot, moves, space, **kw)

    def clock(self) -> float:
        return self.now


def ticking(
    hopper: Hopper, monkeypatch: pytest.MonkeyPatch, outcomes: Sequence[Outcome]
) -> tuple[Ticking, list[int], list[Change]]:
    script = Ticking(outcomes)
    monkeypatch.setattr(search, "play", script)
    did = [hopper.move("right")] * DID
    return script, did, list(changes(did, hopper.space))


def test_time_up_before_the_end_is_not_the_end(
    hopper: Hopper, monkeypatch: pytest.MonkeyPatch
) -> None:
    script, did, _ = ticking(hopper, monkeypatch, [DEAD] * 18)
    late = look_for(hopper, did, no_watch, seconds=17.0, clock=script.clock)
    assert late.tries == 17 and late.out_of_time and not late.lapped
    script.outcomes[:] = [DEAD] * 18
    script.now = 0.0
    in_time = look_for(hopper, did, no_watch, seconds=18.5, clock=script.clock)
    assert in_time.tries == 18 and not in_time.out_of_time and in_time.lapped


# ------------------------------------------------------------ going on from a step on


def test_without_it_a_later_loss_is_a_loss(hopper: Hopper, monkeypatch: pytest.MonkeyPatch) -> None:
    script, did, listed = scripted(hopper, monkeypatch, [lost_at(16)] * 18)
    found = look_for(hopper, did, no_watch)
    assert found == Search(None, 18) and found.deeper_at == TOP
    assert script.calls == [blind(change.apply(did, CARRY_ON)) for change in listed]


def test_a_change_that_got_further_is_gone_on_from(
    hopper: Hopper, monkeypatch: pytest.MonkeyPatch
) -> None:
    script, did, listed = scripted(hopper, monkeypatch, [])
    first = listed[4]
    assert first.hold == 4
    tried = first.apply(did, CARRY_ON)
    stone = tried[:16]  # what that try did up to the step its life went at
    after = seconds(stone, first, hopper.space)
    script.outcomes[:] = [
        *[DEAD] * 4,
        lost_at(16),  # six steps later than what it did: more than the change lasted
        lost_at(16, PAST),  # looked at: on new ground before it went
        DEAD,
        ALIVE,
        ACROSS,
    ]
    lent: list[object] = []

    def watch() -> object:
        lent.append(object())
        return lent[-1]

    found = look_for(hopper, did, watch, deeper=True)
    way = found.way
    assert way == Way(after[1].apply(stone, CARRY_ON), after[1], tries=9, ground=3, first=first)
    assert found.tries == 9 and len(lent) == 2
    assert script.calls == [
        *[blind(change.apply(did, CARRY_ON)) for change in listed[:4]],
        blind(tried),
        looked(tried),
        blind(after[0].apply(stone, CARRY_ON)),
        blind(after[1].apply(stone, CARRY_ON)),
        looked(after[1].apply(stone, CARRY_ON)),
    ]
    # The way is both changes and the carrying on: the first as it was tried, then
    # the second in the moves that try made.
    assert way is not None and way.moves[: first.at + first.hold] == tried[: first.at + first.hold]
    assert way.change.at >= first.at + first.hold and len(way.moves) == 16 + CARRY_ON
    assert way.plain is None and way.lost_at is None
    # Where it stopped is the first change, as for any way found.
    assert found.stopped_at == (TOP[0], spot_of(first, did)) and found.deeper_at == TOP


def test_the_second_changes_come_after_the_first_and_its_move_is_tried_first(
    hopper: Hopper, monkeypatch: pytest.MonkeyPatch
) -> None:
    script, did, listed = scripted(hopper, monkeypatch, [])
    first = listed[4]
    stone = first.apply(did, CARRY_ON)[:16]
    after = seconds(stone, first, hopper.space)
    script.outcomes[:] = [*[DEAD] * 4, lost_at(16), lost_at(16, 5)] + [DEAD] * 200
    found = look_for(hopper, did, no_watch_but_once(), deeper=True, budget=400)
    assert found.way is None
    played = [moves for moves, _ in script.calls[6 : 6 + len(after)]]
    assert played == [change.apply(stone, CARRY_ON) for change in after]
    # By the rule itself: none before the end of the first change, the first
    # change's move at the front, and both kinds are there.
    assert all(change.at >= first.at + first.hold for change in after)
    same = [change.move == first.move for change in after]
    assert same == sorted(same, reverse=True) and True in same and False in same
    assert len(after) >= 6
    # After its list for nothing the search goes on down its own.
    assert script.calls[6 + len(after)] == blind(listed[5].apply(did, CARRY_ON))
    assert found.tries == 18 + 1 + len(after) and found.lapped


def no_watch_but_once() -> Any:
    """A watch for the one look at a step on, and none after it."""
    lent: list[object] = []

    def watch() -> object:
        assert not lent, "only the step on is looked at"
        lent.append(object())
        return lent[-1]

    return watch


def test_before_its_own_loss_was_settled(hopper: Hopper, monkeypatch: pytest.MonkeyPatch) -> None:
    script, did, _ = scripted(hopper, monkeypatch, [])
    # The loss was settled at step 6 of the ten: the game took 4 steps over it.
    listed = list(changes(did, hopper.space, upto=6))
    first = listed[0]
    assert (first.at, first.hold) == (5, 4)
    stone = first.apply(did, CARRY_ON)[:17]
    after = seconds(stone, first, hopper.space, upto=13)  # 4 steps before its own
    assert after and all(9 <= change.at < 13 for change in after)
    assert len(after) < len(seconds(stone, first, hopper.space))
    script.outcomes[:] = [lost_at(17), lost_at(17, PAST)] + [DEAD] * 200
    found = look_for(hopper, did, no_watch_but_once(), deeper=True, live=6, budget=400)
    assert found.way is None
    played = [moves for moves, _ in script.calls[2 : 2 + len(after)]]
    assert played == [change.apply(stone, CARRY_ON) for change in after]
    assert script.calls[2 + len(after)] == blind(listed[1].apply(did, CARRY_ON))


@pytest.mark.parametrize("hold", HOLDS)
def test_no_later_than_the_change_lasted_is_not_looked_at(
    hopper: Hopper, monkeypatch: pytest.MonkeyPatch, hold: int
) -> None:
    script, did, listed = scripted(hopper, monkeypatch, [])
    # Each loses the life as much later as its change lasted: waiting does that.
    script.outcomes[:] = [lost_at(DID + change.hold) for change in listed]
    waited = look_for(hopper, did, no_watch, deeper=True)
    assert waited == Search(None, 18) and waited.lapped
    # One step more, and it is looked at.
    k = next(n for n, change in enumerate(listed) if change.hold == hold)
    script.calls.clear()
    script.outcomes[:] = [lost_at(DID + change.hold) for change in listed]
    script.outcomes[k] = lost_at(DID + hold + 1)
    script.outcomes.insert(k + 1, lost_at(DID + hold + 1, 0))
    later = look_for(hopper, did, no_watch_but_once(), deeper=True)
    assert later == Search(None, 19)
    assert script.calls[k : k + 2] == [
        blind(listed[k].apply(did, CARRY_ON)),
        looked(listed[k].apply(did, CARRY_ON)),
    ]


@pytest.mark.parametrize(
    "seen",
    [
        lost_at(16, PAST - 1),  # the same loss, a little later
        Outcome(steps=30, lost=False, ground=PAST),  # the game did not do the same again
    ],
)
def test_looked_at_and_no_step_on_costs_the_one_play(
    hopper: Hopper, monkeypatch: pytest.MonkeyPatch, seen: Outcome
) -> None:
    script, did, listed = scripted(hopper, monkeypatch, [lost_at(16), seen] + [DEAD] * 17)
    found = look_for(hopper, did, no_watch_but_once(), deeper=True)
    assert found == Search(None, 19) and found.lapped and found.deeper_at == TOP
    assert script.calls[:3] == [
        blind(listed[0].apply(did, CARRY_ON)),
        looked(listed[0].apply(did, CARRY_ON)),
        blind(listed[1].apply(did, CARRY_ON)),
    ]


def test_a_second_change_that_loses_the_life_later_is_not_gone_on_from(
    hopper: Hopper, monkeypatch: pytest.MonkeyPatch
) -> None:
    script, did, listed = scripted(hopper, monkeypatch, [])
    first = listed[0]
    stone = first.apply(did, CARRY_ON)[:16]
    after = seconds(stone, first, hopper.space)
    script.outcomes[:] = [lost_at(16), lost_at(16, PAST)] + [lost_at(30)] * len(after)
    script.outcomes += [DEAD] * 17
    found = look_for(hopper, did, no_watch_but_once(), deeper=True, budget=400)
    assert found.way is None and found.tries == 2 + len(after) + 17
    assert [seen for _, seen in script.calls].count(True) == 1


def test_a_second_change_that_stays_alive_and_gets_nowhere_is_no_way(
    hopper: Hopper, monkeypatch: pytest.MonkeyPatch
) -> None:
    script, did, listed = scripted(hopper, monkeypatch, [])
    first = listed[0]
    stone = first.apply(did, CARRY_ON)[:16]
    after = seconds(stone, first, hopper.space)
    nowhere = Outcome(steps=36, lost=False, ground=PAST - 1)
    script.outcomes[:] = [lost_at(16), lost_at(16, PAST), ALIVE, nowhere, ALIVE, ACROSS]
    found = look_for(hopper, did, object, deeper=True)
    assert found.way == Way(
        after[1].apply(stone, CARRY_ON), after[1], tries=6, ground=3, first=first
    )


# --------------------------------------------------------- cut short, and carried on


def test_a_search_cut_short_in_a_second_list_says_where_and_the_next_begins_there(
    hopper: Hopper, monkeypatch: pytest.MonkeyPatch
) -> None:
    script, did, listed = scripted(hopper, monkeypatch, [])
    first = listed[2]
    stone = first.apply(did, CARRY_ON)[:16]
    after = seconds(stone, first, hopper.space)
    script.outcomes[:] = [DEAD, DEAD, lost_at(16), lost_at(16, PAST), DEAD, DEAD]
    cut = look_for(hopper, did, no_watch_but_once(), deeper=True, budget=6)
    assert cut == Search(None, 6) and not cut.lapped
    assert cut.stopped_at == (TOP[0], spot_of(first, did))
    assert cut.deeper_at == (TOP[0], second_spot(after[2], first, stone))
    # The next search about there: the first change again (it has to be looked at to
    # know its moves), then the second changes from the one not played on, round
    # their list once, and on down the list of single changes.
    script.calls.clear()
    script.outcomes[:] = [lost_at(16), lost_at(16, PAST)] + [DEAD] * 200
    on = look_for(
        hopper,
        did,
        no_watch_but_once(),
        deeper=True,
        budget=400,
        begin_at=cut.stopped_at,
        deeper_at=cut.deeper_at,
    )
    assert on.way is None and on.lapped and on.tries == 18 + 1 + len(after)
    assert script.calls[:2] == [blind(first.apply(did, CARRY_ON)), looked(first.apply(did, 20))]
    played = [moves for moves, _ in script.calls[2 : 2 + len(after)]]
    assert played == [change.apply(stone, CARRY_ON) for change in after[2:] + after[:2]]
    assert script.calls[2 + len(after)] == blind(listed[3].apply(did, CARRY_ON))


def test_where_it_stopped_in_a_second_list_is_for_that_first_change_only(
    hopper: Hopper, monkeypatch: pytest.MonkeyPatch
) -> None:
    script, did, listed = scripted(hopper, monkeypatch, [])
    first = listed[4]  # another step on than the one the last search stopped in
    stone = first.apply(did, CARRY_ON)[:16]
    after = seconds(stone, first, hopper.space)
    script.outcomes[:] = [*[DEAD] * 2, lost_at(16), lost_at(16, PAST), DEAD, DEAD]
    begin = (TOP[0], spot_of(listed[2], did))
    found = look_for(
        hopper,
        did,
        no_watch_but_once(),
        deeper=True,
        budget=6,
        begin_at=begin,
        deeper_at=(TOP[0], (1, 2)),
    )
    # Begun at the third change, the step on is the fifth: its list is begun at the top.
    assert script.calls[4:] == [
        blind(after[0].apply(stone, CARRY_ON)),
        blind(after[1].apply(stone, CARRY_ON)),
    ]
    assert found.deeper_at == (TOP[0], second_spot(after[2], first, stone))


def test_out_of_tries_before_a_step_on_is_looked_at(
    hopper: Hopper, monkeypatch: pytest.MonkeyPatch
) -> None:
    script, did, listed = scripted(hopper, monkeypatch, [DEAD, DEAD, lost_at(16)])
    cut = look_for(hopper, did, no_watch, deeper=True, budget=3)
    assert cut == Search(None, 3) and script.outcomes == []
    # The next begins at that change, and at the top of what comes after it.
    assert cut.stopped_at == (TOP[0], spot_of(listed[2], did)) and cut.deeper_at == TOP


def test_out_of_tries_with_a_second_change_alive_and_not_looked_at(
    hopper: Hopper, monkeypatch: pytest.MonkeyPatch
) -> None:
    script, did, listed = scripted(hopper, monkeypatch, [])
    first = listed[0]
    stone = first.apply(did, CARRY_ON)[:16]
    after = seconds(stone, first, hopper.space)
    script.outcomes[:] = [lost_at(16), lost_at(16, PAST), DEAD, ALIVE]
    cut = look_for(hopper, did, no_watch_but_once(), deeper=True, budget=4)
    # Not settled: the next search plays that second change again.
    assert cut == Search(None, 4)
    assert cut.deeper_at == (TOP[0], second_spot(after[1], first, stone))


def test_the_time_holds_inside_a_second_list(
    hopper: Hopper, monkeypatch: pytest.MonkeyPatch
) -> None:
    script, did, listed = ticking(hopper, monkeypatch, [])
    first = listed[0]
    stone = first.apply(did, CARRY_ON)[:16]
    after = seconds(stone, first, hopper.space)
    script.outcomes[:] = [lost_at(16), lost_at(16, PAST), DEAD, DEAD, DEAD]
    late = look_for(hopper, did, no_watch_but_once(), deeper=True, seconds=4.5, clock=script.clock)
    assert late.tries == 5 and late.out_of_time and late.way is None
    assert late.deeper_at == (TOP[0], second_spot(after[3], first, stone))
    # And before the step on is looked at.
    script.outcomes[:] = [lost_at(16)]
    script.now = 0.0
    sooner = look_for(hopper, did, no_watch, deeper=True, seconds=0.5, clock=script.clock)
    assert sooner.tries == 1 and sooner.out_of_time and sooner.deeper_at == TOP


# ------------------------------------------------------------------ the plain way


def test_from_the_plain_way_too(hopper: Hopper, monkeypatch: pytest.MonkeyPatch) -> None:
    script, _, _ = scripted(hopper, monkeypatch, [])
    back, on = hopper.move("left"), hopper.move("right")
    did = [back] * DID
    base = (on,) * 6  # holding the plain move loses the life at step 6
    listed = list(changes(base, hopper.space))
    first = listed[0]
    tried = first.apply(base, CARRY_ON)
    stone = tried[:11]
    after = seconds(stone, first, hopper.space)
    script.outcomes[:] = [lost_at(6), lost_at(11), lost_at(11, PAST), ALIVE, ACROSS]
    found = look_for(hopper, did, object, deeper=True, plain=on)
    assert found.way == Way(
        after[0].apply(stone, CARRY_ON),
        after[0],
        tries=5,
        ground=3,
        plain=on,
        lost_at=6,
        first=first,
    )
    assert script.calls == [
        blind((on,) * (DID + CARRY_ON)),
        blind(tried),
        looked(tried),
        blind(after[0].apply(stone, CARRY_ON)),
        looked(after[0].apply(stone, CARRY_ON)),
    ]


def test_the_plain_ways_share_holds_inside_its_second_list_and_each_list_says_where(
    hopper: Hopper, monkeypatch: pytest.MonkeyPatch
) -> None:
    script, _, _ = scripted(hopper, monkeypatch, [])
    back, on = hopper.move("left"), hopper.move("right")
    did = [back] * DID
    base = (on,) * 6
    first = next(iter(changes(base, hopper.space)))
    stone = first.apply(base, CARRY_ON)[:11]
    after = seconds(stone, first, hopper.space)
    theirs = list(changes(did, hopper.space))
    # Eight tries, four of them the plain way's: held, its first change, the look at
    # that, one second change. Then four changes to what it did.
    script.outcomes[:] = [lost_at(6), lost_at(11), lost_at(11, PAST), DEAD, *[DEAD] * 4]
    found = look_for(hopper, did, no_watch_but_once(), deeper=True, plain=on, budget=8)
    assert found == Search(None, 8) and script.outcomes == []
    assert found.stopped_at == (spot_of(first, base), spot_of(theirs[4], did))
    assert found.deeper_at == (second_spot(after[1], first, stone), (0, 0))
    assert script.calls[4:] == [blind(change.apply(did, CARRY_ON)) for change in theirs[:4]]
    # Carried on: the plain way's list at its step on and the second change after it.
    script.calls.clear()
    script.outcomes[:] = [lost_at(6), lost_at(11), lost_at(11, PAST), ALIVE, ACROSS]
    on_from = look_for(
        hopper,
        did,
        object,
        deeper=True,
        plain=on,
        begin_at=found.stopped_at,
        deeper_at=found.deeper_at,
    )
    assert on_from.way is not None and on_from.way.change == after[1]
    assert on_from.way.first == first and on_from.way.tries == 5


def test_its_own_loss_is_settled_as_long_before_as_the_plain_ways(
    hopper: Hopper, monkeypatch: pytest.MonkeyPatch
) -> None:
    script, _, _ = scripted(hopper, monkeypatch, [])
    back, on = hopper.move("left"), hopper.move("right")
    did = [back] * 12
    # What it did was settled at step 9 of 12: three steps. The plain way loses the
    # life at step 10, so its changes are placed before step 7, and the second
    # changes three steps before the step on loses its own.
    base = (on,) * 10
    listed = list(changes(base, hopper.space, upto=7))
    first = listed[0]
    stone = first.apply(base, CARRY_ON)[:16]
    after = seconds(stone, first, hopper.space, upto=13)
    assert after and max(change.at for change in after) == 12
    script.outcomes[:] = [lost_at(10), lost_at(16), lost_at(16, PAST), ALIVE, ACROSS]
    found = look_for(hopper, did, object, deeper=True, plain=on, live=9)
    assert found.way is not None and found.way.change == after[0]
    assert script.calls[3] == blind(after[0].apply(stone, CARRY_ON))


# ------------------------------------------------------------------- what is refused


def test_what_is_refused(hopper: Hopper, monkeypatch: pytest.MonkeyPatch) -> None:
    script, did, _ = scripted(hopper, monkeypatch, [])
    for bad in (1, 0, None, "yes"):
        with pytest.raises(ConfigurationError, match="deeper must be true or false"):
            look_for(hopper, did, no_watch, deeper=bad)
    with pytest.raises(ConfigurationError, match="at a standstill none was"):
        look_for(hopper, did, no_watch, deeper=True, stood=True)
    spot = (TOP[0], (0, 2))
    with pytest.raises(ConfigurationError, match="deeper_at is for a search that goes deeper"):
        look_for(hopper, did, no_watch, deeper_at=spot)
    for bad_at in (((0, 0),), ((0, 0), (0, -1)), [(0, 0), (0, 0)], ((0, 0), (0, 0.5)), None):
        with pytest.raises(ConfigurationError, match="deeper_at is two pairs"):
            look_for(hopper, did, no_watch, deeper=True, deeper_at=bad_at)
    # What begin_at is told is as it was.
    with pytest.raises(ConfigurationError, match="begin_at is two pairs"):
        look_for(hopper, did, no_watch, begin_at=((0, 0),))
    assert script.calls == []
    # The top is no place to carry on from, and may be said without going deeper.
    script.outcomes[:] = [DEAD] * 18
    assert look_for(hopper, did, no_watch, deeper_at=TOP) == Search(None, 18)


def test_where_the_screen_stands_still_no_step_on_is_looked_for(
    hopper: Hopper, monkeypatch: pytest.MonkeyPatch
) -> None:
    script, did, listed = scripted(hopper, monkeypatch, [lost_at(16), lost_at(16), ALIVE])
    found = look_for(hopper, did, no_watch, deeper=True, still=True)
    # No new ground to tell a step on by: the first change that keeps the life (T-1058).
    assert found.way == Way(listed[2].apply(did, CARRY_ON), listed[2], tries=3, ground=0)
    assert [seen for _, seen in script.calls] == [False] * 3


# ------------------------------------------------- on a game with two gaps in a row


class TwoGaps(ToyHopper):
    """The practice hopper's rules on a level of its own: gaps close together, with a
    landing between them too short to stop on. One jump does not clear both, and the
    second wants a fresh press: no single change to walking right gets past."""

    GAPS: ClassVar[tuple[tuple[int, int], ...]] = ((14, 3), (19, 3))

    def _enter(self) -> None:
        self._gaps, self._walls, self._walkers, self._coins = self.GAPS, (), (), ()
        self._hole = frozenset(
            column for first, wide in self.GAPS for column in range(first, first + wide)
        )
        self._wall = frozenset()
        self._palette = PALETTES[0]


class OneWideGap(TwoGaps):
    """A gap no jump clears."""

    GAPS: ClassVar[tuple[tuple[int, int], ...]] = ((14, 8),)


FAR_SIDE = (19 + 3) * TILE  # where the ground begins again behind the second gap
PAUSE = 216  # frames from the fall to the life going: 54 steps, as a real game takes


def put_in(hopper: Hopper, game: ToyHopper, pause: int = toy_hopper.DYING_FRAMES) -> None:
    game.dying_frames = pause
    hopper.adapter._game = game
    hopper.game = game
    game.reset()
    hopper.adapter.save_state(SPOT)


@pytest.mark.parametrize("pause", [toy_hopper.DYING_FRAMES, PAUSE])
def test_two_gaps_close_together_want_two_changes(hopper: Hopper, pause: int) -> None:
    put_in(hopper, TwoGaps(0), pause)
    watch, mark, died = a_go_that_fell_in(hopper)
    right = hopper.move("right")
    did = [right] * died
    live = died - pause // FRAMES

    def trial() -> object:
        return watch.trial(mark)

    # Every single change, placed before the loss was settled: none gets past.
    single = look_for(hopper, did, trial, live=live, budget=1000)
    assert single.way is None and single.lapped and single.tries > 50
    # Going on from one that got further: a jump, and a second after it.
    found = look_for(hopper, did, trial, live=live, budget=1000, deeper=True)
    way = found.way
    assert way is not None and way.first is not None and way.tries < 30
    jump = hopper.move("a", "right")
    assert way.first.move == jump and way.change.move == jump
    assert way.change.at >= way.first.at + way.first.hold and way.ground >= PAST
    # The game agrees: played from the save it is over both gaps with every life.
    out = hopper.play(list(way.moves))
    assert not out.lost and hopper.game.state.lives == START_LIVES
    assert hopper.game.state.x > FAR_SIDE
    # And the first change by itself is what it was taken for: over the first gap,
    # into the second.
    alone = hopper.play(list(way.first.apply(did, CARRY_ON)))
    assert alone.lost and alone.steps - died > way.first.hold


def test_a_gap_nothing_crosses_is_searched_for_nothing_either_way(hopper: Hopper) -> None:
    put_in(hopper, OneWideGap(0), PAUSE)
    watch, mark, died = a_go_that_fell_in(hopper)
    did = [hopper.move("right")] * died
    live = died - PAUSE // FRAMES
    lent: list[object] = []

    def trial() -> object:
        lent.append(watch.trial(mark))
        return lent[-1]

    single = look_for(hopper, did, trial, live=live, budget=1000)
    assert single.way is None and lent == []
    deeper = look_for(hopper, did, trial, live=live, budget=1000, deeper=True)
    assert deeper.way is None and deeper.lapped
    # What going deeper cost there: a few looks, and the second lists after them.
    assert 1 <= len(lent) <= 3 and single.tries < deeper.tries < 4 * single.tries


# ------------------------------------------------------------------------ the runner


class Asked:
    """Stands in for ``find_way`` in the runner: notes what each search was told about
    going deeper, and answers from a list (the real search where the list says so)."""

    REAL = "the real search"

    def __init__(self, monkeypatch: pytest.MonkeyPatch, answers: Sequence[object] = ()) -> None:
        self.deeper: list[bool] = []
        self.deeper_at: list[tuple[Spot, Spot]] = []
        self.stood: list[bool] = []
        self.answers = list(answers)
        self.hold_off = False
        self.real = episode.find_way
        monkeypatch.setattr(episode, "find_way", self)

    def __call__(self, *args: Any, **kw: Any) -> Search:
        self.deeper.append(kw["deeper"])
        self.deeper_at.append(kw["deeper_at"])
        self.stood.append(kw["stood"])
        if self.hold_off:
            kw["deeper"], kw["deeper_at"] = False, TOP
        out = self.answers.pop(0) if self.answers else self.REAL
        return out if isinstance(out, Search) else self.real(*args, **kw)


def nothing(lapped: bool, deeper_at: tuple[Spot, Spot] = TOP) -> Search:
    return Search(None, 5, stopped_at=((0, 2), (1, 4)), lapped=lapped, deeper_at=deeper_at)


TOLD = "Every single small change has been tried about here and none got past"


def said(caplog: pytest.LogCaptureFixture, part: str) -> list[str]:
    return [rec.getMessage() for rec in caplog.records if part in rec.getMessage()]


def test_a_place_gets_deeper_searches_once_one_there_has_gone_through_its_list(
    lab: Lab, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    there = ((1, 6), (4, 2))
    asked = Asked(
        monkeypatch,
        [nothing(False), nothing(True, there), nothing(False), nothing(False, there)],
    )
    r, _ = searching(lab, ways_per_go=3, give_up_after=None, max_steps=400)
    with caplog.at_level(logging.INFO, logger=LOG):
        lab.game.reset()
        r.begin()
        while r.step():
            pass
        r.finish()
        # Three lives at the first gap. The second search came to the end of its
        # list, so the third is told to go deeper, from where the second stopped.
        assert asked.deeper == [False, False, True]
        assert asked.deeper_at == [TOP, TOP, there]
        assert len(said(caplog, TOLD)) == 1
        # It is kept for the place from go to go, whatever later searches say, with
        # where the last one stopped.
        lab.game.reset()
        r.begin()
        steps(r, DIED_AT)
        assert asked.deeper[3:] == [True] and asked.deeper_at[3:] == [TOP]
        steps(r, DIED_AT)
        assert asked.deeper[4:] == [True] and asked.deeper_at[4:] == [there]
        assert len(said(caplog, TOLD)) == 3
    assert asked.stood == [False] * 5
    assert said(caplog, TOLD)[0] == (
        "Every single small change has been tried about here and none got past, so this "
        "search also goes on from a change that loses the life later and further on, with "
        "a second change after it."
    )


def test_the_first_search_at_a_place_can_be_the_one_that_goes_through(
    lab: Lab, monkeypatch: pytest.MonkeyPatch
) -> None:
    asked = Asked(monkeypatch, [nothing(True)])
    r, _ = searching(lab, ways_per_go=2, give_up_after=None, max_steps=400)
    r.begin()
    steps(r, 2 * DIED_AT)
    assert asked.deeper == [False, True]
    # The way it then found goes over the gap: further on is another place.
    assert r.ways_found == 1
    for _ in range(300):
        if len(asked.deeper) == 3 or not r.step():
            break
    assert asked.deeper[2:] in ([], [False])


def test_a_life_seen_coming_is_searched_deeper_as_one_lost_is(
    lab: Lab, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    asked = Asked(monkeypatch, [nothing(True), nothing(False)])
    r, _ = looking(lab, ways_per_go=2)
    near = Near(monkeypatch, r.watch)
    r.begin()
    steps(r, 5)
    with caplog.at_level(logging.INFO, logger=LOG):
        near.now = THING
        assert r.step()
        assert (r.looked_ahead, r.ways_ahead, asked.deeper) == (1, 1, [False])
        # It carries on. Ten steps on another thing of the kind is beside the first,
        # and it looks ahead again, about there.
        steps(r, AHEAD_EVERY - 1)
        near.now = (*THING, (2, "thing-9"))
        assert r.step()
    assert asked.deeper == [False, True] and r.ways_sought == 0
    assert len(said(caplog, TOLD)) == 1


def test_not_where_the_screen_stands_still(
    lab: Lab, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    asked = Asked(monkeypatch, [nothing(True), nothing(True), nothing(True)])
    r, _ = searching(lab, ways_per_go=3, give_up_after=None, max_steps=400)
    r.begin()
    steps(r, DIED_AT)
    # The place is known for it. With the ground saying the screen stands still, a
    # search there is for a change that keeps the life, and goes no deeper.
    monkeypatch.setattr(Ground, "still", property(lambda self: True))
    with caplog.at_level(logging.INFO, logger=LOG):
        steps(r, DIED_AT)
    assert asked.deeper == [False, False] and said(caplog, TOLD) == []
    assert asked.deeper_at == [TOP, TOP]


def test_never_at_a_standstill(lab: Lab, monkeypatch: pytest.MonkeyPatch) -> None:
    asked = Asked(monkeypatch, [nothing(True), nothing(True)])
    r, _ = standing(lab, ways_per_go=3)
    r.begin()
    stand, jump = lab.move(), lab.move("a")
    for n in range(STAND_AFTER + 40):
        if r.ways_stood:
            break
        lab.agent.script.clear()
        lab.agent.script.append(jump if n % 8 < 3 else stand)
        r.step()
    # No life was lost there. A search for a way on is not told to go deeper, and
    # what it came to is remembered for no place: the life lost in the next go, at
    # the gap a few steps on, is searched as at a new place.
    assert asked.stood == [True] and asked.deeper == [False] and asked.deeper_at == [TOP]
    r.finish()
    lab.agent.script.clear()
    lab.game.reset()
    r.begin()
    steps(r, DIED_AT)
    assert asked.stood == [True, False] and asked.deeper == [False, False]


def two_gaps(lab: Lab, **kw: object) -> EpisodeRunner:
    """A runner on the level of two gaps, as the app runs one: it asks the game when a
    loss was settled, and the game takes 54 steps over a life."""
    game = TwoGaps(0)
    lab.adapter._game = game
    lab.game = game
    kw.setdefault("ways_per_go", 3)
    kw.setdefault("max_steps", 900)
    r, _ = going(lab, PAUSE, last_chance=True, **kw)
    return r


def to_search(r: EpisodeRunner, number: int) -> None:
    for _ in range(900):
        if r.ways_sought == number:
            return
        assert r.step()
    raise AssertionError(f"it never made search {number}")


def test_through_the_runner_the_second_search_at_two_gaps_finds_the_way_of_two(
    lab: Lab, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    asked = Asked(monkeypatch)
    r = two_gaps(lab)
    with caplog.at_level(logging.INFO, logger=LOG):
        r.begin()
        to_search(r, 1)
        # Every single change, for nothing: the life is lost and the level begun again.
        assert (r.ways_found, asked.deeper) == (0, [False])
        assert lab.game.state.lives == START_LIVES - 1 and said(caplog, TOLD) == []
        to_search(r, 2)
    assert (r.ways_found, r.times_back, asked.deeper) == (1, 1, [False, True])
    assert lab.game.state.lives == START_LIVES - 1  # this one was not lost
    assert len(said(caplog, TOLD)) == 1
    (found,) = said(caplog, "found a way past")
    assert ": holding right instead of what it did, with right + a for 4 steps, starting " in found
    assert " steps before that would lose the life, and " in found
    assert found.endswith(
        " steps after that began right + a for 4 steps. No single change got past here: the "
        "first loses the life later and further on, and the second gets past from there."
    )
    # The way is played: over both gaps, with the lives it had.
    for _ in range(120):
        if lab.game.state.x > FAR_SIDE + 2 * TILE:
            break
        assert r.step()
    assert lab.game.state.x > FAR_SIDE and lab.game.state.lives == START_LIVES - 1
    assert r.ways_sought == 2


def test_held_off_no_search_there_finds_a_way(lab: Lab, monkeypatch: pytest.MonkeyPatch) -> None:
    asked = Asked(monkeypatch)
    asked.hold_off = True  # the search as it was before: single changes, every time
    r = two_gaps(lab)
    for _ in range(2):
        lab.game.reset()
        r.begin()
        while r.step():
            pass
        result = r.finish()
        assert (result.ways_sought, result.ways_found) == (3, 0)
        assert lab.game.state.game_over
    # It was told to go deeper from the second search on, and could not.
    assert asked.deeper == [False] + [True] * 5


def test_the_log_says_a_way_of_two_changes_to_what_it_did(
    lab: Lab, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    first, second = Change(20, lab.move("a", "right"), 4), Change(30, lab.move("a"), 8)
    did = [lab.move("right")] * DIED_AT
    moves = second.apply(first.apply(did, 0)[:DIED_AT], CARRY_ON)
    way = Way(moves, second, tries=7, ground=3, first=first)
    Asked(monkeypatch, [Search(way, 7)])
    r, _ = searching(lab)
    with caplog.at_level(logging.INFO, logger=LOG):
        r.begin()
        steps(r, DIED_AT)
    (found,) = said(caplog, "found a way past")
    assert found.startswith(f"Lost a life at step {DIED_AT}. Went back {DIED_AT} steps and ")
    assert found.endswith(
        f"found a way past on try 7, after 0.0 s: right + a for 4 steps, starting "
        f"{DIED_AT - 20} steps before the life went, and 10 steps after that began a for 8 "
        "steps. No single change got past here: the first loses the life later and further "
        "on, and the second gets past from there."
    )
