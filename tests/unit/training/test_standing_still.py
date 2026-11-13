"""T-1018: a go that stays about one spot, with no life lost, goes back and looks for a way on."""

from __future__ import annotations

import logging
import re

import pytest

from retro_ai_lab.emulator.fake import toy_hopper
from retro_ai_lab.services.experiments import ExperimentHarness
from retro_ai_lab.training import EpisodeRunner, episode
from retro_ai_lab.training.episode import (
    NOW_SLOT,
    STAND_AFTER,
    STAND_NEAR,
    STAND_REACH,
    STAND_TRIES,
    STAND_WITHIN,
    EpisodeResult,
)
from retro_ai_lab.training.moments import EVERY, Moments
from retro_ai_lab.training.route import Route
from retro_ai_lab.training.search import BUDGET, CARRY_ON, Search, Way
from retro_ai_lab.training.trial import Outcome
from retro_ai_lab.training.vision import ScreenWatch
from retro_ai_lab.training.ways import Change
from tests.unit.training import test_going_back as going_back
from tests.unit.training.test_going_back import DIED_AT, SLOTS, Lab, steps
from tests.unit.training.test_remembered_way import remembering
from tests.unit.training.test_search_at_a_loss import GAP_ENDS, searching

lab = going_back.lab  # the practice game behind the real services (pytest fixture)

SIX = tuple(f"back{i}" for i in range(6))  # as many moments as the app keeps
WALL = toy_hopper.TILE * 56  # the left face of the first wall of this level
AT_WALL = WALL - 12  # where the player stands when it is against it
# Two spots on the first stretch of ground, short of the first gap and further apart
# than anything that is still "about one spot" (64 pixels either way of where it came).
NEAR, FAR = 20, 160
NOTHING = Search(None, 4)
STOPPED = (
    "Looking for a way on where it stands still found none 3 times in a row, "
    "so the AI stops doing that in this game: here it only costs time."
)


def standing(lab: Lab, **kw: object) -> tuple[EpisodeRunner, Moments]:
    """A go about to start that may search, also where it stands still."""
    kw.setdefault("stand_after", STAND_AFTER)
    return searching(lab, **kw)


def recalling(lab: Lab, **kw: object) -> tuple[EpisodeRunner, Route, ExperimentHarness]:
    """The same, with a remembered way and goes that start from a checkpoint."""
    kw.setdefault("stand_after", STAND_AFTER)
    return remembering(lab, **kw)


class Asked:
    """Stands in for ``find_way`` in the runner: notes what it was asked, and answers
    from a list (nothing found, once the list is used up)."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch, answers: list[Search] | None = None):
        self.calls: list[tuple[str, tuple[int, ...], bool]] = []
        self.budgets: list[object] = []  # the plays each search was allowed
        self.enough: list[object] = []  # what each was to ask of a try (None: nothing)
        self.answers = list(answers or [])
        monkeypatch.setattr(episode, "find_way", self)

    def __call__(
        self, connection: object, slot: str, trace: tuple[int, ...], *a: object, **kw: object
    ) -> Search:
        self.calls.append((slot, tuple(trace), bool(kw["stood"])))
        self.budgets.append(kw["budget"])
        self.enough.append(kw["enough"])
        return self.answers.pop(0) if self.answers else NOTHING


def hop_about(lab: Lab, r: EpisodeRunner, at: int, count: int = 100) -> None:
    """Walk to about ``at`` and then jump on that spot: ``count`` steps in all."""
    right, left, stand, jump = lab.move("right"), lab.move("left"), lab.move(), lab.move("a")
    there = False
    for n in range(count):
        x = lab.game.state.x
        there = there or abs(x - at) <= 8
        move = (jump if n % 8 < 3 else stand) if there else (right if x < at else left)
        lab.agent.script.clear()
        lab.agent.script.append(move)
        assert r.step()
    lab.agent.script.clear()


def test_a_go_that_stands_at_a_wall_goes_back_and_gets_over(
    lab: Lab, caplog: pytest.LogCaptureFixture
) -> None:
    assert (STAND_AFTER, STAND_WITHIN, STAND_NEAR, STAND_REACH) == (60, 64, 8, 4)
    r, route, _ = recalling(lab, ways_per_go=3, moments=Moments(SIX))
    moments = r.moments
    assert moments is not None
    # The first go only walks right. Its three searches go on the two gaps before the
    # wall, and it ends standing against the wall with none left.
    r.begin()
    played = 0
    while r.step():
        played += 1
    first = r.finish()
    assert first.ended_by == "stuck" and played < 330
    assert lab.game.state.x == AT_WALL and lab.game.state.lives == toy_hopper.START_LIVES
    assert (first.ways_sought, first.ways_found, first.ways_stood) == (3, 3, 0)
    knew = len(route.moves)  # the way to the wall
    # The second go plays what it remembers and comes to the wall with its searches.
    with caplog.at_level(logging.INFO, logger="retro_ai_lab.training.episode"):
        r.begin()
        while lab.game.state.x < AT_WALL:
            assert r.step()
        assert r.ways_sought == 0
        came = moments.steps
        # It stands there for 60 steps and nothing is searched...
        while r.ways_sought == 0:
            assert lab.game.state.x == AT_WALL and r.times_back == 0
            before = moments.steps
            assert r.step() and before - came <= STAND_AFTER
        # The sixty steps count from the last look at which it was more than 64 pixels on
        # from the look that last counted: at 8 pixels a step that is up to 8 steps before
        # the wall, and the step that looks is the sixtieth, so 51 to 59 steps at the wall.
        # (Corrected in T-1020 with Boss's word, 13 Nov: "more than 52" held only while the
        # camera was misread on the way to the wall.)
        assert before - came >= STAND_AFTER - 1 - STAND_WITHIN // 8
    # ...and then, in one step, it has looked and gone back: to a moment from before
    # it came to the wall, so none of the standing is left in the story.
    back = before - (moments.steps - 1)  # the step that looked also played the way's first move
    assert (r.ways_sought, r.ways_found, r.ways_stood, r.times_back) == (1, 1, 1, 1)
    assert lab.saved[-1] == NOW_SLOT and lab.loaded[-1] in SIX
    assert moments.steps - 1 <= before - STAND_AFTER and lab.game.state.x < AT_WALL - 8
    said = [x.getMessage() for x in caplog.records if x.getMessage().startswith("Stayed about")]
    assert len(said) == 1 and re.fullmatch(
        f"Stayed about one spot for {STAND_AFTER} steps, with no life lost. Went back {back} "
        r"steps and found a way on at try \d+, after \d+\.\d s: holding right instead of what "
        r"it did, with right \+ a for \d+ steps from step \d+ of that\.",
        said[0],
    ), said
    assert STAND_AFTER <= back < STAND_AFTER + EVERY
    # The way is played without asking the learner, and it is over the wall with every life.
    asked = lab.agent.asked
    at_the_wall = 0
    while lab.game.state.x <= WALL + toy_hopper.TILE:
        assert r.step()
        at_the_wall += lab.game.state.x == AT_WALL
    assert lab.agent.asked == asked and lab.game.state.lives == toy_hopper.START_LIVES
    # It jumps as it comes to the wall: next to none of the standing is in the way.
    assert at_the_wall <= STAND_REACH
    steps(r, 20)
    second = r.finish()
    assert (second.ways_sought, second.ways_found, second.ways_stood) == (1, 1, 1)
    assert second.describe_ways() == ", went back once (where it stood still) and found a way past"
    # The third go is over the wall from memory: no search, and no standing there.
    r.begin()
    assert r.remembered == len(route.moves) > knew
    at_the_wall = 0
    while lab.game.state.x <= WALL + toy_hopper.TILE:
        assert r.step()
        at_the_wall += lab.game.state.x == AT_WALL
    assert (r.ways_sought, r.times_back) == (0, 0) and at_the_wall <= STAND_REACH
    assert lab.game.state.lives == toy_hopper.START_LIVES
    r.finish()


def test_pacing_over_old_ground_is_no_standstill(lab: Lab, monkeypatch: pytest.MonkeyPatch) -> None:
    asked = Asked(monkeypatch)
    r, _ = standing(lab, ways_per_go=3)
    right, left = lab.move("right"), lab.move("left")
    # To and fro over the first stretch of ground, further than "about one spot" each
    # way: after the first crossing none of it is new ground.
    lab.agent.script.extend([right] * 17 + ([left] * 21 + [right] * 21) * 7)
    r.begin()
    seen = []
    for _ in range(300):
        assert r.step()
        seen.append(lab.game.state.x)
    assert max(seen) - min(seen) > 2 * STAND_WITHIN and r.watch is not None
    assert r.watch.ground.steps == r.watch.ground.steps > 0  # it did get somewhere, once
    assert asked.calls == [] and r.ways_sought == 0
    assert lab.game.state.lives == toy_hopper.START_LIVES
    # The same on a shorter leash, inside "about one spot" either way, is a standstill.
    lab.game.reset()
    r, _ = standing(lab, ways_per_go=3)
    lab.agent.script.clear()
    lab.agent.script.extend([right] * 14 + ([left] * 12 + [right] * 12) * 12)
    r.begin()
    seen.clear()
    for _ in range(300):
        assert r.step()
        seen.append(lab.game.state.x)
    assert STAND_WITHIN < max(seen) - min(seen) < 2 * STAND_WITHIN
    assert len(asked.calls) == 1 and r.ways_sought == r.ways_stood == 1


def test_jumping_on_the_spot_is_searched_once(
    lab: Lab, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    asked = Asked(monkeypatch)
    # Three slots: 40 to 59 steps of moments. A go is not ended here for looking the
    # same to the brain, which knows only how far along it is: the count is what is tested.
    r, moments = standing(lab, ways_per_go=3, unstick_after=None)
    assert len(SLOTS) * EVERY <= STAND_AFTER
    r.begin()
    with caplog.at_level(logging.INFO, logger="retro_ai_lab.training.episode"):
        hop_about(lab, r, FAR, STAND_AFTER // 2)
        while not asked.calls:
            assert lab.loaded == [] and r.steps < STAND_AFTER + EVERY
            before, kept, x = moments.steps, moments.kept, lab.game.state.x
            hop_about(lab, r, FAR, 1)
        assert r.steps > STAND_AFTER
    # Sixty steps about one spot: it looked. Every moment kept is from after it came
    # there, so the oldest is the one, with the first steps it made from it: up to a
    # few past the first look that saw it where it is now (it may be in the air).
    assert len(asked.calls) == 1
    slot, trace, stood = asked.calls[0]
    assert stood and slot == kept[0].slot and STAND_REACH <= len(trace) <= STAND_REACH + 8
    # Twice the plays of a search after a life lost: here most changes cost two.
    assert asked.budgets == [STAND_TRIES * BUDGET] and STAND_TRIES == 2
    # And a try is a way on only where it ends out of the spot: more than 64 pixels,
    # either way and up or down, from where the go came to stand (new ground alone
    # will not do: a small hop counts some).
    (enough,) = asked.enough
    assert callable(enough) and r.watch is not None
    here = r.watch.spot

    def out(dx: int = 0, dy: int = 0) -> bool:
        return bool(enough(Outcome(30, False, 9, spot=(here[0] + dx, here[1] + dy))))

    for reach in (
        [dx for dx in range(-200, 201) if not out(dx=dx)],
        [dy for dy in range(-200, 201) if not out(dy=dy)],
    ):
        assert reach == list(range(reach[0], reach[0] + 2 * STAND_WITHIN + 1)) and 0 in reach
    assert out(dx=200) and out(dy=-200) and STAND_WITHIN == 64
    assert not enough(Outcome(30, False, 9))  # a try that was not watched: no saying
    since = moments.since(kept[0].step)
    assert since is not None and since[0] is kept[0] and since[1][: len(trace)] == trace
    # Nothing found: the game is where it was, a search used, and the moments kept (no
    # life was lost, so they still lead to where the game is).
    assert (r.ways_sought, r.ways_found, r.ways_stood, r.times_back) == (1, 0, 1, 0)
    assert lab.saved[-1] == NOW_SLOT and lab.loaded == [NOW_SLOT]
    assert abs(lab.game.state.x - x) <= 1 and len(moments) == len(SLOTS)
    assert lab.game.state.lives == toy_hopper.START_LIVES
    said = [x.getMessage() for x in caplog.records if x.getMessage().startswith("Stayed about")]
    assert said == [
        f"Stayed about one spot for {STAND_AFTER} steps, with no life lost. Went back "
        f"{before - kept[0].step} steps and made 4 tries in 0.0 s: none got on."
    ]
    # However long it goes on standing there, it is not looked for again...
    hop_about(lab, r, FAR, 3 * STAND_AFTER)
    assert len(asked.calls) == 1
    # ...until it stands somewhere else.
    hop_about(lab, r, NEAR, STAND_AFTER + 30)
    assert len(asked.calls) == 2 and asked.calls[1][2]
    assert r.finish().describe_ways() == (
        ", went back 2 times (each time where it stood still) and found no way past"
    )


def test_a_go_that_got_nowhere_is_not_searched(lab: Lab, monkeypatch: pytest.MonkeyPatch) -> None:
    asked = Asked(monkeypatch)
    r, _ = standing(lab, ways_per_go=3, unstick_after=None)
    stand = lab.move()
    # From its first step it does nothing: a game that has not begun looks the same.
    lab.agent.script.extend([stand] * (2 * STAND_AFTER))
    r.begin()
    steps(r, 2 * STAND_AFTER)
    assert r.watch is not None and r.watch.ground.steps == 0
    assert asked.calls == [] and r.ways_sought == 0
    # Once it has got somewhere, standing is a standstill.
    lab.agent.script.extend([lab.move("right")] * 6 + [stand] * (2 * STAND_AFTER))
    steps(r, 6 + STAND_AFTER + 4)
    assert r.watch.ground.steps > 0
    assert len(asked.calls) == 1 and asked.calls[0][2] and r.ways_stood == 1


def test_a_standstill_is_searched_before_the_go_ends_stuck(
    lab: Lab, monkeypatch: pytest.MonkeyPatch
) -> None:
    stand, right = lab.move(), lab.move("right")
    on = Search(Way((right,) * 8, Change(0, right, 8), tries=2, ground=3), 2)
    asked = Asked(monkeypatch, [on])
    r, moments = standing(lab, ways_per_go=3)
    assert episode.DEFAULT_UNSTICK_AFTER < STAND_AFTER
    # It walks a little and then does nothing, and nothing on the screen changes: the
    # go is about to end "stuck", sooner than sixty steps.
    lab.agent.script.extend([right] * 6 + [stand] * 400)
    r.begin()
    while not asked.calls:
        assert r.step() and r.steps < STAND_AFTER
    # It looked first, and the way it was given is played: the go is not over.
    assert asked.calls[0][2] and (r.ways_sought, r.ways_found, r.times_back) == (1, 1, 1)
    x = lab.game.state.x
    steps(r, 7)
    assert lab.game.state.x > x and r.last_action == right
    # Standing again where the way left it, it looks again; with nothing found the go
    # ends stuck as it always did.
    while r.step():
        pass
    assert len(asked.calls) == 2 and asked.calls[1][2]
    result = r.finish()
    assert result.ended_by == "stuck" and (result.ways_sought, result.ways_found) == (2, 1)
    assert len(moments) > 0


def test_three_with_nothing_found_and_it_stops(
    lab: Lab, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    stand = lab.move()
    on = Search(Way((stand, stand), Change(0, stand, 2), tries=2, ground=3), 2)
    asked = Asked(monkeypatch, [NOTHING, NOTHING, on, NOTHING, NOTHING, NOTHING])
    r, _ = standing(lab, ways_per_go=9, max_steps=5000, unstick_after=None, give_up_after=3)
    r.begin()
    with caplog.at_level(logging.INFO, logger="retro_ai_lab.training.episode"):
        # Two for nothing, then one that finds a way: the count starts again.
        for count, spot in enumerate((FAR, NEAR, FAR, NEAR, FAR), start=1):
            hop_about(lab, r, spot)
            assert len(asked.calls) == count and not r.stood_gave_up
        assert (r.ways_sought, r.ways_found, r.ways_stood) == (5, 1, 5)
        # The third for nothing in a row is the last.
        hop_about(lab, r, NEAR)
        assert len(asked.calls) == 6 and r.stood_gave_up and not r.gave_up
        hop_about(lab, r, FAR)
        hop_about(lab, r, NEAR)
        assert len(asked.calls) == 6
        r.finish()
        # In its next go too.
        lab.game.reset()
        r.begin()
        hop_about(lab, r, FAR)
        assert len(asked.calls) == 6 and r.stood_gave_up
        assert r.finish().ways_sought == 0
    assert [x.getMessage() for x in caplog.records if "stops doing that" in x.getMessage()] == [
        STOPPED
    ]
    # Unless it is told when, it never stops (the training service says after eight,
    # as for the searches after a life lost).
    asked = Asked(monkeypatch)
    r, _ = standing(lab, ways_per_go=9, max_steps=5000, unstick_after=None)
    r.begin()
    for spot in (FAR, NEAR, FAR, NEAR):
        hop_about(lab, r, spot)
    assert len(asked.calls) == 4 and not r.stood_gave_up


def test_nothing_is_searched_without_what_it_needs(
    lab: Lab, monkeypatch: pytest.MonkeyPatch
) -> None:
    asked = Asked(monkeypatch)
    lab.teach()

    def stands(r: EpisodeRunner) -> EpisodeResult:
        lab.game.reset()
        lab.saved.clear()
        lab.loaded.clear()
        r.begin()
        hop_about(lab, r, FAR, 2 * STAND_AFTER)
        assert asked.calls == [] and NOW_SLOT not in lab.saved and lab.loaded == []
        return r.finish()

    # Going back switched off; no saved moments; no screen watch.
    r, _ = standing(lab, ways_per_go=0, unstick_after=None)
    plain = stands(r)
    assert (plain.ways_sought, plain.ways_stood, plain.describe_ways()) == (0, 0, "")
    lab.game.reset()
    watch = lab.watch()
    stands(lab.runner(watch=watch, ways_per_go=3, unstick_after=None))
    stands(lab.runner(moments=Moments(SLOTS), ways_per_go=3, unstick_after=None))
    # A watch that cannot follow the camera has no place for it to stand at.
    monkeypatch.setattr(ScreenWatch, "follows_camera", property(lambda self: False))
    r, _ = standing(lab, ways_per_go=3, unstick_after=None)
    stands(r)
    monkeypatch.undo()
    # With all of it, the same go is searched.
    asked = Asked(monkeypatch)
    r, _ = standing(lab, ways_per_go=3, unstick_after=None)
    lab.saved.clear()
    r.begin()
    hop_about(lab, r, FAR, 2 * STAND_AFTER)
    assert len(asked.calls) == 1


def test_the_record_counts_them_and_is_kept() -> None:
    def ways(sought: int, found: int, stood: int) -> str:
        return EpisodeResult(
            9, 36, 0.0, "max_steps", 3, ways_sought=sought, ways_found=found, ways_stood=stood
        ).describe_ways()

    assert ways(1, 1, 1) == ", went back once (where it stood still) and found a way past"
    assert ways(1, 0, 1) == ", went back once (where it stood still) and found no way past"
    assert ways(3, 3, 1) == (
        ", went back 3 times (once where it stood still) and found a way past each time"
    )
    assert ways(3, 1, 2) == (
        ", went back 3 times (2 times where it stood still) and found a way past once"
    )
    assert ways(2, 2, 2) == (
        ", went back 2 times (each time where it stood still) and found a way past each time"
    )
    # None at a standstill: the words are what they were.
    assert ways(3, 2, 0) == ", went back 3 times and found a way past 2 times"
    result = EpisodeResult(9, 36, 0.0, "max_steps", 3, ways_sought=3, ways_found=2, ways_stood=1)
    assert EpisodeResult.from_dict(result.to_dict()) == result
    # A record saved before this was counted reads as none.
    old = result.to_dict()
    del old["ways_stood"]
    assert EpisodeResult.from_dict(old).ways_stood == 0


# ------------------------------------------------------------ a way found is played out


def test_a_way_found_is_played_to_its_end_before_the_go_ends_for_its_length(lab: Lab) -> None:
    # A go four steps longer than it takes to walk into the first gap. It goes back
    # and finds the way across, which is longer than what is left of the go.
    r, route, _ = recalling(lab, max_steps=DIED_AT + 4)
    r.begin()
    steps(r, DIED_AT)
    assert (r.ways_sought, r.ways_found) == (1, 1) and r.own_steps == DIED_AT
    played = DIED_AT
    while r.step():
        played += 1
        assert played < 200
    played += 1  # the step that ended it was played too
    result = r.finish()
    # It ended for its length, but only once the way was played: every move of it,
    # the steps it went back over again and the carry-on beyond the gap.
    assert result.ended_by == "max_steps" and result.steps == played == 2 * DIED_AT + CARRY_ON
    assert lab.game.state.x > GAP_ENDS and lab.game.state.lives == toy_hopper.START_LIVES
    # So the way is remembered, and the next go, as short, is across from memory with
    # its search unused and its own steps still to play.
    assert len(route.moves) > 23
    r.begin()
    while lab.game.state.x <= GAP_ENDS:
        assert r.step()
    assert (r.ways_sought, r.own_steps) == (0, 0)
    assert lab.game.state.lives == toy_hopper.START_LIVES
    r.finish()
    # A go that finds no way ends at its length as it always did.
    r, _ = standing(lab, way_tries=3, max_steps=DIED_AT + 4)
    r.begin()
    while r.step():
        pass
    result = r.finish()
    assert (result.ended_by, result.steps, result.ways_found) == ("max_steps", DIED_AT + 4, 0)
