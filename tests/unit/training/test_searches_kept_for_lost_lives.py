"""T-1070: a search where the go stands still is not one of its searches after a life lost.

A go may go back and look for a way so many times (three, as the app sets it). The
searches made where it stood still (T-1018) took from the same count, and late in a
long go they find nothing: the user's log of 25 Nov has a go that spent all three
that way and then lost three lives with no search made for any of them, and the
game was over. A search at a standstill now takes from as many of its own, as a look
ahead does (T-1052), and the go's searches wait for the lives.
"""

from __future__ import annotations

import pytest

from retro_ai_lab.emulator.fake import toy_hopper
from retro_ai_lab.training import EpisodeRunner
from retro_ai_lab.training.episode import STAND_AFTER
from retro_ai_lab.training.search import Search, Way
from retro_ai_lab.training.ways import Change
from tests.unit.training import test_going_back as going_back
from tests.unit.training.test_going_back import Lab, steps
from tests.unit.training.test_standing_still import FAR, NEAR, Asked, hop_about, standing

lab = going_back.lab  # the practice game behind the real services (pytest fixture)

ELSEWHERE = 90  # a third spot on the first stretch, more than 64 pixels from the other two


def kinds(asked: Asked) -> str:
    """The searches asked for, in order: "s" where the go stood still, "l" for a life lost."""
    return "".join("s" if stood else "l" for _, _, stood in asked.calls)


def lose_a_life(lab: Lab, r: EpisodeRunner) -> None:
    """Walk right into the gap."""
    lives = lab.game.state.lives
    right = lab.move("right")
    while lab.game.state.lives == lives:
        lab.agent.script.clear()
        lab.agent.script.append(right)
        assert r.step()
    lab.agent.script.clear()


def test_searches_at_a_standstill_leave_the_gos_searches_for_the_lives(
    lab: Lab, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The user's go of 25 Nov: three standstills searched for nothing, then the lives."""
    assert (FAR - ELSEWHERE > 64) and (ELSEWHERE - NEAR > 64)
    asked = Asked(monkeypatch)  # no search finds anything
    r, _ = standing(lab, ways_per_go=2, max_steps=5000, unstick_after=None)
    r.begin()
    for count, spot in enumerate((FAR, NEAR), start=1):
        hop_about(lab, r, spot)
        assert kinds(asked) == "s" * count
    assert (r.ways_sought, r.ways_stood, r.ways_found) == (2, 2, 0)
    # Both of the go's searches are still there for a life lost.
    assert toy_hopper.START_LIVES >= 3
    lose_a_life(lab, r)
    assert kinds(asked) == "ssl" and (r.ways_sought, r.ways_stood) == (3, 2)
    lose_a_life(lab, r)
    assert kinds(asked) == "ssll" and (r.ways_sought, r.ways_stood) == (4, 2)
    result = r.finish()
    assert (result.setbacks, result.ways_sought, result.ways_stood) == (2, 4, 2)
    assert result.ways_found == 0


def test_a_go_searches_only_so_many_standstills(lab: Lab, monkeypatch: pytest.MonkeyPatch) -> None:
    """As many as it may search lives lost, and counted apart."""
    asked = Asked(monkeypatch)
    r, moments = standing(lab, ways_per_go=2, max_steps=5000, unstick_after=None)
    r.begin()
    hop_about(lab, r, FAR)
    hop_about(lab, r, NEAR)
    assert kinds(asked) == "ss"
    # A third spot it stands at is not searched: the two are used.
    saves = len(lab.saved)
    hop_about(lab, r, ELSEWHERE)
    hop_about(lab, r, FAR)
    assert kinds(asked) == "ss" and (r.ways_sought, r.ways_stood) == (2, 2)
    # The moments are still saved all the while: a life lost has its searches yet.
    assert len(lab.saved) > saves and len(moments) > 0
    lose_a_life(lab, r)
    assert kinds(asked) == "ssl"
    r.finish()
    # The next go has both again.
    lab.game.reset()
    r.begin()
    hop_about(lab, r, FAR)
    hop_about(lab, r, NEAR)
    hop_about(lab, r, ELSEWHERE)
    assert kinds(asked) == "ssl" + "ss"
    lose_a_life(lab, r)
    lose_a_life(lab, r)
    assert kinds(asked) == "ssl" + "ssll"
    assert r.finish().ways_stood == 2


def test_with_one_search_a_go_a_standstill_does_not_take_it(
    lab: Lab, monkeypatch: pytest.MonkeyPatch
) -> None:
    asked = Asked(monkeypatch)
    r, _ = standing(lab, ways_per_go=1, max_steps=5000, unstick_after=None)
    r.begin()
    hop_about(lab, r, FAR)
    hop_about(lab, r, NEAR)
    assert kinds(asked) == "s"
    lose_a_life(lab, r)
    assert kinds(asked) == "sl" and (r.ways_sought, r.ways_stood) == (2, 1)


def test_a_way_on_found_at_a_standstill_takes_none_of_the_gos_searches(
    lab: Lab, monkeypatch: pytest.MonkeyPatch
) -> None:
    stand = lab.move()
    on = Search(Way((stand, stand), Change(0, stand, 2), tries=2, ground=3), 2)
    asked = Asked(monkeypatch, [on])
    r, _ = standing(lab, ways_per_go=1, max_steps=5000, unstick_after=None)
    r.begin()
    hop_about(lab, r, FAR)
    assert kinds(asked) == "s" and (r.ways_sought, r.ways_found, r.ways_stood) == (1, 1, 1)
    # It had one of its own, and that is used, found or not.
    hop_about(lab, r, NEAR)
    assert kinds(asked) == "s"
    lose_a_life(lab, r)
    assert kinds(asked) == "sl" and (r.ways_sought, r.ways_found) == (2, 1)


def test_a_go_with_no_search_left_for_a_life_searches_no_standstill(
    lab: Lab, monkeypatch: pytest.MonkeyPatch
) -> None:
    """As before: with its searches gone on lives lost nothing is saved to go back to
    (T-1007), so there is nothing to look from where it then stands."""
    asked = Asked(monkeypatch)
    r, moments = standing(lab, ways_per_go=1, max_steps=5000, unstick_after=None)
    r.begin()
    lose_a_life(lab, r)
    assert kinds(asked) == "l"
    saves = len(lab.saved)
    hop_about(lab, r, FAR)
    hop_about(lab, r, NEAR)
    assert kinds(asked) == "l" and (r.ways_sought, r.ways_stood) == (1, 0)
    assert len(lab.saved) == saves and len(moments) == 0
    lose_a_life(lab, r)
    assert kinds(asked) == "l"


def test_nor_where_the_moments_before_are_still_kept(
    lab: Lab, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The go's one search found a way, so the moment it went back to is still kept:
    there would be something to look from, and it does not look."""
    right, stand = lab.move("right"), lab.move()
    short = Search(Way((right,) * 6 + (stand,) * 2, Change(0, right, 6), tries=2, ground=3), 2)
    asked = Asked(monkeypatch, [short])
    r, moments = standing(lab, ways_per_go=1, max_steps=5000, unstick_after=None)
    r.begin()
    while not asked.calls:  # into the gap: it goes back to the start, the life kept
        lab.agent.script.clear()
        lab.agent.script.append(right)
        assert r.step()
    lab.agent.script.clear()
    assert kinds(asked) == "l" and (r.ways_sought, r.ways_found, r.times_back) == (1, 1, 1)
    steps(r, 8)  # the way it was given, played
    here = lab.game.state.x
    assert lab.game.state.lives == toy_hopper.START_LIVES and 32 < here < 192
    assert len(moments) > 0
    hop_about(lab, r, here, 2 * STAND_AFTER)
    assert abs(lab.game.state.x - here) <= 8 and len(moments) > 0
    assert lab.game.state.lives == toy_hopper.START_LIVES
    assert kinds(asked) == "l" and (r.ways_sought, r.ways_stood) == (1, 0)


def test_a_life_lost_first_leaves_the_standstills_theirs(
    lab: Lab, monkeypatch: pytest.MonkeyPatch
) -> None:
    asked = Asked(monkeypatch)
    r, _ = standing(lab, ways_per_go=2, max_steps=5000, unstick_after=None)
    r.begin()
    lose_a_life(lab, r)
    assert kinds(asked) == "l"
    hop_about(lab, r, FAR)
    hop_about(lab, r, NEAR)
    hop_about(lab, r, ELSEWHERE)
    assert kinds(asked) == "lss" and (r.ways_sought, r.ways_stood) == (3, 2)
    lose_a_life(lab, r)
    assert kinds(asked) == "lssl"


def test_a_go_that_may_not_search_searches_no_standstill(
    lab: Lab, monkeypatch: pytest.MonkeyPatch
) -> None:
    asked = Asked(monkeypatch)
    r, _ = standing(lab, ways_per_go=0, max_steps=5000, unstick_after=None)
    r.begin()
    hop_about(lab, r, FAR)
    lose_a_life(lab, r)
    hop_about(lab, r, NEAR)
    assert kinds(asked) == "" and r.ways_sought == 0
