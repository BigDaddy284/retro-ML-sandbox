"""T-1073: the runner has the log say what a search for nothing came to.

After the line it says of a search after a life lost or seen going that found no way,
the line of :func:`retro_ai_lab.training.search.tell`. Not of a way found, and not of a
search at a standstill, where no life is lost and the line would not fit.
"""

from __future__ import annotations

import logging

import pytest

from retro_ai_lab.training.search import Search, Tally, Way
from retro_ai_lab.training.ways import Change
from tests.unit.training import test_going_back as going_back
from tests.unit.training.test_going_back import DIED_AT, Lab, steps
from tests.unit.training.test_going_back_known import REAL, Asked
from tests.unit.training.test_looking_ahead import THING, Near, looking
from tests.unit.training.test_search_at_a_loss import searching
from tests.unit.training.test_standing_still import FAR, hop_about, standing

lab = going_back.lab  # the practice game behind the real services (pytest fixture)

EPISODE = "retro_ai_lab.training.episode"
SEARCH = "retro_ai_lab.training.search"
TALLY = Tally(sooner=3, later=1, latest=6, nowhere=1, ground=1, unseen=0)
SAYS = (
    "The 5 changes it played came to this: 3 lost the life no later than it went, 1 lost "
    "it later (the latest 6 steps later), 1 kept it and got nowhere new (at most 1 step of "
    "new ground, where 2 is getting past), and 0 kept it and were not looked at, the "
    "tries or the time being up."
)


def lines(caplog: pytest.LogCaptureFixture) -> list[tuple[str, str]]:
    return [(rec.name, rec.getMessage()) for rec in caplog.records if rec.name in (EPISODE, SEARCH)]


@pytest.fixture
def heard(caplog: pytest.LogCaptureFixture) -> pytest.LogCaptureFixture:
    caplog.set_level(logging.INFO, logger=EPISODE)
    caplog.set_level(logging.INFO, logger=SEARCH)
    return caplog


def test_after_a_life_lost_the_line_follows_the_search_for_nothing(
    lab: Lab, monkeypatch: pytest.MonkeyPatch, heard: pytest.LogCaptureFixture
) -> None:
    Asked(monkeypatch, [Search(None, 6, tally=TALLY)])
    r, _ = searching(lab, ways_per_go=1, give_up_after=None, max_steps=400)
    lab.game.reset()
    r.begin()
    steps(r, DIED_AT)
    said = lines(heard)
    told = [n for n, (_, line) in enumerate(said) if line.endswith("none got past.")]
    assert len(told) == 1
    assert said[told[0] + 1] == (SEARCH, SAYS)
    assert [line for name, line in said if name == SEARCH] == [SAYS]
    r.finish()


def test_the_real_search_for_nothing_says_what_it_played(
    lab: Lab, monkeypatch: pytest.MonkeyPatch, heard: pytest.LogCaptureFixture
) -> None:
    asked = Asked(monkeypatch, [REAL])
    r, _ = searching(lab, ways_per_go=1, give_up_after=None, max_steps=400, way_tries=3)
    lab.game.reset()
    r.begin()
    steps(r, DIED_AT)
    out = asked.out[0]
    assert out.way is None and out.tally is not None and out.tally.changes >= 2
    said = [line for name, line in lines(heard) if name == SEARCH]
    assert len(said) == 1
    assert said[0].startswith(f"The {out.tally.changes} changes it played came to this: ")
    r.finish()


def test_a_life_seen_going_with_no_way_past_says_it_too(
    lab: Lab, monkeypatch: pytest.MonkeyPatch, heard: pytest.LogCaptureFixture
) -> None:
    Asked(monkeypatch, [Search(None, 6, tally=TALLY)])
    r, _ = looking(lab)
    near = Near(monkeypatch, r.watch)
    r.begin()
    steps(r, 5)
    near.now = THING
    assert r.step()
    said = lines(heard)
    told = [n for n, (_, line) in enumerate(said) if "none got past, so it carries on" in line]
    assert len(told) == 1 and said[told[0] + 1] == (SEARCH, SAYS)
    r.finish()


def test_a_part_of_the_way_kept_says_it_too(
    lab: Lab, monkeypatch: pytest.MonkeyPatch, heard: pytest.LogCaptureFixture
) -> None:
    walk, jump = lab.move("right"), lab.move("a", "right")
    part = Way((walk,) * 10, Change(2, jump, 4), 3, 4)
    Asked(monkeypatch, [Search(None, 6, part=part, tally=TALLY), Search(None, 5)])
    r, _ = searching(lab, ways_per_go=2, give_up_after=None, max_steps=400)
    lab.game.reset()
    r.begin()
    steps(r, DIED_AT)
    said = lines(heard)
    told = [n for n, (_, line) in enumerate(said) if "so the AI keeps it" in line]
    assert len(told) == 1 and said[told[0] + 1] == (SEARCH, SAYS)
    r.finish()


def test_nothing_is_said_of_a_way_found_or_at_a_standstill(
    lab: Lab, monkeypatch: pytest.MonkeyPatch, heard: pytest.LogCaptureFixture
) -> None:
    # A way found over the gap, with a tally given all the same: nothing.
    walk = lab.move("right")
    asked = Asked(monkeypatch, [REAL])
    r, _ = searching(lab, ways_per_go=1, give_up_after=None, max_steps=400)
    lab.game.reset()
    r.begin()
    steps(r, DIED_AT)
    assert asked.out[0].way is not None
    r.finish()
    # Standing about one spot: a search for nothing there says nothing of its changes.
    asked = Asked(monkeypatch, [Search(None, 6, tally=TALLY)])
    r, _ = standing(lab, ways_per_go=9, max_steps=5000, unstick_after=None, give_up_after=None)
    r.begin()
    hop_about(lab, r, FAR)
    assert asked.stood == [True]
    assert any(line.endswith("none got on.") for _, line in lines(heard))
    r.finish()
    assert [line for name, line in lines(heard) if name == SEARCH] == []
    assert walk == lab.move("right")
