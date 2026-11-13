"""T-1019: the runner and the service ask that a way found can be lived on from."""

from __future__ import annotations

import logging

import pytest

from retro_ai_lab.core import ConfigurationError
from retro_ai_lab.emulator.fake import toy_hopper
from retro_ai_lab.training import episode, service, trial
from retro_ai_lab.training.episode import NOW_SLOT
from retro_ai_lab.training.moments import Moments
from retro_ai_lab.training.search import CUT_BY, END_SLOT, OUTLAST, Search, Way
from retro_ai_lab.training.ways import Change
from tests.unit.training import test_going_back as going_back
from tests.unit.training import test_going_back_in_training as in_training
from tests.unit.training.test_going_back import DIED_AT, SLOTS, Lab, steps
from tests.unit.training.test_going_back_in_training import NO_SAVES, walking_right
from tests.unit.training.test_going_back_in_training import Lab as Services
from tests.unit.training.test_search_at_a_loss import GAP_ENDS, searching

lab = going_back.lab  # the practice game behind a runner (pytest fixture)
services = in_training.lab  # and behind the real services

CUT = "The last %d step%s of that try %s left out: with them it ran on into the next danger, "
CUT += "and where it now stops the AI has room to save itself."
DOWN = "%d of the tries got past and could not be lived on from: the life went just after %s "
DOWN += "whatever was pressed, so %s not taken for a way."


def test_the_runner_passes_it_and_says_what_was_cut(
    lab: Lab, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    walk, jump = lab.move("right"), lab.move("a", "right")
    # On the game itself: walking right into the first gap, the way across is the jump
    # T-1007 found, and it used to carry on for 20 steps, into the pause of the next
    # life lost. It now stops 12 steps sooner, where the AI has room.
    r, moments = searching(lab, way_outlast=OUTLAST)
    assert r.way_outlast == OUTLAST == 30
    r.begin()
    steps(r, DIED_AT - 1)
    start = moments.kept[0]
    kept, lab.saved[:] = list(lab.saved), []  # the moments the go has saved so far
    assert set(kept) <= set(SLOTS) and lab.loaded == []
    with caplog.at_level(logging.INFO, logger="retro_ai_lab.training"):
        assert r.step()
    assert (r.ways_sought, r.ways_found, r.times_back) == (1, 1, 1)
    # The ends of tries were saved in their own slot and asked about from there; the
    # tries themselves and the way are from the moment gone back to, as before.
    assert set(lab.saved) == {NOW_SLOT, END_SLOT} and lab.saved[0] == NOW_SLOT
    assert set(lab.loaded) == {start.slot, END_SLOT} and lab.loaded[-1] == start.slot
    asked = lab.agent.asked
    played = []
    while lab.agent.asked == asked:
        assert r.step()
        played.append(r.last_action)
    played.pop()  # the step the learner was asked again
    assert played == [walk] * 19 + [jump] * 4 + [walk] * (DIED_AT - 23) + [walk] * 8
    assert lab.game.state.lives == toy_hopper.START_LIVES and lab.game.state.x > GAP_ENDS
    said = [x.getMessage() for x in caplog.records]
    assert said.count(CUT % (12, "s", "were")) == 1 and not any("could not be" in x for x in said)
    assert 12 % CUT_BY == 0
    # What the search says is what the go says and plays: scripted, a way cut by one
    # step with one try turned down, and the same in the plural.
    for cut, doomed, lines in (
        (1, 1, [DOWN % (1, "it", "it was"), CUT % (1, "", "was")]),
        (3, 2, [DOWN % (2, "them", "they were"), CUT % (3, "s", "were")]),
        (0, 0, []),
    ):
        asked_for: list[object] = []
        moves = (jump,) * 5

        def scripted(*args: object, **kw: object) -> Search:
            asked_for.append(kw["outlast"])  # noqa: B023
            way = Way(moves, Change(0, jump, 5), tries=9, ground=3, cut=cut)  # noqa: B023
            return Search(way, 9, doomed=doomed)  # noqa: B023

        monkeypatch.setattr(episode, "find_way", scripted)
        caplog.clear()
        r, _ = searching(lab, way_outlast=7)
        r.begin()
        steps(r, DIED_AT - 1)
        with caplog.at_level(logging.INFO, logger="retro_ai_lab.training"):
            assert r.step()
        assert asked_for == [7]
        said = [x.getMessage() for x in caplog.records]
        assert [x for x in said if "left out" in x or "could not be" in x] == lines
        asked = lab.agent.asked
        steps(r, 5)
        assert lab.agent.asked == asked  # the five moves of the way, and no more
        assert r.step() and lab.agent.asked == asked + 1
    # Unless told, the runner asks nothing of a way; nonsense is refused, and so are
    # saved moments that would use the slot the ends of tries are kept in.
    r, _ = searching(lab)
    assert r.way_outlast == 0
    r.begin()
    steps(r, DIED_AT)
    assert asked_for == [7, 0]
    for wrong in (-1, 1.5, True, None, "30"):
        with pytest.raises(ConfigurationError, match="way_outlast"):
            searching(lab, way_outlast=wrong)
    with pytest.raises(ConfigurationError, match=END_SLOT):
        lab.runner(moments=Moments((*SLOTS, END_SLOT)))


def test_the_service_asks_for_it_where_the_go_goes_back(
    services: Services, monkeypatch: pytest.MonkeyPatch
) -> None:
    for kw, connect, wanted in (
        ({}, {}, OUTLAST),
        ({"ways_per_go": 0}, {}, 0),
        ({}, {"capabilities": NO_SAVES}, 0),
    ):
        t = services.training(**kw)
        services.connect(**connect)
        walking_right(t, monkeypatch)
        assert t.step()
        go = t.episode
        assert go is not None and go.way_outlast == wanted
        t.stop()
        services.connection.disconnect()


@pytest.mark.slow
def test_a_way_found_can_be_lived_on_from(
    services: Services, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Three goes of a brain that only walks right, behind the real services, with and
    without the question. After each way found, every move is held for 30 steps on the
    game itself: how many keep the life?"""
    real = episode.find_way

    def run(outlast: int) -> tuple[list[tuple[int, int]], list[int]]:
        monkeypatch.setattr(service, "OUTLAST", outlast)
        ways: list[tuple[int, int]] = []  # how many moves keep the life, and the steps cut

        def measured(connection, slot, trace, space, **kw):  # type: ignore[no-untyped-def]
            assert kw["outlast"] == outlast
            out = real(connection, slot, trace, space, **kw)
            if out.way is not None:
                keep = 0
                for move in range(len(space)):
                    after = trial.play(
                        connection,
                        slot,
                        (*out.way.moves, *([move] * 30)),
                        space,
                        frames_per_step=kw["frames_per_step"],
                        lives=kw["lives"],
                    )
                    keep += not after.lost
                ways.append((keep, out.way.cut))
            return out

        monkeypatch.setattr(episode, "find_way", measured)
        t = services.training()
        game = services.connect()
        walking_right(t, monkeypatch)
        far = []
        for go in range(3):
            furthest = 0
            while t.episodes == go:
                assert t.step()
                furthest = max(furthest, game.state.x)
            far.append(furthest)
        t.stop()
        services.connection.disconnect()
        return ways, far

    # As it was: most of the ways found leave no move that keeps the life.
    before, far_before = run(0)
    assert len(before) >= 5 and all(cut == 0 for _, cut in before)
    assert sum(1 for keep, _ in before if keep == 0) > len(before) // 2
    # With the question: after every way found, at least half the fifteen moves keep
    # it; most ways were cut to get there, each by a few steps.
    with caplog.at_level(logging.INFO, logger="retro_ai_lab.training"):
        after, far_after = run(OUTLAST)
    assert len(after) >= 5 and all(keep >= 8 for keep, _ in after), after
    cuts = [cut for _, cut in after if cut]
    assert len(cuts) > len(after) // 2 and all(cut % CUT_BY == 0 and cut <= 20 for cut in cuts)
    said = [x.getMessage() for x in caplog.records if "left out" in x.getMessage()]
    assert len(said) == len(cuts)
    # It gets about as far in each go: what was left off was only the running on.
    assert all(abs(a - b) <= 64 for a, b in zip(far_after, far_before, strict=True)), (
        far_before,
        far_after,
    )
