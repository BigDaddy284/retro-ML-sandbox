"""T-1026: the runner says so when the way it remembers is paid less than it was."""

from __future__ import annotations

import logging

import pytest

from tests.unit.training import test_going_back as going_back
from tests.unit.training.test_going_back import Lab, steps
from tests.unit.training.test_remembered_way import planted, remembering

lab = going_back.lab  # the practice game behind the real services (pytest fixture)

EPISODE = "retro_ai_lab.training.episode"


def said_at_the_end(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [rec.getMessage() for rec in caplog.records if rec.name == EPISODE]


def test_a_way_paid_less_is_shortened_and_the_log_says_so(
    lab: Lab, caplog: pytest.LogCaptureFixture
) -> None:
    r, route, harness = remembering(lab, ways_per_go=0)
    walk, stand = lab.move("right"), lab.move()
    # A first go, so that the watch knows this screen scrolls: from then on the same
    # steps are paid the same every go.
    lab.agent.script.extend([walk] * 10)
    r.begin()
    steps(r, 10)
    r.finish()
    assert 0 < len(route.moves) <= 10
    # Twelve steps to the right and ten of standing, remembered at a point a step:
    # the practice game pays for new ground only, so the standing pays nothing.
    planted(route, harness, [walk] * 12 + [stand] * 10)
    assert route.paid == 22
    lab.agent.script.extend([stand] * 2)
    r.begin()
    assert r.remembered == 22
    steps(r, 24)
    paid = route.counted
    assert 0 < paid < 22
    with caplog.at_level(logging.INFO, logger=EPISODE):
        r.finish()
    kept = len(route.moves)
    # What is left is walking, and it ends on the last step that was paid for.
    assert 0 < kept <= 12 and route.moves == (walk,) * kept
    assert (route.paid, route.shortened) == (paid, 22 - kept)
    assert said_at_the_end(caplog) == [
        "Played again, the way the AI remembers was paid less than the 22.0 points it was "
        "remembered for: something that paid for it then pays no longer. It now remembers "
        f"{kept} steps of it instead of 22, up to where it was paid the most this time "
        f"({paid:.1f} points)."
    ]
    # The next go begins with those steps and is paid the same for them: one step
    # fewer and it is paid less.
    r.begin()
    assert r.remembered == kept
    steps(r, kept - 1)
    assert route.counted < paid
    steps(r, 1)
    assert route.counted == paid
    caplog.clear()
    with caplog.at_level(logging.INFO, logger=EPISODE):
        r.finish()
    assert said_at_the_end(caplog) == [] and route.moves == (walk,) * kept


def test_a_way_paid_nothing_is_forgotten_and_the_log_says_so(
    lab: Lab, caplog: pytest.LogCaptureFixture
) -> None:
    r, route, harness = remembering(lab, ways_per_go=0)
    stand = lab.move()
    planted(route, harness, [stand] * 8)
    lab.agent.script.extend([stand] * 2)
    r.begin()
    steps(r, 10)
    assert route.counted == 0
    with caplog.at_level(logging.INFO, logger=EPISODE):
        r.finish()
    assert (route.moves, route.paid, route.shortened) == ((), 0.0, 8)
    assert said_at_the_end(caplog) == [
        "Played again, the way the AI remembers was paid nothing (it was remembered for "
        "8.0 points): what paid for it then pays no longer. The AI forgets that way, and "
        "its next go begins without one."
    ]
    r.begin()
    assert r.remembered == 0


def test_a_way_paid_as_before_is_left_alone_and_nothing_is_said(
    lab: Lab, caplog: pytest.LogCaptureFixture
) -> None:
    r, route, _ = remembering(lab, ways_per_go=0)
    walk = lab.move("right")
    lab.agent.script.extend([walk] * 14)
    r.begin()
    steps(r, 14)
    r.finish()
    way, paid = route.moves, route.paid
    assert 0 < len(way) <= 14 and paid > 0
    # Played again and stopped at its end, then played again with two steps more
    # that are paid nothing: the same way, the same figure, not a word.
    for more in (0, 2):
        lab.agent.script.extend([lab.move()] * more)
        r.begin()
        steps(r, len(way) + more)
        with caplog.at_level(logging.INFO, logger=EPISODE):
            r.finish()
        assert (route.moves, route.paid, route.shortened) == (way, paid, 0)
    assert said_at_the_end(caplog) == []
