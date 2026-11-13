"""T-1018: in training, a go that stands at a wall goes back and finds the way on."""

from __future__ import annotations

import logging
import re

import pytest

from retro_ai_lab.core import ConfigurationError
from retro_ai_lab.emulator.fake import toy_hopper
from retro_ai_lab.training.episode import PRESS_FOR, STAND_AFTER
from retro_ai_lab.training.moments import EVERY
from retro_ai_lab.training.service import PRESS_AFTER, WAYS_PER_GO
from tests.unit.training import test_going_back_in_training as in_training
from tests.unit.training.test_going_back_in_training import NO_SAVES, Lab, walking_right

lab = in_training.lab  # the practice game behind the real services (pytest fixture)

WALL = toy_hopper.TILE * 56  # the left face of the first wall of this level
AT_WALL = WALL - 12  # where the player stands when it is against it


def test_training_gets_over_a_wall_and_the_next_go_knows_the_way(
    lab: Lab, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    assert WAYS_PER_GO == 3 and PRESS_AFTER < STAND_AFTER
    t = lab.training()
    game = lab.connect()
    walking_right(t, monkeypatch)
    furthest: list[int] = []
    stood: list[int] = []  # steps each go spent against the wall
    lives: list[int] = []
    with caplog.at_level(logging.INFO, logger="retro_ai_lab.training"):
        for go in range(3):
            far = at_wall = 0
            while t.episodes == go:
                assert t.step()
                far = max(far, game.state.x)
                at_wall += game.state.x == AT_WALL
            furthest.append(far)
            stood.append(at_wall)
            lives.append(game.state.lives)
    t.stop()
    results = [entry.result for entry in t.history]
    # A brain that only walks right. The first go gets across the first gap by going
    # back after the lives lost there, and ends short of the wall.
    assert furthest[0] < AT_WALL and results[0].ways_stood == 0, furthest
    # The second comes to the wall and stands. Sixty steps there (a few more, with the
    # pressing on in between) and it looked for the way on, found it, and is over.
    assert stood[1] >= STAND_AFTER and results[1].ways_stood == 1
    assert furthest[1] > WALL + toy_hopper.TILE, furthest
    said = [r.getMessage() for r in caplog.records if r.getMessage().startswith("Stayed about")]
    assert len(said) == 1 and re.fullmatch(
        r"Stayed about one spot for (\d+) steps, with no life lost. Went back (\d+) steps and "
        r"found a way on at try \d+, after \d+\.\d s: holding right instead of what it did, "
        r"with right \+ a for \d+ steps from step \d+ of that\.",
        said[0],
    ), said
    stayed, back = (int(n) for n in re.findall(r"(\d+) steps[, ]", said[0])[:2])
    assert STAND_AFTER <= stayed <= STAND_AFTER + PRESS_FOR and stayed <= back < stayed + EVERY
    line = next(r.getMessage() for r in caplog.records if r.getMessage().startswith("Go 2 "))
    assert results[1].describe_ways() in line and " where it stood still) and found" in line
    # The way over is part of what it now remembers: the third go does not stand at
    # the wall (the steps counted are the jump rising against it) and looks for
    # nothing there, and it gets further.
    assert stood[2] < STAND_AFTER // 4 and results[2].ways_stood == 0, stood
    assert furthest[2] > furthest[1], (
        stood,
        furthest,
    )
    # Every life lost on the way was got back by going back.
    assert lives == [toy_hopper.START_LIVES] * 3
    assert all(r.ways_found == r.ways_sought <= WAYS_PER_GO for r in results)
    assert "stops doing that" not in caplog.text


SAYS = (
    f"It also goes back where it stays about one spot for {STAND_AFTER} steps without losing "
    "a life (at a wall none is lost), to look for a way on."
)


def test_the_service_switches_it_on_and_says_so(
    lab: Lab, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    for wrong in (-1, 1.5, True, "60", None):
        with pytest.raises(ConfigurationError, match="stand_after"):
            lab.training(stand_after=wrong)
    t = lab.training()
    lab.connect()
    with caplog.at_level(logging.INFO, logger="retro_ai_lab.training"):
        walking_right(t, monkeypatch)
        assert t.step()
        go = t.episode
        assert go is not None and go.stand_after == STAND_AFTER == 60
        while t.episodes < 2:
            assert t.step()
    said = [r.getMessage() for r in caplog.records]
    assert said.count(SAYS) == 1  # once, not once a go
    will = next(i for i, line in enumerate(said) if "goes back a little" in line)
    assert said.index(SAYS) == will + 2  # after the line on what it remembers
    t.stop()
    lab.connection.disconnect()
    # Told not to, or where it cannot go back at all: the runner never looks, and
    # nothing is said about it.
    for kw, connect in (
        ({"stand_after": 0}, {}),
        ({"ways_per_go": 0}, {}),
        ({}, {"capabilities": NO_SAVES}),
    ):
        caplog.clear()
        t = lab.training(**kw)
        lab.connect(**connect)
        with caplog.at_level(logging.INFO, logger="retro_ai_lab.training"):
            walking_right(t, monkeypatch)
            assert t.step()
            go = t.episode
            assert go is not None and go.stand_after == 0
        assert SAYS not in [r.getMessage() for r in caplog.records]
        t.stop()
        lab.connection.disconnect()
