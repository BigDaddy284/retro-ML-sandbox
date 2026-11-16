"""T-1049: a life lost in the second practice level, where the screen cannot show
the game putting the player back.

That level opens on bare ground with one wall in sight. Put back at its start after
a life lost some way in, less than one part in a hundred of the screen is other than
it was and the picture reads a short slide: no jolt (T-993). The watch's camera
carried on from where the life went, and the ground walked again was counted and
paid as new: 178 steps of ground on a level that has 144. With the game's own count
of its camera known (T-1047's go: the player hops on the spot first), the count
standing at what it was at the level's door tells it.
"""

from __future__ import annotations

import logging

import pytest

from retro_ai_lab.emulator.fake import toy_hopper as th
from tests.integration.test_a_new_level_end_to_end import SETTLED
from tests.integration.test_a_new_level_with_the_camera_known import A_STEP, END, Waiting
from tests.unit.training import test_going_back as going_back
from tests.unit.training.test_going_back import Lab
from tests.unit.training.test_vision_put_back_by_count import SAID

lab = going_back.lab  # the practice game behind the real services (pytest fixture)

IN = 60  # steps into the second level before the player stops looking where it goes
GROUND = END // 16  # steps of ground in a level: one for each 16 pixels, the player's size


def said(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [m for m in (rec.getMessage() for rec in caplog.records) if "put it back" in m]


def test_the_ground_walked_again_is_not_paid_again(
    lab: Lab, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.INFO, logger="retro_ai_lab.training"):
        play = Waiting(lab, 40)
        game, ground, r, watch = lab.game, play.ground, play.runner, play.watch
        play.to_the_flag()
        assert watch.camera_bytes == (("ram", th.LAYOUT.scroll), None)
        first = ground.steps
        play.steps(SETTLED + IN)
        assert play.where == (1, 2) and (game.state.level, game.state.lives) == (1, th.START_LIVES)
        # Holding right into what is in the way, until a life goes.
        before = 0, 0, 0
        while game.state.lives == th.START_LIVES:
            before = ground.reach[0], ground.steps, game.state.scroll
            play.step("right")
            assert r.steps < 2000
        reached, counted, scroll = before
        assert reached == scroll > 400 and said(caplog) == []
        # The game has put it back at the level's start. At the watch's next look it
        # is at the door of this place, what it reached there still reached.
        play.step()
        assert game.state.scroll == 0 and game.state.x <= th.START_X + A_STEP
        assert play.where == (1, 2) and ground.place == (0, 0)
        assert ground.reach == (reached, 0, 0, 0) and ground.steps == counted
        assert said(caplog) == [SAID]
        # Walking the same ground again: the watch is never ahead of the game's camera
        # and never more than a step behind it, and until the player is past where it
        # had got to, nothing is counted and nothing is paid.
        paid = r.reward
        again = furthest = 0
        level = game.state.level
        while game.state.level == level:
            furthest = max(furthest, game.state.scroll)
            assert 0 <= game.state.scroll - ground.place[0] <= A_STEP
            if game.state.scroll <= reached:
                again += 1
                assert (ground.reach[0], ground.steps, r.reward) == (reached, counted, paid)
            play.step()
            assert r.steps < 3000
        assert again > reached // A_STEP
        # At its flag: the camera's whole travel, and the level's ground counted once.
        assert (game.state.level, game.state.lives) == (2, th.START_LIVES - 1)
        assert furthest == END == ground.reach[0] and play.where == (1, 2)
        assert ground.steps - first == GROUND
    assert said(caplog) == [SAID]
