"""T-1047: a new level's ground is measured from the camera's first move, once the
game's own count of its camera is known.

T-1041 found the camera's first moves in the second practice level unmeasured: it
opens on bare ground with one small thing in sight, where the picture cannot tell a
camera move, and its player ran from the first step, so the watch had not yet found
which byte of memory moves as the camera does (T-1020: that takes looks with the
camera at rest as well as moving). Here the player hops on the spot first.
"""

from __future__ import annotations

import logging

import pytest

from retro_ai_lab.emulator.fake import toy_hopper as th
from retro_ai_lab.training.moments import Moments
from tests.integration.test_a_new_level_end_to_end import NEW_PLACE, SETTLED, Knowing, Play
from tests.unit.training import test_going_back as going_back
from tests.unit.training.test_going_back import FRAMES, SLOTS, Lab

lab = going_back.lab  # the practice game behind the real services (pytest fixture)

FOUND = (
    f"The AI found where the game keeps count of its camera: ram[{th.LAYOUT.scroll}] moves "
    "as the screen slides across"
)
A_STEP = 8  # pixels the player runs, and the camera slides, in a step
END = th.LEVEL_WIDTH - th.SCREEN_WIDTH  # the furthest a level's camera goes


class Waiting(Play):
    """The same go by a player who hops on the spot before setting off. Its runner
    does not give a go up for standing about: that is not what is looked at here."""

    def __init__(self, lab: Lab, wait: int) -> None:
        self.lab = lab
        lab.game.levels = 2
        lab.teach()
        lab.game.reset()
        self.watch = lab.watch()
        self.runner = lab.runner(
            moments=Moments(SLOTS),
            watch=self.watch,
            ways_per_go=0,
            max_steps=4000,
            unstick_after=None,
        )
        self.player = Knowing()
        self.runner.begin()
        for n in range(wait):
            self.step("a" if n % 8 < 4 else "left")


def found(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [rec.getMessage() for rec in caplog.records if "keeps count" in rec.getMessage()]


@pytest.mark.parametrize("wait", [40, 60])
def test_the_second_levels_ground_is_measured_from_its_first_move(
    lab: Lab, caplog: pytest.LogCaptureFixture, wait: int
) -> None:
    assert FRAMES * 2 == A_STEP  # two pixels a frame
    with caplog.at_level(logging.INFO, logger="retro_ai_lab.training"):
        play = Waiting(lab, wait)
        game, ground, r, watch = lab.game, play.ground, play.runner, play.watch
        # Hopping, the camera has not moved: no byte can have shown itself yet.
        assert game.state.scroll == 0 and game.state.x <= th.START_X
        assert watch.camera_bytes == (None, None) and found(caplog) == []
        # By the first flag the watch knows the game's count: across, and only that.
        play.to_the_flag()
        assert (game.state.level, game.state.lives) == (1, th.START_LIVES)
        assert watch.camera_bytes == (("ram", th.LAYOUT.scroll), None)
        assert len(found(caplog)) == 1 and found(caplog)[0].startswith(FOUND)
        first = ground.steps
        # The second level is a new place, with nothing reached there yet.
        play.steps(SETTLED)
        assert play.where == (1, 2) and ground.reach == (0, 0, 0, 0) and ground.steps == first
        assert game.state.scroll == 0
        # Its first ground is bare, with one wall in sight: the picture alone reads
        # no move there. The watch is never ahead of the game's camera and never more
        # than a step behind it.
        behind = []
        for _ in range(30):
            play.step()
            behind.append(game.state.scroll - ground.reach[0])
        assert game.state.scroll > 20 * A_STEP
        assert min(behind) == 0 and max(behind) <= A_STEP
        # To its flag: the camera's whole travel, to the pixel, and the ground paid.
        paid = r.reward
        counted = ground.steps
        furthest = 0
        level = game.state.level
        while game.state.level == level:
            furthest = max(furthest, game.state.scroll)
            play.step()
            assert r.steps < 2000
        assert (game.state.level, game.state.lives) == (2, th.START_LIVES)
        assert furthest == END == ground.reach[0] and play.where == (1, 2)
        second = ground.steps - first
        assert second >= 140
        assert r.reward - paid >= ground.steps - counted > 100
    assert [m for m in (rec.getMessage() for rec in caplog.records) if "new place" in m] == [
        NEW_PLACE
    ]
