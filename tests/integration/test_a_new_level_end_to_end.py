"""T-1041: a new level is new ground, from end to end on the practice game.

T-1038's rule (a cut with no life lost, to a screen that shows nowhere the go has
been, is a new place and its ground pays from its door) was tested with scripted
screens. Here the real services play a practice cartridge of two levels (T-1040)
through its flags, and the watch is asked what it made of them.
"""

from __future__ import annotations

import logging

import pytest

from retro_ai_lab.emulator.fake import ToyHopper
from retro_ai_lab.emulator.fake import toy_hopper as th
from retro_ai_lab.observation.ground import Ground
from retro_ai_lab.training import EpisodeRunner
from retro_ai_lab.training.moments import Moments
from tests.unit.training import test_going_back as going_back
from tests.unit.training.test_going_back import FRAMES, SLOTS, Lab

lab = going_back.lab  # the practice game behind the real services (pytest fixture)

NEW_PLACE = (
    "The screen changed altogether with no life lost, and it shows nowhere this go has "
    "been: the AI takes it for a new place (number 2 of this go). Ground there is new "
    "from where it came in."
)
SETTLED = 4  # steps after a cut by which the watch has settled where it is
HOLD = 40 // FRAMES  # steps a jump is held for


class Knowing:
    """A player who knows the level the game is on, a step at a time: runs right and
    holds a jump at each thing in the way."""

    def __init__(self) -> None:
        self.hold = 0

    def press(self, game: ToyHopper) -> tuple[str, ...]:
        st = game.state
        front = st.x + th.PLAYER_W
        edges = [col * th.TILE for col, _ in game.gaps] + [col * th.TILE for col in game.walls]
        ahead = [game.walker_at(i) for i in range(len(game.walkers))]
        jump = any(0 <= edge - front <= 8 for edge in edges)
        jump = jump or any(0 <= wx - front <= 16 for wx in ahead)
        if st.on_ground and self.hold == 0 and jump:
            self.hold = HOLD
        if self.hold and st.on_ground and self.hold < HOLD - 1:
            self.hold = 0  # landed: let go, so the next press is a fresh one
        held = ("a", "right") if self.hold else ("right",)
        self.hold = max(0, self.hold - 1)
        return held


class Play:
    """A go on a cartridge of two levels, with a watch and no going back."""

    def __init__(self, lab: Lab) -> None:
        self.lab = lab
        lab.game.levels = 2
        lab.teach()
        lab.game.reset()
        watch = lab.watch()
        self.runner: EpisodeRunner = lab.runner(
            moments=Moments(SLOTS), watch=watch, ways_per_go=0, max_steps=4000
        )
        self.watch = watch
        self.player = Knowing()
        self.runner.begin()

    @property
    def ground(self) -> Ground:
        return self.watch.ground

    @property
    def where(self) -> tuple[int, int]:
        """The place the watch is in, and how many it has made this go."""
        return self.ground.place_number, self.ground.places

    def step(self, *held: str) -> None:
        """One step: as one who knows the level, or with the buttons given."""
        lab = self.lab
        lab.agent.script.clear()
        lab.agent.script.append(lab.move(*(held or self.player.press(lab.game))))
        assert self.runner.step()

    def steps(self, count: int) -> None:
        for _ in range(count):
            self.step()

    def to_the_flag(self) -> None:
        level = self.lab.game.state.level
        for _ in range(800):
            self.step()
            if self.lab.game.state.level != level:
                return
        raise AssertionError("the player who knows the level never passed its flag")

    def until(self, x: int) -> None:
        for _ in range(800):
            if self.lab.game.state.x >= x:
                return
            self.step()
        raise AssertionError(f"the player who knows the level never got to {x}")


def said(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [rec.getMessage() for rec in caplog.records if "new place" in rec.getMessage()]


def test_the_second_level_is_a_new_place_and_the_first_is_known_again(
    lab: Lab, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.INFO, logger="retro_ai_lab.training"):
        play = Play(lab)
        game, ground, r = lab.game, play.ground, play.runner
        # The first level, to its flag: one place, and its ground counted.
        play.to_the_flag()
        assert (game.state.level, game.state.lives) == (1, th.START_LIVES)
        first, far = ground.steps, ground.reach
        assert play.where == (0, 1) and first > 130 and far[0] > 2200 and said(caplog) == []
        # The second level: a new place, said once, with nothing counted for the cut
        # and nothing reached there yet.
        play.steps(SETTLED)
        assert play.where == (1, 2)
        assert ground.steps == first and ground.reach == (0, 0, 0, 0)
        assert said(caplog) == [NEW_PLACE]
        # Its ground is new, and the go is paid for it.
        paid = r.reward
        play.to_the_flag()
        assert (game.state.level, game.state.lives) == (2, th.START_LIVES)
        second = ground.steps - first
        assert second > 100 and ground.reach[0] > 2000 and play.where == (1, 2)
        assert r.reward >= paid + second
        # The first level comes round: the place it began in, known again by its
        # screen, and no ground there counts a second time.
        play.steps(SETTLED)
        assert play.where == (0, 2) and ground.reach == far
        play.steps(120)
        assert game.state.x > 800 and game.state.lives == th.START_LIVES
        assert ground.steps == first + second and ground.reach == far and play.where == (0, 2)
    assert said(caplog) == [NEW_PLACE]


def test_a_life_lost_in_the_second_level_is_the_same_place(
    lab: Lab, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.INFO, logger="retro_ai_lab.training"):
        play = Play(lab)
        game, ground = lab.game, play.ground
        play.to_the_flag()
        play.steps(SETTLED)
        assert play.where == (1, 2)
        # Over what comes first as one who knows it, and then walking on into a gap.
        gap = game.gaps[0][0] * th.TILE
        play.until(gap - 6 * th.TILE)
        while game.state.mode == th.PLAYING:
            play.step("right")
        fell = game.state.x
        assert gap <= fell < gap + 4 * th.TILE
        while game.state.lives == th.START_LIVES:
            play.step("right")
        # The game puts the player back at the start of that level: the same place.
        assert (game.state.level, game.state.lives) == (1, th.START_LIVES - 1)
        assert game.state.x < th.START_X + 16
        counted, reached = ground.steps, ground.reach
        assert reached[0] > 100
        play.player = Knowing()
        play.steps(SETTLED)
        assert play.where == (1, 2) and ground.reach == reached
        # Back to as far as it had counted counts nothing (the game's camera says how
        # far that is), and on from there counts again.
        walked = 0
        while game.state.scroll + th.TILE < reached[0]:
            play.step()
            walked += 1
            assert ground.steps == counted and ground.reach == reached
        assert walked > 20
        play.until(gap + 400)
        assert game.state.lives == th.START_LIVES - 1 and play.where == (1, 2)
        assert ground.steps > counted + 10 and ground.reach[0] > reached[0] + 160
    assert said(caplog) == [NEW_PLACE]
