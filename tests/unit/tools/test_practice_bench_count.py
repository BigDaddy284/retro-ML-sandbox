"""T-1024: the practice bench's own account of a go goes back when the game does.

``_train`` is driven here by a scripted game and stand-ins for the training service
and its watch. The stand-in watch counts as a right one would: the camera's travel
since the go began, each level's ground once.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

from retro_ai_lab.emulator.fake import toy_hopper
from retro_ai_lab.tools.practice_bench import Report, _train

UNIT = 16  # the stand-in watch's step, as the practice game's player has it


@dataclass(frozen=True)
class Step:
    """The game after one step, and what a watch that counts right has reached."""

    level: int
    scroll: int
    counted: int
    lives: int = 3
    back: bool = False  # the go was put back to a saved moment in this step
    mode: int = toy_hopper.PLAYING


class Story:
    """Stands in for the training service: plays the scripted goes a step at a time."""

    def __init__(self, goes: list[list[Step]]) -> None:
        self._goes = goes
        self._at = 0
        self.episodes = 0
        self.history: list[Any] = []
        self.game = SimpleNamespace(state=self._state(Step(0, 0, 0)))
        ground = SimpleNamespace(reach=(0, 0, 0, 0), steps=0, unit=UNIT)
        tracker = SimpleNamespace(scroll=(0, 0), catalogue=SimpleNamespace(kinds=lambda: ()))
        self.watch = SimpleNamespace(
            looks=0,
            me=None,
            seen_me=0,
            around=None,
            ground=ground,
            tracker=tracker,
            lives_after_touch=0,
            lives_after_fall=0,
            roles=lambda: (),
        )
        self.episode = SimpleNamespace(times_back=0)

    @staticmethod
    def _state(step: Step) -> Any:
        # The player is held 32 pixels into the screen, so until the camera is first
        # seen to move its own travel is the camera's.
        return SimpleNamespace(
            x=32 + step.scroll,
            y=toy_hopper.GROUND_Y - toy_hopper.PLAYER_H,
            mode=step.mode,
            lives=step.lives,
            level=step.level,
            scroll=step.scroll,
        )

    def start(self, mode: object) -> None:
        pass

    def step(self) -> None:
        go = self._goes[self.episodes]
        if self._at == 0:
            self.episode = SimpleNamespace(times_back=0)  # a new go, a new runner
        step = go[self._at]
        self._at += 1
        self.game.state = self._state(step)
        self.episode.times_back += step.back
        self.watch.looks += 1
        self.watch.ground.reach = (step.counted, 0, 0, 0)
        self.watch.ground.steps = step.counted // UNIT
        if self._at == len(go):
            self._at = 0
            self.episodes += 1
            result = SimpleNamespace(ways_sought=0, ways_found=0, ways_stood=0)
            self.history.append(SimpleNamespace(result=result))


def bench(*goes: list[Step]) -> Report:
    story = Story(list(goes))
    report = Report(seed=0, goes=len(goes), max_steps=100, algorithm="q_table")
    _train(story, story.game, report)  # type: ignore[arg-type]
    return report


# A level 200 pixels of camera long: walked, finished, and 50 pixels into the next.
TO_THE_NEXT = [
    Step(0, 0, 0),
    Step(0, 100, 100),
    Step(0, 200, 200),
    Step(1, 0, 200),  # the flag: the next level begins with the camera back at 0
    Step(1, 50, 250),
]


def test_a_level_finished_again_after_going_back_counts_once() -> None:
    """The network's go 3 in T-1023's run: a life goes soon after the flag, the search
    puts the game back to before it, and the level is finished a second time. The
    level was counted twice, and its ground with it: 5,161 for the game where the
    watch, rightly, had 2,857."""
    again = [
        *TO_THE_NEXT,
        Step(0, 150, 250, back=True),  # put back to before the flag
        Step(0, 200, 250),
        Step(1, 0, 250),  # finished again: the same level, the same ground
        Step(1, 90, 290),
    ]
    report = bench(again)
    assert report.finished == [1]
    assert (report.ground_goes, report.ground_right) == (1, 1)
    assert report.clean == 1
    # The same go without the going back, for the bench's count either way.
    plain = bench([*TO_THE_NEXT, Step(1, 90, 290)])
    assert plain.finished == [1] and plain.ground_right == 1 and plain.clean == 1
    # Put back to before the second level's flag, three levels in: its ground once too.
    middle = [
        Step(0, 0, 0),
        Step(0, 200, 200),
        Step(1, 0, 200),
        Step(1, 200, 400),
        Step(2, 0, 400),
        Step(2, 20, 420),
        Step(1, 170, 420, back=True),
        Step(1, 200, 420),
        Step(2, 0, 420),
        Step(2, 60, 460),
    ]
    report = bench(middle)
    assert report.finished == [2] and report.ground_right == 1
    assert report.clean == 1  # its first finish stands, and came before any life was lost
    # A watch that had counted the level's ground twice is not "right" in either.
    twice = [
        Step(s.level, s.scroll, s.counted + 200 * (i > 5), back=s.back) for i, s in enumerate(again)
    ]
    assert bench(twice).ground_right == 0


def test_a_finish_undone_and_not_made_again_counts_nothing() -> None:
    undone = [
        *TO_THE_NEXT,
        Step(0, 150, 250, back=True),
        Step(0, 160, 250),  # and it does not get to the flag again
    ]
    report = bench(undone)
    assert report.finished == [0] and report.clean == 0
    assert report.ground_right == 1  # the furthest it got stays the furthest: 250
    # Put back across two flags at once, and one of them made again.
    two = [
        Step(0, 0, 0),
        Step(0, 200, 200),
        Step(1, 0, 200),
        Step(1, 200, 400),
        Step(2, 0, 400),
        Step(2, 30, 430),
        Step(0, 180, 430, back=True),
        Step(0, 200, 430),
        Step(1, 0, 430),
        Step(1, 120, 430),
    ]
    report = bench(two)
    assert report.finished == [1] and report.ground_right == 1 and report.clean == 1
    # A life lost before the finish that stands: finished, and not cleanly.
    hurt = [
        *TO_THE_NEXT,
        Step(1, 0, 250, lives=2),  # a life goes in the next level ...
        Step(0, 150, 250, lives=3, back=True),  # ... and the go is put back before the flag
        Step(0, 200, 250),
        Step(1, 0, 250),
    ]
    report = bench(hurt)
    assert report.finished == [1] and report.lost == [1] and report.clean == 0


def test_the_game_starting_over_takes_nothing_off() -> None:
    """A game over inside a go: the game begins again at its first level by itself.
    The level it finished was finished, and walking the first level again is ground
    the go has been over."""
    over = [
        *TO_THE_NEXT,
        Step(1, 50, 250, lives=0, mode=toy_hopper.GAME_OVER),
        Step(0, 0, 250),  # started over: nothing put it back
        Step(0, 120, 250),
        Step(0, 200, 250),
        Step(1, 0, 250),
        Step(1, 80, 280),  # past where it had been: new ground again
    ]
    report = bench(over)
    assert report.finished == [2]  # the first level twice: both times it was finished
    assert report.lost == [3]
    assert report.ground_right == 1
    assert report.clean == 1  # the first finish came before any life was lost


def test_a_plain_go_is_counted_as_before() -> None:
    """No going back across a flag: two levels and a life lost inside the second, a
    going back within a level, and a second go that starts from the first's end."""
    first = [
        Step(0, 0, 0),
        Step(0, 100, 100),
        Step(0, 60, 100, back=True),  # back within the level: nothing to take off
        Step(0, 200, 200),
        Step(1, 0, 200),
        Step(1, 150, 350),
        Step(1, 0, 350, lives=2),  # a life lost: put back to the level's start by the game
        Step(1, 200, 400, lives=2),
        Step(2, 0, 400, lives=2),
        Step(2, 40, 440, lives=2),
    ]
    second = [Step(0, 0, 0), Step(0, 100, 100), Step(0, 100, 100, lives=2)]
    report = bench(first, second)
    assert report.finished == [2, 0]
    assert report.lost == [1, 1]
    assert report.clean == 1
    assert (report.ground_goes, report.ground_right) == (2, 2)
    assert report.furthest == [32 + 200, 32 + 100]
