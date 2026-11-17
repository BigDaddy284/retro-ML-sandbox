"""T-1054: the learning bench measures the learner alone, by the game's own account.

The practice games are quick, so most of this runs the real thing for a few goes.
What is counted is checked against a real game whose memory a stand-in training
service sets by hand.
"""

from __future__ import annotations

from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

import pytest

from retro_ai_lab.core import ConfigurationError
from retro_ai_lab.core.events import EventBus
from retro_ai_lab.emulator.fake import ToyHopper, toy_hopper, toy_racer
from retro_ai_lab.emulator.fake.toy_racer import ToyRacer
from retro_ai_lab.services import ButtonsHeld
from retro_ai_lab.tools import learning_bench
from retro_ai_lab.tools.learning_bench import (
    Block,
    Go,
    Report,
    _compared,
    _train,
    build_parser,
    how_far,
    main,
    run,
)
from retro_ai_lab.training import Mode
from retro_ai_lab.training.episode import STAND_AFTER
from retro_ai_lab.training.service import WAYS_PER_GO

# ------------------------------------------------------------- the real games


def _whole(report: Report, goes: int, steps: int) -> None:
    assert len(report.played) == goes
    for go in report.played:
        assert 0 <= go.lost <= 3
        assert go.far >= 0
        assert 1 <= go.steps <= steps
        assert go.ended_by


def test_the_racer_is_played_for_as_many_goes_as_asked_with_every_helper_off() -> None:
    report = run("racer", 6, seed=1, algorithm="q_table", max_steps=60, block=3)
    assert (report.game, report.seed, report.goes, report.max_steps) == ("racer", 1, 6, 60)
    assert (report.algorithm, report.block, report.helpers) == ("q_table", 3, False)
    _whole(report, 6, 60)
    assert any(go.far > 0 for go in report.played)  # it drove
    # The learner alone: nothing sought, nothing remembered, nothing looked ahead at.
    assert (report.ways_sought, report.remembered, report.looked_ahead) == (0, 0, 0)
    assert len(report.blocks()) == 2 and len(report.lines()) == 4


def test_the_same_arguments_give_the_same_report_and_another_seed_another() -> None:
    first = run("racer", 5, seed=2, algorithm="q_table", max_steps=80)
    again = run("racer", 5, seed=2, algorithm="q_table", max_steps=80)
    other = run("racer", 5, seed=3, algorithm="q_table", max_steps=80)
    assert first.played == again.played and first.lines() == again.lines()
    assert first.played != other.played


def test_the_network_plays_the_racer_too() -> None:
    report = run("racer", 3, seed=1, max_steps=40)
    assert report.algorithm == "dqn"
    _whole(report, 3, 40)
    assert report.lines()[0].startswith("Learning bench on Toy Racer: neural network, 3 goes")


def test_the_hopper_is_played_likewise() -> None:
    report = run("hopper", 2, seed=1, algorithm="q_table", max_steps=40, block=1)
    _whole(report, 2, 40)
    # It starts 32 into the level, so it is never at nought.
    assert all(go.far >= toy_hopper.START_X for go in report.played)
    assert (report.ways_sought, report.remembered, report.looked_ahead) == (0, 0, 0)
    assert report.lines()[0].startswith("Learning bench on Toy Hopper: simple table, 2 goes")
    assert "(a level is 2,560)" in report.lines()[0]


class _BuiltError(Exception):
    """Raised by the stand-in below: the bench got as far as building the service."""


def _how_built(monkeypatch: pytest.MonkeyPatch, **arguments: Any) -> dict[str, Any]:
    seen: dict[str, Any] = {}

    def service(*args: Any, **options: Any) -> None:
        seen.update(options)
        raise _BuiltError

    monkeypatch.setattr(learning_bench, "TrainingService", service)
    with pytest.raises(_BuiltError):
        run("racer", 1, **arguments)
    return seen


def test_with_the_helpers_off_the_service_is_built_with_every_one_of_them_off(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for arguments in ({}, {"helpers": False}):
        built = _how_built(monkeypatch, **arguments)
        assert built["ways_per_go"] == 0
        assert built["remember"] is False
        assert built["stand_after"] == 0
        assert built["look_ahead"] is False
        assert built["way_seconds"] is None  # no clock in it: the run can be repeated


def test_with_the_helpers_on_the_service_is_built_as_the_app_builds_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    built = _how_built(monkeypatch, helpers=True, seed=9, max_steps=77)
    assert built["ways_per_go"] == WAYS_PER_GO > 0
    assert built["remember"] is True
    assert built["stand_after"] == STAND_AFTER > 0
    assert built["look_ahead"] is True
    assert built["way_seconds"] is None
    assert (built["seed"], built["max_steps"]) == (9, 77)


# ------------------------------------------------- how far, by the game itself


def _put_racer(game: ToyRacer, lives: int, far: int) -> None:
    ram = game._ram
    ram[toy_racer.LAYOUT.lives] = lives
    ram[toy_racer.LAYOUT.lap], ram[toy_racer.LAYOUT.position] = divmod(far, 256)


def _put_hopper(game: ToyHopper, lives: int, far: int) -> None:
    ram = game._ram
    level, x = divmod(far, toy_hopper.LEVEL_WIDTH)
    ram[toy_hopper.LAYOUT.lives], ram[toy_hopper.LAYOUT.level] = lives, level
    ram[toy_hopper.LAYOUT.x_page], ram[toy_hopper.LAYOUT.x_low] = divmod(x, 256)


def test_how_far_is_the_racers_laps_and_its_place_on_the_track() -> None:
    game = ToyRacer(seed=1)
    assert toy_racer.TRACK_LENGTH == 256
    _put_racer(game, 3, 0)
    assert how_far(game) == 0
    game._ram[toy_racer.LAYOUT.position] = 200
    assert how_far(game) == 200
    game._ram[toy_racer.LAYOUT.lap] = 2
    assert how_far(game) == 2 * 256 + 200
    game._ram[toy_racer.LAYOUT.position] = 5
    assert how_far(game) == 517


def test_how_far_is_the_hoppers_levels_and_its_place_in_the_level() -> None:
    game = ToyHopper(seed=1)
    assert toy_hopper.LEVEL_WIDTH == 2560
    assert how_far(game) == toy_hopper.START_X
    game._ram[toy_hopper.LAYOUT.x_low], game._ram[toy_hopper.LAYOUT.x_page] = 44, 1
    assert how_far(game) == 300
    game._ram[toy_hopper.LAYOUT.level] = 1
    assert how_far(game) == 2560 + 300
    game._ram[toy_hopper.LAYOUT.level] = 3
    assert how_far(game) == 3 * 2560 + 300


# --------------------------------------- what a go is counted as, by a stand-in

Act = Callable[["Story"], None]


def at(lives: int, far: int) -> Act:
    """The game is now here, with these lives."""
    return lambda story: story.put(lives, far)


def hold(story: Story) -> None:
    """The step's buttons go down."""
    story.bus.publish(ButtonsHeld(0, frozenset({"up"}), "ai"))


def result(
    steps: int = 1, ended_by: str = "max_steps", sought: int = 0, ahead: int = 0, kept: int = 0
) -> Any:
    return SimpleNamespace(
        steps=steps, ended_by=ended_by, ways_sought=sought, ways_ahead=ahead, remembered=kept
    )


class Story:
    """Stands in for the training service: each go is its steps, each step what
    happens in it, in order. The game is a real one; its memory is set by hand."""

    def __init__(
        self,
        game: ToyRacer | ToyHopper,
        goes: list[list[list[Act]]],
        results: list[Any],
        looked: list[list[int | None]] | None = None,
    ) -> None:
        self.game = game
        self.bus = EventBus()
        self._goes, self._results, self._looked = goes, results, looked
        self._step = 0
        self.episodes = 0
        self.history: list[Any] = []
        self.episode: Any = None
        self.started: list[object] = []

    def put(self, lives: int, far: int) -> None:
        if isinstance(self.game, ToyRacer):
            _put_racer(self.game, lives, far)
        else:
            _put_hopper(self.game, lives, far)

    def start(self, mode: object) -> None:
        self.started.append(mode)

    def step(self) -> None:
        go = self._goes[self.episodes]
        for act in go[self._step]:
            act(self)
        if self._looked is not None:
            looked = self._looked[self.episodes][self._step]
            self.episode = None if looked is None else SimpleNamespace(looked_ahead=looked)
        self._step += 1
        if self._step == len(go):
            self.history.append(SimpleNamespace(result=self._results[self.episodes]))
            self.episodes += 1
            self._step = 0


def _played(story: Story, block: int = 25) -> Report:
    game = "racer" if isinstance(story.game, ToyRacer) else "hopper"
    report = Report(game, 0, len(story._goes), 300, "q_table", block)
    _train(story, story.bus, story.game, report)  # type: ignore[arg-type]
    return report


def test_the_lives_lost_are_the_games_own_count_step_by_step() -> None:
    story = Story(
        ToyRacer(seed=1),
        [
            [
                [at(3, 0), hold, at(3, 10)],
                [hold, at(2, 20)],  # one lost
                [hold, at(2, 30)],
                [hold, at(0, 40)],  # two more in one step
            ],
            [
                # The game is put back to its start before the buttons go down: the
                # three lives it has again were not lost, and none were won.
                [at(3, 0), hold, at(3, 5)],
                [hold, at(3, 9)],
            ],
        ],
        # What the result says of lives is not asked: it has no such thing here.
        [result(4, "game_over"), result(2, "stuck")],
    )
    report = _played(story)
    assert story.started == [Mode.INDEPENDENT]
    assert report.played == [Go(3, 40, 4, "game_over"), Go(0, 9, 2, "stuck")]


def test_a_go_that_begins_with_a_life_already_gone_is_a_life_down() -> None:
    story = Story(
        ToyRacer(seed=1),
        [
            [[at(3, 0), hold, at(3, 10)]],
            # The app moved where its goes start (it does, to get a game going): this
            # go begins further on with a life gone. By the game's account it is a
            # life down, whatever it does next.
            [[at(2, 115), hold, at(2, 120)], [hold, at(1, 130)], [hold, at(1, 131)]],
            [[at(1, 115), hold, at(1, 116)]],
            # Back at the game's start, and with a life more than it starts with:
            # nothing gone, and nothing taken off for the extra.
            [[at(3, 0), hold, at(3, 3)]],
            [[at(4, 0), hold, at(4, 3)], [hold, at(3, 9)]],
        ],
        [result(1), result(3), result(1), result(1), result(2)],
    )
    report = _played(story)
    assert [go.lost for go in report.played] == [0, 2, 2, 0, 1]
    assert report.began_short == 2
    assert report.lines()[-2] == (
        "2 goes began with a life already gone (the app had moved where its goes start): "
        "that life is counted as lost in each."
    )


def test_a_life_given_back_within_the_step_was_not_lost() -> None:
    story = Story(
        ToyRacer(seed=1),
        [
            [
                [at(3, 0), hold, at(3, 30)],
                # It lost a life, went back to before it and played on: the buttons
                # went down more than once in the step, the first time is the one.
                [hold, at(2, 40), at(3, 28), hold, at(3, 33)],
                [hold, at(3, 41)],
            ]
        ],
        [result(3)],
    )
    assert _played(story).played == [Go(0, 41, 3, "max_steps")]


def test_a_life_that_stayed_lost_after_going_back_is_counted() -> None:
    story = Story(
        ToyRacer(seed=1),
        [[[at(3, 0), hold, at(2, 40), at(3, 28), hold, at(2, 35)], [hold, at(2, 36)]]],
        [result(2)],
    )
    assert _played(story).played == [Go(1, 36, 2, "max_steps")]


def test_the_furthest_is_kept_when_the_game_goes_back_and_is_the_gos_own() -> None:
    story = Story(
        ToyRacer(seed=1),
        [
            [[at(3, 0), hold, at(3, 10)], [hold, at(3, 500)], [hold, at(3, 30)]],
            # The last go ended far on. This one begins at the start and is counted
            # from there: where the game stood before its buttons went down is not it.
            [[at(3, 0), hold, at(3, 12)], [hold, at(3, 7)]],
        ],
        [result(3), result(2)],
    )
    report = _played(story)
    assert [go.far for go in report.played] == [500, 12]


def test_a_lap_and_a_level_are_in_how_far_a_go_got() -> None:
    racer = Story(ToyRacer(seed=1), [[[at(3, 0), hold, at(3, 2 * 256 + 5)]]], [result()])
    assert _played(racer).played[0].far == 517
    hopper = Story(
        ToyHopper(seed=1),
        [[[at(3, 32), hold, at(3, 2000)], [hold, at(2, 2560 + 300)], [hold, at(2, 2560 + 40)]]],
        [result(3, "reset")],
    )
    assert _played(hopper).played == [Go(1, 2860, 3, "reset")]


def test_a_step_in_which_no_buttons_went_down_counts_nothing() -> None:
    story = Story(
        ToyRacer(seed=1),
        [[[at(3, 0), hold, at(3, 10)], [at(1, 700)], [hold, at(1, 20)]]],
        [result(2)],
    )
    # The step between only moved the game about (the service putting it somewhere):
    # nothing was played in it. The step after counts from where its buttons went down.
    assert _played(story).played == [Go(0, 20, 2, "max_steps")]


def test_the_steps_and_the_ending_are_the_results_and_the_helpers_work_is_added_up() -> None:
    steps: list[list[Act]] = [[at(3, 0), hold, at(3, 1)], [hold, at(3, 2)], [hold, at(3, 3)]]
    story = Story(
        ToyRacer(seed=1),
        [steps, steps, steps],
        [
            result(120, "stuck", sought=2, ahead=1, kept=40),
            result(7, "game_over"),
            result(300, "max_steps", sought=1, ahead=3, kept=5),
        ],
        # A go's count of looks ahead is read at every step and the most it showed is
        # the go's: by the step that ends a go the service may have begun the next.
        # A go that is not there, or knows nothing of looking ahead, counts as none.
        looked=[[0, 4, 0], [None, None, None], [2, 7, 7]],
    )
    report = _played(story)
    assert [(go.steps, go.ended_by) for go in report.played] == [
        (120, "stuck"),
        (7, "game_over"),
        (300, "max_steps"),
    ]
    assert report.ways_sought == 2 + 1 + 1 + 3
    assert report.remembered == 45
    assert report.looked_ahead == 11
    plain = Story(ToyRacer(seed=1), [steps], [result()])
    plain.episode = SimpleNamespace()  # a runner with no such count
    assert _played(plain).looked_ahead == 0


def test_it_stops_at_the_goes_asked_for() -> None:
    steps: list[list[Act]] = [[at(3, 0), hold, at(3, 1)]]
    story = Story(ToyRacer(seed=1), [steps] * 5, [result()] * 5)
    report = Report("racer", 0, 3, 300, "q_table")
    _train(story, story.bus, story.game, report)  # type: ignore[arg-type]
    assert len(report.played) == 3 and story.episodes == 3


# ------------------------------------------------------------ blocks and lines


def _report(played: list[Go], **options: Any) -> Report:
    made: dict[str, Any] = {
        "game": "racer",
        "seed": 7,
        "goes": len(played),
        "max_steps": 300,
        "algorithm": "dqn",
        "block": 2,
    }
    made.update(options)
    return Report(played=played, **made)


FIVE = [
    Go(3, 100, 300, "max_steps"),
    Go(1, 200, 300, "max_steps"),
    Go(0, 300, 300, "max_steps"),
    Go(0, 1500, 300, "max_steps"),
    Go(2, 50, 12, "stuck"),
]


def test_the_goes_are_added_up_in_blocks_and_those_left_over_are_a_shorter_one() -> None:
    assert _report(FIVE).blocks() == (
        Block(1, 2, 2.0, 150.0, 200, 0),
        Block(3, 4, 0.0, 900.0, 1500, 2),
        Block(5, 5, 2.0, 50.0, 50, 0),
    )
    assert _report(FIVE[:4]).blocks() == (
        Block(1, 2, 2.0, 150.0, 200, 0),
        Block(3, 4, 0.0, 900.0, 1500, 2),
    )
    assert _report(FIVE, block=5).blocks() == (Block(1, 5, 1.2, 430.0, 1500, 2),)
    assert _report(FIVE, block=25).blocks() == (Block(1, 5, 1.2, 430.0, 1500, 2),)
    assert _report([]).blocks() == ()


def test_the_report_in_plain_words() -> None:
    assert _report(FIVE).lines() == [
        "Learning bench on Toy Racer: neural network, 5 goes of up to 300 steps, seed 7, "
        "the learner alone, every helper off. How far is in the game's own units "
        "(a lap is 256).",
        "Goes 1 to 2: 2.00 lives lost a go; got 150 far on average, 200 at best; "
        "0 of 2 with no life lost.",
        "Goes 3 to 4: 0.00 lives lost a go; got 900 far on average, 1,500 at best; "
        "2 of 2 with no life lost.",
        "Go 5: 2.00 lives lost a go; got 50 far on average, 50 at best; 0 of 1 with no life lost.",
        "From the first block to the last: lives lost a go 2.00 to 2.00 (the same), "
        "how far 150 to 50 (less far).",
    ]


def test_the_last_line_says_which_way_it_went() -> None:
    better = _report(FIVE[:4]).lines()[-1]
    assert better == (
        "From the first block to the last: lives lost a go 2.00 to 0.00 (fewer), "
        "how far 150 to 900 (further)."
    )
    worse = _report(list(reversed(FIVE[:4]))).lines()[-1]
    assert worse == (
        "From the first block to the last: lives lost a go 0.00 to 2.00 (more), "
        "how far 900 to 150 (less far)."
    )
    same = _report([FIVE[0], FIVE[1], FIVE[1], FIVE[0]]).lines()[-1]
    assert same == (
        "From the first block to the last: lives lost a go 2.00 to 2.00 (the same), "
        "how far 150 to 150 (the same)."
    )


def test_what_looks_the_same_in_the_report_is_called_the_same() -> None:
    assert _compared(1.004, 1.001, "fewer", "more", 2) == "the same"
    assert _compared(1.01, 1.02, "fewer", "more", 2) == "fewer"
    assert _compared(1.02, 1.01, "fewer", "more", 2) == "more"
    assert _compared(150.3, 150.1, "less far", "further", 0) == "the same"
    assert _compared(151.0, 150.0, "less far", "further", 0) == "further"
    assert _compared(149.0, 150.0, "less far", "further", 0) == "less far"


def test_one_block_one_go_and_the_other_words_of_the_heading() -> None:
    one = _report([Go(1, 2600, 300, "max_steps")], game="hopper", algorithm="q_table", block=25)
    assert one.lines() == [
        "Learning bench on Toy Hopper: simple table, 1 go of up to 300 steps, seed 7, "
        "the learner alone, every helper off. How far is in the game's own units "
        "(a level is 2,560).",
        "Go 1: 1.00 lives lost a go; got 2,600 far on average, 2,600 at best; "
        "0 of 1 with no life lost.",
        "One block only: nothing to compare it with.",
    ]
    helped = _report(FIVE, helpers=True, goes=1200, max_steps=2000).lines()[0]
    assert helped == (
        "Learning bench on Toy Racer: neural network, 1,200 goes of up to 2,000 steps, "
        "seed 7, trained as the app does, with its helpers on. How far is in the game's "
        "own units (a lap is 256)."
    )
    assert _report([]).lines()[1:] == ["No goes were played."]
    said = "began with a life already gone (the app had moved where its goes start): "
    assert _report(FIVE, began_short=1).lines()[-2] == (
        f"1 go {said}that life is counted as lost in each."
    )
    assert _report(FIVE, began_short=1200).lines()[-2].startswith(f"1,200 goes {said}")
    assert not any("already gone" in line for line in _report(FIVE).lines())


# ------------------------------------------------------------------- arguments


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (
            {"game": "kart"},
            r"The learning bench has no practice game called 'kart' \(choices: racer, hopper\)\.",
        ),
        ({"game": ""}, "The learning bench has no practice game called ''"),
        ({"goes": 0}, "The learning bench needs a whole number of goes, 1 or more."),
        ({"goes": -3}, "The learning bench needs a whole number of goes, 1 or more."),
        ({"goes": 2.0}, "The learning bench needs a whole number of goes, 1 or more."),
        ({"goes": True}, "The learning bench needs a whole number of goes, 1 or more."),
        ({"max_steps": 0}, "The learning bench needs a whole number of steps, 1 or more."),
        ({"max_steps": "9"}, "The learning bench needs a whole number of steps, 1 or more."),
        ({"block": 0}, "The learning bench needs a whole number of goes to a block, 1 or more."),
        (
            {"block": None},
            "The learning bench needs a whole number of goes to a block, 1 or more.",
        ),
        ({"seed": -1}, "The learning bench needs a seed from 0 to 65535."),
        ({"seed": 65536}, "The learning bench needs a seed from 0 to 65535."),
        ({"seed": 1.5}, "The learning bench needs a seed from 0 to 65535."),
        ({"seed": False}, "The learning bench needs a seed from 0 to 65535."),
        ({"algorithm": "magic"}, r"No way of learning called 'magic' \(choices: .*dqn"),
        ({"helpers": "yes"}, "The learning bench needs helpers to be True or False."),
        ({"helpers": 1}, "The learning bench needs helpers to be True or False."),
        ({"helpers": None}, "The learning bench needs helpers to be True or False."),
    ],
)
def test_what_it_cannot_run_is_refused_in_plain_words(
    arguments: dict[str, Any], message: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    def never(*args: Any, **options: Any) -> None:
        raise AssertionError("refused before anything is built")

    monkeypatch.setattr(learning_bench, "TrainingService", never)
    asked: dict[str, Any] = {"game": "racer", "goes": 1}
    asked.update(arguments)
    with pytest.raises(ConfigurationError, match=message):
        run(asked.pop("game"), asked.pop("goes"), **asked)


def test_the_edges_of_what_it_takes(monkeypatch: pytest.MonkeyPatch) -> None:
    # The lowest and the highest seed, and one of everything else, get as far as
    # the service being built.
    for seed in (0, 65535):
        built = _how_built(monkeypatch, seed=seed, max_steps=1, block=1)
        assert (built["seed"], built["max_steps"]) == (seed, 1)


# ---------------------------------------------------------------- command line


def _asked(monkeypatch: pytest.MonkeyPatch, argv: list[str]) -> tuple[int, tuple[Any, ...], Any]:
    seen: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def stand_in(*args: Any, **options: Any) -> Report:
        seen.append((args, options))
        return _report(FIVE)

    monkeypatch.setattr(learning_bench, "run", stand_in)
    code = main(argv)
    assert len(seen) == 1
    return code, seen[0][0], seen[0][1]


def test_the_command_line_prints_the_reports_lines(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    code, args, options = _asked(monkeypatch, ["racer"])
    assert code == 0
    assert capsys.readouterr().out.splitlines() == _report(FIVE).lines()
    # Asked for nothing but the game: 100 goes of 300 steps for the network, in
    # blocks of 25, the first track, every helper off.
    assert args == ("racer", 100)
    assert options == {
        "seed": 0,
        "algorithm": "dqn",
        "max_steps": 300,
        "block": 25,
        "helpers": False,
    }


def test_each_option_of_the_command_line_reaches_the_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    code, args, options = _asked(
        monkeypatch,
        [
            "hopper",
            "--goes",
            "40",
            "--steps",
            "150",
            "--seed",
            "65535",
            "--block",
            "8",
            "--algorithm",
            "q_table",
            "--helpers",
        ],
    )
    assert code == 0 and args == ("hopper", 40)
    assert options == {
        "seed": 65535,
        "algorithm": "q_table",
        "max_steps": 150,
        "block": 8,
        "helpers": True,
    }


@pytest.mark.parametrize(
    ("argv", "said"),
    [
        ([], "the following arguments are required: game"),
        (["kart"], "invalid choice: 'kart'"),
        (["racer", "--goes", "0"], "argument --goes: must be 1 or more."),
        (["racer", "--goes", "many"], "argument --goes: 'many' is not a whole number."),
        (["racer", "--steps", "-1"], "argument --steps: must be 1 or more."),
        (["racer", "--block", "0"], "argument --block: must be 1 or more."),
        (["racer", "--seed", "65536"], "argument --seed: must be from 0 to 65535."),
        (["racer", "--seed", "-1"], "argument --seed: must be from 0 to 65535."),
        (["racer", "--seed", "x"], "argument --seed: 'x' is not a whole number."),
        (["racer", "--algorithm", "magic"], "argument --algorithm: invalid choice: 'magic'"),
    ],
)
def test_the_command_line_refuses_what_the_run_would(
    argv: list[str], said: str, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as stopped:
        build_parser().parse_args(argv)
    assert stopped.value.code == 2
    assert said in capsys.readouterr().err


def test_the_command_line_runs_the_real_thing(capsys: pytest.CaptureFixture[str]) -> None:
    argv = ["racer", "--goes", "4", "--steps", "30", "--block", "2", "--seed", "1"]
    code = main([*argv, "--algorithm", "q_table"])
    out = capsys.readouterr().out.splitlines()
    assert code == 0 and len(out) == 4
    assert out[0] == (
        "Learning bench on Toy Racer: simple table, 4 goes of up to 30 steps, seed 1, "
        "the learner alone, every helper off. How far is in the game's own units "
        "(a lap is 256)."
    )
    assert out[1].startswith("Goes 1 to 2: ") and out[2].startswith("Goes 3 to 4: ")
    assert out[3].startswith("From the first block to the last: lives lost a go ")
    assert out == run("racer", 4, seed=1, algorithm="q_table", max_steps=30, block=2).lines()
