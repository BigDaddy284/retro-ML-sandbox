"""T-1056: the bench counts how far a go got from where the go began.

The app may move where its goes start (past a life it takes for lost whatever is
pressed). A go begun further on did not get the ground it was put past.
"""

from __future__ import annotations

from retro_ai_lab.emulator.fake import ToyHopper, toy_hopper
from retro_ai_lab.emulator.fake.toy_racer import ToyRacer
from retro_ai_lab.tools.learning_bench import Go, build_parser
from tests.unit.tools.test_learning_bench import Story, _played, at, hold, result


def _far(story: Story) -> list[int]:
    return [go.far for go in _played(story).played]


def test_a_go_begun_further_on_is_counted_from_where_it_began() -> None:
    story = Story(
        ToyRacer(seed=1),
        [
            [[at(3, 0), hold, at(3, 10)]],
            [[at(2, 115), hold, at(2, 120)], [hold, at(1, 130)], [hold, at(1, 131)]],
            [[at(1, 115), hold, at(1, 116)]],
            [[at(3, 0), hold, at(3, 3)]],
        ],
        [result(1), result(3), result(1), result(1)],
    )
    report = _played(story)
    # 131 and 116 are where the game had it: 115 of each was the start moved.
    assert [go.far for go in report.played] == [10, 16, 1, 3]
    assert [go.began for go in report.played] == [0, 115, 115, 0]


def test_a_go_begun_at_the_games_start_says_so_and_is_counted_as_it_was() -> None:
    story = Story(ToyRacer(seed=1), [[[at(3, 0), hold, at(3, 40)]]], [result(4, "game_over")])
    # Equal to a go written without where it began: that is nought unless said.
    assert _played(story).played == [Go(0, 40, 4, "game_over")]
    assert Go(0, 40, 4, "game_over").began == 0
    assert Go(0, 40, 4, "game_over") != Go(0, 40, 4, "game_over", 7)


def test_the_hoppers_start_is_not_nought_and_is_not_taken_off() -> None:
    story = Story(
        ToyHopper(seed=1),
        [
            [[at(3, toy_hopper.START_X), hold, at(3, 50)]],
            # Begun at 200 and got to 260: 60 of ground, on top of the game's own 32.
            [[at(2, 200), hold, at(2, 260)]],
        ],
        [result(), result()],
    )
    report = _played(story)
    assert [go.far for go in report.played] == [50, 92]
    assert [go.began for go in report.played] == [0, 200 - toy_hopper.START_X]


def test_a_lap_and_a_level_finished_in_a_go_begun_further_on_are_counted() -> None:
    racer = Story(ToyRacer(seed=1), [[[at(2, 200), hold, at(2, 256 + 10)]]], [result()])
    assert _far(racer) == [66]
    hopper = Story(
        ToyHopper(seed=1), [[[at(2, 2000), hold, at(2, 2560 + 100)]]], [result(1, "reset")]
    )
    assert _far(hopper) == [2560 + 100 - (2000 - toy_hopper.START_X)]


def test_the_furthest_is_kept_when_the_game_goes_back_in_a_go_begun_further_on() -> None:
    story = Story(
        ToyRacer(seed=1),
        [[[at(2, 115), hold, at(2, 300)], [hold, at(2, 120)], [hold, at(2, 118)]]],
        [result(3)],
    )
    assert _far(story) == [185]


def test_where_a_go_began_is_where_its_first_buttons_went_down() -> None:
    story = Story(
        ToyRacer(seed=1),
        [
            [
                # It lost a life, was put back to before it and played on, all within
                # the go's first step: the buttons went down twice. The first time is
                # where the go began; from 100 it would have been 40.
                [at(2, 115), hold, at(1, 130), at(2, 100), hold, at(2, 104)],
                [hold, at(2, 140)],
            ]
        ],
        [result(2)],
    )
    report = _played(story)
    assert [(go.far, go.began) for go in report.played] == [(25, 115)]


def test_a_go_that_never_gets_past_where_it_began_got_nowhere() -> None:
    stood = Story(ToyRacer(seed=1), [[[at(2, 115), hold, at(2, 115)]]], [result()])
    assert _far(stood) == [0]
    # Behind where it began is not less than nowhere.
    back = Story(
        ToyRacer(seed=1), [[[at(2, 115), hold, at(2, 110)], [hold, at(2, 90)]]], [result(2)]
    )
    assert _far(back) == [0]


def test_a_go_begun_behind_the_games_start_is_counted_by_the_games_place() -> None:
    # No game does it, but nothing is ever added on for it: a go is never said to
    # have got further than the game had it.
    story = Story(ToyHopper(seed=1), [[[at(3, 400), hold, at(3, 450)]]], [result(1, "max_steps")])
    story.put(3, 500)
    assert _played(story).played == [Go(0, 450, 1, "max_steps", 0)]


def test_a_go_in_which_no_buttons_went_down_began_nowhere_and_got_nowhere() -> None:
    story = Story(
        ToyRacer(seed=1),
        [[[at(2, 115), hold, at(2, 131)]], [[at(2, 300)]], [[at(2, 115), hold, at(2, 120)]]],
        [result(), result(1, "stuck"), result()],
    )
    assert _played(story).played == [
        Go(1, 16, 1, "max_steps", 115),
        Go(0, 0, 1, "stuck", 0),
        Go(1, 5, 1, "max_steps", 115),
    ]


def test_a_step_with_no_buttons_down_moves_nothing_in_a_go_begun_further_on() -> None:
    story = Story(
        ToyRacer(seed=1),
        [[[at(2, 115), hold, at(2, 120)], [at(2, 700)], [hold, at(2, 125)]]],
        [result(2)],
    )
    # The step between only moved the game about: 700 was never got to, and the go
    # still began at 115.
    assert [(go.far, go.began) for go in _played(story).played] == [(10, 115)]


def test_every_go_is_counted_from_where_it_began_itself() -> None:
    story = Story(
        ToyRacer(seed=1),
        [
            [[at(2, 115), hold, at(2, 131)]],
            [[at(1, 240), hold, at(1, 250)]],
            [[at(3, 0), hold, at(3, 7)]],
            [[at(2, 115), hold, at(2, 117)]],
        ],
        [result(), result(), result(), result()],
    )
    report = _played(story)
    assert [go.far for go in report.played] == [16, 10, 7, 2]
    assert [go.began for go in report.played] == [115, 240, 0, 115]


def test_the_games_start_is_read_when_training_begins() -> None:
    # Whatever the game's start is, it is where the game stands when training
    # begins: nothing here knows a game's start by heart.
    hopper = Story(
        ToyHopper(seed=1),
        [[[at(3, 500), hold, at(3, 560)]], [[at(2, 700), hold, at(2, 730)]]],
        [result(), result()],
    )
    hopper.put(3, 500)
    report = _played(hopper)
    assert [(go.far, go.began) for go in report.played] == [(560, 0), (530, 200)]
    racer = Story(ToyRacer(seed=1), [[[at(3, 40), hold, at(3, 50)]]], [result()])
    racer.put(3, 40)
    assert [(go.far, go.began) for go in _played(racer).played] == [(50, 0)]


def test_the_lives_and_the_reports_lines_are_as_they_were_for_goes_begun_further_on() -> None:
    # The story of T-1054's own test of a go begun a life down.
    story = Story(
        ToyRacer(seed=1),
        [
            [[at(3, 0), hold, at(3, 10)]],
            [[at(2, 115), hold, at(2, 120)], [hold, at(1, 130)], [hold, at(1, 131)]],
            [[at(1, 115), hold, at(1, 116)]],
            [[at(3, 0), hold, at(3, 3)]],
            [[at(4, 0), hold, at(4, 3)], [hold, at(3, 9)]],
        ],
        [result(1), result(3), result(1), result(1), result(2)],
    )
    report = _played(story)
    assert [go.lost for go in report.played] == [0, 2, 2, 0, 1]
    assert report.began_short == 2
    assert [go.far for go in report.played] == [10, 16, 1, 3, 9]
    assert report.lines() == [
        "Learning bench on Toy Racer: simple table, 5 goes of up to 300 steps, seed 0, "
        "the learner alone, every helper off. How far is in the game's own units "
        "(a lap is 256).",
        # 7.8 is the mean of 10, 16, 1, 3 and 9: with the moved start in, it was 53.8.
        "Goes 1 to 5: 1.00 lives lost a go; got 8 far on average, 16 at best; "
        "2 of 5 with no life lost.",
        "2 goes began with a life already gone (the app had moved where its goes start): "
        "that life is counted as lost in each.",
        "One block only: nothing to compare it with.",
    ]


def test_the_blocks_add_up_what_the_goes_themselves_covered() -> None:
    story = Story(
        ToyRacer(seed=1),
        [
            [[at(3, 0), hold, at(3, 10)]],
            [[at(2, 115), hold, at(2, 131)]],
            [[at(2, 115), hold, at(2, 119)]],
            [[at(2, 115), hold, at(2, 121)]],
        ],
        [result(), result(), result(), result()],
    )
    first, last = _played(story, block=2).blocks()
    assert (first.far, first.best) == (13.0, 16)
    assert (last.far, last.best) == (5.0, 6)


def test_the_help_says_what_how_far_is() -> None:
    said = " ".join(build_parser().format_help().split())
    assert "How far a go got is counted from where the go began" in said
