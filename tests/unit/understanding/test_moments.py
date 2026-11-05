"""T-901: a person says what just happened; the AI narrows the byte down."""

from __future__ import annotations

import pytest

from retro_ai_lab.core import ConfigurationError
from retro_ai_lab.observation.memory import MemorySnapshot
from retro_ai_lab.understanding import Moment, MomentFinder, VariableKind


def snap(frame: int, ram: list[int]) -> MemorySnapshot:
    return MemorySnapshot(frame, {"ram": bytes(ram)})


def test_moments_name_their_kind() -> None:
    assert Moment.LOST_LIFE.kind is VariableKind.LIVES and Moment.LOST_LIFE.name == "Lives"
    assert Moment.GOT_FURTHER.kind is VariableKind.PROGRESS and Moment.SCORED.name == "Score"
    assert Moment.LOST_LIFE.words == "lost a life" and Moment.SCORED.movement == "went up"


def test_a_lost_life_keeps_the_bytes_that_went_down_by_one() -> None:
    finder = MomentFinder()
    before = snap(100, [3, 10, 7, 0, 200])
    after = snap(160, [2, 11, 6, 0, 150])  # byte 0 and 2 down by one; 4 down by fifty
    n = finder.tell(Moment.LOST_LIFE, [before, after])
    assert n.candidates == (("ram", 0), ("ram", 2)) and n.presses == 1 and n.settled is None
    assert n.text == (
        "You lost a life: 2 bytes went down by one in the last 5 seconds. Tell the AI again "
        "the next time it happens."
    )
    # The next death: only byte 0 goes down by one again; two presses agree.
    n = finder.tell(
        Moment.LOST_LIFE, [snap(400, [2, 11, 6, 0, 150]), snap(430, [1, 11, 6, 0, 150])]
    )
    assert n.settled == ("ram", 0) and n.presses == 2 and n.count == 1
    assert n.text.startswith('Got it: "Lives" is ram[0]. You said you lost a life 2 times')
    assert "that byte went down by one each time" in n.text
    assert finder.presses(Moment.LOST_LIFE) == 0  # settled: ready for the next game


def test_score_and_progress_keep_the_bytes_that_went_up() -> None:
    finder = MomentFinder(settle_after=1)
    n = finder.tell(Moment.SCORED, [snap(0, [5, 5, 5]), snap(10, [5, 9, 4])])
    assert n.settled == ("ram", 1) and "You said you scored once" in n.text
    n = finder.tell(Moment.GOT_FURTHER, [snap(0, [1, 1]), snap(10, [2, 2])])
    assert n.settled is None and n.count == 2 and n.presses == 1


def test_one_byte_left_waits_for_a_second_press() -> None:
    finder = MomentFinder()
    n = finder.tell(Moment.LOST_LIFE, [snap(0, [3, 3]), snap(10, [2, 3])])
    assert n.count == 1 and n.settled is None
    assert "one byte went down by one" in n.text and "once more" in n.text


def test_nothing_in_common_starts_again() -> None:
    finder = MomentFinder()
    finder.tell(Moment.LOST_LIFE, [snap(0, [3, 3]), snap(10, [2, 3])])
    n = finder.tell(Moment.LOST_LIFE, [snap(20, [2, 3]), snap(30, [2, 2])])
    assert n.candidates == (("ram", 1),) and n.presses == 1
    assert n.text.startswith("Nothing went down by one both times, so the AI started again")
    n = finder.tell(Moment.LOST_LIFE, [snap(40, [2, 2]), snap(50, [2, 2])])
    assert n.count == 0 and "no byte went down by one" in n.text and n.presses == 1
    finder.forget()
    assert finder.presses(Moment.LOST_LIFE) == 0 and finder.candidates(Moment.LOST_LIFE) == ()


def test_the_window_is_the_last_five_seconds_and_stops_at_a_restore() -> None:
    finder = MomentFinder(window_frames=300)
    ring = [snap(0, [9, 9]), snap(100, [3, 9]), snap(250, [3, 9]), snap(500, [2, 8])]
    # 100 is more than 300 frames before 500: the window starts at frame 250, so byte 1
    # (down by one since 250) counts and byte 0 (down by one since 250 too) as well.
    n = finder.tell(Moment.LOST_LIFE, ring)
    assert n.candidates == (("ram", 0), ("ram", 1))
    finder.forget()
    # A restore sends the frame counter backwards: the window starts after it.
    ring = [snap(900, [5, 5]), snap(10, [5, 4]), snap(40, [4, 4])]
    n = finder.tell(Moment.LOST_LIFE, ring)
    assert n.candidates == (("ram", 0),)


def test_misuse() -> None:
    with pytest.raises(ConfigurationError, match="Play for a few seconds"):
        MomentFinder().tell(Moment.LOST_LIFE, [snap(0, [1])])
    with pytest.raises(ConfigurationError):
        MomentFinder(window_frames=0)
    with pytest.raises(ConfigurationError):
        MomentFinder(settle_after=0)
    finder = MomentFinder()
    # A region that changed size or is missing is skipped, not a crash.
    n = finder.tell(
        Moment.LOST_LIFE,
        [
            MemorySnapshot(0, {"ram": b"\x03", "x": b"\x01"}),
            MemorySnapshot(5, {"ram": b"\x02\x02", "x": b"\x00"}),
        ],
    )
    assert n.candidates == (("x", 0),)


# ------------------------------------------------------------ T-904: restarts


def test_restarts_vote_for_the_byte_that_went_down_by_one_before_each() -> None:
    from retro_ai_lab.understanding import RestartWatch
    from retro_ai_lab.understanding.moments import RESTARTS_FORGET, RESTARTS_NEEDED

    assert RESTARTS_NEEDED == 3 and RESTARTS_FORGET == 8
    watch = RestartWatch(window_frames=100)

    def restart(lives_before: int, speed_before: int, n: int) -> tuple[str, int] | None:
        # 100 frames before the restart, the moment before it, and the restart itself.
        before = snap(n * 1000, [lives_before, speed_before, 9])
        death = snap(n * 1000 + 50, [lives_before - 1, 0, 9])
        home = snap(n * 1000 + 51, [3, 0, 0])
        return watch.restart([before, death, home])

    assert restart(3, 1, 1) is None and watch.restarts == 1
    assert watch.votes(("ram", 0)) == 1 and watch.votes(("ram", 1)) == 1
    assert restart(2, 1, 2) is None  # a tie so far
    assert restart(1, 0, 3) == ("ram", 0)  # speed was already 0: lives alone has 3 votes
    assert watch.restarts == 0 and watch.votes(("ram", 0)) == 0  # a winner clears the count
    # Bytes the caller knows better about get no vote.
    for n in range(4, 7):
        got = watch.restart(
            [
                snap(n * 1000, [3, 1, 9]),
                snap(n * 1000 + 50, [2, 0, 9]),
                snap(n * 1000 + 51, [3, 0, 0]),
            ],
            skip={("ram", 1)},
        )
    assert got == ("ram", 0)
    # Too few snapshots: no vote, no count.
    assert watch.restart([snap(0, [3]), snap(1, [2])]) is None and watch.restarts == 0


def test_restarts_with_no_clear_winner_are_forgotten() -> None:
    from retro_ai_lab.understanding import RestartWatch
    from retro_ai_lab.understanding.moments import RESTARTS_FORGET

    watch = RestartWatch(window_frames=100)
    for n in range(RESTARTS_FORGET - 1):
        # Two bytes always go down by one together: a tie that never breaks.
        assert (
            watch.restart(
                [snap(n * 1000, [3, 3]), snap(n * 1000 + 50, [2, 2]), snap(n * 1000 + 51, [3, 3])]
            )
            is None
        )
    assert watch.restarts == RESTARTS_FORGET - 1
    assert watch.restart([snap(9000, [3, 3]), snap(9050, [2, 2]), snap(9051, [3, 3])]) is None
    assert watch.restarts == 0  # forgotten: a game whose restarts are not deaths

    # A winner needs two thirds of the restarts, not just three votes.
    def one(n: int, byte: int) -> tuple[str, int] | None:
        before = [3, 3]
        after = [3, 3]
        after[byte] = 2
        return watch.restart(
            [snap(n * 100, before), snap(n * 100 + 5, after), snap(n * 100 + 6, [3, 3])]
        )

    assert [one(n, b) for n, b in enumerate((1, 0, 1, 0, 1))] == [None] * 5
    assert watch.votes(("ram", 1)) == 3 and watch.votes(("ram", 0)) == 2  # 3 of 5 is not enough
    assert one(5, 1) == ("ram", 1)  # 4 of 6 is
    with pytest.raises(ConfigurationError):
        RestartWatch(needed=0)
    with pytest.raises(ConfigurationError):
        RestartWatch(needed=3, forget_after=2)
