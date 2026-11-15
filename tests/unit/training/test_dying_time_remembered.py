"""T-1033: the AI remembers how long this game takes over a lost life, and asks for
the last chance there first."""

from __future__ import annotations

import logging

import pytest

from retro_ai_lab.emulator.fake import toy_hopper
from retro_ai_lab.training import episode
from retro_ai_lab.training.chance import WALK, Chance
from retro_ai_lab.training.episode import ROOM, EpisodeRunner
from retro_ai_lab.training.vision import ScreenWatch
from tests.unit.training import test_going_back as going_back
from tests.unit.training.test_going_back import SLOTS, Lab, steps
from tests.unit.training.test_last_chance_in_a_go import LOG, LONG, LONG_DIED, going, said
from tests.unit.training.test_last_chance_out_of_time import Searching

lab = going_back.lab  # the practice game behind the real services (pytest fixture)


class Compared:
    """Stands before ``last_chance`` in the runner: notes what each asking was given and
    what it said, and what the same asking says without the guess."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.real = episode.last_chance
        self.calls: list[dict[str, object]] = []
        monkeypatch.setattr(episode, "last_chance", self)

    def __call__(
        self, connection: object, slot: str, trace: tuple[int, ...], space: object, **kw: object
    ) -> Chance | None:
        got = self.real(connection, slot, trace, space, **kw)  # type: ignore[arg-type]
        plain = got
        if kw["guess"] is not None:
            plain = self.real(connection, slot, trace, space, **{**kw, "guess": None})  # type: ignore[arg-type]
        self.calls.append(
            {
                "slot": slot,
                "steps": len(trace),
                "shut": kw["shut"],
                "guess": kw["guess"],
                "said": got,
                "plain": plain,
            }
        )
        return got


def lost(r: EpisodeRunner, sought: int) -> int:
    """Carry the go on until it has looked for a way that many times: the step it is at."""
    while r.ways_sought < sought:
        assert r.step()
    return r.steps


@pytest.mark.parametrize(
    ("frames", "dying", "first", "back"),
    [
        # The pause, the steps the first loss takes, the first asking's plays, and how
        # far the three losses go back.
        (LONG, 53, 73, (74, 78, 86)),
        (toy_hopper.DYING_FRAMES, 15, 48, (36, 40, 48)),
    ],
)
def test_the_second_life_costs_far_fewer_plays(
    lab: Lab,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    frames: int,
    dying: int,
    first: int,
    back: tuple[int, int, int],
) -> None:
    asking = Compared(monkeypatch)
    searching = Searching(monkeypatch)
    r, _ = going(lab, frames, last_chance=True, ways_per_go=3)
    assert r.dying_steps is None
    r.begin()
    with caplog.at_level(logging.INFO, logger=LOG):
        at = [lost(r, sought) for sought in (1, 2, 3)]
    # The first life: nothing is remembered, and it asks as it did (T-1032), play for
    # play: no guess, and with the long pause two moments.
    began = [c for c in asking.calls if c["guess"] is None]
    assert asking.calls[: len(began)] == began and len(began) == (2 if frames == LONG else 1)
    plays = sum(c["said"].plays for c in began)  # type: ignore[attr-defined]
    assert plays == first
    if frames == LONG:
        assert [(c["steps"], c["shut"], c["said"]) for c in began] == [
            (34, None, Chance(0, 15)),
            (54, 20, Chance(1, 58)),
        ]
    # What that showed is remembered: the steps from the last one that could change
    # the loss to the one the count dropped at.
    assert r.dying_steps == dying
    # The second and third: one asking each, at the moment that time leaves room at,
    # with the step it puts the last chance at as the guess.
    second, third = asking.calls[len(began) :]
    assert (second["steps"], second["shut"], second["guess"]) == (
        back[1],
        None,
        back[1] - dying - 1,
    )
    assert (third["steps"], third["shut"], third["guess"]) == (back[2], None, back[2] - dying - 1)
    # The second loss took three steps less than the first: four steps walked, each a
    # play, and the next too late (a play for every other move). The third took the
    # same: the guess, and the step after it.
    others = len(lab.space) - 1
    assert second["said"] == Chance(back[1] - dying + 3, 1 + 4 + others)
    assert third["said"] == Chance(back[2] - dying, 1 + 1 + others)
    assert second["said"].plays == 19 < first / 2 and third["said"].plays == 16  # type: ignore[attr-defined]
    # The step is the one the asking finds without the guess, at more plays.
    for call in (second, third):
        plain, got = call["plain"], call["said"]
        assert isinstance(plain, Chance) and isinstance(got, Chance)
        assert (plain.live, plain.sure) == (got.live, True) and plain.plays > got.plays
    # A loss three steps shorter is the same kind of loss: the longest is kept.
    assert r.dying_steps == dying
    # Each search is told the step, and each finds a way: no life is down at the end.
    assert [(steps_back, live) for _, steps_back, live in searching.calls] == [
        (back[0], back[0] - dying),
        (back[1], back[1] - dying + 3),
        (back[2], back[2] - dying),
    ]
    assert (r.ways_sought, r.ways_found, r.times_back) == (3, 3, 3)
    assert lab.game.state.lives == toy_hopper.START_LIVES
    settled = [line for line in said(caplog) if "That was settled" in line]
    assert len(settled) == 3
    assert settled[1].startswith(
        f"Lost a life at step {at[1]}. That was settled {dying - 3} steps before: nothing "
        "pressed from then on changes it (19 plays, "
    )
    assert f"The AI goes back {back[1]} steps" in settled[1]
    assert settled[2].startswith(
        f"Lost a life at step {at[2]}. That was settled {dying} steps before: nothing "
        "pressed from then on changes it (16 plays, "
    )
    r.finish()


class Told:
    """Stands in for ``last_chance``: notes what it was asked, and answers from a list."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.answers: list[Chance | None] = []
        self.asked: list[tuple[str, int, object, object]] = []
        monkeypatch.setattr(episode, "last_chance", self)

    def __call__(
        self, connection: object, slot: str, trace: tuple[int, ...], *a: object, **kw: object
    ) -> Chance | None:
        self.asked.append((slot, len(trace), kw["shut"], kw["guess"]))
        return self.answers.pop(0)


SIX = tuple(f"back{i}" for i in range(6))
GOES = 6  # goes of the same walk this lab's life count holds for (see ``Goes``)
went: dict[int, int] = {}  # goes made on each lab


class Goes:
    """One runner, go after go: each walks right from the start into the first gap with
    the long pause, and loses the life on step 74 with moments 74, 54 and 34 steps back
    (and one 14 back, newer than the one a search starts from). No search finds a way.

    Six goes at most on one lab: nobody has confirmed the life count there, and after
    six goes of the same walk its discovery takes another value for it, so that a
    seventh go sees no life lost. That is the lab's, and not what is under test.
    """

    def __init__(self, lab: Lab, monkeypatch: pytest.MonkeyPatch, slots: tuple[str, ...]) -> None:
        self.lab = lab
        self.told = Told(monkeypatch)
        self.searching = Searching(monkeypatch, finds=False)
        self.r, self.moments = going(lab, LONG, slots, last_chance=True)
        self.fresh = True

    def lose(self, *answers: Chance | None) -> list[tuple[int, object, object]]:
        """The next go up to its search, with the asking answering so: what was asked
        (steps back, the step given as too late, the guess)."""
        went[id(self.lab)] = went.get(id(self.lab), 0) + 1
        assert went[id(self.lab)] <= GOES, "one go too many for this lab: see Goes"
        if not self.fresh:
            self.lab.game.reset()
        self.fresh = False
        self.told.answers[:] = answers
        self.told.asked.clear()
        self.searching.calls.clear()
        self.r.begin()
        steps(self.r, LONG_DIED)
        assert self.r.ways_sought == 1 and self.told.answers == []
        self.r.finish()
        return [(length, shut, guess) for _, length, shut, guess in self.told.asked]

    @property
    def searched(self) -> tuple[int, object]:
        """The search that followed: steps back, and the steps before the settled one."""
        ((_, length, live),) = self.searching.calls
        return length, live


@pytest.fixture(autouse=True)
def _count_goes_anew() -> None:
    went.clear()


def test_what_is_remembered(
    lab: Lab, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    goes = Goes(lab, monkeypatch, SIX)
    r = goes.r
    # Nothing remembered: it asks at the moment it goes to anyway, with no guess, as it
    # did (T-1032). Five steps there are 25 at the one before: 29 steps of dying.
    assert r.dying_steps is None
    assert goes.lose(Chance(5, 9)) == [(34, None, None)]
    assert goes.searched == (54, 25) and r.dying_steps == 29
    # By that time the moment 34 back has 5 steps and no room: passed over unasked. The
    # one 54 back is asked, with the guess on the last step in time by that time.
    assert goes.lose(Chance(30, 9)) == [(54, None, 24)]
    assert goes.searched == (54, 30)
    assert r.dying_steps == 29  # 24 this time: the longest is kept
    # An answer the time cut short is not remembered (the true step may be earlier).
    assert goes.lose(Chance(20, 9, sure=False)) == [(54, None, 24)]
    assert goes.searched == (74, 40) and r.dying_steps == 29
    # Nor is no answer: the search is the one it was, from the moment it was at. What
    # it says names the steps that were played again, not those of that moment.
    with caplog.at_level(logging.INFO, logger=LOG):
        assert goes.lose(None) == [(54, None, 24)]
    assert goes.searched == (34, None) and r.dying_steps == 29
    assert said(caplog)[0] == (
        "Lost a life at step 74. What it did in the 54 steps before, played again, did "
        "not lose the life on the same step, so the AI cannot tell from when on that was "
        "settled, and looks for a way past as it did before."
    )
    # Nor a moment too late from its first step, which says no step. The one before it
    # is asked as in T-1032, with the steps between as too late and no guess: the
    # time remembered was wrong for this loss.
    assert goes.lose(Chance(0, 9), Chance(0, 9)) == [(54, None, 24), (74, 20, None)]
    assert goes.searched == (74, 0) and r.dying_steps == 29
    assert goes.lose(Chance(0, 9), Chance(3, 9)) == [(54, None, 24), (74, 20, None)]
    assert goes.searched == (74, 3) and r.dying_steps == 71  # longer: remembered


def test_a_shorter_loss_takes_its_place_only_when_far_shorter(
    lab: Lab, monkeypatch: pytest.MonkeyPatch
) -> None:
    goes = Goes(lab, monkeypatch, SIX)
    r = goes.r
    assert goes.lose(Chance(0, 9), Chance(0, 9), Chance(3, 9)) == [
        (34, None, None),
        (54, 20, None),
        (74, 20, None),
    ]
    assert r.dying_steps == 71
    # A time no moment has room for: the oldest is asked all the same.
    assert goes.lose(Chance(3, 9)) == [(74, None, 2)]
    assert r.dying_steps == 71
    # Shorter by as much as the asking walks is the same kind of loss; by one more it
    # is another, and takes its place.
    assert goes.lose(Chance(3 + WALK, 9)) == [(74, None, 2)]
    assert r.dying_steps == 71
    assert goes.lose(Chance(4 + WALK, 9)) == [(74, None, 2)]
    assert goes.searched == (74, 4 + WALK) and r.dying_steps == 70 - WALK == 62
    # A loss whose last step could still have saved the life took no steps at all.
    assert goes.lose(Chance(74, 2)) == [(74, None, 11)]
    assert r.dying_steps == 0
    # Then every moment has room, and the newest is asked with its last step as guess.
    assert goes.lose(Chance(34, 2)) == [(34, None, 33)]
    assert goes.searched == (34, 34) and r.dying_steps == 0


@pytest.mark.parametrize(("dying", "asked"), [(39, 54), (40, 74)])
def test_room_enough_is_fifteen_steps(
    lab: Lab, monkeypatch: pytest.MonkeyPatch, dying: int, asked: int
) -> None:
    # With 39 steps of dying the moment 54 back has fifteen steps before the settled
    # one, with 40 it has fourteen: then it is the one 74 back that is asked.
    goes = Goes(lab, monkeypatch, SIX)
    goes.lose(Chance(0, 9), Chance(54 - dying, 9))
    assert ROOM == 15 and goes.r.dying_steps == dying
    assert goes.lose(Chance(asked - dying, 2)) == [(asked, None, asked - dying - 1)]
    assert goes.searched == (asked, asked - dying) and goes.r.dying_steps == dying


def test_moments_passed_over(lab: Lab, monkeypatch: pytest.MonkeyPatch) -> None:
    # Three moments kept: the loss finds those 54, 34 and 14 steps back.
    goes = Goes(lab, monkeypatch, SLOTS)
    r = goes.r
    assert goes.lose(Chance(0, 9), Chance(2, 9)) == [(34, None, None), (54, 20, None)]
    assert r.dying_steps == 52
    # By that time neither has room. The newer is passed over because an older one is
    # there; the oldest is asked, with the step that time gives as the guess.
    assert goes.lose(Chance(2, 9)) == [(54, None, 1)]
    assert goes.searched == (54, 2) and r.dying_steps == 52
    # One step longer, and that step would be the first: asking the first step is what
    # the asking does anyway, so no guess is given.
    assert goes.lose(Chance(1, 9)) == [(54, None, 1)]
    assert r.dying_steps == 53
    assert goes.lose(Chance(1, 9)) == [(54, None, None)]
    assert goes.searched == (54, 1)


def test_an_older_moment_must_be_one_to_search_from(
    lab: Lab, monkeypatch: pytest.MonkeyPatch
) -> None:
    goes = Goes(lab, monkeypatch, SIX)
    r = goes.r
    assert goes.lose(Chance(0, 9), Chance(0, 9), Chance(1, 9)) == [
        (34, None, None),
        (54, 20, None),
        (74, 20, None),
    ]
    assert r.dying_steps == 73
    assert goes.lose(Chance(0, 9)) == [(74, None, None)]
    # A moment saved with no look at the screen cannot be searched from, so it is no
    # older moment to pass one over for: the one before it is asked, room or none, and
    # with no guess where the time remembered is longer than what it did from there.
    real = ScreenWatch.mark
    marks: list[object] = []

    def marking(self: ScreenWatch) -> object:
        marks.append(real(self))
        return None if len(marks) == 1 else marks[-1]

    monkeypatch.setattr(ScreenWatch, "mark", marking)
    assert goes.lose(Chance(1, 9)) == [(54, None, None)]
    assert goes.searched == (54, 1)
    assert r.dying_steps == 53  # twenty steps shorter: it takes the place
    # The next go's first moment was looked at again: it is the one with room.
    assert goes.lose(Chance(21, 9)) == [(74, None, 20)]


def test_where_nothing_was_found_nothing_is_guessed(
    lab: Lab, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A runner that never found a dying time asks as in T-1032, go after go.
    goes = Goes(lab, monkeypatch, SIX)
    for _ in range(2):
        assert goes.lose(Chance(0, 9), Chance(0, 9), Chance(0, 9)) == [
            (34, None, None),
            (54, 20, None),
            (74, 20, None),
        ]
        assert goes.searched == (74, 0) and goes.r.dying_steps is None
    for _ in range(2):
        assert goes.lose(None) == [(34, None, None)]
        assert goes.searched == (34, None) and goes.r.dying_steps is None
    # And a runner of its own remembers nothing of another's.
    assert goes.lose(Chance(ROOM, 9)) == [(34, None, None)]
    assert goes.r.dying_steps == 34 - ROOM
    other = Goes(lab, monkeypatch, SIX)
    assert other.r.dying_steps is None
    assert other.lose(Chance(ROOM, 9)) == [(34, None, None)]
