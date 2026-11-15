"""T-1036: during a go the AI asks the game whether a rise it was paid for was earned."""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence

import pytest

from retro_ai_lab.core import ConfigurationError, EmulatorError
from retro_ai_lab.core.events import EventBus
from retro_ai_lab.emulator.fake import toy_hopper
from retro_ai_lab.emulator.fake.toy_hopper import LAYOUT, ToyHopper
from retro_ai_lab.services.discovery import ON_ITS_OWN, DiscoveryUpdated
from retro_ai_lab.services.rewards import RewardEarned
from retro_ai_lab.training import episode
from retro_ai_lab.training.earned import Rise
from retro_ai_lab.training.episode import (
    NOW_SLOT,
    RISE_BACK,
    RISE_MOST,
    EpisodeRunner,
    EpisodeSignals,
)
from retro_ai_lab.training.moments import EVERY, Moments
from retro_ai_lab.understanding.events import EventKind, GameEvent
from retro_ai_lab.understanding.knowledge import Variable, VariableKind
from tests.unit.training import test_going_back as going_back
from tests.unit.training.test_going_back import DIED_AT, FRAMES, SURE, Lab
from tests.unit.training.test_last_chance_in_a_go import LOG, SIX, said

lab = going_back.lab  # the practice game behind the real services (pytest fixture)

SECONDS = 5.0
EARNED = Rise(False, 2)
ANYWAY = Rise(True, 15)


def rise(var_id: str | None, before: int | None = 3, after: int | None = 4) -> RewardEarned:
    """A payment for that value going from ``before`` to ``after``."""
    return RewardEarned(
        7, 1.0, ("row",), GameEvent(EventKind.PROGRESS, 7, var_id, "", before, after)
    )


class Keeper:
    """Stands in for whoever keeps count: says which values are worth asking about,
    and notes what it is told."""

    def __init__(self, worth: Callable[[str], bool] = lambda var_id: True) -> None:
        self.is_worth = worth
        self.asked: list[str] = []
        self.told: list[tuple[str, bool]] = []

    def worth(self, var_id: str) -> bool:
        self.asked.append(var_id)
        return self.is_worth(var_id)

    def answered(self, var_id: str, anyway: bool) -> None:
        self.told.append((var_id, anyway))


class Asked:
    """Stands before ``rose_anyway`` in the runner: notes what each asking was given.
    It answers what the test says; told nothing, it lets the game answer."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch, lab: Lab, answer: object = "game") -> None:
        self.real = episode.rose_anyway
        self.lab = lab
        self.answer = answer
        self.calls: list[dict[str, object]] = []
        monkeypatch.setattr(episode, "rose_anyway", self)

    def __call__(
        self, connection: object, slot: str, trace: Sequence[int], space: object, **kw: object
    ) -> Rise | None:
        assert connection is self.lab.connection and space is self.lab.space
        self.calls.append({"slot": slot, "trace": tuple(trace), **kw})
        answer = self.answer
        if answer == "game":
            return self.real(connection, slot, trace, space, **kw)  # type: ignore[arg-type]
        if callable(answer):
            answer = answer(kw["value"])
        if isinstance(answer, Exception):
            raise answer
        assert answer is None or isinstance(answer, Rise)
        return answer


def learn(lab: Lab, name: str, address: int) -> Variable:
    """The AI reads that byte as progress."""
    value = lab.knowledge.propose(
        "ram", address, kind=VariableKind.PROGRESS, name=name, evidence=SURE
    )
    lab.bus.publish(DiscoveryUpdated((value,), (), (), ()))
    return value


def go(
    lab: Lab,
    keeper: Keeper | None,
    *,
    frames: int = toy_hopper.DYING_FRAMES,
    moments: Moments | None = None,
    **kw: object,
) -> EpisodeRunner:
    """A go about to start that keeps moments, may search once, and asks ``keeper``."""
    if lab.knowledge.best(VariableKind.LIVES) is None:
        lab.teach()
    lab.game.dying_frames = frames
    lab.game.reset()
    watch = lab.watch()
    lab.saved.clear()  # finding the player made saves of its own
    lab.loaded.clear()
    if keeper is not None:
        kw.update(rise_worth_asking=keeper.worth, rise_answered=keeper.answered)
    kw.setdefault("ways_per_go", 1)
    kw.setdefault("unstick_after", None)  # standing about is not what is under test
    kw.setdefault("way_seconds", SECONDS)
    r = lab.runner(moments=Moments(SIX) if moments is None else moments, watch=watch, **kw)
    r.begin()
    return r


def test_the_signals_note_what_was_paid_for_rising() -> None:
    bus = EventBus()
    signals = EpisodeSignals(bus)
    bus.publish(rise("a"))  # no go is running: nobody's
    signals.start()
    assert signals.take_rises() == ()
    # Each value once, in the order they first paid.
    for var_id in ("b", "a", "b", "c", "a"):
        bus.publish(rise(var_id))
    # Not rises: it fell, it stayed, the payment gives no numbers, or names no value.
    bus.publish(rise("fell", 4, 3))
    bus.publish(rise("stayed", 3, 3))
    bus.publish(rise("unknown", None, None))
    bus.publish(rise("half", None, 4))
    bus.publish(rise(None))
    assert signals.take_rises() == ("b", "a", "c")
    assert signals.take_rises() == ()  # taking clears them
    assert signals.take() == (10.0, 0)  # and every payment is counted as ever
    # A rise by more than one, from nothing, and of any kind of event.
    bus.publish(rise("far", 0, 200))
    for kind in (EventKind.UP, EventKind.SCORE_GAIN):
        bus.publish(RewardEarned(7, 1.0, ("row",), GameEvent(kind, 7, kind.value, "", 3, 4)))
    assert signals.take_rises() == ("far", "up", "score_gain")
    # A new go starts with none, and none is noted once the go is over.
    bus.publish(rise("a"))
    signals.start()
    assert signals.take_rises() == ()
    signals.stop()
    bus.publish(rise("a"))
    assert signals.take_rises() == ()


def test_the_runner_asks_from_the_newest_moment_eight_steps_back(
    lab: Lab, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    assert (RISE_BACK, RISE_MOST, EVERY) == (8, 40, 20)
    page = learn(lab, "Page", LAYOUT.x_page)
    keeper = Keeper()
    asked = Asked(monkeypatch, lab, EARNED)
    # It stands, steps left and jumps on the spot, in a pattern no 20 steps of which
    # are like the next 20: what it did since a moment is told apart from any other.
    kinds = (lab.move(), lab.move("left"), lab.move("a"))
    script = [kinds[(i * i) % 3 if i % 7 else 1] for i in range(70)]
    lab.agent.script.extend(script)
    r = go(lab, keeper)
    with caplog.at_level(logging.INFO, logger=LOG):
        for step in range(1, 71):
            lab.bus.publish(rise(page.id))  # paid on this step
            before = len(asked.calls)
            assert r.step() and r.steps == step
            if step < RISE_BACK:
                # Every moment is closer than eight steps: too little was done since
                # for holding another move to tell anything.
                assert len(asked.calls) == before, step
                continue
            assert len(asked.calls) == before + 1, step
            moment = (step - RISE_BACK) // EVERY  # the newest eight steps back or more
            assert asked.calls[-1] == {
                "slot": SIX[moment % len(SIX)],
                "trace": tuple(script[moment * EVERY : step]),
                "frames_per_step": FRAMES,
                "value": lab.knowledge.get(page.id),
                "seconds": SECONDS,
            }, step
            assert RISE_BACK <= len(asked.calls[-1]["trace"]) < RISE_BACK + EVERY  # type: ignore[arg-type]
    assert lab.game.state.lives == toy_hopper.START_LIVES  # no life went: a plain go
    times = 70 - RISE_BACK + 1
    assert len(asked.calls) == times
    # Each answer is told, and the present is saved before each asking and put back
    # after it.
    assert keeper.told == [(page.id, False)] * times
    assert lab.saved.count(NOW_SLOT) == times and lab.loaded == [NOW_SLOT] * times
    lines = [line for line in said(caplog) if "asked the game" in line]
    assert len(lines) == times
    assert lines[0] == (
        "Page went up at step 8 and was paid. The AI asked the game: played again from "
        "8 steps before with another move held, it stayed lower (2 plays), so that rise "
        "was earned."
    )
    assert "at step 28 " in lines[20] and "from 8 steps before" in lines[20]
    assert "at step 27 " in lines[19] and "from 27 steps before" in lines[19]
    r.finish()


def test_several_values_share_one_save_and_one_load(
    lab: Lab, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    page = learn(lab, "Page", LAYOUT.x_page)
    clock = learn(lab, "Clock", LAYOUT.clock + 1)
    keeper = Keeper()
    asked = Asked(monkeypatch, lab, lambda value: ANYWAY if value.id == clock.id else EARNED)
    stand = lab.move()
    lab.agent.script.extend([stand] * 30)
    r = go(lab, keeper)
    for _ in range(29):
        assert r.step()
    assert asked.calls == [] and keeper.asked == []  # nothing was paid for rising
    lab.bus.publish(rise(clock.id))
    lab.bus.publish(rise(page.id))
    lab.bus.publish(rise(clock.id))
    with caplog.at_level(logging.INFO, logger=LOG):
        assert r.step()
    assert keeper.asked == [clock.id, page.id]
    assert [call["value"] for call in asked.calls] == [
        lab.knowledge.get(clock.id),
        lab.knowledge.get(page.id),
    ]
    assert {(call["slot"], call["trace"]) for call in asked.calls} == {(SIX[1], (stand,) * 10)}
    assert keeper.told == [(clock.id, True), (page.id, False)]
    assert lab.saved.count(NOW_SLOT) == 1 and lab.loaded == [NOW_SLOT]
    assert lab.saved[-1] == NOW_SLOT  # saved after the step, before anything was played
    lines = [line for line in said(caplog) if "asked the game" in line]
    assert lines == [
        "Clock went up at step 30 and was paid. The AI asked the game: played again from "
        "10 steps before with each other move held, it went up just the same (15 plays). "
        "What goes up whatever is pressed is not progress, so that counts against it.",
        "Page went up at step 30 and was paid. The AI asked the game: played again from "
        "10 steps before with another move held, it stayed lower (2 plays), so that rise "
        "was earned.",
    ]
    # The next step was paid for nothing: nothing is asked.
    assert r.step()
    assert len(asked.calls) == 2 and lab.saved.count(NOW_SLOT) == 1
    r.finish()


def test_when_it_does_not_ask(lab: Lab, monkeypatch: pytest.MonkeyPatch) -> None:
    page = learn(lab, "Page", LAYOUT.x_page)
    asked = Asked(monkeypatch, lab, EARNED)
    stand = lab.move()

    def paid_each_step(r: EpisodeRunner, count: int, var_id: str = page.id) -> None:
        for _ in range(count):
            lab.bus.publish(rise(var_id))
            assert r.step()

    # Nobody to ask or to tell.
    lab.agent.script.extend([stand] * 30)
    r = go(lab, None)
    paid_each_step(r, 30)
    assert asked.calls == [] and NOW_SLOT not in lab.saved and lab.loaded == []
    r.finish()
    # No moments to go back to.
    keeper = Keeper()
    lab.agent.script.extend([stand] * 30)
    lab.game.reset()
    lab.saved.clear()
    r = lab.runner(
        rise_worth_asking=keeper.worth, rise_answered=keeper.answered, unstick_after=None
    )
    r.begin()
    paid_each_step(r, 30)
    assert asked.calls == [] and keeper.told == [] and lab.saved == [] and lab.loaded == []
    r.finish()
    # A value not worth asking about costs not a play and not a save.
    keeper = Keeper(lambda var_id: False)
    lab.agent.script.extend([stand] * 30)
    r = go(lab, keeper)
    paid_each_step(r, 30)
    assert keeper.asked == [page.id] * 30 and keeper.told == []
    assert asked.calls == [] and NOW_SLOT not in lab.saved and lab.loaded == []
    # A value the AI no longer knows is nobody's to ask about.
    keeper.is_worth = lambda var_id: True
    paid_each_step(r, 1, "gone")
    assert keeper.asked[30:] == [] and asked.calls == [] and NOW_SLOT not in lab.saved
    paid_each_step(r, 1)  # the same step for a value it knows: asked
    assert len(asked.calls) == 1 and keeper.told == [(page.id, False)]
    r.finish()
    # The newest moment far enough back is more than forty steps back: what was done
    # since is too much for one move held to stand in for.
    asked.calls.clear()
    keeper = Keeper()
    lab.agent.script.extend([stand] * 50)
    r = go(lab, keeper, moments=Moments(SIX, every=100))
    paid_each_step(r, RISE_MOST)
    assert [len(call["trace"]) for call in asked.calls] == list(range(RISE_BACK, RISE_MOST + 1))  # type: ignore[arg-type]
    paid_each_step(r, 10)
    assert len(asked.calls) == RISE_MOST - RISE_BACK + 1  # steps 41 to 50: none
    assert len(keeper.asked) == 50 and lab.saved.count(NOW_SLOT) == len(asked.calls)
    r.finish()


def test_a_rise_as_a_life_goes_is_not_asked_about(
    lab: Lab, monkeypatch: pytest.MonkeyPatch
) -> None:
    page = learn(lab, "Page", LAYOUT.x_page)
    asked = Asked(monkeypatch, lab, EARNED)
    keeper = Keeper()
    r = go(lab, keeper)  # it walks right, and the life goes at the gap
    for _ in range(DIED_AT - 2):
        assert r.step()
    lab.bus.publish(rise(page.id))
    assert r.step()  # the step before the life goes: asked as ever
    assert lab.game.state.lives == toy_hopper.START_LIVES
    assert [(call["slot"], len(call["trace"])) for call in asked.calls] == [(SIX[1], DIED_AT - 21)]  # type: ignore[arg-type]
    lab.bus.publish(rise(page.id))
    assert r.step()  # the life goes on this one, and the go looks for a way past
    assert r.steps == DIED_AT and r.ways_sought == 1
    assert len(asked.calls) == 1 and keeper.told == [(page.id, False)]
    # Not on a later step either: that rise is the loss's business.
    for _ in range(12):
        assert r.step()
    assert len(asked.calls) == 1 and keeper.asked == [page.id]
    r.finish()


def twin(script: Sequence[tuple[str, ...]], frames: int) -> list[bytes]:
    """The practice game played straight with those buttons: all of it, after each step."""
    game = ToyHopper(0)
    game.dying_frames = frames
    game.reset()
    out = []
    for buttons in script:
        for _ in range(FRAMES):
            game.step(buttons)
        out.append(game.snapshot())
    return out


@pytest.mark.parametrize("frames", [toy_hopper.DYING_FRAMES, 200])
def test_the_game_is_where_it_was(lab: Lab, monkeypatch: pytest.MonkeyPatch, frames: int) -> None:
    learn(lab, "Page", LAYOUT.x_page)
    clock = learn(lab, "Clock", LAYOUT.clock + 1)
    asked = Asked(monkeypatch, lab)  # the game itself answers
    keeper = Keeper()
    # It walks, jumps the gap, walks on and stands: the page and the clock both rise.
    buttons = [("right",)] * 20 + [("right", "a")] * 4 + [("right",)] * 8 + [()] * 108
    lab.agent.script.extend(lab.move(*held) for held in buttons)
    straight = twin(buttons, frames)
    r = go(lab, keeper, frames=frames)
    start = lab.observation.frame
    for step, expected in enumerate(straight, 1):
        assert r.step()
        # All of the game, its dice included, is what playing straight gives.
        assert lab.game.snapshot() == expected, step
    assert lab.game.state.lives == toy_hopper.START_LIVES
    # The game was asked, and it played: other moves, from saved moments.
    assert len(asked.calls) >= 2 and (clock.id, True) in keeper.told
    assert lab.loaded.count(NOW_SLOT) == len(asked.calls)
    assert len(lab.loaded) > 4 * len(asked.calls)
    # None of that was the go's: its steps, its frames, and what the learner saw.
    assert r.steps == len(buttons) == lab.agent.asked
    assert lab.observation.frame == start + FRAMES * len(buttons)
    assert len(lab.agent.learned) <= len(buttons)
    r.finish()


def test_no_answer_tells_nothing(
    lab: Lab, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    page = learn(lab, "Page", LAYOUT.x_page)
    clock = learn(lab, "Clock", LAYOUT.clock + 1)
    stand = lab.move()
    # Out of time for the first value: the second is still asked, and the game put back.
    asked = Asked(monkeypatch, lab, lambda value: None if value.id == page.id else ANYWAY)
    keeper = Keeper()
    lab.agent.script.extend([stand] * 12)
    r = go(lab, keeper)
    for _ in range(9):
        assert r.step()
    lab.bus.publish(rise(page.id))
    lab.bus.publish(rise(clock.id))
    with caplog.at_level(logging.INFO, logger=LOG):
        assert r.step()
    assert len(asked.calls) == 2 and keeper.told == [(clock.id, True)]
    assert lab.saved.count(NOW_SLOT) == 1 and lab.loaded == [NOW_SLOT]
    assert [line.split(" ")[0] for line in said(caplog) if "asked the game" in line] == ["Clock"]
    # The game itself has no answer where the value does not go up when played again
    # (this payment was made up: the page never moved).
    asked.answer = "game"
    before = lab.game.snapshot()
    lab.bus.publish(rise(page.id))
    assert r.step()
    assert len(asked.calls) == 3 and keeper.told == [(clock.id, True)]
    assert lab.loaded[-1] == NOW_SLOT and lab.loaded.count(NOW_SLOT) == 2
    after = ToyHopper(0)
    after.restore(before)
    for _ in range(FRAMES):
        after.step(())
    assert lab.game.snapshot() == after.snapshot()
    r.finish()


def test_a_dropped_link_ends_the_go(lab: Lab, monkeypatch: pytest.MonkeyPatch) -> None:
    page = learn(lab, "Page", LAYOUT.x_page)
    asked = Asked(monkeypatch, lab, EmulatorError("The link dropped."))
    keeper = Keeper()
    lab.agent.script.extend([lab.move()] * 12)
    r = go(lab, keeper)
    for _ in range(9):
        assert r.step()
    lab.bus.publish(rise(page.id))
    assert not r.step()
    assert len(asked.calls) == 1 and keeper.told == []
    assert not r.step()  # the go is over
    assert r.finish().ended_by == "disconnected"


def test_a_clock_paid_as_progress_is_dropped_and_real_progress_kept(
    lab: Lab, caplog: pytest.LogCaptureFixture
) -> None:
    """The practice game and the real services, nothing standing in."""
    page = learn(lab, "Page", LAYOUT.x_page)
    clock = learn(lab, "Clock", LAYOUT.clock + 1)
    paid: list[tuple[int, str | None]] = []
    discovery = lab.discovery
    # It stands still, then walks, jumps the gap and walks on, and stands again.
    buttons = [()] * 130 + [("right",)] * 20 + [("right", "a")] * 4 + [("right",)] * 8
    buttons += [()] * 38
    lab.agent.script.extend(lab.move(*held) for held in buttons)
    r = go(
        lab,
        None,
        frames=200,
        rise_worth_asking=discovery.rise_worth_asking,
        rise_answered=discovery.rise_answered,
    )
    lab.bus.subscribe(RewardEarned, lambda e: paid.append((r.steps + 1, e.event.variable_id)))

    def rises() -> list[tuple[int, str | None]]:
        return [(step, var_id) for step, var_id in paid if var_id in (page.id, clock.id)]

    with caplog.at_level(logging.INFO, logger=LOG):
        while r.steps < 130:
            assert r.step()
    # The clock's top byte went up every 256 frames (64 steps), and was paid for
    # nothing but the time going by. The game was asked both times.
    assert rises() == [(64, clock.id), (128, clock.id)]
    assert discovery.unearned.counts(clock.id) == (2, 2)
    assert lab.saved.count(NOW_SLOT) == 2
    lines = [line for line in said(caplog) if "asked the game" in line]
    assert [line.split(" went up at step ")[0] for line in lines] == ["Clock", "Clock"]
    assert all("it went up just the same (15 plays)." in line for line in lines)
    # The AI's next two looks at its guesses: doubted, then dropped.
    assert discovery.rise_worth_asking(page.id) and not discovery.rise_worth_asking(clock.id)
    assert [v.id for v in discovery.doubt_bouncing()] == [clock.id]
    assert not lab.knowledge.get(clock.id).rejected
    assert [v.id for v in discovery.doubt_bouncing()] == [clock.id]
    assert lab.knowledge.get(clock.id).rejected
    assert any(
        e.kind == ON_ITS_OWN and "went up whatever was pressed 2 of the 2 times" in e.detail.lower()
        for e in lab.knowledge.get(clock.id).evidence
    )
    # Across the gap the page turns: paid, asked about, earned, and kept. The clock
    # goes on ticking (step 192), and is paid nothing and asked about no more.
    caplog.clear()
    with caplog.at_level(logging.INFO, logger=LOG):
        while r.steps < 200:
            assert r.step()
    assert rises() == [(64, clock.id), (128, clock.id), (159, page.id)]
    assert lab.adapter.read_memory("ram", LAYOUT.clock + 1, 1) == bytes([3])
    assert discovery.unearned.counts(page.id) == (1, 0)
    assert discovery.unearned.counts(clock.id) == (2, 2) and lab.saved.count(NOW_SLOT) == 3
    lines = [line for line in said(caplog) if "asked the game" in line]
    assert len(lines) == 1 and lines[0].startswith("Page went up at step 159 and was paid.")
    assert lines[0].endswith("it stayed lower (2 plays), so that rise was earned.")
    assert not lab.knowledge.get(page.id).rejected and discovery.rise_worth_asking(page.id)
    assert lab.game.state.lives == toy_hopper.START_LIVES
    r.finish()


def test_misuse_is_refused(lab: Lab) -> None:
    keeper = Keeper()
    for kw in ({"rise_worth_asking": keeper.worth}, {"rise_answered": keeper.answered}):
        with pytest.raises(ConfigurationError, match="go together"):
            lab.runner(**kw)
    for worth, answered in (("yes", keeper.answered), (keeper.worth, 3), (True, False)):
        with pytest.raises(ConfigurationError, match="must be callable"):
            lab.runner(rise_worth_asking=worth, rise_answered=answered)
    lab.runner(rise_worth_asking=keeper.worth, rise_answered=keeper.answered)  # both: fine
