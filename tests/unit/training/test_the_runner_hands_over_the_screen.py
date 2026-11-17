"""T-1055: a learner that watches the screen is handed it with every situation."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import pytest

from retro_ai_lab.core import Capabilities, EmulatorError, Frame
from retro_ai_lab.core.events import EventBus
from retro_ai_lab.emulator.fake.adapter import FakeEmulatorAdapter
from retro_ai_lab.emulator.fake.toy_hopper import LAYOUT
from retro_ai_lab.emulator.registry import default_registry
from retro_ai_lab.learning import (
    NO_MOVE,
    PICTURE_SIZE,
    ActionSpace,
    QLearningAgent,
    ScreenConfig,
    ScreenLearner,
    Settings,
    Sight,
    State,
    StateEncoder,
    shrink,
)
from retro_ai_lab.services import (
    ConnectionService,
    DiscoveryService,
    DiscoveryUpdated,
    ExperimentHarness,
    ObservationService,
    RewardService,
)
from retro_ai_lab.storage import RewardStore
from retro_ai_lab.training import EpisodeRunner
from retro_ai_lab.training.moments import Moments

FRAMES = 4
SURE_SLOTS = ("back0", "back1", "back2")
NO_SCREEN = "This emulator gives no picture of the screen"


class Recording(QLearningAgent):
    """A table learner that keeps what it is handed, and what the screen showed then."""

    def __init__(self, space: ActionSpace, game: Any) -> None:
        super().__init__(space, Settings(epsilon=0.5), seed=1)
        self.game = game
        self.always: int | None = None  # a move to make whatever it thinks
        self.asked: list[tuple[State, bytes, int]] = []  # the state, the screen then, its move
        self.learned: list[tuple[State, int, State, bool, int]] = []

    def choose(self, state: State) -> int:
        action = super().choose(state) if self.always is None else self.always
        self.asked.append((state, shrink(self.game.render()), action))
        return action

    def learn(  # type: ignore[override]
        self,
        state: State,
        action: int,
        reward: float,
        next_state: State,
        done: bool = False,
        *,
        steps: int = 1,
    ) -> float:
        self.learned.append((state, action, next_state, done, steps))
        return super().learn(state, action, reward, next_state, done, steps=steps)


class Watching(Recording):
    wants_picture = True


@dataclass
class Lab:
    bus: EventBus
    connection: ConnectionService
    observation: ObservationService
    discovery: DiscoveryService
    shots: list[int] = field(default_factory=list)  # one entry a screen the emulator gave

    def connect(self, game: str, **kw: Any) -> FakeEmulatorAdapter:
        self.connection.connect("fake", game=game, seed=3, **kw)
        adapter = self.connection.adapter
        assert isinstance(adapter, FakeEmulatorAdapter)
        real = adapter._do_capture_frame

        def capture() -> Frame:
            self.shots.append(self.observation.frame)
            return real()

        adapter._do_capture_frame = capture  # type: ignore[method-assign]
        return adapter

    def runner(self, agent: Any, **kw: Any) -> EpisodeRunner:
        knowledge = self.discovery.knowledge
        assert knowledge is not None
        kw.setdefault("frames_per_step", FRAMES)
        kw.setdefault("max_steps", 30)
        return EpisodeRunner(
            self.connection, self.observation, self.bus, agent, StateEncoder(knowledge), **kw
        )


@pytest.fixture
def lab(tmp_path: Path) -> Iterator[Lab]:
    bus = EventBus(on_error=lambda exc, event, handler: (_ for _ in ()).throw(exc))
    connection = ConnectionService(default_registry(), bus)
    observation = ObservationService(connection, bus)
    discovery = DiscoveryService(connection, observation, bus, experiment_every=None)
    RewardService(connection, discovery, bus, store=RewardStore(tmp_path / "r"))
    yield Lab(bus, connection, observation, discovery)
    if connection.adapter is not None:
        connection.disconnect()


def play(r: EpisodeRunner) -> None:
    while r.step():
        pass


@pytest.mark.parametrize("game", ["racer", "hopper"])
def test_every_situation_shows_the_screen_then_the_one_before_and_the_move_made(
    lab: Lab, game: str
) -> None:
    adapter = lab.connect(game)
    agent = Watching(ActionSpace.from_buttons(adapter.buttons()), adapter.game)
    r = lab.runner(agent)
    first = r.begin()
    start = shrink(adapter.game.render())
    assert first.sight == Sight(start, start, NO_MOVE)  # nothing went before
    play(r)
    steps = len(agent.asked)
    assert steps >= 20
    states = [state for state, _, _ in agent.asked]
    assert states[0] is first
    for i, (state, screen, _) in enumerate(agent.asked):
        sight = state.sight
        assert sight is not None and len(sight.now) == PICTURE_SIZE
        assert sight.now == screen  # the screen as it was when it was asked
        if i:
            assert sight.before == states[i - 1].sight.now  # type: ignore[union-attr]
            assert sight.held == agent.asked[i - 1][2]
    # The screen does change, and so does the move: these are not one picture throughout.
    assert len({state.sight.now for state in states}) > 5  # type: ignore[union-attr]
    assert len({state.sight.held for state in states}) > 2  # type: ignore[union-attr]
    # One screen a step and one at the start, and no more.
    assert len(lab.shots) == steps + 1
    last = r.state
    assert last is not None and last.sight is not None
    assert last.sight.now == shrink(adapter.game.render())
    assert last.sight.before == states[-1].sight.now  # type: ignore[union-attr]
    assert last.sight.held == agent.asked[-1][2]
    assert r.finish().steps == steps
    # Every step is learnt from the very states it was asked about, with what they showed.
    assert len(agent.learned) == steps
    for i, (state, action, next_state, _done, spanned) in enumerate(agent.learned):
        assert state is states[i] and action == agent.asked[i][2]
        after = states[i + spanned] if i + spanned < steps else last
        assert next_state is after and next_state.sight is not None


def test_a_new_go_begins_with_no_picture_before_it(lab: Lab) -> None:
    adapter = lab.connect("hopper")
    space = ActionSpace.from_buttons(adapter.buttons())
    agent = Watching(space, adapter.game)
    agent.always = space.index_of(["right"])  # walk on, so the screen ends somewhere else
    harness = ExperimentHarness(lab.connection)
    harness.checkpoint()
    r = lab.runner(agent, max_steps=12, harness=harness)
    opening = r.begin().sight
    play(r)
    ended = r.state
    assert ended is not None and ended.sight is not None and opening is not None
    assert r.finish().steps == 12
    again = r.begin()  # the game is put back to where the goes start
    start = shrink(adapter.game.render())
    assert again.sight == Sight(start, start, NO_MOVE) == opening
    assert start != ended.sight.now  # not the last go's last screen
    assert len(lab.shots) == 12 + 2  # the new start is looked at afresh
    # Carried on without putting the game back, the go still begins with nothing before.
    assert r.step()
    moved = r.state
    r.finish()
    carried = r.begin(restore=False)
    assert moved is not None and moved.sight is not None and carried.sight is not None
    assert carried.sight == Sight(moved.sight.now, moved.sight.now, NO_MOVE)
    assert moved.sight.before != moved.sight.now
    r.finish()


def test_put_back_to_a_saved_moment_it_shows_that_screen_and_nothing_before(lab: Lab) -> None:
    adapter = lab.connect("hopper")
    knowledge = lab.discovery.knowledge
    assert knowledge is not None
    from retro_ai_lab.understanding import Evidence, VariableKind

    sure = [Evidence("test", "", 6.0)]
    added = (
        knowledge.propose("ram", LAYOUT.x_low, kind=VariableKind.POSITION, name="X", evidence=sure),
        knowledge.propose("ram", LAYOUT.lives, kind=VariableKind.LIVES, name="L", evidence=sure),
    )
    lab.bus.publish(DiscoveryUpdated(added, (), (), ()))
    space = ActionSpace.from_buttons(adapter.buttons())
    agent = Watching(space, adapter.game)
    agent.configure(Settings(epsilon=0.0, epsilon_min=0.0))
    moments = Moments(SURE_SLOTS, every=5, lead=6)
    r = lab.runner(agent, moments=moments, max_steps=200)
    r.begin()
    for _ in range(20):
        assert r.step()
    now = r.state
    assert now is not None and now.sight is not None and len(moments.kept) >= 2
    moment = moments.kept[0]
    shots = len(lab.shots)
    back = r.go_back(moment)
    there = shrink(adapter.game.render())
    assert back is r.state and back.sight == Sight(there, there, NO_MOVE)
    assert there != now.sight.now  # the game is somewhere else, and that is what it shows
    assert len(lab.shots) == shots + 1
    # A move played from there is a step like any other: the picture before is this one.
    jump = space.index_of(["a", "right"])
    assert r.go_back(moment, then=[jump]).sight == Sight(there, there, NO_MOVE)
    asked = len(agent.asked)
    assert r.step()
    after = r.state
    assert after is not None and len(agent.asked) == asked  # the move was the list's
    assert after.sight == Sight(shrink(adapter.game.render()), there, jump)
    r.finish()


def test_a_step_somebody_else_played_shows_the_screen_after_it(lab: Lab) -> None:
    adapter = lab.connect("racer")
    space = ActionSpace.from_buttons(adapter.buttons())
    agent = Watching(space, adapter.game)
    r = lab.runner(agent)
    first = r.begin(restore=False)
    assert first.sight is not None
    held = frozenset(b for b in adapter.buttons() if b.name == "up")
    for _ in range(3):
        adapter.step(FRAMES, pressed=held)
    lab.observation.advance(3 * FRAMES)
    assert r.observe_step(["up"], 3 * FRAMES)
    state = r.state
    assert state is not None
    up = space.index_of(["up"])
    assert state.sight == Sight(shrink(adapter.game.render()), first.sight.now, up)
    assert state.sight.now != first.sight.now  # the road moved
    assert len(lab.shots) == 2  # the start, and this step
    r.finish()
    assert [(s, a, n) for s, a, n, _, _ in agent.learned] == [(first, up, state)]
    assert agent.learned[0][0].sight is first.sight and agent.learned[0][2].sight is state.sight


@pytest.mark.parametrize("game", ["racer", "hopper"])
def test_a_learner_that_does_not_watch_is_handed_no_picture_and_no_screen_is_asked_for(
    lab: Lab, game: str
) -> None:
    adapter = lab.connect(game)
    agent = Recording(ActionSpace.from_buttons(adapter.buttons()), adapter.game)
    r = lab.runner(agent)
    first = r.begin()
    assert first.sight is None
    play(r)
    last = r.state
    steps = r.finish().steps
    assert steps >= 20 and len(agent.learned) == steps == len(agent.asked)
    assert all(state.sight is None for state, _, _ in agent.asked)
    assert all(s.sight is None and n.sight is None for s, _, n, _, _ in agent.learned)
    assert last is not None and last.sight is None
    assert lab.shots == []  # the go costs what it cost before


def test_the_go_is_the_same_go_watched_or_not(lab: Lab) -> None:
    """The picture rides on the state; the state, the moves and the game are as they were."""
    told: list[list[tuple[str, int, str, bool, int]]] = []
    ends = []
    for kind in (Recording, Watching):
        adapter = lab.connect("hopper")
        agent = kind(ActionSpace.from_buttons(adapter.buttons()), adapter.game)
        r = lab.runner(agent)
        r.begin()
        play(r)
        result = r.finish()
        told.append([(s.key, a, n.key, d, k) for s, a, n, d, k in agent.learned])
        ends.append((result, adapter.game.state, [a for _, _, a in agent.asked]))
        lab.connection.disconnect()
    assert told[0] == told[1] and ends[0] == ends[1] and len(told[0]) >= 20


def test_an_emulator_with_no_screen_gives_no_picture_and_says_so_once(
    lab: Lab, caplog: pytest.LogCaptureFixture
) -> None:
    adapter = lab.connect("racer", capabilities=replace(Capabilities.all(), screen_capture=False))
    space = ActionSpace.from_buttons(adapter.buttons())
    agent = Watching(space, object())  # nothing to render: choose() is not reached here
    agent.choose = lambda state: 0  # type: ignore[method-assign]
    caplog.set_level(logging.WARNING)
    r = lab.runner(agent, max_steps=8)
    first = r.begin()
    assert first.sight is None
    play(r)
    assert r.state is not None and r.state.sight is None
    assert r.finish().steps == 8 == len(agent.learned)  # the go is played all the same
    assert all(s.sight is None and n.sight is None for s, _, n, _, _ in agent.learned)
    assert lab.shots == [] and caplog.text.count(NO_SCREEN) == 1
    assert "a learner that watches the screen has nothing to learn from here." in caplog.text
    r.begin()
    r.finish()
    assert caplog.text.count(NO_SCREEN) == 1  # said once, not every go
    # A learner that does not watch is told nothing.
    caplog.clear()
    plain = Recording(space, object())
    plain.choose = lambda state: 0  # type: ignore[method-assign]
    quiet = lab.runner(plain, max_steps=3)
    quiet.begin()
    play(quiet)
    quiet.finish()
    assert NO_SCREEN not in caplog.text


def test_the_screen_network_counts_what_an_emulator_with_no_screen_cannot_show(lab: Lab) -> None:
    adapter = lab.connect("racer", capabilities=replace(Capabilities.all(), screen_capture=False))
    space = ActionSpace.from_buttons(adapter.buttons())
    agent = ScreenLearner(space, seed=1, config=ScreenConfig(hidden=(8,)))
    r = lab.runner(agent, max_steps=8)
    r.begin()
    play(r)
    assert r.finish().steps == 8
    assert agent.unseen == 8 and agent.updates == 0 and agent.known == 0


def test_a_screen_that_cannot_be_had_for_one_step_leaves_that_step_without_a_picture(
    lab: Lab, monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter = lab.connect("racer")
    space = ActionSpace.from_buttons(adapter.buttons())
    agent = Watching(space, adapter.game)
    r = lab.runner(agent)
    first = r.begin(restore=False)
    held = frozenset(b for b in adapter.buttons() if b.name == "up")
    adapter.step(FRAMES, pressed=held)
    lab.observation.advance(FRAMES)
    real = adapter.capture_frame

    def broken() -> Frame:
        raise EmulatorError("The emulator did not answer.")

    monkeypatch.setattr(adapter, "capture_frame", broken)
    assert r.observe_step(["up"], FRAMES)
    blind = r.state
    assert blind is not None and blind.sight is None and blind.parts == first.parts
    monkeypatch.setattr(adapter, "capture_frame", real)
    adapter.step(FRAMES, pressed=held)
    lab.observation.advance(FRAMES)
    assert r.observe_step(["up"], FRAMES)
    state = r.state
    assert state is not None and state.sight is not None
    now = shrink(adapter.game.render())
    # The picture before the missing one is not passed off as the one a step earlier.
    assert state.sight == Sight(now, now, space.index_of(["up"]))
    r.finish()


def test_the_screen_network_plays_a_go_and_remembers_every_step_with_its_pictures(
    lab: Lab,
) -> None:
    adapter = lab.connect("hopper")
    space = ActionSpace.from_buttons(adapter.buttons())
    small = ScreenConfig(hidden=(8,), batch_size=4, warmup=8, target_sync=10, memory=100)
    agent = ScreenLearner(space, Settings(epsilon=0.5), seed=1, config=small)
    r = lab.runner(agent)
    r.begin()
    play(r)
    steps = r.finish().steps
    assert steps >= 20
    assert agent.updates == steps == agent.known and agent.unseen == 0 and agent.trained > 0
    assert len(lab.shots) == steps + 1
    pictures = {step[0].now for step in agent._memory} | {step[3].now for step in agent._memory}
    assert len(pictures) > 5 and all(len(p) == PICTURE_SIZE for p in pictures)
    for seen, action, _reward, after, _done, spanned in agent._memory:
        assert 0 <= action < len(space) and spanned >= 1
        assert after.held in range(len(space))  # a move was made to get there
        assert seen.held in (NO_MOVE, *range(len(space)))
