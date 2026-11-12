"""T-1004: a go keeps saved moments and can be put back to one."""

from __future__ import annotations

import logging
from collections import deque
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from retro_ai_lab.core import ConfigurationError, EmulatorError, UnsupportedCapabilityError
from retro_ai_lab.core.events import EventBus
from retro_ai_lab.emulator.fake import toy_hopper
from retro_ai_lab.emulator.fake.adapter import FakeEmulatorAdapter
from retro_ai_lab.emulator.fake.toy_hopper import LAYOUT, ToyHopper
from retro_ai_lab.emulator.registry import default_registry
from retro_ai_lab.learning import ActionSpace, QLearningAgent, Settings, State, StateEncoder
from retro_ai_lab.learning.replay import Transition
from retro_ai_lab.services import (
    ConnectionService,
    DiscoveryService,
    DiscoveryUpdated,
    ObservationService,
    RewardService,
)
from retro_ai_lab.storage import RewardStore
from retro_ai_lab.training import EpisodeRunner
from retro_ai_lab.training.moments import Moment, Moments
from retro_ai_lab.training.vision import Mark, ScreenWatch
from retro_ai_lab.understanding import Evidence, Knowledge, VariableKind
from retro_ai_lab.understanding.player import find_player_in_game

SURE = [Evidence("test", "", 6.0)]
SLOTS = ("back0", "back1", "back2")
FRAMES = 4
DIED_AT = 36  # walking right from the start, the life goes on this step (1-based)


class Scripted(QLearningAgent):
    """A learner that makes the moves the test gives it, then keeps walking right."""

    def __init__(self, space: ActionSpace) -> None:
        super().__init__(space, Settings(epsilon=0.5), seed=1)
        self.script: deque[int] = deque()
        self.walk = space.index_of(["right"])
        self.asked = 0
        self.learned: list[tuple[str, int, float, str, bool, int]] = []

    def choose(self, state: State) -> int:
        self.asked += 1
        return self.script.popleft() if self.script else self.walk

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
        self.learned.append((state.key, action, reward, next_state.key, done, steps))
        return super().learn(state, action, reward, next_state, done, steps=steps)


@dataclass
class Lab:
    bus: EventBus
    connection: ConnectionService
    observation: ObservationService
    discovery: DiscoveryService
    adapter: FakeEmulatorAdapter
    game: ToyHopper
    space: ActionSpace
    agent: Scripted
    knowledge: Knowledge
    saved: list[str] = field(default_factory=list)
    loaded: list[str] = field(default_factory=list)

    def move(self, *buttons: str) -> int:
        return self.space.index_of(buttons)

    def teach(self, *, lives: bool = True) -> None:
        k = self.knowledge
        added = [
            k.propose("ram", LAYOUT.x_low, kind=VariableKind.POSITION, name="X", evidence=SURE)
        ]
        if lives:
            added.append(
                k.propose("ram", LAYOUT.lives, kind=VariableKind.LIVES, name="Lives", evidence=SURE)
            )
        self.bus.publish(DiscoveryUpdated(tuple(added), (), (), ()))

    def runner(self, **kw: object) -> EpisodeRunner:
        kw.setdefault("frames_per_step", FRAMES)
        kw.setdefault("max_steps", 400)
        return EpisodeRunner(
            self.connection,
            self.observation,
            self.bus,
            self.agent,
            StateEncoder(self.knowledge),
            **kw,  # type: ignore[arg-type]
        )

    def watch(self) -> ScreenWatch:
        names = [b.name for b in self.adapter.buttons() if b.name not in ("start", "select")]
        player = find_player_in_game(self.connection, names)
        assert player is not None and player.colours is not None
        self.game.reset()
        return ScreenWatch(player)


@pytest.fixture
def lab(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Lab]:
    bus = EventBus(on_error=lambda exc, event, handler: (_ for _ in ()).throw(exc))
    connection = ConnectionService(default_registry(), bus)
    observation = ObservationService(connection, bus)
    discovery = DiscoveryService(connection, observation, bus, experiment_every=None)
    RewardService(connection, discovery, bus, store=RewardStore(tmp_path / "r"))
    connection.connect("fake", game="hopper", seed=0)
    adapter = connection.adapter
    assert isinstance(adapter, FakeEmulatorAdapter)
    game = adapter.game
    assert isinstance(game, ToyHopper)
    knowledge = discovery.knowledge
    assert knowledge is not None
    space = ActionSpace.from_buttons(adapter.buttons())
    made = Lab(
        bus, connection, observation, discovery, adapter, game, space, Scripted(space), knowledge
    )
    save, load = adapter.save_state, adapter.load_state

    def saving(slot: str) -> None:
        made.saved.append(slot)
        save(slot)

    def loading(slot: str) -> None:
        made.loaded.append(slot)
        load(slot)

    monkeypatch.setattr(adapter, "save_state", saving)
    monkeypatch.setattr(adapter, "load_state", loading)
    yield made
    connection.disconnect()


def steps(r: EpisodeRunner, count: int) -> None:
    for _ in range(count):
        assert r.step()


def test_no_saves_without_moments_or_without_a_life_count(lab: Lab) -> None:
    lab.teach()
    plain = lab.runner()
    plain.begin()
    steps(plain, 45)
    without = plain.finish()
    assert lab.saved == [] and without.setbacks == 1
    where = lab.game.state
    learned = list(lab.agent.learned)

    # The same go with moments kept is the same go: saving changes nothing in the game.
    lab.game.reset()
    lab.agent.learned.clear()
    moments = Moments(SLOTS)
    kept = lab.runner(moments=moments)
    kept.begin()
    steps(kept, 45)
    assert kept.finish() == without
    assert lab.game.state == where and lab.agent.learned == learned
    assert lab.saved == ["back0", "back1", "back2"] and lab.loaded == []


def test_nothing_is_saved_while_no_life_or_health_value_is_known(lab: Lab) -> None:
    lab.teach(lives=False)
    moments = Moments(SLOTS)
    r = lab.runner(moments=moments)
    r.begin()
    steps(r, 45)
    r.finish()
    assert lab.saved == [] and len(moments) == 0


def test_a_go_saves_moments_and_remembers_its_moves(lab: Lab) -> None:
    lab.teach()
    jump, right = lab.move("a", "right"), lab.move("right")
    lab.agent.script.extend([right] * 5 + [jump] * 3)
    moments = Moments(SLOTS, every=10, lead=12)
    r = lab.runner(moments=moments)
    r.begin()
    assert lab.saved == []  # the save belongs to the first step, not to the go's start
    xs = []
    for _ in range(25):
        xs.append(lab.game.state.x)
        assert r.step()
    assert lab.saved == ["back0", "back1", "back2"]
    assert [m.step for m in moments.kept] == [0, 10, 20]
    assert all(m.mark is None for m in moments.kept)  # no screen watch: nothing to mark
    assert moments.steps == 25
    found = moments.before_loss()
    assert found is not None
    moment, since = found
    assert moment.step == 10 and since == (right,) * 15
    # Each save was made before that step's move: loading it is the game as it was then.
    for kept in moments.kept:
        lab.adapter.load_state(kept.slot)
        assert lab.game.state.x == xs[kept.step]
    r.finish()
    # A new go starts a new story.
    r.begin()
    assert len(moments) == 0 and moments.due()
    r.finish()


def test_going_back_puts_the_game_and_the_go_where_they_were(lab: Lab) -> None:
    lab.teach()
    moments = Moments(SLOTS, every=10, lead=12)
    r = lab.runner(moments=moments)
    r.begin()
    at: dict[int, tuple[object, State | None]] = {}
    for step in range(30):
        at[step] = (lab.game.state, r.state)
        assert r.step()
    middle = moments.kept[1]
    assert middle.step == 10
    reward, played = r.reward, r.steps
    state = r.go_back(middle)
    assert lab.loaded == [middle.slot]
    assert lab.game.state == at[10][0]
    assert state == at[10][1] and r.state == state
    assert moments.steps == 10 and moments.kept[-1] is middle
    # The go itself goes on: its steps and its reward are what was really played.
    assert r.running and r.steps == played and r.reward == reward and r.times_back == 1
    # And carries on from there as it did the first time.
    for step in range(10, 20):
        assert (lab.game.state, r.state) == at[step]
        assert r.step()
    assert r.steps == played + 10
    assert [m.step for m in moments.kept] == [0, 10]
    assert r.step()  # step 20 of the story: the next save is due
    assert [m.step for m in moments.kept] == [0, 10, 20]
    r.finish()


def test_the_jump_is_nobody_s_loss_gain_or_restart(lab: Lab) -> None:
    lab.teach()
    moments = Moments(SLOTS, every=10, lead=12)
    events: list[object] = []
    r = lab.runner(moments=moments)
    r.begin()
    steps(r, DIED_AT)
    assert lab.game.state.lives == toy_hopper.START_LIVES - 1
    assert r.last_reward == -10.0 and r.reward == -10.0
    found = moments.before_loss()
    assert found is not None
    moment, since = found
    assert moment.step == 20 and len(since) == DIED_AT - 20
    lab.bus.subscribe(object, events.append)
    r.go_back(moment)
    # The life is back in the game, and that is not something the game did.
    assert lab.game.state.lives == toy_hopper.START_LIVES
    assert [e for e in events if isinstance(e, DiscoveryUpdated) and e.events] == []
    assert r.step()
    assert r.last_reward == 0.0 and r.reward == -10.0 and r.running
    # Walking on it falls in again: one more life lost, counted once.
    steps(r, DIED_AT - 21)
    result = r.finish()
    assert result.setbacks == 2 and result.reward == -20.0 and result.ended_by == "stopped"


def test_steps_waiting_to_be_learned_are_settled_before_the_jump(lab: Lab) -> None:
    lab.teach()
    moments = Moments(SLOTS, every=10, lead=12)
    r = lab.runner(moments=moments, lookahead=8)
    r.begin()
    steps(r, DIED_AT)
    # Seven steps are still waiting for their lookahead: the last of them is the death.
    assert len(lab.agent.learned) == DIED_AT - 7
    after_death = r.state
    assert after_death is not None
    moment = moments.kept[-1]
    back_at = r.go_back(moment)
    assert len(lab.agent.learned) == DIED_AT
    settled = lab.agent.learned[-7:]
    assert all(done and next_key == after_death.key for _, _, _, next_key, done, _ in settled)
    # Each is paid the life it led to, as far off as it was, and nothing beyond it.
    gamma = lab.agent.settings.gamma
    assert [round(reward, 6) for _, _, reward, _, _, _ in settled] == [
        round(-10.0 * gamma**ahead, 6) for ahead in range(6, -1, -1)
    ]
    # What is learned after the jump starts from the moment and knows nothing of the
    # death behind it: three steps on (the life goes again at the sixth), none is paid
    # anything and none ends in or looks ahead to the state the death left.
    assert back_at != after_death
    steps(r, 3)
    assert len(lab.agent.learned) == DIED_AT  # they wait for their lookahead
    now = r.state
    assert now is not None and now != after_death
    r.finish()
    later = lab.agent.learned[DIED_AT:]
    assert len(later) == 3 and later[0][0] == back_at.key
    assert [(reward, next_key, done) for _, _, reward, next_key, done, _ in later] == [
        (0.0, now.key, False)
    ] * 3


def test_the_moves_given_are_played_next_and_learned_from(lab: Lab) -> None:
    lab.teach()
    jump, right = lab.move("a", "right"), lab.move("right")
    moments = Moments(SLOTS, every=10, lead=12)
    told: list[Transition] = []
    r = lab.runner(moments=moments, on_learned=told.append)
    r.begin()
    steps(r, DIED_AT)
    moment, since = moments.before_loss() or (None, ())
    assert moment is not None and moment.step == 20
    asked, heard = lab.agent.asked, len(told)

    # Its own moves again: the same story, to the same death, without asking the learner.
    r.go_back(moment, then=since)
    for move in since:
        assert r.step()
        assert r.last_action == move
    assert lab.agent.asked == asked
    assert lab.game.state.lives == toy_hopper.START_LIVES - 1 and r.last_reward == -10.0
    assert [t.action for t in told[heard:]] == list(since)
    assert all(t.source == "ai" for t in told[heard:])
    assert moments.steps == DIED_AT

    # Other moves: a jump at the edge gets across, and then the learner is asked again.
    moment, since = moments.before_loss() or (None, ())
    assert moment is not None and moment.step == 20
    way = [jump] * 8 + [right] * 9  # the save is at the very edge: jump at once
    r.go_back(moment, then=way)
    for move in way:
        assert r.step()
        assert r.last_action == move
        assert r.last_buttons == lab.space.buttons_for(move)
    assert lab.agent.asked == asked
    across = lab.game.state
    assert across.lives == toy_hopper.START_LIVES and across.mode == 0 and across.on_ground
    assert across.x > 240  # the far side of the first gap
    assert r.step()
    assert lab.agent.asked == asked + 1 and r.last_action == right
    assert moments.steps == 20 + len(way) + 1 and r.times_back == 2
    # The memory holds the story as it now stands: the way across, not the fall.
    newest = moments.kept[-1]
    assert newest.step == 30
    assert moments.before_loss() == (moment, (*way, right))  # the newest is not 12 back yet
    steps(r, 4)
    assert moments.before_loss() == (newest, (*way[10:], right, right, right, right, right))
    r.finish()


def test_the_screen_watch_goes_back_too(lab: Lab) -> None:
    lab.teach()
    watch = lab.watch()
    moments = Moments(SLOTS, every=10, lead=12)
    r = lab.runner(moments=moments, watch=watch)
    r.begin()
    steps(r, DIED_AT + 3)  # the life goes, and the watch sees the player back at the start
    assert all(isinstance(m.mark, Mark) for m in moments.kept)
    moment = moments.kept[-1]
    assert moment.step == 30
    counted = watch.mark()
    assert counted != moment.mark  # the death and the way back to the start moved it on
    r.go_back(moment)
    # What it had counted at the moment is what it has again.
    assert watch.mark() == moment.mark
    # Walking on to where it had already been pays nothing a second time.
    paid = 0
    for _ in range(4):
        assert r.step()
        paid += watch.take_new_ground()
    assert paid == 0
    r.finish()


def test_holes_and_consoles_without_save_states(
    lab: Lab, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    lab.teach()
    right = lab.move("right")
    moments = Moments(SLOTS, every=10, lead=12)
    r = lab.runner(moments=moments)
    r.begin()
    steps(r, 12)
    assert len(moments) == 2
    # A step somebody else drove is a hole in the list of moves.
    r.go_back(moments.kept[0], then=[right] * 5)
    assert r.observe_step(["right"], FRAMES)
    assert len(moments) == 0
    asked = lab.agent.asked
    assert r.step()
    assert lab.agent.asked == asked + 1  # the moves still to be played were dropped
    r.finish()

    # The runner's own unstick press is one too.
    lab.game.reset()
    lab.agent.script.extend([lab.move()] * 40)
    stuck = lab.runner(moments=Moments(SLOTS, every=3, lead=4), unstick_after=5)
    stuck.set_unstick_buttons(("start",))
    stuck.begin()
    kept = []
    for _ in range(8):
        assert stuck.step()
        kept.append((stuck.last_action, len(stuck.moments or ())))
    assert stuck.unsticks == 1
    hole = next(i for i, (action, _) in enumerate(kept) if action is None)
    assert kept[hole][1] == 0 and kept[hole - 1][1] > 0 and kept[hole + 1][1] == 1
    stuck.finish()

    # A console that cannot save: the go carries on without moments, and says so once.
    lab.game.reset()
    lab.saved.clear()

    def cannot(slot: str) -> None:
        raise UnsupportedCapabilityError("save_states")

    monkeypatch.setattr(lab.adapter, "save_state", cannot)
    none = Moments(SLOTS, every=3, lead=4)
    plain = lab.runner(moments=none)
    with caplog.at_level(logging.INFO):
        plain.begin()
        steps(plain, 10)
        plain.finish()
        plain.begin()
        steps(plain, 5)
        plain.finish()
    assert len(none) == 0
    said = [rec.getMessage() for rec in caplog.records if "save" in rec.getMessage()]
    assert len(said) == 1 and "can't" in said[0]

    # The link dropping at a save ends the go like any other dropped link.
    def gone(slot: str) -> None:
        raise EmulatorError("the link dropped")

    monkeypatch.setattr(lab.adapter, "save_state", gone)
    dropped = lab.runner(moments=Moments(SLOTS, every=3, lead=4))
    dropped.begin()
    assert not dropped.step()
    assert dropped.finish().ended_by == "disconnected"


def test_misuse_is_refused(lab: Lab) -> None:
    lab.teach()
    moments = Moments(SLOTS, every=10, lead=12)
    r = lab.runner(moments=moments, max_steps=15)
    stray = Moment(slot="back0", step=0, mark=None)
    with pytest.raises(ConfigurationError):
        r.go_back(stray)  # no go is running
    r.begin()
    steps(r, 12)
    first = moments.kept[0]
    where, state, learned = lab.game.state, r.state, len(lab.agent.learned)
    for bad in (
        lambda: r.go_back(stray),  # not one of those kept
        lambda: r.go_back("back0"),  # type: ignore[arg-type]
        lambda: r.go_back(first, then=[len(lab.space)]),
        lambda: r.go_back(first, then=[-1]),
        lambda: r.go_back(first, then=[1.5]),  # type: ignore[list-item]
    ):
        with pytest.raises(ConfigurationError):
            bad()
    # A refused call changed nothing.
    assert lab.loaded == [] and lab.game.state == where and r.state == state
    assert len(lab.agent.learned) == learned and len(moments) == 2 and r.times_back == 0
    # A go that is over cannot go back, and neither can a runner that keeps no moments.
    while r.step():
        pass
    with pytest.raises(ConfigurationError):
        r.go_back(first)
    r.finish()
    plain = lab.runner()
    plain.begin()
    with pytest.raises(ConfigurationError):
        plain.go_back(first)
    assert plain.moments is None and plain.times_back == 0
    plain.finish()
