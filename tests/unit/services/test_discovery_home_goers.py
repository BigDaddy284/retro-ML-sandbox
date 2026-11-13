"""T-1021: values seen to come home in play are said once, and a landing is no restart."""

from __future__ import annotations

import logging

import pytest

from retro_ai_lab.core.events import EventBus
from retro_ai_lab.emulator.fake import FakeEmulatorAdapter
from retro_ai_lab.emulator.fake.toy_hopper import LAYOUT, ToyHopper
from retro_ai_lab.emulator.registry import default_registry
from retro_ai_lab.observation import MemorySnapshot
from retro_ai_lab.services import (
    ConnectionService,
    DiscoveryService,
    DiscoveryUpdated,
    ObservationService,
)
from retro_ai_lab.understanding import EventKind, Evidence, Variable, VariableKind
from retro_ai_lab.understanding.events import CYCLES, HOME_OFTEN

FRAMES = 4  # a step
SURE = [Evidence("test", "", 3.0)]  # Medium
QUIET = "retro_ai_lab.services.discovery"


class Game:
    """The practice game behind the real services, played a step at a time."""

    def __init__(self) -> None:
        self.bus = EventBus(on_error=lambda exc, event, handler: (_ for _ in ()).throw(exc))
        self.connection = ConnectionService(default_registry(), self.bus)
        self.observation = ObservationService(self.connection, self.bus)
        # No reviews and no experiments: what the test teaches is what is known.
        self.discovery = DiscoveryService(
            self.connection, self.observation, self.bus, review_every=10**9, experiment_every=None
        )
        self.connection.connect("fake", game="hopper", seed=0)
        adapter = self.connection.adapter
        assert isinstance(adapter, FakeEmulatorAdapter)
        self.adapter = adapter
        game = adapter.game
        assert isinstance(game, ToyHopper)
        self.game = game
        game.reset()
        self.buttons = {b.name: b for b in adapter.buttons()}
        knowledge = self.discovery.knowledge
        assert knowledge is not None
        self.knowledge = knowledge
        self.updates: list[DiscoveryUpdated] = []
        self.bus.subscribe(DiscoveryUpdated, self.updates.append)
        self.seen: list[MemorySnapshot] = []
        first = self.observation.observe()
        assert first is not None
        self.seen.append(first.snapshot)

    def teach(self, offset: int, kind: VariableKind, name: str) -> Variable:
        return self.knowledge.propose("ram", offset, kind=kind, name=name, evidence=SURE)

    def step(self, *held: str, steps: int = 1) -> None:
        for _ in range(steps):
            self.adapter.step(FRAMES, pressed=[self.buttons[name] for name in held])
            event = self.observation.advance(FRAMES)
            assert event is not None
            self.seen.append(event.snapshot)

    def jump(self) -> None:
        """Up and down again on the spot."""
        assert self.game.state.on_ground
        self.step("a", steps=3)
        assert not self.game.state.on_ground
        for _ in range(40):
            self.step()
            if self.game.state.on_ground:
                return
        raise AssertionError("it never came down")

    def marked(self, variable: Variable) -> list[Evidence]:
        return [e for e in self.knowledge.get(variable.id).evidence if e.kind == CYCLES]

    def close(self) -> None:
        self.connection.disconnect()


@pytest.fixture
def game() -> Game:
    made = Game()
    yield made  # type: ignore[misc]
    made.close()


def taught(game: Game) -> tuple[Variable, Variable]:
    """The fall speed taken for a score and "on the ground" for a switch, as the AI's
    own guessing has them, and three values that move once and stay away."""
    speed = game.teach(LAYOUT.vy, VariableKind.SCORE, "Score")
    ground = game.teach(LAYOUT.on_ground, VariableKind.FLAG, "Flag")
    game.teach(LAYOUT.x_low, VariableKind.PROGRESS, "Place")
    game.teach(LAYOUT.scroll, VariableKind.PROGRESS, "Camera")
    game.teach(LAYOUT.frame + 1, VariableKind.PROGRESS, "Time")
    game.step()  # the detector's first sight of them
    game.step(steps=66)  # the top of the frame count moves on
    game.step("right", steps=10)  # ... and so do its place and the camera
    return speed, ground


def restarts(game: Game) -> int:
    return sum(e.kind is EventKind.RESET for u in game.updates for e in u.events)


def said(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [r.getMessage() for r in caplog.records if "most other values" in r.getMessage()]


def test_home_goers_are_said_once_and_nothing_else_changes(
    game: Game, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO, logger=QUIET)
    speed, ground = taught(game)
    known = [v.to_dict() for v in game.knowledge]
    announced = len(game.updates)
    for _ in range(HOME_OFTEN - 1):
        game.jump()
    assert said(caplog) == []
    game.jump()
    # Three landings with the game playing on, and no restart called at any of them.
    assert restarts(game) == 0
    assert said(caplog) == [
        f"{speed.display_name}, {ground.display_name} went back to where they started "
        f"{HOME_OFTEN} times while most other values stayed where they were. That alone is "
        "no longer read as the game starting over."
    ]
    # Said once, however often it lands after that.
    for _ in range(3):
        game.jump()
    assert len(said(caplog)) == 1
    # Nothing the AI knows of any value was changed, and no value announced as changed.
    assert [v.to_dict() for v in game.knowledge] == known
    assert not any(u.added or u.changed for u in game.updates[announced:])
    assert not game.marked(speed) and not game.marked(ground)


def test_the_apps_own_restores_make_no_home_goer(
    game: Game, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO, logger=QUIET)
    taught(game)
    for _ in range(HOME_OFTEN + 2):
        # The same jumps as above, but on the way down the app loads a saved state
        # before every look at the memory (here the very moment it saved, so the
        # game is none the wiser): where the two values are when they are home
        # again, the app may have put them.
        assert game.game.state.on_ground
        game.step("a", steps=3)
        assert not game.game.state.on_ground
        for _ in range(40):
            game.adapter.step(FRAMES, pressed=[])
            game.adapter.save_state("t1021_now")
            game.adapter.load_state("t1021_now")
            event = game.observation.advance(FRAMES)
            assert event is not None and event.put_back
            if game.game.state.on_ground:
                break
        assert game.game.state.on_ground
    assert restarts(game) == 0
    assert said(caplog) == []  # five times home, every one of them after a load
    # Left to itself the game makes them home-goers as ever.
    for _ in range(HOME_OFTEN):
        game.jump()
    assert len(said(caplog)) == 1


def landing_with_nothing_else_moved(game: Game) -> int:
    """The AI learns more about the three other values (their records change, and the
    detector starts watching them anew), then jumps: the start of a go that plays the
    way it remembers, where nothing else has moved yet. The restarts called."""
    for offset in (LAYOUT.x_low, LAYOUT.scroll, LAYOUT.frame + 1):
        variable = game.knowledge.at("ram", offset)
        assert variable is not None
        game.knowledge.support(variable.id, Evidence("test", "Seen again.", 0.5))
    before = restarts(game)
    game.step()
    game.jump()
    return restarts(game) - before


def test_a_landing_is_no_restart_once_it_has_been_seen(
    game: Game, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO, logger=QUIET)
    # As it is today, and still is where the two have not been seen to come home:
    # the two home together at the landing is called the game starting over.
    other = Game()
    try:
        speed, ground = taught(other)
        assert landing_with_nothing_else_moved(other) == 1
        blamed = [e.variables for u in other.updates for e in u.events if e.variables]
        assert blamed == [(speed.id, ground.id)]
    finally:
        other.close()
    assert said(caplog) == []
    # Seen three times in play first: it is what those two do.
    taught(game)
    for _ in range(HOME_OFTEN):
        game.jump()
    assert len(said(caplog)) == 1 and restarts(game) == 0
    assert landing_with_nothing_else_moved(game) == 0
    for _ in range(5):
        game.jump()
    assert restarts(game) == 0
