"""T-1057: "is the game listening" is not fooled by buttons that undo each other.

On a game where the pad only walks something left and right at one speed, holding
left and then right puts it back where it was. The check used to look only at where
the memory ended up after every button had its turn, and took such a game for one
that ignores the pad.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pytest

from retro_ai_lab.core.events import EventBus
from retro_ai_lab.emulator.fake import FakeEmulatorAdapter
from retro_ai_lab.emulator.fake.toy_racer import _BUTTON_BIT, BUTTON_NAMES, LAYOUT, ToyRacer
from retro_ai_lab.emulator.registry import default_registry
from retro_ai_lab.learning import Q_TABLE, Settings
from retro_ai_lab.services import ConnectionService, DiscoveryService, ObservationService
from retro_ai_lab.storage import BrainStore
from retro_ai_lab.training import TrainingService
from retro_ai_lab.training import service as training_service
from retro_ai_lab.understanding import find_wake_button, inputs_matter, probe_inputs
from retro_ai_lab.understanding.wakeup import DEFAULT_PER_BUTTON, DEFAULT_SETTLE

MIDDLE = 100  # where the walker starts: far from either end of its floor
PACE = 2  # what it walks in a frame
NOT_LISTENING = "The game is not listening to the AI; trying each button to get it going."
PAD = [n for n in BUTTON_NAMES if n != "start"]


class SideStepper(ToyRacer):
    """A one-screen game in the practice racer's memory: the pad walks its place left
    and right at one speed and does nothing else. Nothing scrolls and nothing speeds
    up, so left for a while and then right for as long puts it back where it was.

    ``deaf`` ignores the pad altogether (the byte that echoes the pad still does);
    ``restless`` keeps a byte that counts the frames ever played, which a save state
    does not put back: it differs from one run to the next whatever is held.
    """

    def __init__(self, seed: int = 0, *, deaf: bool = False, restless: bool = False) -> None:
        self.deaf, self.restless, self.played = deaf, restless, 0
        super().__init__(seed)

    def reset(self) -> None:
        super().reset()
        self._ram[LAYOUT.position] = MIDDLE

    def step(self, pressed: Iterable[str] = ()) -> None:
        held = frozenset(pressed)
        ram = self._ram
        ram[LAYOUT.input] = sum(_BUTTON_BIT[name] for name in held)
        if ram[LAYOUT.game_over]:
            if "start" in held:
                self._restart_after_game_over()
        elif not self.deaf:
            want = ("right" in held) - ("left" in held)
            ram[LAYOUT.position] = max(0, min(2 * MIDDLE, ram[LAYOUT.position] + PACE * want))
        self.played += 1
        if self.restless:
            ram[LAYOUT.lap] = self.played & 0xFF
        self._set_u16(LAYOUT.frame, (self._u16(LAYOUT.frame) + 1) & 0xFFFF)
        ram[LAYOUT.timer] = (ram[LAYOUT.timer] + 1) & 0xFF
        self._write_noise()


@pytest.fixture
def connection() -> ConnectionService:
    bus = EventBus(on_error=lambda exc, event, handler: (_ for _ in ()).throw(exc))
    connection = ConnectionService(default_registry(), bus)
    connection.connect("fake", seed=3)
    return connection


def adapter_of(connection: ConnectionService) -> FakeEmulatorAdapter:
    adapter = connection.adapter
    assert isinstance(adapter, FakeEmulatorAdapter)
    return adapter


def stepper(connection: ConnectionService, **kind: bool) -> SideStepper:
    game = SideStepper(3, **kind)
    adapter_of(connection)._game = game
    return game


def played(connection: ConnectionService, monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    """Every run of frames the adapter is asked for: (the buttons held, the frames)."""
    seen: list[Any] = []
    real = FakeEmulatorAdapter.step

    def spy(self: FakeEmulatorAdapter, frames: int, **options: Any) -> Any:
        seen.append((tuple(b.name for b in options.get("pressed", ())), frames))
        return real(self, frames, **options)

    monkeypatch.setattr(FakeEmulatorAdapter, "step", spy)
    return seen


def together(names: list[str], per_button: int, settle: int) -> list[Any]:
    """The runs the check has always made: nothing held twice over, then each button
    in turn in one run, and the frames in which the pad is let go."""
    whole = per_button * len(names) + settle
    return [((), whole), ((), whole), *[((n,), per_button) for n in names], ((), settle)]


def alone(names: list[str], per_button: int, settle: int) -> list[Any]:
    """The runs it makes besides where those showed nothing: nothing held twice over
    for one button's frames, then each button by itself."""
    runs: list[Any] = [((), per_button + settle), ((), per_button + settle)]
    for name in names:
        runs += [((name,), per_button), ((), settle)]
    return runs


# ------------------------------------------------------------ the answer


def test_the_stand_in_is_a_game_whose_left_and_right_undo_each_other(
    connection: ConnectionService,
) -> None:
    game = stepper(connection)
    assert game.state.position == MIDDLE
    for name in ("left",) * 5 + ("right",) * 5 + ("up", "down", "a", "b"):
        game.step([name])
    assert game.state.position == MIDDLE  # back where it was, by another way
    game.step(["left"])
    assert game.state.position == MIDDLE - PACE
    assert game.state.speed == 0 and game.state.score == 0 and game.state.lives == 3


@pytest.mark.parametrize(
    "names",
    [
        ["left", "right"],
        ["right", "left"],
        PAD,
        list(BUTTON_NAMES),
        ["up", "a", "left", "b", "right", "down"],
    ],
)
def test_buttons_that_undo_each_other_still_matter(
    connection: ConnectionService, names: list[str]
) -> None:
    game = stepper(connection)
    before = game.snapshot()
    assert inputs_matter(connection, names)
    assert game.snapshot() == before  # the game is put back where it was
    assert probe_inputs(connection, names, per_button=2, settle=3).matters
    assert game.snapshot() == before


@pytest.mark.parametrize("names", [["left"], ["right"], ["up", "left"], ["right", "a", "b"]])
def test_one_button_that_matters_is_enough_wherever_it_comes(
    connection: ConnectionService, names: list[str]
) -> None:
    stepper(connection)
    assert inputs_matter(connection, names)


def test_buttons_that_do_nothing_still_do_not_matter(connection: ConnectionService) -> None:
    game = stepper(connection)
    before = game.snapshot()
    # Up, down, A and B do nothing in this game, together or one at a time.
    assert not inputs_matter(connection, ["up", "down", "a", "b"])
    assert game.snapshot() == before


def test_a_game_that_only_echoes_the_pad_is_still_not_listening(
    connection: ConnectionService,
) -> None:
    game = stepper(connection, deaf=True)
    before = game.snapshot()
    game.step(["left"])
    assert game.ram[LAYOUT.input] == _BUTTON_BIT["left"]  # the pad is echoed, all the same
    game.restore(before)
    assert not inputs_matter(connection, PAD)
    assert not inputs_matter(connection, ["left", "right"], per_button=1, settle=1)
    assert game.snapshot() == before


def test_with_no_frames_to_let_go_in_the_echo_of_a_button_alone_would_count(
    connection: ConnectionService,
) -> None:
    # What the frames at the end of a button's own run are for: without them the byte
    # that echoes the pad differs, and a game that ignores the pad would seem to hear.
    stepper(connection, deaf=True)
    assert inputs_matter(connection, ["left", "right"], settle=0)


def test_what_differs_from_run_to_run_whatever_is_held_does_not_count(
    connection: ConnectionService,
) -> None:
    game = stepper(connection, deaf=True, restless=True)
    game.step([])
    one = game.snapshot()
    game.step([])
    game.restore(one)
    game.step([])
    assert game.ram[LAYOUT.lap] == 3  # the save state did not put that byte back
    assert not inputs_matter(connection, PAD)
    # ... and where the pad is heard, that byte does not hide it.
    stepper(connection, restless=True)
    assert inputs_matter(connection, PAD)


# ------------------------------------------------------- what is played


def test_where_the_buttons_together_change_something_nothing_more_is_played(
    connection: ConnectionService, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The practice racer in play: up speeds it up, so the run of every button in turn
    already ends somewhere else. The runs are the ones the check has always made."""
    seen = played(connection, monkeypatch)
    assert inputs_matter(connection, PAD)
    assert seen == together(PAD, DEFAULT_PER_BUTTON, DEFAULT_SETTLE)
    del seen[:]
    assert probe_inputs(connection, ["up", "b"], per_button=3, settle=4).matters
    assert seen == together(["up", "b"], 3, 4)


def test_where_they_change_nothing_each_button_is_asked_alone(
    connection: ConnectionService, monkeypatch: pytest.MonkeyPatch
) -> None:
    stepper(connection)
    seen = played(connection, monkeypatch)
    assert inputs_matter(connection, PAD)
    assert seen == [
        *together(PAD, DEFAULT_PER_BUTTON, DEFAULT_SETTLE),
        *alone(PAD, DEFAULT_PER_BUTTON, DEFAULT_SETTLE),
    ]
    del seen[:]
    # Every button is asked, also after one has answered: what each drives is wanted.
    assert probe_inputs(connection, ["left", "right", "a"], per_button=2, settle=3).matters
    assert seen == [*together(["left", "right", "a"], 2, 3), *alone(["left", "right", "a"], 2, 3)]


def test_a_game_that_ignores_the_pad_is_asked_button_by_button_and_put_back(
    connection: ConnectionService, monkeypatch: pytest.MonkeyPatch
) -> None:
    game = stepper(connection, deaf=True)
    before = game.snapshot()
    seen = played(connection, monkeypatch)
    assert not inputs_matter(connection, ["left", "right"], per_button=2, settle=3)
    assert seen == [*together(["left", "right"], 2, 3), *alone(["left", "right"], 2, 3)]
    assert game.snapshot() == before


def test_the_game_is_put_back_when_asking_a_button_alone_goes_wrong(
    connection: ConnectionService, monkeypatch: pytest.MonkeyPatch
) -> None:
    game = stepper(connection)
    before = game.snapshot()
    real = FakeEmulatorAdapter.step
    runs = 0

    def failing(self: FakeEmulatorAdapter, frames: int, **options: Any) -> Any:
        nonlocal runs
        runs += 1
        if runs > len(together(PAD, DEFAULT_PER_BUTTON, DEFAULT_SETTLE)) + 3:
            raise RuntimeError("the link went quiet")
        return real(self, frames, **options)

    monkeypatch.setattr(FakeEmulatorAdapter, "step", failing)
    with pytest.raises(RuntimeError, match="the link went quiet"):
        inputs_matter(connection, PAD)
    assert game.snapshot() == before


# ------------------------------------------------- what moves by itself


def test_what_a_single_button_drives_is_not_called_self_moving(
    connection: ConnectionService,
) -> None:
    stepper(connection)
    probe = probe_inputs(connection, PAD)
    assert probe.matters
    offsets = {offset for region, offset in probe.self_moving if region == "ram"}
    # The frame counter, the timer and the replayed noise move whoever plays.
    assert {LAYOUT.frame, LAYOUT.timer} <= offsets
    assert set(range(LAYOUT.noise, LAYOUT.noise + 4)) <= offsets
    # Where the pad walks to is driven, by left alone and by right alone.
    assert LAYOUT.position not in offsets
    assert offsets <= {LAYOUT.frame, LAYOUT.frame + 1, LAYOUT.timer, *range(12, 16)}


def test_on_a_game_that_ignores_the_pad_what_moves_is_still_named(
    connection: ConnectionService,
) -> None:
    stepper(connection, deaf=True)
    probe = probe_inputs(connection, PAD)
    assert not probe.matters
    offsets = {offset for region, offset in probe.self_moving if region == "ram"}
    assert {LAYOUT.frame, LAYOUT.timer} <= offsets and LAYOUT.position not in offsets


# ------------------------------------------------------ the wake search


def test_the_button_that_starts_such_a_game_is_found(
    connection: ConnectionService, caplog: pytest.LogCaptureFixture
) -> None:
    game = stepper(connection)
    game._ram[LAYOUT.game_over] = 1
    assert not inputs_matter(connection, ["left", "right"])  # over: the pad is ignored
    caplog.set_level(logging.INFO, logger="retro_ai_lab.understanding.wakeup")
    order = [*PAD, "start"]
    found = find_wake_button(connection, order, ["left", "right"], wait=4, per_button=2)
    assert found == "start"
    assert not game.state.game_over  # left running after the press
    assert "Pressing Start gets the game going." in caplog.text


# ---------------------------------------------------------- the service


def service(tmp_path: Path) -> tuple[TrainingService, ConnectionService]:
    bus = EventBus(on_error=lambda exc, event, handler: (_ for _ in ()).throw(exc))
    connection = ConnectionService(default_registry(), bus)
    observation = ObservationService(connection, bus)
    discovery = DiscoveryService(connection, observation, bus, experiment_every=None)
    training = TrainingService(
        connection,
        observation,
        discovery,
        bus,
        frames_per_step=4,
        max_steps=12,
        settings=Settings(epsilon=0.5),
        seed=1,
        store=BrainStore(tmp_path / "b"),
        algorithm=Q_TABLE,
    )
    connection.connect("fake", seed=3)
    return training, connection


def searches(monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    made: list[Any] = []
    real = training_service.find_wake_button

    def spy(*args: Any, **options: Any) -> Any:
        made.append(args[1:])
        return real(*args, **options)

    monkeypatch.setattr(training_service, "find_wake_button", spy)
    return made


def test_the_trainer_does_not_take_such_a_game_for_one_that_is_not_listening(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    training, connection = service(tmp_path)
    game = stepper(connection)
    made = searches(monkeypatch)
    caplog.set_level(logging.INFO, logger="retro_ai_lab.training.service")
    training.train(1)
    assert NOT_LISTENING not in caplog.text and made == []
    assert training.episodes == 1 and training.history[-1].result.steps == 12
    # The go was played from where the game stood: 12 steps of 4 frames, and the frames
    # of a wake search (hundreds, holding nothing) were never run.
    assert game.state.frame == 12 * 4


def test_the_trainer_still_looks_for_a_way_into_a_game_that_ignores_the_pad(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    training, connection = service(tmp_path)
    stepper(connection, deaf=True)
    made = searches(monkeypatch)
    caplog.set_level(logging.INFO, logger="retro_ai_lab.training.service")
    training.train(1)
    assert NOT_LISTENING in caplog.text and made
