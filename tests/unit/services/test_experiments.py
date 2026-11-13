"""T-205: save-state experiment harness."""

from __future__ import annotations

import pytest

from retro_ai_lab.core import (
    Capabilities,
    ConfigurationError,
    EmulatorDisconnectedError,
    UnsupportedCapabilityError,
)
from retro_ai_lab.core.events import EventBus
from retro_ai_lab.emulator.fake import FakeEmulatorAdapter
from retro_ai_lab.emulator.fake.toy_racer import LAYOUT
from retro_ai_lab.emulator.registry import default_registry
from retro_ai_lab.services import ConnectionService
from retro_ai_lab.services.experiments import (
    REQUIRED,
    ExperimentHarness,
    Trial,
    constant_plan,
)


@pytest.fixture
def connection() -> ConnectionService:
    return ConnectionService(default_registry(), EventBus())


@pytest.fixture
def harness(connection: ConnectionService) -> ExperimentHarness:
    connection.connect("fake", seed=5)
    return ExperimentHarness(connection)


def game(connection: ConnectionService) -> FakeEmulatorAdapter:
    a = connection.adapter
    assert isinstance(a, FakeEmulatorAdapter)
    return a


# ----------------------------------------------------------- plumbing


def test_constant_plan() -> None:
    assert constant_plan(["up", "a"], 3) == (frozenset({"a", "up"}),) * 3
    for bad in (0, -1, True, 1.5):
        with pytest.raises(ConfigurationError):
            constant_plan(["up"], bad)  # type: ignore[arg-type]


def test_not_available_until_connected_and_capable(connection: ConnectionService) -> None:
    h = ExperimentHarness(connection)
    assert not h.available()
    with pytest.raises(EmulatorDisconnectedError):
        h.checkpoint()
    connection.connect("fake", capabilities=Capabilities(memory_read=True, input=True))
    assert not h.available()
    with pytest.raises(UnsupportedCapabilityError) as info:
        h.checkpoint()
    assert info.value.capability in REQUIRED
    connection.disconnect()
    connection.connect("fake")
    assert h.available()


def test_bad_slot() -> None:
    with pytest.raises(ConfigurationError):
        ExperimentHarness(ConnectionService(default_registry(), EventBus()), slot="")


def test_needs_a_checkpoint_first(harness: ExperimentHarness) -> None:
    assert not harness.has_checkpoint
    with pytest.raises(ConfigurationError, match="checkpoint"):
        harness.run(constant_plan(["up"], 1))
    with pytest.raises(ConfigurationError, match="checkpoint"):
        harness.restore()


def test_it_counts_the_checkpoints_taken(harness: ExperimentHarness) -> None:
    """T-1014: what was worked out from one checkpoint can tell it is another now."""
    assert harness.taken == 0
    harness.checkpoint()
    assert harness.taken == 1 and harness.has_checkpoint
    harness.restore()
    harness.run(constant_plan(["up"], 2))
    assert harness.taken == 1  # going back to it, and trials from it, take no new one
    harness.checkpoint()
    assert harness.taken == 2


# -------------------------------------------------------------- trials


def test_trial_runs_from_checkpoint_and_restores(
    harness: ExperimentHarness, connection: ConnectionService
) -> None:
    adapter = game(connection)
    connection.call(lambda a: a.advance_frames(10))
    start = adapter.game.ram
    harness.checkpoint()
    assert harness.has_checkpoint and harness.slot in adapter.slots
    trial = harness.run(constant_plan(["up"], 8), label="accelerate")
    assert isinstance(trial, Trial)
    assert trial.label == "accelerate" and trial.frames == 8
    assert trial.before.get("ram") == start
    assert trial.after.frame == 8
    assert ("ram", LAYOUT.speed) in trial.changed
    assert trial.after.byte("ram", LAYOUT.speed) > trial.before.byte("ram", LAYOUT.speed)
    assert trial.screen_before is not None and trial.screen_diff is not None
    assert adapter.game.ram == start, "the game was not put back"
    assert adapter.held == frozenset()
    assert harness.trials == 1


def test_trials_are_repeatable(harness: ExperimentHarness) -> None:
    harness.checkpoint()
    a = harness.run(constant_plan(["up"], 5))
    b = harness.run(constant_plan(["up"], 5))
    assert a.before == b.before and a.after == b.after
    assert a.label == "trial-1" and b.label == "trial-2"


def test_per_frame_plan_and_no_restore(
    harness: ExperimentHarness, connection: ConnectionService
) -> None:
    harness.checkpoint()
    plan = [["up"], ["up"], ["up", "left"], [], ["b"]]
    trial = harness.run(plan, restore=False)
    assert trial.plan == tuple(frozenset(p) for p in plan)
    assert game(connection).game.ram == trial.after.get("ram")
    harness.restore()
    assert game(connection).game.ram == trial.before.get("ram")


def test_rejects_empty_plan_and_unknown_buttons(harness: ExperimentHarness) -> None:
    harness.checkpoint()
    with pytest.raises(ConfigurationError, match="at least one frame"):
        harness.run([])
    with pytest.raises(ConfigurationError, match="Unknown buttons"):
        harness.run([["up"], ["jump"]])


def test_no_screen_capture_still_works(connection: ConnectionService) -> None:
    connection.connect(
        "fake",
        capabilities=Capabilities(
            memory_read=True, input=True, frame_advance=True, save_states=True
        ),
    )
    h = ExperimentHarness(connection)
    h.checkpoint()
    trial = h.run(constant_plan(["up"], 2))
    assert trial.screen_before is None and trial.screen_after is None
    assert trial.screen_diff is None
    assert not trial.diff.is_empty


def test_a_link_that_drops_at_the_check_is_not_available(connection: ConnectionService) -> None:
    """T-980: the user's log ended with EmulatorDisconnectedError raised out of
    available(), from a memory handler that ran as the app shut down."""
    connection.connect("fake")
    h = ExperimentHarness(connection)
    assert h.available()
    adapter = connection.adapter
    assert adapter is not None
    adapter.drop_link()  # the emulator is gone, the service does not know yet
    assert connection.is_connected
    assert not h.available()
    assert not connection.is_connected  # the connection service noticed as it called


def test_link_loss_mid_trial_forgets_checkpoint(connection: ConnectionService) -> None:
    connection.connect("fake", drop_after_ops=12)
    h = ExperimentHarness(connection)
    h.checkpoint()
    with pytest.raises(EmulatorDisconnectedError):
        h.run(constant_plan(["up"], 50))
    assert not h.has_checkpoint
    assert not connection.is_connected


# ---------------------------------------------------------- comparison


def test_compare_finds_bytes_that_depend_on_the_input(harness: ExperimentHarness) -> None:
    harness.checkpoint()
    trials = harness.run_many(
        {
            "idle": constant_plan([], 6),
            "up": constant_plan(["up"], 6),
            "left": constant_plan(["left"], 6),
        }
    )
    cmp = harness.compare(trials)
    assert cmp.labels == ("idle", "up", "left")
    # speed depends on "up", lane depends on "left"
    assert ("ram", LAYOUT.speed) in cmp.discriminating
    assert ("ram", LAYOUT.lane) in cmp.discriminating
    assert cmp.values("ram", LAYOUT.speed)["up"] > cmp.values("ram", LAYOUT.speed)["idle"]
    assert cmp.values("ram", LAYOUT.lane)["left"] != cmp.values("ram", LAYOUT.lane)["idle"]
    # header bytes never discriminate
    assert all(off < LAYOUT.header for (_, off) in cmp.discriminating)
    # the frame counter changed in every trial
    assert ("ram", LAYOUT.frame) in cmp.common
    with pytest.raises(ConfigurationError):
        cmp.values("ram", LAYOUT.header)


def test_compare_rejects_bad_sets(harness: ExperimentHarness) -> None:
    harness.checkpoint()
    one = harness.run(constant_plan(["up"], 2), label="x")
    with pytest.raises(ConfigurationError, match="two"):
        harness.compare([one])
    with pytest.raises(ConfigurationError, match="unique"):
        harness.compare([one, one])
    harness.run(constant_plan(["up"], 30), restore=False)
    harness.checkpoint()  # new starting point
    other = harness.run(constant_plan(["up"], 2), label="y")
    with pytest.raises(ConfigurationError, match="same state"):
        harness.compare([one, other])
    with pytest.raises(ConfigurationError):
        harness.run_many({})


def test_the_harness_reports_every_restore(connection: ConnectionService) -> None:
    """T-935: whoever watches the memory is told after each load_state."""
    connection.connect("fake", seed=5)
    restores = 0

    def noted() -> None:
        nonlocal restores
        restores += 1

    with pytest.raises(ConfigurationError, match="callable"):
        ExperimentHarness(connection, on_restore="later")  # type: ignore[arg-type]
    harness = ExperimentHarness(connection, on_restore=noted)
    with pytest.raises(ConfigurationError):
        harness.restore()  # no checkpoint: nothing was loaded
    assert restores == 0
    harness.checkpoint()
    harness.restore()
    assert restores == 1
    harness.run(constant_plan(["up"], 2), label="x")
    assert restores == 3  # the start of the trial and the restore after it
    harness.run(constant_plan(["up"], 2), label="y", restore=False)
    assert restores == 4
