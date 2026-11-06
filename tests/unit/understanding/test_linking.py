"""T-303: which bytes each button moves, folded into the knowledge."""

from __future__ import annotations

import random

import pytest

from retro_ai_lab.core import ConfigurationError
from retro_ai_lab.core.events import EventBus
from retro_ai_lab.emulator.fake import FakeEmulatorAdapter
from retro_ai_lab.emulator.fake.toy_racer import BUTTON_NAMES, LAYOUT
from retro_ai_lab.emulator.registry import default_registry
from retro_ai_lab.services import ConnectionService, ExperimentHarness
from retro_ai_lab.understanding import (
    Confidence,
    Evidence,
    InputLinks,
    Knowledge,
    VariableKind,
    apply_input_links,
    find_input_links,
)

RAM = "ram"


@pytest.fixture
def connection() -> ConnectionService:
    c = ConnectionService(default_registry(), EventBus())
    c.connect("fake", seed=5)
    return c


@pytest.fixture
def harness(connection: ConnectionService) -> ExperimentHarness:
    # Get the car moving first so braking and steering have something to act on.
    connection.call(lambda a: a.set_input([b for b in a.buttons() if b.name == "up"]))
    connection.call(lambda a: a.advance_frames(2))
    connection.call(lambda a: a.set_input(()))
    h = ExperimentHarness(connection)
    h.checkpoint()
    return h


def adapter(connection: ConnectionService) -> FakeEmulatorAdapter:
    a = connection.adapter
    assert isinstance(a, FakeEmulatorAdapter)
    return a


# ------------------------------------------------------------ finding


def test_links_on_the_toy_racer(harness: ExperimentHarness, connection: ConnectionService) -> None:
    before = adapter(connection).game.ram
    links = find_input_links(harness, BUTTON_NAMES)
    assert adapter(connection).game.ram == before, "the game was put back"
    assert links.buttons == BUTTON_NAMES and links.frames == 8
    assert links.unstable == frozenset()  # the toy racer's noise is part of its save state
    up = links.by_button["up"]
    assert (RAM, LAYOUT.speed) in up and (RAM, LAYOUT.position) in up
    assert (RAM, LAYOUT.lane) in links.by_button["left"]
    assert (RAM, LAYOUT.lane) in links.by_button["right"]
    assert (RAM, LAYOUT.speed) in links.by_button["b"]  # brake
    assert (RAM, LAYOUT.lane) not in up
    assert links.buttons_for((RAM, LAYOUT.lane)) == ("left", "right")
    # The input mask mirrors every button, even ones that do nothing in play.
    assert links.buttons_for((RAM, LAYOUT.input)) == BUTTON_NAMES
    assert "a" in links.dead_buttons() or links.by_button["a"] == {(RAM, LAYOUT.input)}
    for off in range(LAYOUT.header, LAYOUT.header + 15):
        assert (RAM, off) not in links.locations()
    for off in range(LAYOUT.noise, LAYOUT.noise + 4):
        assert (RAM, off) not in links.locations()


class _PausingGame:
    """A harness stub: idle advances forty bytes, Start freezes them, A and Left move their own."""

    has_checkpoint = True

    def run(self, plan: tuple[frozenset[str], ...], *, label: str = "") -> object:
        from retro_ai_lab.observation.diff import diff_snapshots
        from retro_ai_lab.observation.memory import MemorySnapshot
        from retro_ai_lab.services.experiments import Trial

        held = plan[0]
        before = bytearray(256)
        after = bytearray(256)
        if "start" not in held:  # the game runs on: forty bytes advance
            for i in range(40):
                after[i] = 1
        if "a" in held:  # a jump touches ten of those and a byte of its own
            for i in range(10):
                after[i] = 7
            after[101] = 9
        if "left" in held:
            after[100] = 5
        x = MemorySnapshot(0, {RAM: bytes(before)})
        y = MemorySnapshot(len(plan), {RAM: bytes(after)})
        return Trial(label, plan, x, y, diff_snapshots(x, y), None, None, None)


def test_a_button_that_stops_the_game_links_to_nothing() -> None:
    """T-930: Start paused the user's game, and every enemy sprite was linked to it."""
    from typing import cast

    from retro_ai_lab.understanding.linking import SWEEP_LEAST, SWEEP_SHARE

    assert SWEEP_SHARE == 0.5 and SWEEP_LEAST == 8
    links = find_input_links(cast(ExperimentHarness, _PausingGame()), ("a", "left", "start"))
    assert links.sweeping == ("start",) and links.active() == ("a", "left")
    assert links.by_button["start"] == frozenset()
    assert links.by_button["left"] == {(RAM, 100)}
    assert links.by_button["a"] == {(RAM, i) for i in range(10)} | {(RAM, 101)}
    assert links.dead_buttons() == () and "start" not in links.buttons_for((RAM, 5))
    assert links.locations() == {(RAM, i) for i in range(10)} | {(RAM, 100), (RAM, 101)}


def test_a_sweeping_button_does_not_count_for_the_mirror() -> None:
    """T-930: the controller mirror is the byte every active button moves."""
    k = Knowledge("g")
    by_button = {
        "a": frozenset({(RAM, 1), (RAM, 9)}),
        "b": frozenset({(RAM, 1)}),
        "left": frozenset({(RAM, 1), (RAM, 2)}),
        "start": frozenset(),
    }
    links = InputLinks(("a", "b", "left", "start"), 8, by_button, frozenset(), ("start",))
    touched = {v.offset: v for v in apply_input_links(k, links)}
    assert touched[1].kind is VariableKind.INPUT and touched[1].name == "Controller input"
    assert touched[2].kind is VariableKind.POSITION and touched[9].kind is VariableKind.UNKNOWN
    # with only two active buttons there is no mirror to find
    two = {b: by_button[b] for b in ("a", "b", "start")}
    few = InputLinks(("a", "b", "start"), 8, two, frozenset(), ("start",))
    assert all(v.kind is not VariableKind.INPUT for v in apply_input_links(Knowledge("h"), few))


def test_unstable_bytes_are_excluded(
    harness: ExperimentHarness, connection: ConnectionService
) -> None:
    game = adapter(connection).game
    original = adapter(connection)._do_load_state
    rng = random.Random(3)

    def noisy_load(slot: str) -> None:
        original(slot)
        game._noise = rng.randrange(1, 2**31)  # a console whose RNG is not in the save

    adapter(connection)._do_load_state = noisy_load  # type: ignore[method-assign]
    links = find_input_links(harness, ["up", "a"], repeats=3)
    assert {(RAM, off) for off in range(LAYOUT.noise, LAYOUT.noise + 4)} <= links.unstable
    assert not links.unstable & links.locations()
    assert (RAM, LAYOUT.speed) in links.by_button["up"]


def test_find_rejects_bad_arguments(
    harness: ExperimentHarness, connection: ConnectionService
) -> None:
    with pytest.raises(ConfigurationError, match="at least one"):
        find_input_links(harness, [])
    with pytest.raises(ConfigurationError, match="positive"):
        find_input_links(harness, ["up"], frames=0)
    with pytest.raises(ConfigurationError, match="positive"):
        find_input_links(harness, ["up"], repeats=0)
    fresh = ExperimentHarness(connection, slot="other")
    with pytest.raises(ConfigurationError, match="checkpoint"):
        find_input_links(fresh, ["up"])
    with pytest.raises(ConfigurationError, match="Unknown buttons"):
        find_input_links(harness, ["jump"])


# ----------------------------------------------------------- applying


def links_fixture() -> InputLinks:
    lane, speed, score, mask, mystery = (RAM, 3), (RAM, 4), (RAM, 8), (RAM, 11), (RAM, 20)
    buttons = ("up", "left", "right", "a")
    by_button = {
        "up": frozenset({speed, score, mask}),
        "left": frozenset({lane, mask}),
        "right": frozenset({lane, mask, mystery}),
        "a": frozenset({mask}),
    }
    return InputLinks(buttons, 8, by_button, frozenset())


def test_apply_adds_evidence_relabels_and_proposes() -> None:
    k = Knowledge("g")
    k.propose(RAM, 3, kind=VariableKind.PROGRESS, evidence=[Evidence("few-values", "", 1.0)])
    k.propose(RAM, 4, kind=VariableKind.SPEED, evidence=[Evidence("both-ways", "", 1.0)])
    k.propose(RAM, 8, kind=VariableKind.SCORE, evidence=[Evidence("monotonic", "", 2.0)])
    k.propose(RAM, 11, evidence=[Evidence("changes", "", 1.0)])
    touched = apply_input_links(k, links_fixture())
    assert {v.id for v in touched} == {"v1", "v2", "v3", "v4", "v5"}
    lane = k.at(RAM, 3)
    assert lane is not None and lane.kind is VariableKind.POSITION
    assert lane.evidence[-1].kind == "input-linked"
    assert "Left or Right" in lane.evidence[-1].detail
    speed = k.at(RAM, 4)
    assert speed is not None and speed.kind is VariableKind.SPEED  # a gauge stays a gauge
    assert (
        speed.confidence is Confidence.LOW
        and speed.evidence[-1].detail == "Changes when Up is held."
    )
    mask = k.at(RAM, 11)
    assert mask is not None and mask.kind is VariableKind.INPUT and mask.name == "Controller input"
    mystery = k.at(RAM, 20)
    assert mystery is not None and mystery.kind is VariableKind.POSITION  # new, direction-linked
    assert mystery.evidence[0].kind == "input-linked"
    # Running again adds nothing new.
    assert apply_input_links(k, links_fixture()) == ()


def test_apply_respects_confirmed_and_rejected() -> None:
    k = Knowledge("g")
    k.propose(RAM, 3, kind=VariableKind.PROGRESS, name="Lap")
    k.confirm("v1")
    k.propose(RAM, 11)
    k.reject("v2")
    touched = apply_input_links(k, links_fixture())
    lap = k.get("v1")
    assert lap.kind is VariableKind.PROGRESS and lap.confirmed  # label kept, evidence added
    assert lap.evidence[-1].kind == "input-linked"
    assert k.get("v2").rejected and k.get("v2").evidence[-1].kind == "user"
    assert {v.id for v in touched} == {"v1", "v3", "v4", "v5"}


def test_apply_with_too_few_buttons_never_declares_a_mirror() -> None:
    k = Knowledge("g")
    links = InputLinks(
        ("a", "b"), 4, {"a": frozenset({(RAM, 1)}), "b": frozenset({(RAM, 1)})}, frozenset()
    )
    (v,) = apply_input_links(k, links)
    assert v.kind is VariableKind.UNKNOWN and v.name == ""


# ----------------------------------------------- end to end on the toy


def test_discovery_then_linking_fixes_lane_and_finds_the_controller(
    harness: ExperimentHarness,
) -> None:
    from retro_ai_lab.emulator.fake import ToyRacer
    from retro_ai_lab.observation import ChangeTracker, MemorySnapshot
    from retro_ai_lab.understanding import find_candidates, propose_candidates

    game = ToyRacer(seed=5)
    tracker = ChangeTracker()
    tracker.observe(MemorySnapshot(0, {RAM: game.ram}))
    for i in range(1, 700):
        game.step(["up", "left"] if i % 50 < 4 else ["up", "right"] if i % 70 < 4 else ["up"])
        tracker.observe(MemorySnapshot(i, {RAM: game.ram}))
    k = Knowledge("toyracer-5")
    propose_candidates(k, find_candidates(tracker))
    apply_input_links(k, find_input_links(harness, BUTTON_NAMES))
    lane = k.at(RAM, LAYOUT.lane)
    assert lane is not None and lane.kind is VariableKind.POSITION
    mask = k.at(RAM, LAYOUT.input)
    assert mask is not None and mask.kind is VariableKind.INPUT
    speed = k.at(RAM, LAYOUT.speed)
    assert speed is not None and speed.confidence >= Confidence.MEDIUM
    score = k.best(VariableKind.SCORE)
    assert score is not None and score.offset == LAYOUT.score
    assert any(e.kind == "input-linked" for e in score.evidence)
