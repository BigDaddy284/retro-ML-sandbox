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
