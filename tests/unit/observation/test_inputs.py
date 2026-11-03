"""T-204: input events and the input log."""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from retro_ai_lab.core import ConfigurationError
from retro_ai_lab.observation.inputs import SOURCES, HeldSpan, InputEvent, InputLog

# -------------------------------------------------------------- events


def test_event_normalises_buttons_and_round_trips() -> None:
    e = InputEvent(3, ["b", "a", "a"], "ai")  # type: ignore[arg-type]
    assert e.buttons == frozenset({"a", "b"})
    assert e.to_dict() == {"frame": 3, "buttons": ["a", "b"], "source": "ai"}
    assert InputEvent.from_dict(e.to_dict()) == e
    assert InputEvent.from_dict({"frame": 0, "buttons": []}).source == "human"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"frame": -1, "buttons": ()},
        {"frame": 1.5, "buttons": ()},
        {"frame": True, "buttons": ()},
        {"frame": 0, "buttons": ("",)},
        {"frame": 0, "buttons": (1,)},
        {"frame": 0, "buttons": (), "source": "cat"},
    ],
)
def test_event_rejects_bad_values(kwargs: dict[str, object]) -> None:
    with pytest.raises(ConfigurationError):
        InputEvent(**kwargs)  # type: ignore[arg-type]


def test_event_from_bad_dict() -> None:
    with pytest.raises(ConfigurationError):
        InputEvent.from_dict({"buttons": []})
    with pytest.raises(ConfigurationError):
        InputEvent.from_dict({"frame": 1, "buttons": 5})


def test_sources_are_the_documented_three() -> None:
    assert SOURCES == ("human", "ai", "replay")


# ----------------------------------------------------------------- log


def test_log_records_only_changes() -> None:
    log = InputLog()
    assert log.latest is None and len(log) == 0
    assert log.record(0, []) is None  # releasing nothing at the start
    assert log.record(2, ["up"]) == InputEvent(2, {"up"})  # type: ignore[arg-type]
    assert log.record(5, ["up"]) is None  # unchanged
    assert log.record(7, ["up", "a"]) is not None
    assert log.record(9, []) is not None
    assert [e.frame for e in log] == [2, 7, 9]
    assert log.events == tuple(log)


def test_same_frame_replaces_previous_change() -> None:
    log = InputLog()
    log.record(0, ["left"])
    log.record(4, ["up"])
    assert log.record(4, ["up", "a"]) is not None
    assert [e.buttons for e in log] == [frozenset({"left"}), frozenset({"up", "a"})]
    # change and change back within one frame leaves no trace
    assert log.record(4, ["left"]) is None
    assert [e.buttons for e in log] == [frozenset({"left"})]


def test_frames_must_not_go_backwards() -> None:
    log = InputLog()
    log.record(5, ["a"])
    with pytest.raises(ConfigurationError, match="go back"):
        log.record(4, ["b"])


def test_held_at_and_spans() -> None:
    log = InputLog()
    log.record(2, ["up"], "human")
    log.record(6, ["up", "a"], "human")
    log.record(9, [], "ai")
    assert log.held_at(0) == frozenset()
    assert log.held_at(2) == {"up"}
    assert log.held_at(5) == {"up"}
    assert log.held_at(6) == {"up", "a"}
    assert log.held_at(100) == frozenset()
    assert log.spans() == (
        HeldSpan(2, 6, frozenset({"up"}), "human"),
        HeldSpan(6, 9, frozenset({"up", "a"}), "human"),
        HeldSpan(9, None, frozenset(), "ai"),
    )
    assert log.spans()[0].length == 4 and log.spans()[-1].length is None
    assert log.by_source("ai") == (InputEvent(9, frozenset(), "ai"),)


def test_frames_held_and_presses() -> None:
    log = InputLog()
    log.record(0, ["a"])
    log.record(3, [])
    log.record(5, ["a", "b"])
    log.record(8, ["b"])
    assert log.frames_held("a", until=20) == 3 + 3
    assert log.frames_held("a", until=6) == 3 + 1
    assert log.presses("a") == 2
    assert log.presses("b") == 1
    assert log.presses("x") == 0
    with pytest.raises(ConfigurationError, match="until"):
        log.frames_held("b")  # still held
    assert log.frames_held("b", until=10) == 5


def test_capacity_drops_oldest() -> None:
    log = InputLog(capacity=3)
    for i in range(5):
        log.record(i, ["a"] if i % 2 == 0 else [])
    assert len(log) == 3 and log.dropped == 2
    assert log.capacity == 3
    assert [e.frame for e in log] == [2, 3, 4]
    log.clear()
    assert len(log) == 0 and log.dropped == 0
    with pytest.raises(ConfigurationError):
        InputLog(capacity=0)


def test_serialisation_round_trip_and_order_check() -> None:
    log = InputLog()
    log.record(1, ["up"])
    log.record(4, ["up", "a"], "ai")
    copy = InputLog.from_list(log.to_list())
    assert copy.events == log.events
    with pytest.raises(ConfigurationError, match="order"):
        InputLog.from_list([{"frame": 5, "buttons": []}, {"frame": 2, "buttons": ["a"]}])


@settings(max_examples=40)
@given(
    st.lists(
        st.tuples(st.integers(0, 5), st.frozensets(st.sampled_from(["a", "b", "up"]))),
        max_size=30,
    )
)
def test_log_is_consistent_with_a_naive_model(steps: list[tuple[int, frozenset[str]]]) -> None:
    """Feed monotonic frames; held_at must equal the last state set at or before each frame."""
    log = InputLog()
    frame = 0
    state: dict[int, frozenset[str]] = {}
    for gap, buttons in steps:
        frame += gap
        log.record(frame, buttons)
        state[frame] = buttons
    for f in range(frame + 2):
        past = [k for k in state if k <= f]
        expected = state[max(past)] if past else frozenset()
        assert log.held_at(f) == expected
