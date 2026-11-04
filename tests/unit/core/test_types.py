"""T-002: domain types — validation, immutability, equality and serialisation."""

from __future__ import annotations

import dataclasses
import json
from typing import Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from retro_ai_lab.core import (
    Button,
    Capabilities,
    ConfigurationError,
    Frame,
    GameInfo,
    MemoryRegion,
    StepResult,
)

# ---------------------------------------------------------------- Button


def test_button_valid() -> None:
    b = Button("a", "A")
    assert (b.name, b.label) == ("a", "A")


@pytest.mark.parametrize("name", ["", "A", "1a", "a-b", "a b", "ü", None, 3])
def test_button_rejects_bad_names(name: Any) -> None:
    with pytest.raises(ConfigurationError):
        Button(name, "A")


@pytest.mark.parametrize("label", ["", "   ", None, 1])
def test_button_rejects_bad_labels(label: Any) -> None:
    with pytest.raises(ConfigurationError):
        Button("a", label)


def test_button_is_immutable_hashable_and_comparable() -> None:
    b = Button("start", "Start")
    with pytest.raises(dataclasses.FrozenInstanceError):
        b.name = "x"  # type: ignore[misc]
    assert b == Button("start", "Start")
    assert b != Button("start", "Begin")
    assert len({b, Button("start", "Start")}) == 1


def test_button_round_trip() -> None:
    b = Button("dpad_up", "Up")
    assert Button.from_dict(json.loads(json.dumps(b.to_dict()))) == b


@pytest.mark.parametrize("data", [{}, {"name": "a"}, {"name": "a", "label": "A", "x": 1}, "a"])
def test_button_from_dict_rejects_bad_shapes(data: Any) -> None:
    with pytest.raises(ConfigurationError):
        Button.from_dict(data)


# ---------------------------------------------------------- Capabilities


def test_capabilities_default_all_false() -> None:
    assert Capabilities().enabled() == ()


def test_capabilities_all_and_names() -> None:
    assert Capabilities.all().enabled() == Capabilities.names()
    assert Capabilities.names() == (
        "memory_read",
        "screen_capture",
        "input",
        "frame_advance",
        "speed_control",
        "save_states",
    )


def test_capabilities_supports() -> None:
    caps = Capabilities(input=True, save_states=True)
    assert caps.supports("input")
    assert caps.supports("save_states")
    assert not caps.supports("memory_read")
    assert caps.enabled() == ("input", "save_states")


def test_capabilities_supports_unknown_raises() -> None:
    with pytest.raises(ConfigurationError):
        Capabilities().supports("teleport")


@pytest.mark.parametrize("bad", [1, "yes", None])
def test_capabilities_rejects_non_bool(bad: Any) -> None:
    with pytest.raises(ConfigurationError):
        Capabilities(input=bad)


@given(st.builds(Capabilities, **{n: st.booleans() for n in Capabilities.names()}))
def test_capabilities_round_trip(caps: Capabilities) -> None:
    assert Capabilities.from_dict(json.loads(json.dumps(caps.to_dict()))) == caps


def test_capabilities_from_dict_rejects_unknown_or_missing() -> None:
    with pytest.raises(ConfigurationError):
        Capabilities.from_dict({**Capabilities().to_dict(), "teleport": True})
    with pytest.raises(ConfigurationError):
        Capabilities.from_dict({"input": True})


# ---------------------------------------------------------- MemoryRegion


def test_memory_region_valid_and_contains() -> None:
    r = MemoryRegion("wram", 8192, "Work RAM")
    assert r.contains(0, 8192)
    assert r.contains(8191, 1)
    assert r.contains(100, 0)
    assert not r.contains(8192, 1)
    assert not r.contains(-1, 1)
    assert not r.contains(0, 8193)
    assert not r.contains(10, -1)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"name": "", "size": 1},
        {"name": "WRAM", "size": 1},
        {"name": "wram", "size": 0},
        {"name": "wram", "size": -5},
        {"name": "wram", "size": 1.5},
        {"name": "wram", "size": True},
        {"name": "wram", "size": 1, "description": None},
    ],
)
def test_memory_region_rejects_bad_values(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ConfigurationError):
        MemoryRegion(**kwargs)


def test_memory_region_round_trip() -> None:
    r = MemoryRegion("vram", 16, "video")
    assert MemoryRegion.from_dict(json.loads(json.dumps(r.to_dict()))) == r


# -------------------------------------------------------------- GameInfo


def test_game_info_valid_and_optional_hash() -> None:
    g = GameInfo("GB", "Toy Racer")
    assert g.rom_hash is None
    h = GameInfo("GB", "Toy Racer", "abc123")
    assert h.rom_hash == "abc123"
    assert g != h


@pytest.mark.parametrize(
    "kwargs",
    [
        {"system": "", "title": "x"},
        {"system": "GB", "title": " "},
        {"system": "GB", "title": "x", "rom_hash": ""},
        {"system": None, "title": "x"},
    ],
)
def test_game_info_rejects_bad_values(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ConfigurationError):
        GameInfo(**kwargs)


def test_game_info_round_trip() -> None:
    g = GameInfo("GBA", "Toy Racer", "ff00")
    assert GameInfo.from_dict(json.loads(json.dumps(g.to_dict()))) == g


# ----------------------------------------------------------------- Frame


def test_frame_valid_and_pixel_access() -> None:
    # 2x2 RGB: red, green / blue, white
    px = bytes([255, 0, 0, 0, 255, 0, 0, 0, 255, 255, 255, 255])
    f = Frame(2, 2, 3, px, number=7)
    assert f.size == (2, 2)
    assert f.pixel(0, 0) == (255, 0, 0)
    assert f.pixel(1, 0) == (0, 255, 0)
    assert f.pixel(0, 1) == (0, 0, 255)
    assert f.pixel(1, 1) == (255, 255, 255)
    assert f.number == 7


def test_frame_grey_single_channel() -> None:
    f = Frame(3, 1, 1, bytes([1, 2, 3]))
    assert f.pixel(2, 0) == (3,)


@pytest.mark.parametrize("xy", [(-1, 0), (0, -1), (2, 0), (0, 2)])
def test_frame_pixel_out_of_range(xy: tuple[int, int]) -> None:
    f = Frame(2, 2, 1, bytes(4))
    with pytest.raises(IndexError):
        f.pixel(*xy)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"width": 0, "height": 1, "channels": 1, "pixels": b""},
        {"width": 1, "height": 0, "channels": 1, "pixels": b""},
        {"width": 1, "height": 1, "channels": 2, "pixels": bytes(2)},
        {"width": 1, "height": 1, "channels": 5, "pixels": bytes(5)},
        {"width": 1, "height": 1, "channels": 1, "pixels": bytes(2)},
        {"width": 2, "height": 2, "channels": 3, "pixels": bytes(11)},
        {"width": 1, "height": 1, "channels": 1, "pixels": "\x00"},
        {"width": 1, "height": 1, "channels": 1, "pixels": bytearray(1)},
        {"width": 1, "height": 1, "channels": 1, "pixels": bytes(1), "number": -1},
        {"width": 1.0, "height": 1, "channels": 1, "pixels": bytes(1)},
    ],
)
def test_frame_rejects_bad_values(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ConfigurationError):
        Frame(**kwargs)


@settings(max_examples=50)
@given(
    width=st.integers(1, 8),
    height=st.integers(1, 8),
    channels=st.sampled_from([1, 3, 4]),
    number=st.integers(0, 10**6),
    data=st.data(),
)
def test_frame_round_trip(width: int, height: int, channels: int, number: int, data: Any) -> None:
    pixels = data.draw(
        st.binary(min_size=width * height * channels, max_size=width * height * channels)
    )
    f = Frame(width, height, channels, pixels, number)
    back = Frame.from_dict(json.loads(json.dumps(f.to_dict())))
    assert back == f
    assert back.pixels == pixels


def test_frame_from_dict_rejects_bad_hex() -> None:
    bad = {**Frame(1, 1, 1, b"\x00").to_dict(), "pixels": "zz"}
    with pytest.raises(ConfigurationError):
        Frame.from_dict(bad)
    not_str = {**Frame(1, 1, 1, b"\x00").to_dict(), "pixels": b"\x00"}
    with pytest.raises(ConfigurationError):
        Frame.from_dict(not_str)


def test_frame_is_immutable() -> None:
    f = Frame(1, 1, 1, b"\x00")
    with pytest.raises(dataclasses.FrozenInstanceError):
        f.width = 2  # type: ignore[misc]


def test_step_result_pressed_is_names_or_none() -> None:
    """T-887: a watched step names the buttons the person held."""
    assert StepResult(1).pressed is None
    assert StepResult(1, pressed=frozenset({"a", "up"})).pressed == frozenset({"a", "up"})
    with pytest.raises(ConfigurationError, match="pressed"):
        StepResult(1, pressed={"a"})  # type: ignore[arg-type]
    with pytest.raises(ConfigurationError, match="pressed"):
        StepResult(1, pressed=frozenset({1}))  # type: ignore[arg-type]
