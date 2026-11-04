"""Immutable domain types shared across layers.

These are deliberately plain: frozen dataclasses with validation in
``__post_init__`` and symmetric ``to_dict`` / ``from_dict`` so they can be
stored as JSON (see DESIGN.md §14) without any extra machinery.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields
from typing import Any, Self

from retro_ai_lab.core.errors import ConfigurationError

_IDENTIFIER = re.compile(r"^[a-z][a-z0-9_]*$")


def _require_identifier(value: object, what: str) -> str:
    if not isinstance(value, str) or not _IDENTIFIER.match(value):
        raise ConfigurationError(
            f"{what} must be a lowercase identifier (letters, digits, underscores), got {value!r}."
        )
    return value


def _require_non_empty(value: object, what: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigurationError(f"{what} must be a non-empty string, got {value!r}.")
    return value


def _require_int(value: object, what: str, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigurationError(f"{what} must be an integer, got {value!r}.")
    if value < minimum:
        raise ConfigurationError(f"{what} must be >= {minimum}, got {value}.")
    return value


def _pick(data: Mapping[str, Any], cls: type, what: str) -> dict[str, Any]:
    """Extract exactly the dataclass fields from ``data``; reject unknown keys."""
    if not isinstance(data, Mapping):
        raise ConfigurationError(f"{what} must be a mapping, got {type(data).__name__}.")
    names = {f.name for f in fields(cls)}
    unknown = set(data) - names
    if unknown:
        raise ConfigurationError(f"{what} has unknown fields: {sorted(unknown)}.")
    missing = names - set(data)
    if missing:
        raise ConfigurationError(f"{what} is missing fields: {sorted(missing)}.")
    return {name: data[name] for name in names}


@dataclass(frozen=True, slots=True)
class Button:
    """One controller input, e.g. ``Button("a", "A")``.

    ``name`` is the stable machine identifier used in protocols and storage;
    ``label`` is what children see.
    """

    name: str
    label: str

    def __post_init__(self) -> None:
        _require_identifier(self.name, "Button name")
        _require_non_empty(self.label, "Button label")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Self:
        return cls(**_pick(data, cls, "Button"))


@dataclass(frozen=True, slots=True)
class Capabilities:
    """What a connected emulator can do. Callers must check before asking."""

    memory_read: bool = False
    screen_capture: bool = False
    input: bool = False
    frame_advance: bool = False
    speed_control: bool = False
    save_states: bool = False

    def __post_init__(self) -> None:
        for f in fields(self):
            if not isinstance(getattr(self, f.name), bool):
                raise ConfigurationError(f"Capability '{f.name}' must be a bool.")

    @classmethod
    def names(cls) -> tuple[str, ...]:
        return tuple(f.name for f in fields(cls))

    @classmethod
    def all(cls) -> Self:
        return cls(**dict.fromkeys(cls.names(), True))

    def supports(self, capability: str) -> bool:
        if capability not in self.names():
            raise ConfigurationError(f"Unknown capability {capability!r}.")
        return bool(getattr(self, capability))

    def enabled(self) -> tuple[str, ...]:
        return tuple(name for name in self.names() if getattr(self, name))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Self:
        return cls(**_pick(data, cls, "Capabilities"))


@dataclass(frozen=True, slots=True)
class MemoryRegion:
    """A readable block of emulator memory, e.g. work RAM."""

    name: str
    size: int
    description: str = ""

    def __post_init__(self) -> None:
        _require_identifier(self.name, "Memory region name")
        _require_int(self.size, "Memory region size", minimum=1)
        if not isinstance(self.description, str):
            raise ConfigurationError("Memory region description must be a string.")

    def contains(self, offset: int, length: int) -> bool:
        """True if ``[offset, offset+length)`` lies inside this region."""
        return offset >= 0 and length >= 0 and offset + length <= self.size

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Self:
        return cls(**_pick(data, cls, "MemoryRegion"))


@dataclass(frozen=True, slots=True)
class GameInfo:
    """What the emulator tells us about the loaded game."""

    system: str
    title: str
    rom_hash: str | None = None

    def __post_init__(self) -> None:
        _require_non_empty(self.system, "Game system")
        _require_non_empty(self.title, "Game title")
        if self.rom_hash is not None:
            _require_non_empty(self.rom_hash, "ROM hash")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Self:
        return cls(**_pick(data, cls, "GameInfo"))


@dataclass(frozen=True, slots=True)
class Frame:
    """One captured screen image: packed 8-bit pixels, row-major.

    ``pixels`` holds ``width * height * channels`` bytes. ``channels`` is 1
    (grey), 3 (RGB) or 4 (RGBA). ``number`` is the emulator frame counter.
    """

    width: int
    height: int
    channels: int
    pixels: bytes
    number: int = 0

    def __post_init__(self) -> None:
        _require_int(self.width, "Frame width", minimum=1)
        _require_int(self.height, "Frame height", minimum=1)
        _require_int(self.channels, "Frame channels", minimum=1)
        if self.channels not in (1, 3, 4):
            raise ConfigurationError(f"Frame channels must be 1, 3 or 4, got {self.channels}.")
        _require_int(self.number, "Frame number", minimum=0)
        if not isinstance(self.pixels, bytes):
            raise ConfigurationError("Frame pixels must be bytes.")
        expected = self.width * self.height * self.channels
        if len(self.pixels) != expected:
            raise ConfigurationError(
                f"Frame pixels has {len(self.pixels)} bytes; "
                f"expected {expected} for {self.width}x{self.height}x{self.channels}."
            )

    @property
    def size(self) -> tuple[int, int]:
        return (self.width, self.height)

    def pixel(self, x: int, y: int) -> tuple[int, ...]:
        """Return the channel values at ``(x, y)``; raises IndexError outside the frame."""
        if not (0 <= x < self.width and 0 <= y < self.height):
            raise IndexError(f"({x}, {y}) is outside a {self.width}x{self.height} frame.")
        start = (y * self.width + x) * self.channels
        return tuple(self.pixels[start : start + self.channels])

    def to_dict(self) -> dict[str, Any]:
        return {
            "width": self.width,
            "height": self.height,
            "channels": self.channels,
            "pixels": self.pixels.hex(),
            "number": self.number,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Self:
        picked = _pick(data, cls, "Frame")
        raw = picked["pixels"]
        if not isinstance(raw, str):
            raise ConfigurationError("Frame pixels must be a hex string in serialised form.")
        try:
            picked["pixels"] = bytes.fromhex(raw)
        except ValueError as exc:
            raise ConfigurationError("Frame pixels is not valid hex.") from exc
        return cls(**picked)


@dataclass(frozen=True, slots=True)
class StepResult:
    """What one bundled ``step`` brought back (T-822).

    ``reads`` holds one ``bytes`` per requested read, in request order;
    ``frame`` is the screenshot taken after the last frame, if one was asked
    for; ``pressed`` names the buttons the person held on the real controller
    during a watched step (T-887), ``None`` when the step did not watch.
    """

    frames: int
    reads: tuple[bytes, ...] = ()
    frame: Frame | None = None
    pressed: frozenset[str] | None = None

    def __post_init__(self) -> None:
        _require_int(self.frames, "StepResult frames", minimum=1)
        if not isinstance(self.reads, tuple) or not all(isinstance(r, bytes) for r in self.reads):
            raise ConfigurationError("StepResult reads must be a tuple of bytes.")
        if self.frame is not None and not isinstance(self.frame, Frame):
            raise ConfigurationError("StepResult frame must be a Frame or None.")
        if self.pressed is not None and (
            not isinstance(self.pressed, frozenset)
            or not all(isinstance(name, str) for name in self.pressed)
        ):
            raise ConfigurationError("StepResult pressed must be a frozenset of names or None.")


__all__ = ["Button", "Capabilities", "Frame", "GameInfo", "MemoryRegion", "StepResult"]
