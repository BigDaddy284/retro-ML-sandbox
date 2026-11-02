"""The emulator adapter contract.

Every emulator (real or fake) is driven through :class:`EmulatorAdapter`.
Subclasses implement the ``_do_*`` hooks; the public methods here add the
checks that every adapter must make in exactly the same way:

* every operation except ``connect``/``disconnect`` requires a connection;
* every optional operation checks :class:`~retro_ai_lab.core.Capabilities`
  first and raises :class:`~retro_ai_lab.core.UnsupportedCapabilityError`
  rather than letting a backend fail in its own way;
* arguments are validated before they reach the backend.

This keeps DESIGN.md §7 honest: callers detect capabilities, never assume them.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from typing import ClassVar, final

from retro_ai_lab.core import (
    Button,
    Capabilities,
    ConfigurationError,
    EmulatorDisconnectedError,
    Frame,
    GameInfo,
    MemoryRegion,
    UnsupportedCapabilityError,
)

DEFAULT_CONNECT_TIMEOUT_S = 5.0


class EmulatorAdapter(ABC):
    """Base class for all emulator adapters.

    ``name`` is the stable identifier shown by the adapter registry (T-006);
    ``display_name`` is what children see.
    """

    name: ClassVar[str] = ""
    display_name: ClassVar[str] = ""

    def __init__(self) -> None:
        self._connected = False

    # ------------------------------------------------------------ lifecycle

    @final
    @property
    def is_connected(self) -> bool:
        return self._connected

    @final
    def connect(self, timeout_s: float = DEFAULT_CONNECT_TIMEOUT_S) -> None:
        """Open the connection. Calling it while connected is a no-op."""
        if not isinstance(timeout_s, int | float) or isinstance(timeout_s, bool):
            raise ConfigurationError("Connect timeout must be a number of seconds.")
        if timeout_s <= 0:
            raise ConfigurationError("Connect timeout must be greater than zero.")
        if self._connected:
            return
        self._do_connect(float(timeout_s))
        self._connected = True

    @final
    def disconnect(self) -> None:
        """Close the connection. Calling it while disconnected is a no-op."""
        if not self._connected:
            return
        try:
            self._do_disconnect()
        finally:
            self._connected = False

    @final
    def mark_disconnected(self) -> None:
        """Record that the link dropped without our asking (backend use)."""
        self._connected = False

    # ---------------------------------------------------------- description

    @final
    def capabilities(self) -> Capabilities:
        self._require_connected()
        return self._do_capabilities()

    @final
    def game_info(self) -> GameInfo:
        self._require_connected()
        return self._do_game_info()

    @final
    def buttons(self) -> Sequence[Button]:
        self._require_connected()
        return tuple(self._do_buttons())

    @final
    def memory_regions(self) -> Sequence[MemoryRegion]:
        self._require_connected()
        self._require("memory_read")
        return tuple(self._do_memory_regions())

    # ----------------------------------------------------------- operations

    @final
    def read_memory(self, region: str, offset: int, length: int) -> bytes:
        self._require_connected()
        self._require("memory_read")
        regions = {r.name: r for r in self._do_memory_regions()}
        if region not in regions:
            raise ConfigurationError(
                f"Unknown memory region {region!r}; available: {sorted(regions)}."
            )
        _check_int(offset, "offset", minimum=0)
        _check_int(length, "length", minimum=0)
        if not regions[region].contains(offset, length):
            raise ConfigurationError(
                f"Read of {length} bytes at {offset} is outside region "
                f"{region!r} (size {regions[region].size})."
            )
        data = self._do_read_memory(region, offset, length)
        if len(data) != length:
            raise ConfigurationError(
                f"Backend returned {len(data)} bytes for a {length}-byte read."
            )
        return data

    @final
    def capture_frame(self) -> Frame:
        self._require_connected()
        self._require("screen_capture")
        return self._do_capture_frame()

    @final
    def set_input(self, pressed: Iterable[Button]) -> None:
        self._require_connected()
        self._require("input")
        pressed_set = frozenset(pressed)
        known = set(self._do_buttons())
        unknown = pressed_set - known
        if unknown:
            raise ConfigurationError(
                f"Unknown buttons for this emulator: {sorted(b.name for b in unknown)}."
            )
        self._do_set_input(pressed_set)

    @final
    def advance_frames(self, n: int = 1) -> None:
        self._require_connected()
        self._require("frame_advance")
        _check_int(n, "frame count", minimum=1)
        self._do_advance_frames(n)

    @final
    def set_speed(self, multiplier: float) -> None:
        self._require_connected()
        self._require("speed_control")
        if not isinstance(multiplier, int | float) or isinstance(multiplier, bool):
            raise ConfigurationError("Speed multiplier must be a number.")
        if multiplier <= 0:
            raise ConfigurationError("Speed multiplier must be greater than zero.")
        self._do_set_speed(float(multiplier))

    @final
    def save_state(self, slot: str) -> None:
        self._require_connected()
        self._require("save_states")
        _check_slot(slot)
        self._do_save_state(slot)

    @final
    def load_state(self, slot: str) -> None:
        self._require_connected()
        self._require("save_states")
        _check_slot(slot)
        self._do_load_state(slot)

    # -------------------------------------------------------------- helpers

    @final
    def _require_connected(self) -> None:
        if not self._connected:
            raise EmulatorDisconnectedError()

    @final
    def _require(self, capability: str) -> None:
        if not self._do_capabilities().supports(capability):
            raise UnsupportedCapabilityError(capability)

    # ------------------------------------------------- backend hooks (abstract)

    @abstractmethod
    def _do_connect(self, timeout_s: float) -> None: ...

    @abstractmethod
    def _do_disconnect(self) -> None: ...

    @abstractmethod
    def _do_capabilities(self) -> Capabilities: ...

    @abstractmethod
    def _do_game_info(self) -> GameInfo: ...

    @abstractmethod
    def _do_buttons(self) -> Sequence[Button]: ...

    # The remaining hooks have safe defaults so an adapter that reports a
    # capability as unsupported need not implement it.

    def _do_memory_regions(self) -> Sequence[MemoryRegion]:
        raise UnsupportedCapabilityError("memory_read")

    def _do_read_memory(self, region: str, offset: int, length: int) -> bytes:
        raise UnsupportedCapabilityError("memory_read")

    def _do_capture_frame(self) -> Frame:
        raise UnsupportedCapabilityError("screen_capture")

    def _do_set_input(self, pressed: frozenset[Button]) -> None:
        raise UnsupportedCapabilityError("input")

    def _do_advance_frames(self, n: int) -> None:
        raise UnsupportedCapabilityError("frame_advance")

    def _do_set_speed(self, multiplier: float) -> None:
        raise UnsupportedCapabilityError("speed_control")

    def _do_save_state(self, slot: str) -> None:
        raise UnsupportedCapabilityError("save_states")

    def _do_load_state(self, slot: str) -> None:
        raise UnsupportedCapabilityError("save_states")


def _check_int(value: object, what: str, *, minimum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigurationError(f"The {what} must be an integer, got {value!r}.")
    if value < minimum:
        raise ConfigurationError(f"The {what} must be >= {minimum}, got {value}.")


def _check_slot(slot: object) -> None:
    if not isinstance(slot, str) or not slot or not slot.replace("_", "").isalnum():
        raise ConfigurationError(
            f"Save-state slot must be a non-empty name of letters, digits or underscores, "
            f"got {slot!r}."
        )


__all__ = ["DEFAULT_CONNECT_TIMEOUT_S", "EmulatorAdapter"]
