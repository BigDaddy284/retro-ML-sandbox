"""The adapter contract test suite, reused for every emulator adapter.

Subclass :class:`AdapterContract` in a test module as ``TestXxxContract`` and
implement :meth:`make_adapter`. Every real adapter (BizHawk, VBA-M, ...) runs
the same suite so behaviour stays identical across emulators (DESIGN.md §7).

Tests that need a capability the adapter lacks skip with a recorded reason;
that is the one place a skip is acceptable, because the capability model is
itself what makes the operation inapplicable (TESTING_STRATEGY.md §2.3).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest

from retro_ai_lab.core import (
    Button,
    Capabilities,
    ConfigurationError,
    EmulatorDisconnectedError,
    Frame,
    GameInfo,
    UnsupportedCapabilityError,
)
from retro_ai_lab.emulator.base import EmulatorAdapter

_OPERATIONS: list[tuple[str, Callable[[EmulatorAdapter], Any]]] = [
    ("capabilities", lambda a: a.capabilities()),
    ("game_info", lambda a: a.game_info()),
    ("buttons", lambda a: a.buttons()),
    ("memory_regions", lambda a: a.memory_regions()),
    ("read_memory", lambda a: a.read_memory("ram", 0, 1)),
    ("capture_frame", lambda a: a.capture_frame()),
    ("set_input", lambda a: a.set_input(())),
    ("advance_frames", lambda a: a.advance_frames(1)),
    ("set_speed", lambda a: a.set_speed(1.0)),
    ("save_state", lambda a: a.save_state("contract")),
    ("load_state", lambda a: a.load_state("contract")),
]

_GATED: list[tuple[str, str]] = [
    ("memory_read", "memory_regions"),
    ("memory_read", "read_memory"),
    ("screen_capture", "capture_frame"),
    ("input", "set_input"),
    ("frame_advance", "advance_frames"),
    ("speed_control", "set_speed"),
    ("save_states", "save_state"),
    ("save_states", "load_state"),
]


class AdapterContract:
    """Behaviour every adapter must exhibit. See module docstring."""

    def make_adapter(self, capabilities: Capabilities | None = None) -> EmulatorAdapter:
        """Return a fresh, *disconnected* adapter.

        ``capabilities`` asks for a specific capability set when the adapter
        can be configured; adapters that cannot may ignore it.
        """
        raise NotImplementedError

    @pytest.fixture
    def adapter(self) -> EmulatorAdapter:
        a = self.make_adapter()
        a.connect()
        return a

    @staticmethod
    def _need(adapter: EmulatorAdapter, capability: str) -> None:
        if not adapter.capabilities().supports(capability):
            pytest.skip(f"adapter does not support {capability}")

    # ------------------------------------------------------------ identity

    def test_has_name_and_display_name(self, adapter: EmulatorAdapter) -> None:
        assert adapter.name and adapter.name.isidentifier() and adapter.name.islower()
        assert adapter.display_name.strip()

    def test_describes_itself(self, adapter: EmulatorAdapter) -> None:
        assert isinstance(adapter.capabilities(), Capabilities)
        assert isinstance(adapter.game_info(), GameInfo)
        buttons = adapter.buttons()
        assert buttons, "an adapter must expose at least one button"
        assert all(isinstance(b, Button) for b in buttons)
        assert len(set(buttons)) == len(buttons)

    def test_description_is_stable(self, adapter: EmulatorAdapter) -> None:
        assert adapter.capabilities() == adapter.capabilities()
        assert adapter.game_info() == adapter.game_info()
        assert adapter.buttons() == adapter.buttons()

    # ----------------------------------------------------------- lifecycle

    def test_starts_disconnected_and_connects(self) -> None:
        a = self.make_adapter()
        assert not a.is_connected
        a.connect()
        assert a.is_connected
        a.disconnect()
        assert not a.is_connected

    def test_connect_and_disconnect_are_idempotent(self) -> None:
        a = self.make_adapter()
        a.disconnect()
        a.connect()
        a.connect()
        assert a.is_connected
        a.disconnect()
        a.disconnect()
        assert not a.is_connected

    def test_reconnect_works(self) -> None:
        a = self.make_adapter()
        a.connect()
        a.disconnect()
        a.connect()
        assert a.is_connected
        assert a.buttons()

    @pytest.mark.parametrize(("op", "call"), _OPERATIONS, ids=[o for o, _ in _OPERATIONS])
    def test_operations_require_connection(
        self, op: str, call: Callable[[EmulatorAdapter], Any]
    ) -> None:
        a = self.make_adapter()
        with pytest.raises(EmulatorDisconnectedError):
            call(a)

    # -------------------------------------------------------- capabilities

    @pytest.mark.parametrize(("cap", "op"), _GATED, ids=[f"{c}:{o}" for c, o in _GATED])
    def test_disabled_capability_raises(self, cap: str, op: str) -> None:
        caps = Capabilities.from_dict({**Capabilities.all().to_dict(), cap: False})
        a = self.make_adapter(caps)
        a.connect()
        if a.capabilities().supports(cap):
            pytest.skip("adapter capabilities are not configurable")
        call = dict(_OPERATIONS)[op]
        with pytest.raises(UnsupportedCapabilityError) as info:
            call(a)
        assert info.value.capability == cap

    # -------------------------------------------------------------- memory

    def test_memory_regions_and_reads(self, adapter: EmulatorAdapter) -> None:
        self._need(adapter, "memory_read")
        regions = adapter.memory_regions()
        assert regions
        for region in regions:
            data = adapter.read_memory(region.name, 0, region.size)
            assert isinstance(data, bytes)
            assert len(data) == region.size
            assert adapter.read_memory(region.name, 0, 0) == b""
            last = adapter.read_memory(region.name, region.size - 1, 1)
            assert last == data[-1:]

    def test_memory_read_is_repeatable_without_advancing(self, adapter: EmulatorAdapter) -> None:
        self._need(adapter, "memory_read")
        region = adapter.memory_regions()[0]
        a = adapter.read_memory(region.name, 0, region.size)
        b = adapter.read_memory(region.name, 0, region.size)
        assert a == b

    def test_memory_read_rejects_out_of_range(self, adapter: EmulatorAdapter) -> None:
        self._need(adapter, "memory_read")
        region = adapter.memory_regions()[0]
        with pytest.raises(ConfigurationError):
            adapter.read_memory(region.name, region.size, 1)
        with pytest.raises(ConfigurationError):
            adapter.read_memory("no_such_region", 0, 1)

    # -------------------------------------------------------------- screen

    def test_capture_frame(self, adapter: EmulatorAdapter) -> None:
        self._need(adapter, "screen_capture")
        f = adapter.capture_frame()
        assert isinstance(f, Frame)
        g = adapter.capture_frame()
        assert g.size == f.size and g.channels == f.channels

    # --------------------------------------------------------------- input

    def test_set_input_accepts_each_button(self, adapter: EmulatorAdapter) -> None:
        self._need(adapter, "input")
        for b in adapter.buttons():
            adapter.set_input([b])
        adapter.set_input(adapter.buttons())
        adapter.set_input(())

    def test_set_input_rejects_foreign_button(self, adapter: EmulatorAdapter) -> None:
        self._need(adapter, "input")
        with pytest.raises(ConfigurationError):
            adapter.set_input([Button("not_a_real_button_zz", "zz")])

    # ------------------------------------------------------------- frames

    def test_advance_frames_moves_time_forward(self, adapter: EmulatorAdapter) -> None:
        self._need(adapter, "frame_advance")
        self._need(adapter, "screen_capture")
        before = adapter.capture_frame().number
        adapter.advance_frames(3)
        after = adapter.capture_frame().number
        assert after > before

    def test_advance_frames_rejects_zero(self, adapter: EmulatorAdapter) -> None:
        self._need(adapter, "frame_advance")
        with pytest.raises(ConfigurationError):
            adapter.advance_frames(0)

    # -------------------------------------------------------------- speed

    def test_set_speed_accepts_positive_values(self, adapter: EmulatorAdapter) -> None:
        self._need(adapter, "speed_control")
        adapter.set_speed(1)
        adapter.set_speed(0.5)
        adapter.set_speed(8)
        with pytest.raises(ConfigurationError):
            adapter.set_speed(0)

    # --------------------------------------------------------- save states

    def test_save_and_load_restore_memory_exactly(self, adapter: EmulatorAdapter) -> None:
        self._need(adapter, "save_states")
        self._need(adapter, "memory_read")
        self._need(adapter, "frame_advance")
        region = adapter.memory_regions()[0]
        adapter.save_state("contract_a")
        snapshot = adapter.read_memory(region.name, 0, region.size)
        if adapter.capabilities().supports("input"):
            adapter.set_input(adapter.buttons()[:1])
        adapter.advance_frames(25)
        assert adapter.read_memory(region.name, 0, region.size) != snapshot
        adapter.load_state("contract_a")
        assert adapter.read_memory(region.name, 0, region.size) == snapshot

    def test_load_unknown_slot_is_an_error(self, adapter: EmulatorAdapter) -> None:
        self._need(adapter, "save_states")
        with pytest.raises(ConfigurationError):
            adapter.load_state("slot_that_was_never_saved_zz")


__all__ = ["AdapterContract"]
