"""T-006: adapter registry."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pytest

from retro_ai_lab.core import Button, Capabilities, ConfigurationError, GameInfo
from retro_ai_lab.emulator.base import EmulatorAdapter
from retro_ai_lab.emulator.fake import FakeEmulatorAdapter
from retro_ai_lab.emulator.registry import AdapterInfo, AdapterRegistry, default_registry


class Stub(EmulatorAdapter):
    name = "stub"
    display_name = "Stub"

    def __init__(self, flavour: str = "plain") -> None:
        super().__init__()
        self.flavour = flavour

    def _do_connect(self, timeout_s: float) -> None: ...

    def _do_disconnect(self) -> None: ...

    def _do_capabilities(self) -> Capabilities:
        return Capabilities()

    def _do_game_info(self) -> GameInfo:
        return GameInfo("X", "Stub")

    def _do_buttons(self) -> Sequence[Button]:
        return []


def test_register_list_and_create() -> None:
    reg = AdapterRegistry()
    assert len(reg) == 0
    assert reg.register(Stub) is Stub
    assert reg.names() == ("stub",)
    assert reg.describe() == (AdapterInfo("stub", "Stub"),)
    assert "stub" in reg
    assert list(reg) == ["stub"]
    a = reg.create("stub", flavour="mint")
    assert isinstance(a, Stub)
    assert a.flavour == "mint"
    assert reg.get("stub") is Stub


def test_unknown_name_is_a_configuration_error() -> None:
    reg = AdapterRegistry()
    with pytest.raises(ConfigurationError, match="No emulator adapter named 'nope'"):
        reg.create("nope")
    with pytest.raises(ConfigurationError):
        reg.get("nope")


def test_bad_options_are_a_configuration_error() -> None:
    reg = AdapterRegistry()
    reg.register(Stub)
    with pytest.raises(ConfigurationError, match="Bad options"):
        reg.create("stub", colour="red")


def test_duplicate_name_rejected() -> None:
    reg = AdapterRegistry()
    reg.register(Stub)

    class Stub2(Stub):
        pass

    with pytest.raises(ConfigurationError, match="already registered"):
        reg.register(Stub2)


@pytest.mark.parametrize("bad_name", ["", "Stub", "has space", "1st"])
def test_invalid_adapter_names_rejected(bad_name: str) -> None:
    class Bad(Stub):
        name = bad_name

    with pytest.raises(ConfigurationError):
        AdapterRegistry().register(Bad)


def test_missing_display_name_rejected() -> None:
    class Bad(Stub):
        name = "bad"
        display_name = "  "

    with pytest.raises(ConfigurationError):
        AdapterRegistry().register(Bad)


@pytest.mark.parametrize("not_adapter", [object, int, "stub", Stub()])
def test_non_adapter_classes_rejected(not_adapter: Any) -> None:
    with pytest.raises(ConfigurationError):
        AdapterRegistry().register(not_adapter)


def test_default_registry_ships_the_fake_and_bizhawk_adapters() -> None:
    from retro_ai_lab.emulator.bizhawk import BizHawkAdapter

    reg = default_registry()
    assert reg.names() == ("fake", "bizhawk")
    assert reg.get("bizhawk") is BizHawkAdapter
    assert isinstance(reg.create("bizhawk", port=43111), BizHawkAdapter)
    assert "fake" in reg
    assert reg.get("fake") is FakeEmulatorAdapter
    a = reg.create("fake", seed=4)
    assert isinstance(a, FakeEmulatorAdapter)
    assert a.game.seed == 4
    # Independent instances each time.
    assert default_registry() is not reg
