"""T-002: error hierarchy."""

from __future__ import annotations

import pytest

from retro_ai_lab.core import (
    ConfigurationError,
    EmulatorDisconnectedError,
    EmulatorError,
    ProtocolError,
    RetroAILabError,
    UnsupportedCapabilityError,
)


@pytest.mark.parametrize(
    "cls",
    [ConfigurationError, EmulatorError, EmulatorDisconnectedError, UnsupportedCapabilityError],
)
def test_all_errors_derive_from_base(cls: type[Exception]) -> None:
    assert issubclass(cls, RetroAILabError)
    assert issubclass(cls, Exception)


@pytest.mark.parametrize(
    "cls", [EmulatorDisconnectedError, UnsupportedCapabilityError, ProtocolError]
)
def test_emulator_errors_derive_from_emulator_error(cls: type[Exception]) -> None:
    assert issubclass(cls, EmulatorError)


def test_configuration_error_is_not_an_emulator_error() -> None:
    assert not issubclass(ConfigurationError, EmulatorError)


def test_disconnected_has_default_message() -> None:
    assert str(EmulatorDisconnectedError()) == "The emulator is not connected."
    assert str(EmulatorDisconnectedError("link dropped")) == "link dropped"


def test_unsupported_capability_records_name() -> None:
    err = UnsupportedCapabilityError("save_states")
    assert err.capability == "save_states"
    assert "save_states" in str(err)
    custom = UnsupportedCapabilityError("input", "No pad.")
    assert custom.capability == "input"
    assert str(custom) == "No pad."


def test_errors_can_be_caught_by_base() -> None:
    with pytest.raises(RetroAILabError):
        raise ProtocolError("bad frame")
