"""T-1002: the app's side of the "starting" window a built app shows while it loads."""

from __future__ import annotations

import builtins
import logging
import sys
import types

import pytest

from retro_ai_lab.core import starting


class Splash(types.ModuleType):
    """A stand-in for the build tool's ``pyi_splash`` module."""

    def __init__(self, *, alive: bool = True, broken: bool = False) -> None:
        super().__init__("pyi_splash")
        self.alive = alive
        self.broken = broken
        self.said: list[str] = []
        self.closed = 0

    def is_alive(self) -> bool:
        if self.broken:
            raise RuntimeError("This module is not initialized; did it fail to load?")
        return self.alive

    def update_text(self, text: str) -> None:
        if self.broken:
            raise ConnectionError("Unable to send command")
        self.said.append(text)

    def close(self) -> None:
        if self.broken:
            raise OSError("the socket is gone")
        self.closed += 1
        self.alive = False


@pytest.fixture(autouse=True)
def _not_looked_for_yet(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(starting, "_window", starting._UNKNOWN)


def test_from_source_it_does_nothing(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setitem(sys.modules, "pyi_splash", None)  # importing it now fails, as from source
    with caplog.at_level(logging.DEBUG):
        assert starting.showing() is False
        starting.say("Step 1 of 4: loading the window toolkit")
        starting.close()
        starting.close()
        assert starting.showing() is False
    assert caplog.records == []


def test_it_says_and_closes_once(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    splash = Splash()
    monkeypatch.setitem(sys.modules, "pyi_splash", splash)
    with caplog.at_level(logging.INFO):
        starting.say("one")
        assert starting.showing() is True
        starting.say("two")
        starting.close()
        assert starting.showing() is False
        starting.say("three")
        starting.close()
    assert splash.said == ["one", "two"]
    assert splash.closed == 1
    lines = [record.getMessage() for record in caplog.records]
    assert lines == ["The starting window is showing", "The starting window is closed"]


def test_the_steps_are_numbered_in_plain_words(monkeypatch: pytest.MonkeyPatch) -> None:
    splash = Splash()
    monkeypatch.setitem(sys.modules, "pyi_splash", splash)
    for number in range(0, len(starting.STEPS) + 2):
        starting.step(number)
    assert splash.said == [
        "Step 1 of 4: loading the window toolkit",
        "Step 2 of 4: loading the app",
        "Step 3 of 4: starting the window toolkit",
        "Step 4 of 4: building the window",
    ]
    # The window reads its text from between the first "(" and the last ")" of what it is sent.
    assert not any(set(text) & set('()[]{}$\\"') for text in splash.said)


def test_a_splash_that_was_never_up_is_left_alone(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    splash = Splash(alive=False)
    monkeypatch.setitem(sys.modules, "pyi_splash", splash)
    with caplog.at_level(logging.INFO):
        starting.say("one")
        starting.close()
    assert splash.said == [] and splash.closed == 0
    assert starting.showing() is False
    assert [record.getMessage() for record in caplog.records] == []


def test_a_splash_that_cannot_be_loaded_or_fails_is_swallowed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # It raises whatever is asked of it.
    monkeypatch.setitem(sys.modules, "pyi_splash", Splash(broken=True))
    starting.say("one")
    starting.close()
    assert starting.showing() is False

    # It is alive but cannot be spoken to or closed.
    half = Splash()
    half.update_text = Splash(broken=True).update_text  # type: ignore[method-assign]
    half.close = Splash(broken=True).close  # type: ignore[method-assign]
    monkeypatch.setattr(starting, "_window", starting._UNKNOWN)
    monkeypatch.setitem(sys.modules, "pyi_splash", half)
    starting.say("one")
    starting.close()

    # Loading it fails with something other than "not there".
    real = builtins.__import__

    def failing(name: str, *args: object, **kwargs: object) -> object:
        if name == "pyi_splash":
            raise RuntimeError("the bootloader said no")
        return real(name, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(starting, "_window", starting._UNKNOWN)
    monkeypatch.delitem(sys.modules, "pyi_splash")
    monkeypatch.setattr(builtins, "__import__", failing)
    starting.say("one")
    starting.close()
    assert starting.showing() is False
