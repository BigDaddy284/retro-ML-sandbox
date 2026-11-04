"""Registry of available emulator adapters.

The registry maps an adapter's ``name`` to its class so the UI can list
emulators and the connection service can create one by name. Real adapters
register themselves in :func:`default_registry` as they are implemented.
"""

from __future__ import annotations

import os
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Any

from retro_ai_lab.core import ConfigurationError
from retro_ai_lab.emulator.base import EmulatorAdapter


@dataclass(frozen=True, slots=True)
class AdapterInfo:
    name: str
    display_name: str


PRACTICE_ENV_VAR = "RETRO_AI_LAB_PRACTICE_CONSOLE"


def practice_console_wanted(environ: Mapping[str, str] | None = None) -> bool:
    """Whether the practice console belongs in the Console list (T-892).

    The built app lists only real emulators; tests, the profile tool and a
    developer set ``RETRO_AI_LAB_PRACTICE_CONSOLE=1`` to see the Toy Racer.
    """
    value = (os.environ if environ is None else environ).get(PRACTICE_ENV_VAR, "")
    return value.strip().lower() in {"1", "true", "yes", "on"}


class AdapterRegistry:
    def __init__(self) -> None:
        self._classes: dict[str, type[EmulatorAdapter]] = {}
        self._unlisted: set[str] = set()

    def register(self, cls: type[EmulatorAdapter], *, listed: bool = True) -> type[EmulatorAdapter]:
        """Add an adapter class. Usable as a decorator.

        An unlisted adapter can still be created by name (tests and tools
        connect to it) but :meth:`describe` leaves it out of the Console list.
        """
        if not isinstance(cls, type) or not issubclass(cls, EmulatorAdapter):
            raise ConfigurationError("Only EmulatorAdapter subclasses can be registered.")
        name = cls.name
        if not name or not name.isidentifier() or not name.islower():
            raise ConfigurationError(
                f"Adapter class {cls.__name__} needs a lowercase identifier 'name', got {name!r}."
            )
        if not cls.display_name.strip():
            raise ConfigurationError(f"Adapter {name!r} needs a display_name.")
        if name in self._classes:
            raise ConfigurationError(f"An adapter named {name!r} is already registered.")
        self._classes[name] = cls
        if not listed:
            self._unlisted.add(name)
        return cls

    def names(self) -> tuple[str, ...]:
        return tuple(self._classes)

    def describe(self) -> tuple[AdapterInfo, ...]:
        """The adapters to offer a person, in registration order (unlisted ones left out)."""
        return tuple(
            AdapterInfo(c.name, c.display_name)
            for c in self._classes.values()
            if c.name not in self._unlisted
        )

    def get(self, name: str) -> type[EmulatorAdapter]:
        try:
            return self._classes[name]
        except KeyError:
            raise ConfigurationError(
                f"No emulator adapter named {name!r}. Available: {list(self._classes)}."
            ) from None

    def create(self, name: str, **options: Any) -> EmulatorAdapter:
        """Instantiate adapter ``name`` with keyword ``options``."""
        cls = self.get(name)
        try:
            return cls(**options)
        except TypeError as exc:
            raise ConfigurationError(f"Bad options for adapter {name!r}: {exc}") from exc

    def __contains__(self, name: object) -> bool:
        return name in self._classes

    def __iter__(self) -> Iterator[str]:
        return iter(self._classes)

    def __len__(self) -> int:
        return len(self._classes)


def default_registry(*, practice_console: bool | None = None) -> AdapterRegistry:
    """The registry the application uses: every adapter that ships.

    The practice console (Toy Racer) is always registered, so anything can
    connect to it by name, but it is listed for a person only when
    ``practice_console`` is true (``None`` asks :func:`practice_console_wanted`).
    """
    from retro_ai_lab.emulator.bizhawk import BizHawkAdapter
    from retro_ai_lab.emulator.fake import FakeEmulatorAdapter
    from retro_ai_lab.emulator.mgba import MgbaAdapter

    if practice_console is None:
        practice_console = practice_console_wanted()
    registry = AdapterRegistry()
    registry.register(FakeEmulatorAdapter, listed=practice_console)
    registry.register(BizHawkAdapter)
    registry.register(MgbaAdapter)
    return registry


__all__ = [
    "PRACTICE_ENV_VAR",
    "AdapterInfo",
    "AdapterRegistry",
    "default_registry",
    "practice_console_wanted",
]
