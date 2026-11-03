"""Registry of available emulator adapters.

The registry maps an adapter's ``name`` to its class so the UI can list
emulators and the connection service can create one by name. Real adapters
register themselves in :func:`default_registry` as they are implemented.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

from retro_ai_lab.core import ConfigurationError
from retro_ai_lab.emulator.base import EmulatorAdapter


@dataclass(frozen=True, slots=True)
class AdapterInfo:
    name: str
    display_name: str


class AdapterRegistry:
    def __init__(self) -> None:
        self._classes: dict[str, type[EmulatorAdapter]] = {}

    def register(self, cls: type[EmulatorAdapter]) -> type[EmulatorAdapter]:
        """Add an adapter class. Usable as a decorator."""
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
        return cls

    def names(self) -> tuple[str, ...]:
        return tuple(self._classes)

    def describe(self) -> tuple[AdapterInfo, ...]:
        return tuple(AdapterInfo(c.name, c.display_name) for c in self._classes.values())

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


def default_registry() -> AdapterRegistry:
    """The registry the application uses: every adapter that ships."""
    from retro_ai_lab.emulator.bizhawk import BizHawkAdapter
    from retro_ai_lab.emulator.fake import FakeEmulatorAdapter

    registry = AdapterRegistry()
    registry.register(FakeEmulatorAdapter)
    registry.register(BizHawkAdapter)
    return registry


__all__ = ["AdapterInfo", "AdapterRegistry", "default_registry"]
