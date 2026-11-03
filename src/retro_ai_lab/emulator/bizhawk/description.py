"""What the bridge's ``hello`` tells us, in the app's own types.

EmuHawk names things its way (``System Bus``, ``P1 A``, ``Up``); the app's
:class:`~retro_ai_lab.core.Button` and :class:`~retro_ai_lab.core.MemoryRegion`
want lowercase identifiers. :func:`identifier` does the mapping and
:class:`Description` keeps both directions so later commands can name
things the way the emulator does. Capabilities are derived from the
commands the script says it has, so the adapter never claims more than the
bridge it is talking to implements.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from retro_ai_lab.core import Button, Capabilities, GameInfo, MemoryRegion, ProtocolError

NO_GAME = "NULL"  # what emu.getsystemid() returns with no ROM loaded
_NOT_IDENTIFIER = re.compile(r"[^a-z0-9]+")


def identifier(name: str, taken: Mapping[str, str] | None = None) -> str:
    """A lowercase identifier for an EmuHawk name, unique among ``taken``."""
    base = _NOT_IDENTIFIER.sub("_", name.strip().lower()).strip("_")
    if not base:
        base = "item"
    if base[0].isdigit():
        base = "n" + base
    if taken is None:
        return base
    candidate = base
    n = 1
    while candidate in taken and taken[candidate] != name:
        n += 1
        candidate = f"{base}_{n}"
    return candidate


@dataclass(frozen=True, slots=True)
class Description:
    """A validated ``hello``: game, buttons, memory domains and commands."""

    system: str
    title: str
    rom_hash: str | None
    bizhawk: str  # the emulator's name and version text (``emulator`` or ``bizhawk`` in hello)
    buttons: tuple[Button, ...]
    regions: tuple[MemoryRegion, ...]
    commands: frozenset[str]
    button_names: Mapping[str, str]  # identifier -> EmuHawk name
    domain_names: Mapping[str, str]  # identifier -> EmuHawk name

    @classmethod
    def from_hello(cls, hello: Mapping[str, Any]) -> Description:
        system = _text(hello, "system")
        title = _text(hello, "rom")
        if system.upper() == NO_GAME or not title:
            raise ProtocolError("Load a game in the emulator first, then connect.")
        rom_hash = _text(hello, "hash") or None
        bizhawk = _text(hello, "emulator") or _text(hello, "bizhawk") or "unknown"
        commands = frozenset(_strings(hello, "commands"))

        button_names: dict[str, str] = {}
        buttons: list[Button] = []
        for name in _strings(hello, "buttons"):
            if not name.strip():
                raise ProtocolError("The bridge listed a button with no name.")
            ident = identifier(name, button_names)
            if ident in button_names:
                continue
            button_names[ident] = name
            buttons.append(Button(ident, name))

        domain_names: dict[str, str] = {}
        regions: list[MemoryRegion] = []
        raw_domains = hello.get("domains", [])
        if not isinstance(raw_domains, list):
            raise ProtocolError("The bridge's 'domains' must be a list.")
        for entry in raw_domains:
            if not isinstance(entry, dict) or not isinstance(entry.get("name"), str):
                raise ProtocolError("Each memory domain needs a 'name'.")
            size = entry.get("size")
            if isinstance(size, bool) or not isinstance(size, int) or size < 1:
                raise ProtocolError(f"Memory domain {entry['name']!r} needs a size above zero.")
            name = entry["name"]
            if not name.strip():
                raise ProtocolError("The bridge listed a memory domain with no name.")
            ident = identifier(name, domain_names)
            if ident in domain_names:
                continue
            domain_names[ident] = name
            regions.append(MemoryRegion(ident, size, description=name))

        return cls(
            system=system,
            title=title,
            rom_hash=rom_hash,
            bizhawk=bizhawk,
            buttons=tuple(buttons),
            regions=tuple(regions),
            commands=commands,
            button_names=button_names,
            domain_names=domain_names,
        )

    @property
    def game_info(self) -> GameInfo:
        return GameInfo(self.system, self.title, rom_hash=self.rom_hash)

    def capabilities(self, *, free_run: bool = False) -> Capabilities:
        """What this bridge can do, from the commands it says it has."""
        has = self.commands.__contains__
        return Capabilities(
            memory_read=has("read") and bool(self.regions),
            screen_capture=has("screenshot"),
            input=has("input") and bool(self.buttons),
            frame_advance=has("step") and not free_run,
            speed_control=has("speed"),
            save_states=has("save") and has("load"),
        )


def _text(hello: Mapping[str, Any], key: str) -> str:
    value = hello.get(key)
    if value is None:
        return ""
    if not isinstance(value, str):
        raise ProtocolError(f"The bridge's '{key}' must be text.")
    return value.strip()


def _strings(hello: Mapping[str, Any], key: str) -> list[str]:
    value = hello.get(key, [])
    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
        raise ProtocolError(f"The bridge's '{key}' must be a list of names.")
    return value


__all__ = ["NO_GAME", "Description", "identifier"]
