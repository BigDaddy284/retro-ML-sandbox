"""T-112: the bridge's hello as buttons, regions, game info and capabilities."""

from __future__ import annotations

from typing import Any

import pytest

from retro_ai_lab.core import Button, Capabilities, MemoryRegion, ProtocolError
from retro_ai_lab.emulator.bizhawk.description import Description, identifier

HELLO: dict[str, Any] = {
    "script": 1,
    "bizhawk": "2.10",
    "system": "NES",
    "rom": "Micro Racer (USA)",
    "hash": "0123ABCD",
    "domains": [{"name": "RAM", "size": 2048}, {"name": "System Bus", "size": 65536}],
    "buttons": ["A", "B", "Down", "Left", "Right", "Select", "Start", "Up"],
    "commands": ["bye", "hello", "mode", "ping"],
}


def test_identifiers_are_lowercase_unique_and_readable() -> None:
    assert identifier("System Bus") == "system_bus"
    assert identifier("P1 A") == "p1_a"
    assert identifier("Up") == "up"
    assert identifier("  Fast-Forward!! ") == "fast_forward"
    assert identifier("1st Player") == "n1st_player"
    assert identifier("***") == "item"
    taken = {"a": "A"}
    assert identifier("A", taken) == "a"  # same name, same identifier
    assert identifier("a", taken) == "a_2"  # a different name that collides
    taken["a_2"] = "a"
    assert identifier("A!", taken) == "a_3"
    assert Button(identifier("P1 A"), "P1 A").name == "p1_a"


def test_description_from_a_good_hello() -> None:
    d = Description.from_hello(HELLO)
    assert d.game_info.system == "NES"
    assert d.game_info.title == "Micro Racer (USA)"
    assert d.game_info.rom_hash == "0123ABCD"
    assert d.bizhawk == "2.10"
    assert d.buttons == tuple(Button(identifier(n), n) for n in HELLO["buttons"])
    assert d.button_names["select"] == "Select"
    assert d.regions == (
        MemoryRegion("ram", 2048, description="RAM"),
        MemoryRegion("system_bus", 65536, description="System Bus"),
    )
    assert d.domain_names == {"ram": "RAM", "system_bus": "System Bus"}
    assert d.commands == frozenset(HELLO["commands"])
    assert d.capabilities() == Capabilities()
    # optional fields may be missing
    bare = Description.from_hello({"system": "GB", "rom": "X"})
    assert bare.rom_hash is None and bare.bizhawk == "unknown"
    assert bare.buttons == () and bare.regions == () and bare.commands == frozenset()
    # duplicates collapse to one entry
    twice = Description.from_hello(
        {**HELLO, "buttons": ["A", "A"], "domains": HELLO["domains"] * 2}
    )
    assert [b.name for b in twice.buttons] == ["a"]
    assert [r.name for r in twice.regions] == ["ram", "system_bus"]


@pytest.mark.parametrize(
    ("patch", "match"),
    [
        ({"system": "NULL"}, "Load a game in the emulator first"),
        ({"rom": ""}, "Load a game in the emulator first"),
        ({"rom": None}, "Load a game in the emulator first"),
        ({"system": 7}, "'system' must be text"),
        ({"buttons": "A"}, "'buttons' must be a list of names"),
        ({"buttons": ["A", 3]}, "'buttons' must be a list of names"),
        ({"buttons": [" "]}, "button with no name"),
        ({"commands": [None]}, "'commands' must be a list of names"),
        ({"domains": {"name": "RAM"}}, "'domains' must be a list"),
        ({"domains": ["RAM"]}, "needs a 'name'"),
        ({"domains": [{"name": "RAM", "size": 0}]}, "needs a size above zero"),
        ({"domains": [{"name": "RAM", "size": True}]}, "needs a size above zero"),
        ({"domains": [{"name": "", "size": 4}]}, "domain with no name"),
    ],
)
def test_bad_hellos_are_protocol_errors(patch: dict[str, Any], match: str) -> None:
    with pytest.raises(ProtocolError, match=match):
        Description.from_hello({**HELLO, **patch})


@pytest.mark.parametrize(
    ("commands", "free_run", "expected"),
    [
        ([], False, {}),
        (["read"], False, {"memory_read": True}),
        (["screenshot"], False, {"screen_capture": True}),
        (["input"], False, {"input": True}),
        (["step"], False, {"frame_advance": True}),
        (["step"], True, {}),
        (["speed"], False, {"speed_control": True}),
        (["save"], False, {}),
        (["load"], False, {}),
        (["save", "load"], False, {"save_states": True}),
        (
            ["read", "screenshot", "input", "step", "speed", "save", "load"],
            False,
            dict.fromkeys(Capabilities.names(), True),
        ),
    ],
)
def test_capabilities_follow_the_commands(
    commands: list[str], free_run: bool, expected: dict[str, bool]
) -> None:
    d = Description.from_hello({**HELLO, "commands": commands})
    assert d.capabilities(free_run=free_run) == Capabilities(**expected)
    # without buttons or domains the matching flags stay off
    bare = Description.from_hello(
        {**HELLO, "commands": ["read", "input"], "buttons": [], "domains": []}
    )
    assert bare.capabilities() == Capabilities()
