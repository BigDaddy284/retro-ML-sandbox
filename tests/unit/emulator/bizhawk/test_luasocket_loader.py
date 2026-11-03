"""T-816: socket.lua loads the LuaSocket library shipped next to the bridge script."""

from __future__ import annotations

from typing import Any

import lupa.lua54 as lua54
import pytest

from retro_ai_lab.emulator.bizhawk.adapter import SCRIPT_PATH

LOADER = SCRIPT_PATH.with_name("socket.lua")

FAKE_CORE = """
calls = {}
local function note(text) calls[#calls + 1] = text; return 1 end
local sock = {
  setoption = function(self, name, value) return note("setoption:" .. name) end,
  bind = function(self, host, port) return note("bind:" .. host .. ":" .. port) end,
  listen = function(self, backlog) return note("listen:" .. tostring(backlog)) end,
  connect = function(self, host, port) return note("connect:" .. host .. ":" .. port) end,
}
return { tcp = function() return sock end, select = function() return "selected" end }
"""


class Loader:
    """Runs socket.lua in a fresh Lua with a recorded ``package.loadlib``."""

    def __init__(
        self,
        *,
        version: str = "Lua 5.4",
        windows: bool = True,
        env: dict[str, str] | None = None,
        fail: str | None = None,
    ) -> None:
        self.runtime = lua54.LuaRuntime(encoding=None, unpack_returned_tuples=True)
        self.loadlib_calls: list[tuple[str, str]] = []
        core = self.runtime.execute(FAKE_CORE.encode())

        def loadlib(path: bytes, symbol: bytes) -> Any:
            self.loadlib_calls.append((path.decode(), symbol.decode()))
            if fail is not None:
                return None, fail.encode()
            return lambda: core

        values = env if env is not None else {"PROCESSOR_ARCHITECTURE": "AMD64"}

        def getenv(name: bytes) -> bytes | None:
            value = values.get(name.decode())
            return None if value is None else value.encode()

        g = self.runtime.globals()
        g._VERSION = version.encode()
        g.package.loadlib = loadlib
        g.os.getenv = getenv
        sep = "\\" if windows else "/"
        g.package.config = (sep + "\n;\n?\n!\n-\n").encode()
        g.package.path = b"./?.lua"

    def run(self) -> Any:
        dofile = self.runtime.globals().dofile
        return dofile(str(LOADER).encode())


def test_windows_lua54_loads_the_x64_dll_next_to_the_script() -> None:
    loader = Loader()
    module = loader.run()
    assert loader.loadlib_calls == [
        (f"{LOADER.parent}/luasocket/x64/socket-windows-5-4.dll", "luaopen_socket_core")
    ]
    assert module.select() == b"selected" and module.core is not None
    assert loader.runtime.eval(b'package.loaded["socket"] == require("socket")') is True


@pytest.mark.parametrize(
    ("version", "env", "expected"),
    [
        ("Lua 5.1", {"PROCESSOR_ARCHITECTURE": "AMD64"}, "x64/socket-windows-5-1.dll"),
        ("Lua 5.4", {"PROCESSOR_ARCHITECTURE": "x86"}, "x86/socket-windows-5-4.dll"),
        (
            "Lua 5.4",
            {"PROCESSOR_ARCHITECTURE": "x86", "PROCESSOR_ARCHITEW6432": "AMD64"},
            "x64/socket-windows-5-4.dll",
        ),
    ],
)
def test_lua_version_and_process_bits_pick_the_file(
    version: str, env: dict[str, str], expected: str
) -> None:
    loader = Loader(version=version, env=env)
    loader.run()
    assert loader.loadlib_calls[0][0].endswith("luasocket/" + expected)


def test_other_platforms_ask_for_a_shared_object() -> None:
    loader = Loader(windows=False)
    loader.run()
    assert loader.loadlib_calls[0][0].endswith("luasocket/x64/socket-linux-5-4.so")


def test_a_failed_load_says_where_it_looked_and_what_to_do() -> None:
    loader = Loader(fail="The specified module could not be found.")
    with pytest.raises(lua54.LuaError) as info:
        loader.run()
    text = str(info.value)
    assert "socket-windows-5-4.dll" in text and "could not be found" in text
    assert "64-bit" in text and "luasocket folder" in text


def test_bind_and_connect_drive_the_core_socket() -> None:
    loader = Loader()
    module = loader.run()
    assert module.bind(b"127.0.0.1", 43110) is not None
    assert module.connect(b"127.0.0.1", 43110) is not None
    calls = [c.decode() for c in loader.runtime.globals().calls.values()]
    assert calls == [
        "setoption:reuseaddr",
        "bind:127.0.0.1:43110",
        "listen:32",
        "connect:127.0.0.1:43110",
    ]


def test_the_bridge_script_puts_its_folder_first_on_the_path() -> None:
    """Loaded from a file (as EmuHawk does), the script prepends its own folder."""
    runtime = lua54.LuaRuntime(encoding=None, unpack_returned_tuples=False)
    seen: list[bytes] = []
    runtime.execute(
        b"RETRO_AI_LAB_INTERNALS = {}; package.preload['socket'] = function() "
        b"SEEN(package.path); return {} end"
    )
    runtime.globals().SEEN = seen.append
    runtime.globals().dofile(str(SCRIPT_PATH).encode())
    assert seen and seen[0].decode().startswith(f"{SCRIPT_PATH.parent}/?.lua;")


@pytest.mark.parametrize("name", ["socket-windows-5-4.dll", "socket-windows-5-1.dll"])
def test_the_shipped_dlls_are_windows_binaries_with_their_licence(name: str) -> None:
    folder = LOADER.parent / "luasocket"
    assert (folder / "x64" / name).read_bytes()[:2] == b"MZ"
    licence = (folder / "LICENSE").read_text(encoding="utf-8")
    assert "Diego Nehab" in licence and "Archipelago" in licence
