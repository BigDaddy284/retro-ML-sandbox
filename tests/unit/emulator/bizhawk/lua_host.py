"""Run the real bridge script without EmuHawk.

The script is loaded into an in-process Lua 5.4 (``lupa``, the same Lua
version EmuHawk embeds) as a coroutine, the way EmuHawk resumes scripts once
per frame. The emulator globals it uses (``emu``, ``client``, ``gameinfo``,
``memory``, ``joypad``, ``savestate``, ``print``, ``event``) are stubs that
record calls and play the Toy Racer, and ``require("socket")`` returns a
LuaSocket look-alike over real Python sockets, so the adapter connects to
the script over genuine TCP.
"""

from __future__ import annotations

import socket
import threading
import time
from pathlib import Path
from typing import Any

import lupa.lua54 as lua54

from retro_ai_lab.emulator.fake.toy_racer import BUTTONS, RAM_SIZE, ToyRacer

SCRIPT = Path(__file__).resolve().parents[4] / (
    "src/retro_ai_lab/emulator/bizhawk/retro_ai_lab.lua"
)
SYSTEM = b"NES"
ROM_NAME = b"Toy Racer (stub)"
ROM_HASH = b"ABCDEF0123456789"
BIZHAWK_VERSION = b"2.10"
DOMAINS = (b"RAM", b"System Bus")
PAD_BUTTONS = tuple(b.label.encode() for b in BUTTONS)
TICK_S = 0.0005

_GLUE = r"""
return function(py, chunk, lua_port)
  RETRO_AI_LAB_PORT = lua_port
  local function wrap_client(h)
    return {
      settimeout = function(self, t) py.client_settimeout(h, t) end,
      receive = function(self, pattern)
        local r = py.client_receive(h)
        return r.line, r.err, r.partial
      end,
      send = function(self, data) return py.client_send(h, data) end,
      close = function(self) py.client_close(h) end,
    }
  end
  local socket = {}
  function socket.bind(host, port)
    local h = py.bind(host, port)
    return {
      settimeout = function(self, t) py.server_settimeout(h, t) end,
      accept = function(self)
        local c = py.server_accept(h)
        if c == nil then return nil, "timeout" end
        return wrap_client(c)
      end,
      getsockname = function(self) return host, py.server_port(h) end,
      close = function(self) py.server_close(h) end,
    }
  end
  package.preload["socket"] = function() return socket end

  emu = {
    framecount = function() return py.framecount() end,
    getsystemid = function() return py.systemid() end,
    frameadvance = function() coroutine.yield("frame") end,
    yield = function() coroutine.yield("yield") end,
  }
  client = {
    pause = function() py.pause() end,
    unpause = function() py.unpause() end,
    ispaused = function() return py.ispaused() end,
    getversion = function() return py.version() end,
    speedmode = function(p) py.speedmode(p) end,
    screenshot = function(path) py.screenshot(path) end,
  }
  gameinfo = {
    getromname = function() return py.romname() end,
    getromhash = function() return py.romhash() end,
  }
  memory = {
    getmemorydomainlist = function() return py.domainlist() end,
    getmemorydomainsize = function(name) return py.domainsize(name) end,
    read_bytes_as_array = function(addr, len, domain) return py.readbytes(addr, len, domain) end,
  }
  joypad = {
    get = function(controller) return py.joypad_get(controller) end,
    set = function(buttons, controller) py.joypad_set(buttons, controller) end,
  }
  savestate = {
    save = function(path) py.save(path) end,
    load = function(path) return py.load(path) end,
  }
  event = { onexit = function(f) py.onexit(f) end }
  print = function(...) py.log(table.concat({...}, " ")) end

  local co = coroutine.create(chunk)
  return function()
    local ok, kind = coroutine.resume(co)
    return ok, kind, coroutine.status(co)
  end
end
"""


class LuaScriptHost:
    """The bridge script running against stubs; ``port`` is where it listens."""

    def __init__(self, *, seed: int = 0, port: int = 0) -> None:
        self._runtime = lua54.LuaRuntime(encoding=None, unpack_returned_tuples=False)
        self._game = ToyRacer(seed)
        self._requested_port = port
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._servers: dict[int, socket.socket] = {}
        self._clients: dict[int, socket.socket] = {}
        self._buffers: dict[int, bytearray] = {}
        self._timeouts: dict[int, float | None] = {}
        self._next_handle = 1
        self.port = 0
        self.paused = False
        self.speed: float | None = None
        self.pressed: tuple[str, ...] = ()
        self.frames = 0
        self.log: list[bytes] = []
        self.calls: list[str] = []
        self.error: str | None = None
        self.exit_hooks = 0
        self.screenshots: list[bytes] = []
        self.states: dict[bytes, bytes] = {}
        self._bridge: dict[str, Any] = {
            "bind": self._bind,
            "server_settimeout": self._server_settimeout,
            "server_accept": self._server_accept,
            "server_port": self._server_port,
            "server_close": self._server_close,
            "client_settimeout": self._client_settimeout,
            "client_receive": self._client_receive,
            "client_send": self._client_send,
            "client_close": self._client_close,
            "framecount": lambda: self._game.state.frame,
            "systemid": lambda: SYSTEM,
            "pause": lambda: self._record("pause", paused=True),
            "unpause": lambda: self._record("unpause", paused=False),
            "ispaused": lambda: self.paused,
            "version": lambda: BIZHAWK_VERSION,
            "speedmode": self._speedmode,
            "screenshot": self._screenshot,
            "romname": lambda: ROM_NAME,
            "romhash": lambda: ROM_HASH,
            "domainlist": lambda: self._runtime.table_from(DOMAINS),
            "domainsize": self._domainsize,
            "readbytes": self._readbytes,
            "joypad_get": self._joypad_get,
            "joypad_set": self._joypad_set,
            "save": self._save,
            "load": self._load,
            "onexit": self._onexit,
            "log": self.log.append,
        }

    # ---------------------------------------------------------- lifecycle

    def __enter__(self) -> LuaScriptHost:
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.stop()

    @property
    def game(self) -> ToyRacer:
        return self._game

    def start(self) -> None:
        if self._thread is not None:
            return
        runtime = self._runtime
        chunk = runtime.compile(SCRIPT.read_text(encoding="utf-8"), name="retro_ai_lab.lua")
        glue = runtime.execute(_GLUE)
        py = runtime.table_from({name.encode(): fn for name, fn in self._bridge.items()})
        self._step = glue(py, chunk, self._requested_port)
        # Run until the script has bound its port (its first yield) so that
        # callers can connect as soon as start() returns.
        self._tick()
        if self.error:
            raise RuntimeError(self.error)
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="lua-script-host", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
        for sock in list(self._clients.values()) + list(self._servers.values()):
            sock.close()
        self._clients.clear()
        self._servers.clear()

    # ------------------------------------------------------------- running

    def _run(self) -> None:
        while not self._stop.is_set():
            if not self._tick():
                return
            time.sleep(TICK_S)

    def _tick(self) -> bool:
        ok, kind, status = self._step()
        if not ok:
            self.error = kind.decode("utf-8", "replace") if isinstance(kind, bytes) else str(kind)
            return False
        if kind == b"frame":
            self._game.step(self.pressed)
            self.pressed = ()
            self.frames += 1
        return status != b"dead"

    def _record(self, call: str, *, paused: bool | None = None) -> None:
        with self._lock:
            self.calls.append(call)
            if paused is not None:
                self.paused = paused

    # ----------------------------------------------------- socket stand-in

    def _handle(self) -> int:
        handle = self._next_handle
        self._next_handle += 1
        return handle

    def _bind(self, host: bytes, port: int) -> int:
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((host.decode(), int(port)))
        server.listen(1)
        handle = self._handle()
        self._servers[handle] = server
        self.port = server.getsockname()[1]
        return handle

    def _server_settimeout(self, handle: int, timeout: float | None) -> None:
        self._servers[handle].settimeout(timeout)

    def _server_port(self, handle: int) -> int:
        return int(self._servers[handle].getsockname()[1])

    def _server_accept(self, handle: int) -> int | None:
        try:
            client, _ = self._servers[handle].accept()
        except (TimeoutError, BlockingIOError):
            return None
        except OSError:
            return None
        client_handle = self._handle()
        self._clients[client_handle] = client
        self._buffers[client_handle] = bytearray()
        self._timeouts[client_handle] = None
        return client_handle

    def _server_close(self, handle: int) -> None:
        server = self._servers.pop(handle, None)
        if server is not None:
            server.close()

    def _client_settimeout(self, handle: int, timeout: float | None) -> None:
        self._timeouts[handle] = timeout
        self._clients[handle].settimeout(timeout)

    def _client_receive(self, handle: int) -> Any:
        """LuaSocket ``receive("*l")``: line, or nil + "timeout"/"closed" + partial."""
        client = self._clients.get(handle)
        if client is None:
            return self._result(err=b"closed", partial=b"")
        buffer = self._buffers[handle]
        cut = buffer.find(b"\n")
        if cut < 0:
            try:
                chunk = client.recv(65536)
            except (TimeoutError, BlockingIOError):
                partial = bytes(buffer)
                buffer.clear()
                return self._result(err=b"timeout", partial=partial)
            except OSError:
                return self._result(err=b"closed", partial=bytes(buffer))
            if not chunk:
                return self._result(err=b"closed", partial=bytes(buffer))
            buffer += chunk
            cut = buffer.find(b"\n")
            if cut < 0:
                partial = bytes(buffer)
                buffer.clear()
                return self._result(err=b"timeout", partial=partial)
        line = bytes(buffer[:cut]).rstrip(b"\r")
        del buffer[: cut + 1]
        return self._result(line=line)

    def _result(self, **fields: bytes) -> Any:
        # Lua strings are bytes on this runtime, keys included.
        return self._runtime.table_from({name.encode(): value for name, value in fields.items()})

    def _client_send(self, handle: int, data: bytes) -> int:
        client = self._clients.get(handle)
        if client is None:
            return 0
        try:
            client.sendall(data)
        except OSError:
            return 0
        return len(data)

    def _client_close(self, handle: int) -> None:
        client = self._clients.pop(handle, None)
        self._buffers.pop(handle, None)
        if client is not None:
            client.close()

    # --------------------------------------------------- emulator stand-ins

    def _speedmode(self, percent: float) -> None:
        self._record("speedmode")
        self.speed = float(percent)

    def _screenshot(self, path: bytes) -> None:
        self._record("screenshot")
        self.screenshots.append(path)

    def _domainsize(self, name: bytes) -> int:
        if name == b"RAM":
            return RAM_SIZE
        if name == b"System Bus":
            return 65536
        return 0

    def _readbytes(self, addr: int, length: int, domain: bytes) -> Any:
        self._record("read")
        ram = self._game.ram
        data = ram[addr : addr + length] if domain == b"RAM" else bytes(length)
        return self._runtime.table_from(list(data))

    def _joypad_get(self, controller: int | None) -> Any:
        self._record("joypad_get")
        if controller == 1:
            pad: dict[bytes, Any] = dict.fromkeys(PAD_BUTTONS, False)
            pad[b"X Axis"] = 0  # an analog stick is not a button
            return self._runtime.table_from(pad)
        return self._runtime.table_from({b"P1 " + name: False for name in PAD_BUTTONS})

    def _joypad_set(self, buttons: Any, controller: int | None) -> None:
        self._record("joypad_set")
        labels = {b.label.encode(): b.name for b in BUTTONS}
        pressed = []
        for name, value in buttons.items():
            if value and name in labels:
                pressed.append(labels[name])
        self.pressed = tuple(pressed)

    def _save(self, path: bytes) -> None:
        self._record("save")
        self.states[path] = self._game.snapshot()

    def _load(self, path: bytes) -> bool:
        self._record("load")
        snapshot = self.states.get(path)
        if snapshot is None:
            return False
        self._game.restore(snapshot)
        return True

    def _onexit(self, fn: Any) -> None:
        self.exit_hooks += 1


def load_internals(seed: int = 0) -> Any:
    """The script's ``json`` and ``base64`` tables, for direct tests."""
    runtime = lua54.LuaRuntime(encoding=None, unpack_returned_tuples=False)
    runtime.execute(
        "RETRO_AI_LAB_INTERNALS = {}; package.preload['socket'] = function() return {} end"
    )
    runtime.execute(SCRIPT.read_text(encoding="utf-8"))
    return runtime.globals().RETRO_AI_LAB_INTERNALS


__all__ = ["BIZHAWK_VERSION", "ROM_HASH", "ROM_NAME", "SCRIPT", "SYSTEM", "LuaScriptHost"]
