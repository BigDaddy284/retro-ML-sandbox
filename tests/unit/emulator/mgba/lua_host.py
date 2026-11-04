"""Run the real mGBA bridge script without mGBA.

The script is loaded into an in-process Lua 5.4 (``lupa``) with stubs of
mGBA's scripting globals (``emu``, ``callbacks``, ``console``, ``C`` and the
event-driven ``socket``) that record calls and play the Toy Racer. The host
thread runs one emulated frame per tick the way mGBA does: socket events are
delivered, ``keysRead`` callbacks fire, the game advances with the keys the
script set, then ``frame`` callbacks fire. The sockets are real, so the app's
adapter connects to the script over genuine TCP.
"""

from __future__ import annotations

import contextlib
import select
import socket
import threading
import time
from pathlib import Path
from typing import Any

import lupa.lua54 as lua54

from retro_ai_lab.emulator.bizhawk.image import encode_png
from retro_ai_lab.emulator.fake.toy_racer import RAM_SIZE, ToyRacer
from retro_ai_lab.emulator.mgba import SCRIPT_PATH

SCRIPT = SCRIPT_PATH
TITLE = b"TOYRACER"
CODE = b"ATRE"
CRC32 = 0xABCDEF01
WRAM_BASE = 0x02000000
IWRAM_BASE = 0x03000000
IWRAM_SIZE = 32768
GBA_KEYS: dict[bytes, int] = {
    b"A": 0,
    b"B": 1,
    b"SELECT": 2,
    b"START": 3,
    b"RIGHT": 4,
    b"LEFT": 5,
    b"UP": 6,
    b"DOWN": 7,
    b"R": 8,
    b"L": 9,
}
TICK_S = 0.0005

_GLUE = r"""
return function(py, chunk, lua_port)
  RETRO_AI_LAB_PORT = lua_port
  local function wrap_client(h)
    return {
      receive = function(self, max)
        local r = py.client_receive(h, max)
        return r.chunk, r.err
      end,
      send = function(self, data) return py.client_send(h, data) end,
      close = function(self) py.client_close(h) end,
      add = function(self, event, fn) return py.client_add(h, event, fn) end,
      hasdata = function(self) return py.client_hasdata(h) end,
      poll = function(self) end,
    }
  end
  socket = {}
  function socket.bind(host, port)
    local h = py.bind(host, port)
    if h == nil then return nil, "address in use" end
    return {
      listen = function(self, backlog) return py.server_listen(h, backlog) end,
      accept = function(self)
        local c = py.server_accept(h)
        if c == nil then return nil, "timeout" end
        return wrap_client(c)
      end,
      close = function(self) py.server_close(h) end,
      add = function(self, event, fn) return py.server_add(h, event, fn) end,
      poll = function(self) end,
    }
  end
  socket.tcp = function() error("not stubbed") end

  C = {
    PLATFORM = { GBA = 1, GB = 0 },
    GBA_KEY = py.gba_keys(),
    GB_KEY = { A = 0, B = 1, SELECT = 2, START = 3, RIGHT = 4, LEFT = 5, UP = 6, DOWN = 7 },
    CHECKSUM = { CRC32 = 0 },
  }
  local function domain(key)
    return {
      size = function(self) return py.domain_size(key) end,
      base = function(self) return py.domain_base(key) end,
      name = function(self) return key end,
      readRange = function(self, addr, len) return py.read_range(key, addr, len) end,
    }
  end
  emu = {
    platform = function(self) return C.PLATFORM.GBA end,
    getGameTitle = function(self) return py.title() end,
    getGameCode = function(self) return py.code() end,
    checksum = function(self, kind) return py.checksum(kind) end,
    currentFrame = function(self) return py.framecount() end,
    setKeys = function(self, mask) py.set_keys(mask) end,
    getKeys = function(self) return py.get_keys() end,
    screenshot = function(self, path) py.screenshot(path) end,
    saveStateBuffer = function(self, flags) return py.save_state() end,
    loadStateBuffer = function(self, buffer, flags) return py.load_state(buffer) end,
    memory = { wram = domain("wram"), iwram = domain("iwram") },
  }
  callbacks = {
    add = function(self, name, fn) return py.callbacks_add(name, fn) end,
    remove = function(self, id) end,
  }
  console = {
    log = function(self, text) py.log(text) end,
    warn = function(self, text) py.log(text) end,
    error = function(self, text) py.log(text) end,
  }
  chunk()
  return true
end
"""


class MgbaScriptHost:
    """The bridge script running against stubs; ``port`` is where it listens."""

    def __init__(self, *, seed: int = 0, port: int = 0) -> None:
        self._runtime = lua54.LuaRuntime(encoding=None, unpack_returned_tuples=False)
        self._game = ToyRacer(seed)
        self._requested_port = port
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._running = threading.Event()
        self._running.set()
        self._budget = 0
        self._servers: dict[int, socket.socket] = {}
        self._clients: dict[int, socket.socket] = {}
        self._events: dict[tuple[int, bytes], list[Any]] = {}
        self._callbacks: dict[bytes, list[Any]] = {}
        self._next_handle = 1
        self.port = 0
        self.mask = 0
        self.real_mask = 0  # what the host's person holds on the controller (T-887)
        self.masks: list[int] = []  # setKeys per frame
        self.frames = 0
        self.log: list[bytes] = []
        self.calls: list[str] = []
        self.error: str | None = None
        self.screenshots: list[bytes] = []
        self.write_screenshots = True
        self._bridge: dict[str, Any] = {
            "bind": self._bind,
            "server_listen": self._server_listen,
            "server_accept": self._server_accept,
            "server_close": self._server_close,
            "server_add": self._server_add,
            "client_receive": self._client_receive,
            "client_send": self._client_send,
            "client_close": self._client_close,
            "client_add": self._client_add,
            "client_hasdata": self._client_hasdata,
            "gba_keys": lambda: self._runtime.table_from(dict(GBA_KEYS)),
            "title": lambda: TITLE,
            "code": lambda: CODE,
            "checksum": self._checksum,
            "framecount": lambda: self._game.state.frame,
            "set_keys": self._set_keys,
            "get_keys": lambda: self.mask,
            "screenshot": self._screenshot,
            "save_state": self._save_state,
            "load_state": self._load_state,
            "domain_size": self._domain_size,
            "domain_base": self._domain_base,
            "read_range": self._read_range,
            "callbacks_add": self._callbacks_add,
            "log": self.log.append,
        }

    # ---------------------------------------------------------- lifecycle

    def __enter__(self) -> MgbaScriptHost:
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.stop()

    @property
    def game(self) -> ToyRacer:
        return self._game

    @property
    def pressed(self) -> tuple[str, ...]:
        """The Toy Racer buttons the current key mask holds down."""
        names = {b"A": "a", b"B": "b", b"START": "start"}
        names.update({k: k.decode().lower() for k in (b"UP", b"DOWN", b"LEFT", b"RIGHT")})
        return tuple(
            names[key] for key, bit in GBA_KEYS.items() if key in names and self.mask & (1 << bit)
        )

    def callback_names(self) -> tuple[str, ...]:
        return tuple(sorted(name.decode() for name, fns in self._callbacks.items() if fns))

    def start(self) -> None:
        if self._thread is not None:
            return
        runtime = self._runtime
        chunk = runtime.compile(SCRIPT.read_text(encoding="utf-8"), name="retro_ai_lab.lua")
        glue = runtime.execute(_GLUE)
        py = runtime.table_from({name.encode(): fn for name, fn in self._bridge.items()})
        try:
            glue(py, chunk, self._requested_port)
        except lua54.LuaError as exc:
            self.error = str(exc)
            raise RuntimeError(self.error) from exc
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="mgba-script-host", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
        for fn in self._callbacks.get(b"stop", []):
            with contextlib.suppress(lua54.LuaError):
                fn()
        for sock in list(self._clients.values()) + list(self._servers.values()):
            sock.close()
        self._clients.clear()
        self._servers.clear()

    # ------------------------------------------------------------- running

    def hold(self) -> None:
        """Freeze the game: requests are still served, but no frames pass."""
        self._running.clear()
        time.sleep(TICK_S * 4)

    def release(self) -> None:
        self._running.set()

    def run_frames(self, n: int, *, timeout_s: float = 5.0) -> None:
        """While held: let exactly ``n`` frames pass, then return."""
        with self._lock:
            self._budget += n
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            with self._lock:
                if self._budget == 0:
                    return
            time.sleep(TICK_S)
        raise AssertionError("the host did not run the frames in time")

    def _run(self) -> None:
        while not self._stop.is_set():
            if self._running.is_set():
                if not self.tick():
                    return
            else:
                with self._lock:
                    budget = self._budget
                if budget > 0:
                    if not self.tick():
                        return
                    with self._lock:
                        self._budget -= 1
                else:
                    try:
                        self._deliver_socket_events()
                    except lua54.LuaError as exc:
                        self.error = str(exc)
                        return
            time.sleep(TICK_S)

    def tick(self) -> bool:
        """One emulated frame, the way mGBA pumps a script."""
        try:
            self._deliver_socket_events()
            self.mask = self.real_mask  # the person's keys, unless keysRead overrides them
            self._fire(b"keysRead")
            self._game.step(self.pressed)
            self.masks.append(self.mask)
            self.frames += 1
            self._fire(b"frame")
        except lua54.LuaError as exc:
            self.error = str(exc)
            return False
        return True

    def _fire(self, name: bytes) -> None:
        for fn in list(self._callbacks.get(name, [])):
            fn()

    def _deliver_socket_events(self) -> None:
        for handle, server in list(self._servers.items()):
            ready, _, _ = select.select([server], [], [], 0)
            if ready:
                for fn in list(self._events.get((handle, b"received"), [])):
                    fn()
        for handle, client in list(self._clients.items()):
            ready, _, _ = select.select([client], [], [], 0)
            if ready:
                for fn in list(self._events.get((handle, b"received"), [])):
                    fn()

    def _record(self, call: str) -> None:
        with self._lock:
            self.calls.append(call)

    # ----------------------------------------------------- socket stand-in

    def _handle(self) -> int:
        handle = self._next_handle
        self._next_handle += 1
        return handle

    def _bind(self, host: bytes, port: int) -> int | None:
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            server.bind((host.decode(), int(port)))
        except OSError:
            server.close()
            return None
        handle = self._handle()
        self._servers[handle] = server
        return handle

    def _server_listen(self, handle: int, backlog: int | None) -> bool:
        server = self._servers[handle]
        server.listen(int(backlog or 1))
        server.setblocking(False)
        self.port = server.getsockname()[1]
        return True

    def _server_accept(self, handle: int) -> int | None:
        try:
            client, _ = self._servers[handle].accept()
        except (BlockingIOError, OSError):
            return None
        client.setblocking(False)
        client_handle = self._handle()
        self._clients[client_handle] = client
        return client_handle

    def _server_close(self, handle: int) -> None:
        server = self._servers.pop(handle, None)
        if server is not None:
            server.close()

    def _server_add(self, handle: int, event: bytes, fn: Any) -> int:
        self._events.setdefault((handle, event), []).append(fn)
        return len(self._events[(handle, event)])

    def _client_add(self, handle: int, event: bytes, fn: Any) -> int:
        self._events.setdefault((handle, event), []).append(fn)
        return len(self._events[(handle, event)])

    def _client_hasdata(self, handle: int) -> bool:
        client = self._clients.get(handle)
        if client is None:
            return False
        ready, _, _ = select.select([client], [], [], 0)
        return bool(ready)

    def _client_receive(self, handle: int, max_bytes: int) -> Any:
        """mGBA ``receive(max)``: data, or nil + error ("timeout" when nothing waits)."""
        client = self._clients.get(handle)
        if client is None:
            return self._result(err=b"closed")
        try:
            chunk = client.recv(int(max_bytes))
        except BlockingIOError:
            return self._result(err=b"timeout")
        except OSError:
            return self._result(err=b"closed")
        if not chunk:
            return self._result(err=b"closed")
        return self._result(chunk=chunk)

    def _result(self, **fields: bytes) -> Any:
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
        for key in [k for k in self._events if k[0] == handle]:
            del self._events[key]
        if client is not None:
            client.close()

    # --------------------------------------------------- emulator stand-ins

    def _checksum(self, kind: int) -> int:
        self._record("checksum")
        return CRC32

    def _set_keys(self, mask: int) -> None:
        self._record("setKeys")
        self.mask = int(mask)

    def _screenshot(self, path: bytes) -> None:
        self._record("screenshot")
        self.screenshots.append(path)
        if self.write_screenshots:
            Path(path.decode()).write_bytes(encode_png(self._game.render()))

    def _save_state(self) -> bytes:
        self._record("saveStateBuffer")
        return self._game.snapshot()

    def _load_state(self, buffer: bytes) -> bool:
        self._record("loadStateBuffer")
        try:
            self._game.restore(bytes(buffer))
        except Exception:
            return False
        return True

    def _domain_size(self, key: bytes) -> int:
        return RAM_SIZE if key == b"wram" else IWRAM_SIZE

    def _domain_base(self, key: bytes) -> int:
        return WRAM_BASE if key == b"wram" else IWRAM_BASE

    def _read_range(self, key: bytes, addr: int, length: int) -> bytes:
        self._record("readRange")
        base = self._domain_base(key)
        start = int(addr) - base
        if key == b"wram":
            return self._game.ram[start : start + int(length)]
        return bytes(int(length))

    def _callbacks_add(self, name: bytes, fn: Any) -> int:
        self._callbacks.setdefault(bytes(name), []).append(fn)
        return len(self._callbacks[bytes(name)])


def load_internals() -> Any:
    """The script's ``json`` and ``base64`` tables, for direct tests."""
    runtime = lua54.LuaRuntime(encoding=None, unpack_returned_tuples=False)
    runtime.execute("RETRO_AI_LAB_INTERNALS = {}")
    runtime.execute(SCRIPT.read_text(encoding="utf-8"))
    return runtime.globals().RETRO_AI_LAB_INTERNALS


__all__ = ["CODE", "CRC32", "GBA_KEYS", "SCRIPT", "TITLE", "MgbaScriptHost", "load_internals"]
