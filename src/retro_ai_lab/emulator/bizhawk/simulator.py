"""``BizHawkSimulator``: a Python stand-in for EmuHawk running the bridge.

It speaks the protocol of ARCHITECTURE.md §5 over a real TCP socket, with
the Toy Racer behind it, so the adapter and everything above it can be
tested without an emulator, and so faults can be injected that a real
emulator would only show by accident: never answering, speaking another
protocol version, sending garbage, or dropping the link after N requests.
"""

from __future__ import annotations

import base64
import contextlib
import json
import socket
import threading
from collections.abc import Callable
from typing import Any

from retro_ai_lab.core import ConfigurationError
from retro_ai_lab.emulator.bizhawk.protocol import DEFAULT_HOST, PROTOCOL_VERSION
from retro_ai_lab.emulator.fake.toy_racer import ToyRacer

SYSTEM = "TOY"
ROM_NAME = "Toy Racer"
BIZHAWK_VERSION = "simulator"
_ACCEPT_POLL_S = 0.05

Handler = Callable[[dict[str, Any]], dict[str, Any]]


class BizHawkSimulator:
    """Context manager: ``with BizHawkSimulator() as sim: adapter(port=sim.port)``."""

    def __init__(
        self,
        *,
        host: str = DEFAULT_HOST,
        port: int = 0,
        seed: int = 0,
        script_version: int = PROTOCOL_VERSION,
        answer: bool = True,
        garbage: bool = False,
        drop_after: int | None = None,
    ) -> None:
        if drop_after is not None and drop_after < 0:
            raise ConfigurationError("drop_after must be >= 0 or None.")
        self._host = host
        self._port = port
        self._game = ToyRacer(seed)
        self._script_version = script_version
        self._answer = answer
        self._garbage = garbage
        self._drop_after = drop_after
        self._server: socket.socket | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._connections = 0
        self._requests: list[dict[str, Any]] = []
        self._paused = False
        self._free_run = False
        self._client: socket.socket | None = None
        self._closing = False

    # ---------------------------------------------------------- lifecycle

    def __enter__(self) -> BizHawkSimulator:
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.stop()

    def start(self) -> None:
        if self._server is not None:
            return
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((self._host, self._port))
        server.listen(1)
        server.settimeout(_ACCEPT_POLL_S)
        self._server = server
        self._port = server.getsockname()[1]
        self._stop.clear()
        self._thread = threading.Thread(target=self._serve, name="bizhawk-simulator", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self.drop_client()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
        if self._server is not None:
            self._server.close()
            self._server = None

    def drop_client(self) -> None:
        """Close the app's connection without a ``bye``, like a crash would."""
        with self._lock:
            client, self._client = self._client, None
        if client is not None:
            with contextlib.suppress(OSError):
                client.shutdown(socket.SHUT_RDWR)
            client.close()

    # ------------------------------------------------------------ queries

    @property
    def host(self) -> str:
        return self._host

    @property
    def port(self) -> int:
        return self._port

    @property
    def game(self) -> ToyRacer:
        return self._game

    @property
    def connections(self) -> int:
        """How many times the app has connected."""
        with self._lock:
            return self._connections

    @property
    def requests(self) -> tuple[dict[str, Any], ...]:
        """Every request received, in order."""
        with self._lock:
            return tuple(self._requests)

    @property
    def paused(self) -> bool:
        with self._lock:
            return self._paused

    @property
    def free_run(self) -> bool:
        with self._lock:
            return self._free_run

    # ------------------------------------------------------------ serving

    def _serve(self) -> None:
        assert self._server is not None
        while not self._stop.is_set():
            try:
                client, _ = self._server.accept()
            except TimeoutError:
                continue
            except OSError:
                return
            with self._lock:
                self._connections += 1
                self._client = client
            try:
                self._session(client)
            finally:
                with self._lock:
                    if self._client is client:
                        self._client = None
                    self._paused = False
                client.close()

    def _session(self, client: socket.socket) -> None:
        client.settimeout(_ACCEPT_POLL_S)
        buffer = bytearray()
        served = 0
        while not self._stop.is_set():
            cut = buffer.find(b"\n")
            if cut < 0:
                try:
                    chunk = client.recv(65536)
                except TimeoutError:
                    continue
                except OSError:
                    return
                if not chunk:
                    return
                buffer += chunk
                continue
            line = bytes(buffer[:cut])
            del buffer[: cut + 1]
            if self._drop_after is not None and served >= self._drop_after:
                return
            served += 1
            reply = self._reply(line)
            if reply is None:
                continue
            try:
                client.sendall(reply)
            except OSError:
                return
            if self._closing:
                self._closing = False
                return

    def _reply(self, line: bytes) -> bytes | None:
        try:
            request = json.loads(line.decode("utf-8"))
        except ValueError:
            request = None
        if not isinstance(request, dict):
            return _encode({"id": None, "ok": False, "error": "That was not a JSON object."})
        with self._lock:
            self._requests.append(request)
        if not self._answer:
            return None
        if self._garbage:
            return b"this is not json\n"
        request_id = request.get("id")
        cmd = request.get("cmd")
        handler = self._handlers().get(str(cmd))
        if handler is None:
            return _encode({"id": request_id, "ok": False, "error": f"Unknown command {cmd!r}."})
        try:
            body = handler(request)
        except ConfigurationError as exc:
            return _encode({"id": request_id, "ok": False, "error": str(exc)})
        return _encode({"id": request_id, "ok": True, "frame": self._game.state.frame, **body})

    def _handlers(self) -> dict[str, Handler]:
        return {
            "hello": self._hello,
            "ping": lambda _r: {},
            "bye": self._bye,
            "mode": self._mode,
        }

    def _hello(self, request: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            self._paused = not self._free_run
        return {
            "script": self._script_version,
            "bizhawk": BIZHAWK_VERSION,
            "system": SYSTEM,
            "rom": ROM_NAME,
            "hash": f"toyracer-{self._game.seed}",
            "domains": [],
            "buttons": [],
        }

    def _bye(self, request: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            self._paused = False
        self._closing = True
        return {}

    def _mode(self, request: dict[str, Any]) -> dict[str, Any]:
        free_run = request.get("free_run")
        if not isinstance(free_run, bool):
            raise ConfigurationError("'free_run' must be true or false.")
        with self._lock:
            self._free_run = free_run
            self._paused = not free_run
        return {}


def _encode(body: dict[str, Any]) -> bytes:
    return (json.dumps(body, separators=(",", ":")) + "\n").encode("utf-8")


def b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


__all__ = ["BIZHAWK_VERSION", "ROM_NAME", "SYSTEM", "BizHawkSimulator"]
