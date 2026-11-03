"""Wire format of the BizHawk bridge (ARCHITECTURE.md §5).

One JSON object per line in each direction. The app numbers its requests;
the script answers each in order with the same ``id`` and ``ok`` true or
false. Bytes travel as base64. Everything the script could get wrong is
turned into :class:`~retro_ai_lab.core.ProtocolError` here, and anything the
socket does wrong into :class:`~retro_ai_lab.core.EmulatorDisconnectedError`
or :class:`~retro_ai_lab.core.EmulatorError`, so the adapter only deals in
the app's own errors.
"""

from __future__ import annotations

import contextlib
import json
import socket
from typing import Any

from retro_ai_lab.core import EmulatorDisconnectedError, EmulatorError, ProtocolError

PROTOCOL_VERSION = 1
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 43110
MAX_LINE = 16 * 1024 * 1024  # a screenshot in base64 is well under this
_CHUNK = 65536


def encode_request(request_id: int, cmd: str, **fields: Any) -> bytes:
    """One request line, newline included."""
    if "id" in fields or "cmd" in fields:
        raise ProtocolError("Request fields can't be called 'id' or 'cmd'.")
    body = {"id": request_id, "cmd": cmd, **fields}
    return (json.dumps(body, separators=(",", ":"), ensure_ascii=True) + "\n").encode("ascii")


def decode_response(line: bytes, expected_id: int) -> dict[str, Any]:
    """Parse one response line and check it answers request ``expected_id``.

    Returns the whole object. ``ok`` false is reported as an
    :class:`EmulatorError` carrying the script's message.
    """
    try:
        data = json.loads(line.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise ProtocolError("The emulator sent something that was not JSON.") from exc
    if not isinstance(data, dict):
        raise ProtocolError("The emulator sent a JSON value that was not an object.")
    if data.get("id") != expected_id:
        raise ProtocolError(
            f"The emulator answered request {data.get('id')!r} instead of {expected_id}."
        )
    ok = data.get("ok")
    if ok is True:
        return data
    if ok is False:
        raise EmulatorError(str(data.get("error") or "The emulator refused the request."))
    raise ProtocolError("The emulator's answer had no 'ok' field.")


class LineChannel:
    """Newline-framed messages over a connected socket, with a read buffer."""

    def __init__(self, sock: socket.socket) -> None:
        self._sock = sock
        self._buffer = bytearray()
        self._open = True

    @property
    def is_open(self) -> bool:
        return self._open

    def send_line(self, line: bytes) -> None:
        try:
            self._sock.sendall(line)
        except OSError as exc:
            self.close()
            raise EmulatorDisconnectedError("The emulator went away.") from exc

    def receive_line(self, timeout_s: float) -> bytes:
        """The next line without its newline; waits at most ``timeout_s``."""
        while True:
            cut = self._buffer.find(b"\n")
            if cut >= 0:
                line = bytes(self._buffer[:cut])
                del self._buffer[: cut + 1]
                return line
            if len(self._buffer) > MAX_LINE:
                self.close()
                raise ProtocolError("The emulator sent a message that was far too long.")
            self._sock.settimeout(timeout_s)
            try:
                chunk = self._sock.recv(_CHUNK)
            except TimeoutError as exc:
                raise EmulatorError(
                    f"The emulator did not answer within {timeout_s:g} seconds."
                ) from exc
            except OSError as exc:
                self.close()
                raise EmulatorDisconnectedError("The emulator went away.") from exc
            if not chunk:
                self.close()
                raise EmulatorDisconnectedError("The emulator closed the connection.")
            self._buffer += chunk

    def close(self) -> None:
        if not self._open:
            return
        self._open = False
        with contextlib.suppress(OSError):
            self._sock.shutdown(socket.SHUT_RDWR)
        self._sock.close()


class Link:
    """Numbered request/response calls over a :class:`LineChannel`."""

    def __init__(self, channel: LineChannel, *, timeout_s: float) -> None:
        self._channel = channel
        self._timeout_s = timeout_s
        self._next_id = 1

    @property
    def is_open(self) -> bool:
        return self._channel.is_open

    @property
    def timeout_s(self) -> float:
        return self._timeout_s

    def call(self, cmd: str, *, timeout_s: float | None = None, **fields: Any) -> dict[str, Any]:
        """Send ``cmd`` and return the script's answer (an ``ok`` response)."""
        if not self._channel.is_open:
            raise EmulatorDisconnectedError()
        request_id = self._next_id
        self._next_id += 1
        self._channel.send_line(encode_request(request_id, cmd, **fields))
        line = self._channel.receive_line(self._timeout_s if timeout_s is None else timeout_s)
        try:
            return decode_response(line, request_id)
        except ProtocolError:
            self._channel.close()
            raise

    def close(self) -> None:
        self._channel.close()


def open_link(host: str, port: int, *, timeout_s: float) -> Link:
    """Connect to the bridge; failures become child-readable errors."""
    try:
        sock = socket.create_connection((host, port), timeout=timeout_s)
    except TimeoutError as exc:
        raise EmulatorError(
            f"BizHawk did not answer on port {port} within {timeout_s:g} seconds."
        ) from exc
    except OSError as exc:
        raise EmulatorError(
            f"Couldn't find BizHawk on port {port}. Open retro_ai_lab.lua in EmuHawk's "
            "Lua Console (Tools > Lua Console > Open Script) and try again."
        ) from exc
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    return Link(LineChannel(sock), timeout_s=timeout_s)


__all__ = [
    "DEFAULT_HOST",
    "DEFAULT_PORT",
    "MAX_LINE",
    "PROTOCOL_VERSION",
    "LineChannel",
    "Link",
    "decode_response",
    "encode_request",
    "open_link",
]
