"""``BizHawkAdapter``: the app's side of the bridge (ARCHITECTURE.md §5).

The adapter talks to ``retro_ai_lab.lua`` running inside EmuHawk. This
module holds the connection lifecycle: open the socket, exchange ``hello``,
say ``bye``, and turn a dropped link into ``EmulatorDisconnectedError`` on
the next call. The capabilities, game description, frames, input, memory
and state commands are added task by task (T-112 to T-116).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from retro_ai_lab.core import (
    Button,
    Capabilities,
    ConfigurationError,
    EmulatorDisconnectedError,
    GameInfo,
    ProtocolError,
)
from retro_ai_lab.emulator.base import EmulatorAdapter
from retro_ai_lab.emulator.bizhawk.protocol import (
    DEFAULT_HOST,
    DEFAULT_PORT,
    PROTOCOL_VERSION,
    Link,
    open_link,
)

DEFAULT_CALL_TIMEOUT_S = 5.0


class BizHawkAdapter(EmulatorAdapter):
    name = "bizhawk"
    display_name = "BizHawk (EmuHawk)"

    def __init__(
        self,
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
        *,
        call_timeout_s: float = DEFAULT_CALL_TIMEOUT_S,
    ) -> None:
        super().__init__()
        if not isinstance(host, str) or not host.strip():
            raise ConfigurationError("The BizHawk host must be a non-empty string.")
        if isinstance(port, bool) or not isinstance(port, int) or not 1 <= port <= 65535:
            raise ConfigurationError("The BizHawk port must be a whole number from 1 to 65535.")
        if (
            isinstance(call_timeout_s, bool)
            or not isinstance(call_timeout_s, int | float)
            or call_timeout_s <= 0
        ):
            raise ConfigurationError("The call timeout must be a number of seconds above zero.")
        self._host = host
        self._port = port
        self._call_timeout_s = float(call_timeout_s)
        self._link: Link | None = None
        self._hello: dict[str, Any] = {}

    # ------------------------------------------------------------ queries

    @property
    def host(self) -> str:
        return self._host

    @property
    def port(self) -> int:
        return self._port

    @property
    def hello(self) -> dict[str, Any]:
        """The script's answer to ``hello`` (empty when not connected)."""
        return dict(self._hello)

    def ping(self) -> None:
        """Round trip to the script; raises if the link is gone."""
        self._require_connected()
        self._call("ping")

    # -------------------------------------------------------------- hooks

    def _do_connect(self, timeout_s: float) -> None:
        link = open_link(self._host, self._port, timeout_s=timeout_s)
        try:
            answer = link.call("hello", timeout_s=timeout_s, script=PROTOCOL_VERSION)
            version = answer.get("script")
            if version != PROTOCOL_VERSION:
                raise ProtocolError(
                    f"The bridge script speaks protocol {version!r} but this app needs "
                    f"{PROTOCOL_VERSION}. Use the retro_ai_lab.lua that came with this app."
                )
        except BaseException:
            link.close()
            raise
        self._link = link
        self._hello = answer

    def _do_disconnect(self) -> None:
        link, self._link = self._link, None
        self._hello = {}
        if link is None:
            return
        try:
            if link.is_open:
                link.call("bye")
        except Exception:
            pass
        finally:
            link.close()

    def _do_capabilities(self) -> Capabilities:
        return Capabilities()

    def _do_game_info(self) -> GameInfo:
        system = str(self._hello.get("system") or "").strip() or "Unknown system"
        rom = str(self._hello.get("rom") or "").strip() or "Untitled game"
        rom_hash = self._hello.get("hash")
        return GameInfo(system, rom, rom_hash=str(rom_hash) if rom_hash else None)

    def _do_buttons(self) -> Sequence[Button]:
        return ()

    # ------------------------------------------------------------ helpers

    def _call(self, cmd: str, **fields: Any) -> dict[str, Any]:
        link = self._link
        if link is None or not link.is_open:
            self._drop()
            raise EmulatorDisconnectedError()
        try:
            return link.call(cmd, **fields)
        except (EmulatorDisconnectedError, ProtocolError):
            self._drop()
            raise

    def _drop(self) -> None:
        link, self._link = self._link, None
        self._hello = {}
        if link is not None:
            link.close()
        self.mark_disconnected()


__all__ = ["DEFAULT_CALL_TIMEOUT_S", "BizHawkAdapter"]
