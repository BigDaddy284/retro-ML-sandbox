"""``BizHawkAdapter``: the app's side of the bridge (ARCHITECTURE.md §5).

The adapter talks to ``retro_ai_lab.lua`` running inside EmuHawk. This
module holds the connection lifecycle: open the socket, exchange ``hello``,
say ``bye``, and turn a dropped link into ``EmulatorDisconnectedError`` on
the next call. The capabilities, game description, frames, input, memory
and state commands are added task by task (T-112 to T-116).
"""

from __future__ import annotations

import base64
import binascii
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any, ClassVar

from retro_ai_lab.core import (
    Button,
    Capabilities,
    ConfigurationError,
    EmulatorDisconnectedError,
    EmulatorError,
    Frame,
    GameInfo,
    MemoryRegion,
    ProtocolError,
)
from retro_ai_lab.emulator.base import EmulatorAdapter
from retro_ai_lab.emulator.bizhawk.description import Description
from retro_ai_lab.emulator.bizhawk.image import decode_image
from retro_ai_lab.emulator.bizhawk.protocol import (
    BIZHAWK_HINT,
    DEFAULT_HOST,
    DEFAULT_PORT,
    PROTOCOL_VERSION,
    Link,
    open_link,
)

DEFAULT_CALL_TIMEOUT_S = 5.0
SCRIPT_PATH = Path(__file__).with_name("retro_ai_lab.lua")  # the bridge the app ships
DEFAULT_MAX_REGION_BYTES = 256 * 1024
"""Memory domains larger than this are not offered as regions.

The memory observer reads every region whole, every frame; a multi-megabyte
bus would swamp both the line protocol and the discovery engine.
"""


class BizHawkAdapter(EmulatorAdapter):
    name = "bizhawk"
    display_name = "BizHawk (EmuHawk)"
    emulator_name: ClassVar[str] = "BizHawk"  # in messages
    script_path: ClassVar[Path] = SCRIPT_PATH  # the bridge script this adapter talks to
    open_hint: ClassVar[str] = BIZHAWK_HINT  # what to do when nothing listens

    def __init__(
        self,
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
        *,
        call_timeout_s: float = DEFAULT_CALL_TIMEOUT_S,
        max_region_bytes: int = DEFAULT_MAX_REGION_BYTES,
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
        if isinstance(max_region_bytes, bool) or not isinstance(max_region_bytes, int):
            raise ConfigurationError("max_region_bytes must be a whole number of bytes.")
        if max_region_bytes < 1:
            raise ConfigurationError("max_region_bytes must be at least 1.")
        self._host = host
        self._port = port
        self._call_timeout_s = float(call_timeout_s)
        self._max_region_bytes = max_region_bytes
        self._link: Link | None = None
        self._hello: dict[str, Any] = {}
        self._description: Description | None = None
        self._free_run = False
        self._hold = True
        self._held: frozenset[Button] = frozenset()

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

    @property
    def description(self) -> Description:
        """The validated ``hello``: game, buttons, domains, commands."""
        self._require_connected()
        assert self._description is not None
        return self._description

    @property
    def held(self) -> frozenset[Button]:
        """The buttons currently held down in the emulator."""
        return self._held

    @property
    def hold(self) -> bool:
        """True when the bridge holds the game still between requests (ARCHITECTURE §5.3).

        A bridge that can't pause (mGBA) says ``hold: false`` in its handshake: the game
        runs on between steps, and a step's reads describe the frame it was answered on.
        """
        return self._hold

    @property
    def free_run(self) -> bool:
        """True when the game runs between the app's steps (ARCHITECTURE.md §5.3)."""
        return self._free_run

    def ping(self) -> None:
        """Round trip to the script; raises if the link is gone."""
        self._require_connected()
        self._call("ping")

    def set_free_run(self, free_run: bool) -> None:
        """Let the game run between steps (True) or hold it (False, the default).

        In free-run the adapter no longer offers ``frame_advance``: see
        ARCHITECTURE.md §5.3.
        """
        if not isinstance(free_run, bool):
            raise ConfigurationError("free_run must be True or False.")
        self._require_connected()
        self._call("mode", free_run=free_run)
        self._free_run = free_run

    # -------------------------------------------------------------- hooks

    def check_link(self) -> bool:
        link = self._link
        if not self._connected or link is None:
            return False
        if link.peer_closed():
            self._drop()
            return False
        return True

    def _do_connect(self, timeout_s: float) -> None:
        cls = type(self)
        try:
            link = open_link(
                self._host,
                self._port,
                timeout_s=timeout_s,
                emulator=cls.emulator_name,
                hint=cls.open_hint,
            )
        except EmulatorError as exc:
            raise EmulatorError(f"{exc} The script is at {cls.script_path}.") from exc
        try:
            answer = link.call("hello", timeout_s=timeout_s, script=PROTOCOL_VERSION)
            version = answer.get("script")
            if version != PROTOCOL_VERSION:
                raise ProtocolError(
                    f"The bridge script speaks protocol {version!r} but this app needs "
                    f"{PROTOCOL_VERSION}. Use the retro_ai_lab.lua that came with this app."
                )
            description = Description.from_hello(answer)
        except BaseException:
            link.close()
            raise
        self._link = link
        self._hello = answer
        self._description = description
        self._free_run = False
        self._hold = answer.get("hold") is not False

    def _do_disconnect(self) -> None:
        link, self._link = self._link, None
        self._hello = {}
        self._description = None
        self._held = frozenset()
        self._hold = True
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
        caps = self.description.capabilities(free_run=self._free_run)
        if caps.memory_read and not self._do_memory_regions():
            caps = replace(caps, memory_read=False)
        return caps

    def _do_game_info(self) -> GameInfo:
        return self.description.game_info

    def _do_buttons(self) -> Sequence[Button]:
        return self.description.buttons

    def _do_memory_regions(self) -> Sequence[MemoryRegion]:
        return tuple(r for r in self.description.regions if r.size <= self._max_region_bytes)

    def _do_read_memory(self, region: str, offset: int, length: int) -> bytes:
        domain = self.description.domain_names[region]
        answer = self._call("read", reads=[{"domain": domain, "addr": offset, "len": length}])
        reads = answer.get("reads")
        if not isinstance(reads, list) or len(reads) != 1 or not isinstance(reads[0], str):
            raise ProtocolError("The emulator's answer to a read was not one base64 string.")
        return _b64decode(reads[0], "memory")

    def _do_capture_frame(self) -> Frame:
        answer = self._call("screenshot")
        return self._frame_from(answer)

    def _do_advance_frames(self, n: int) -> None:
        self._call("step", frames=n)

    def _do_set_speed(self, multiplier: float) -> None:
        self._call("speed", percent=max(1, round(multiplier * 100)))

    def _do_save_state(self, slot: str) -> None:
        self._call("save", slot=slot)

    def _do_load_state(self, slot: str) -> None:
        answer = self._call("load", slot=slot)
        if answer.get("loaded") is not True:
            raise ConfigurationError(f"No save state in slot {slot!r}.")

    def _do_set_input(self, pressed: frozenset[Button]) -> None:
        names = self.description.button_names
        down = {b.name for b in pressed}
        self._call("input", buttons={label: ident in down for ident, label in names.items()})
        self._held = pressed

    # ------------------------------------------------------------ helpers

    def _frame_from(self, answer: dict[str, Any]) -> Frame:
        image = answer.get("image")
        if image is None:
            raise EmulatorError("The emulator did not take a screenshot.")
        if not isinstance(image, str):
            raise ProtocolError("The emulator's screenshot was not base64 text.")
        data = _b64decode(image, "screenshot")
        number = answer.get("frame")
        return decode_image(data, number=number if isinstance(number, int) else 0)

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
        self._description = None
        self._held = frozenset()
        self._hold = True
        if link is not None:
            link.close()
        self.mark_disconnected()


def _b64decode(text: str, what: str) -> bytes:
    try:
        return base64.b64decode(text, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ProtocolError(f"The emulator's {what} was not valid base64.") from exc


__all__ = ["DEFAULT_CALL_TIMEOUT_S", "DEFAULT_MAX_REGION_BYTES", "BizHawkAdapter"]
