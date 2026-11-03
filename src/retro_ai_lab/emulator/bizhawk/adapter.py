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
import contextlib
import logging
import os
import re
import tempfile
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
    StepResult,
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

IMAGE_TRANSFERS = ("auto", "file", "socket")
LOCAL_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})

DEFAULT_CALL_TIMEOUT_S = 5.0
SLOWEST_FPS = 30  # a step's frames are allowed to run this slowly before the link gives up
SCRIPT_PATH = Path(__file__).with_name("retro_ai_lab.lua")  # the bridge the app ships
DEFAULT_MAX_REGION_BYTES = 256 * 1024
"""Memory domains larger than this are not offered as regions.

The memory observer reads every region whole, every frame; a multi-megabyte
bus would swamp both the line protocol and the discovery engine.
"""

NOT_LIVE_MEMORY_WORDS = frozenset(
    {
        "rom",
        "vrom",
        "bus",
        "bios",
        "registers",
        # Picture and sound memory (T-835): see the note below.
        "oam",
        "vram",
        "ciram",
        "nametables",
        "palram",
        "palette",
        "palettes",
        "cgram",
        "cram",
        "vsram",
        "apuram",
    }
)
"""Words in a domain's name that mark it as not worth snapshotting (T-827).

Only the game's own working memory tells the AI anything: ROM and BIOS never
change, and a bus (EmuHawk's ``System Bus``, ``PPU Bus``) mirrors RAM and ROM
plus hardware registers, some of which change the game's state when read. On
a NES those domains are 100 KB a frame against 2 KB of RAM. A "registers"
domain (NesHawk's ``CPU registers``) is a view of the CPU, not memory, and
reading it whole threw on the user's EmuHawk (T-833).

Picture memory (sprite attributes in ``OAM``, tile maps in ``CIRAM
(nametables)`` or ``VRAM``, palettes in ``PALRAM``/``CGRAM``/``CRAM``) and
sound memory (``APURAM``) are rewritten every frame from the state in RAM,
and the picture is already what the screen capture shows. On the user's
Super Mario Bros. they made 1646 watched values out of a 2 KB game: sprite
bytes bounce between a few values and land back on their first, which reads
as lives lost and as a reset, so every go ended after a step or two (T-835).
"""

UNREADABLE_REGION = re.compile(r"could not read \d+ bytes at \d+ of '(?P<domain>[^']+)'")
"""The bridge script's message for a read the emulator refused (T-832)."""


log = logging.getLogger(__name__)


def is_live_memory(ident: str) -> bool:
    """Is region ``ident`` (the app's lowercase name) worth reading every frame?"""
    words = ident.split("_")
    return not (set(words) & NOT_LIVE_MEMORY_WORDS or ident.startswith("cart"))


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
        image_transfer: str = "auto",
    ) -> None:
        super().__init__()
        if not isinstance(host, str) or not host.strip():
            raise ConfigurationError("The BizHawk host must be a non-empty string.")
        if image_transfer not in IMAGE_TRANSFERS:
            raise ConfigurationError(
                f"image_transfer must be one of {', '.join(IMAGE_TRANSFERS)}, "
                f"not {image_transfer!r}."
            )
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
        # Screenshots as a file when the emulator is on this PC (T-825): the
        # script writes the PNG where we say and only the path crosses the
        # socket, instead of a base64 copy made byte by byte in Lua.
        if image_transfer == "auto":
            image_transfer = "file" if host.strip().lower() in LOCAL_HOSTS else "socket"
        self._image_transfer = image_transfer
        self._image_path = os.path.join(
            tempfile.gettempdir(), f"retro_ai_lab_{os.getpid()}_{id(self):x}.png"
        )
        self._link: Link | None = None
        self._hello: dict[str, Any] = {}
        self._description: Description | None = None
        self._free_run = False
        self._hold = True
        self._held: frozenset[Button] = frozenset()
        self._unreadable: set[str] = set()  # regions the emulator refused to read (T-833)

    # ------------------------------------------------------------ queries

    @property
    def host(self) -> str:
        return self._host

    @property
    def image_transfer(self) -> str:
        """How screenshots travel: ``file`` (same PC) or ``socket`` (base64)."""
        return self._image_transfer

    @property
    def link_summary(self) -> str:
        """What every call since connect cost, slowest command first (T-830)."""
        return "no calls" if self._link is None else self._link.summary(lifetime=True)

    @property
    def image_path(self) -> str:
        """Where the script writes screenshots when they travel as a file."""
        return self._image_path

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

    def _do_check_link(self) -> bool:
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
                call_timeout_s=self._call_timeout_s,
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
        return tuple(
            r
            for r in self.description.regions
            if r.size <= self._max_region_bytes
            and is_live_memory(r.name)
            and r.name not in self._unreadable
        )

    @property
    def unreadable_regions(self) -> frozenset[str]:
        """Regions the emulator refused to read since connect; no longer offered (T-833)."""
        return frozenset(self._unreadable)

    def _do_read_memory(self, region: str, offset: int, length: int) -> bytes:
        return self._do_read_many([(region, offset, length)])[0]

    def _do_read_many(self, reads: Sequence[tuple[str, int, int]]) -> Sequence[bytes]:
        answer = self._call("read", reads=self._read_requests(reads))
        return self._reads_from(answer, len(reads))

    def _do_step(
        self,
        frames: int,
        pressed: frozenset[Button] | None,
        reads: Sequence[tuple[str, int, int]],
        screenshot: bool,
    ) -> StepResult:
        """One ``step`` request carries buttons, frames, reads and the screenshot.

        At normal speed the frames take as long as they take (300 frames are five
        seconds), so the answer is waited for accordingly (T-853).
        """
        fields: dict[str, Any] = {"frames": frames}
        fields["timeout_s"] = self._call_timeout_s + frames / SLOWEST_FPS
        if pressed is not None:
            fields["buttons"] = self._button_fields(pressed)
        if reads:
            fields["reads"] = self._read_requests(reads)
        if screenshot:
            fields["screenshot"] = True
            fields.update(self._image_fields())
        answer = self._call("step", **fields)
        if pressed is not None:
            self._held = pressed
        data = tuple(self._reads_from(answer, len(reads))) if reads else ()
        frame = self._frame_from(answer) if screenshot else None
        return StepResult(frames, data, frame)

    def _do_capture_frame(self) -> Frame:
        answer = self._call("screenshot", **self._image_fields())
        return self._frame_from(answer)

    def _image_fields(self) -> dict[str, Any]:
        if self._image_transfer == "file":
            return {"image_file": self._image_path}
        return {}

    def _do_advance_frames(self, n: int) -> None:
        self._call("step", frames=n, timeout_s=self._call_timeout_s + n / SLOWEST_FPS)

    def _do_set_speed(self, multiplier: float) -> None:
        self._call("speed", percent=max(1, round(multiplier * 100)))

    def _do_save_state(self, slot: str) -> None:
        self._call("save", slot=slot)

    def _do_load_state(self, slot: str) -> None:
        answer = self._call("load", slot=slot)
        if answer.get("loaded") is not True:
            raise ConfigurationError(f"No save state in slot {slot!r}.")

    def _do_set_input(self, pressed: frozenset[Button]) -> None:
        self._call("input", buttons=self._button_fields(pressed))
        self._held = pressed

    # ------------------------------------------------------------ helpers

    def _button_fields(self, pressed: frozenset[Button]) -> dict[str, bool]:
        down = {b.name for b in pressed}
        return {label: ident in down for ident, label in self.description.button_names.items()}

    def _read_requests(self, reads: Sequence[tuple[str, int, int]]) -> list[dict[str, Any]]:
        domains = self.description.domain_names
        return [
            {"domain": domains[region], "addr": offset, "len": length}
            for region, offset, length in reads
        ]

    @staticmethod
    def _reads_from(answer: dict[str, Any], count: int) -> list[bytes]:
        reads = answer.get("reads")
        if (
            not isinstance(reads, list)
            or len(reads) != count
            or not all(isinstance(r, str) for r in reads)
        ):
            raise ProtocolError(
                f"The emulator's answer did not hold {count} base64 read(s) as asked."
            )
        return [_b64decode(text, "memory") for text in reads]

    def _frame_from(self, answer: dict[str, Any]) -> Frame:
        image = answer.get("image")
        path = answer.get("image_file")
        if path is not None and image is None:
            data = self._read_image_file(path)
        else:
            if image is None:
                raise EmulatorError("The emulator did not take a screenshot.")
            if not isinstance(image, str):
                raise ProtocolError("The emulator's screenshot was not base64 text.")
            data = _b64decode(image, "screenshot")
        number = answer.get("frame")
        return decode_image(data, number=number if isinstance(number, int) else 0)

    def _read_image_file(self, path: object) -> bytes:
        if not isinstance(path, str) or path != self._image_path:
            raise ProtocolError("The emulator named a screenshot file the app did not ask for.")
        try:
            with open(path, "rb") as file:
                data = file.read()
        except OSError as exc:
            raise EmulatorError(
                f"The emulator says it wrote a screenshot to {path}, but this app cannot "
                f"read it ({exc.strerror or exc}). If the emulator runs on another PC, "
                "connect with image_transfer='socket'."
            ) from exc
        finally:
            with contextlib.suppress(OSError):
                os.remove(path)
        if not data:
            raise EmulatorError("The emulator wrote an empty screenshot.")
        return data

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
        except EmulatorError as exc:
            self._note_unreadable(str(exc))
            raise

    def _note_unreadable(self, message: str) -> None:
        """A read the emulator refused drops that region from the offer (T-833)."""
        match = UNREADABLE_REGION.search(message)
        if match is None or self._description is None:
            return
        label = match.group("domain")
        idents = [i for i, name in self._description.domain_names.items() if name == label]
        if not idents or idents[0] in self._unreadable:
            return
        self._unreadable.add(idents[0])
        log.warning(
            "%s refused to read memory region '%s'; it is left out from now on (%s).",
            self.emulator_name,
            label,
            message,
        )

    def _drop(self) -> None:
        link, self._link = self._link, None
        self._hello = {}
        self._description = None
        self._held = frozenset()
        self._hold = True
        self._unreadable = set()
        if link is not None:
            link.close()
        self.mark_disconnected()


def _b64decode(text: str, what: str) -> bytes:
    try:
        return base64.b64decode(text, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ProtocolError(f"The emulator's {what} was not valid base64.") from exc


__all__ = [
    "DEFAULT_CALL_TIMEOUT_S",
    "DEFAULT_MAX_REGION_BYTES",
    "IMAGE_TRANSFERS",
    "LOCAL_HOSTS",
    "NOT_LIVE_MEMORY_WORDS",
    "SLOWEST_FPS",
    "UNREADABLE_REGION",
    "BizHawkAdapter",
    "is_live_memory",
]
