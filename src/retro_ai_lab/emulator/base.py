"""The emulator adapter contract.

Every emulator (real or fake) is driven through :class:`EmulatorAdapter`.
Subclasses implement the ``_do_*`` hooks; the public methods here add the
checks that every adapter must make in exactly the same way:

* every operation except ``connect``/``disconnect`` requires a connection;
* every optional operation checks :class:`~retro_ai_lab.core.Capabilities`
  first and raises :class:`~retro_ai_lab.core.UnsupportedCapabilityError`
  rather than letting a backend fail in its own way;
* arguments are validated before they reach the backend.

This keeps DESIGN.md §7 honest: callers detect capabilities, never assume them.
"""

from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from typing import ClassVar, final

from retro_ai_lab.core import (
    Button,
    Capabilities,
    ConfigurationError,
    EmulatorDisconnectedError,
    Frame,
    GameInfo,
    MemoryRegion,
    StepResult,
    UnsupportedCapabilityError,
)

DEFAULT_CONNECT_TIMEOUT_S = 5.0


class EmulatorAdapter(ABC):
    """Base class for all emulator adapters.

    ``name`` is the stable identifier shown by the adapter registry (T-006);
    ``display_name`` is what children see: the class's is what the list of
    emulators shows, and an adapter may say more once it knows what it runs (T-985).
    """

    name: ClassVar[str] = ""
    display_name: str = ""

    def __init__(self) -> None:
        self._connected = False
        # One conversation with the emulator at a time: the frame pump's worker
        # thread, the GUI thread (play clock, training) and the link watchdog
        # all go through here, and a socket cannot interleave two requests.
        self._lock = threading.RLock()

    # ------------------------------------------------------------ lifecycle

    @final
    @property
    def is_connected(self) -> bool:
        return self._connected

    @final
    def connect(self, timeout_s: float = DEFAULT_CONNECT_TIMEOUT_S) -> None:
        """Open the connection. Calling it while connected is a no-op."""
        with self._lock:
            if not isinstance(timeout_s, int | float) or isinstance(timeout_s, bool):
                raise ConfigurationError("Connect timeout must be a number of seconds.")
            if timeout_s <= 0:
                raise ConfigurationError("Connect timeout must be greater than zero.")
            if self._connected:
                return
            self._do_connect(float(timeout_s))
            self._connected = True

    @final
    def disconnect(self) -> None:
        """Close the connection. Calling it while disconnected is a no-op."""
        with self._lock:
            if not self._connected:
                return
            try:
                self._do_disconnect()
            finally:
                self._connected = False

    @final
    def mark_disconnected(self) -> None:
        """Record that the link dropped without our asking (backend use)."""
        self._connected = False

    @final
    def check_link(self) -> bool:
        """Is the link still up? Called by a watchdog while nothing is being sent.

        If another thread is talking to the emulator right now, the link is
        evidently up and this answers True at once rather than waiting its
        turn; a watchdog on the GUI thread must never stall behind a capture.
        """
        if not self._lock.acquire(blocking=False):
            return self._connected
        try:
            return self._do_check_link()
        finally:
            self._lock.release()

    # ------------------------------------------------------------ run mode

    @property
    def hold(self) -> bool:
        """True when the game stands still between requests (ARCHITECTURE.md §5.3).

        A backend that cannot pause its game (mGBA) says False: the game runs
        on between steps, and a step's reads describe the frame it was answered on.
        """
        return True

    @property
    def free_run(self) -> bool:
        """True while the game has been let run between the app's steps (T-942)."""
        return False

    @property
    def supports_free_run(self) -> bool:
        """Whether ``set_free_run`` can let a holding game run on its own (T-942)."""
        return False

    def set_free_run(self, free_run: bool) -> None:
        """Let the game run between steps (True) or hold it still (False).

        The base cannot switch: the practice console only moves when stepped.
        """
        if not isinstance(free_run, bool):
            raise ConfigurationError("free_run must be True or False.")
        raise UnsupportedCapabilityError("This emulator cannot run the game on its own.")

    def _do_check_link(self) -> bool:
        """Hook: probe the transport. The default trusts ``is_connected``.

        A backend whose transport can be probed without a request (a socket,
        say) overrides this, marks itself disconnected when the other end has
        gone, and returns False.
        """
        return self._connected

    # ---------------------------------------------------------- description

    @final
    def capabilities(self) -> Capabilities:
        with self._lock:
            self._require_connected()
            return self._do_capabilities()

    @final
    def game_info(self) -> GameInfo:
        with self._lock:
            self._require_connected()
            return self._do_game_info()

    @final
    def buttons(self) -> Sequence[Button]:
        with self._lock:
            self._require_connected()
            return tuple(self._do_buttons())

    @final
    def memory_regions(self) -> Sequence[MemoryRegion]:
        with self._lock:
            self._require_connected()
            self._require("memory_read")
            return tuple(self._do_memory_regions())

    # ----------------------------------------------------------- operations

    @final
    def read_memory(self, region: str, offset: int, length: int) -> bytes:
        with self._lock:
            self._require_connected()
            self._require("memory_read")
            self._check_reads([(region, offset, length)])
            data = self._do_read_memory(region, offset, length)
            _check_read_length(data, length)
            return data

    @final
    def read_many(self, reads: Sequence[tuple[str, int, int]]) -> tuple[bytes, ...]:
        """Several reads in one go (one request on a socket backend, T-822).

        Each read is ``(region, offset, length)``; the result holds the bytes
        in the same order.
        """
        with self._lock:
            self._require_connected()
            self._require("memory_read")
            wanted = self._check_reads(reads)
            if not wanted:
                return ()
            data = tuple(self._do_read_many(wanted))
            if len(data) != len(wanted):
                raise ConfigurationError(
                    f"Backend answered {len(data)} reads for {len(wanted)} requested."
                )
            for chunk, (_, _, length) in zip(data, wanted, strict=True):
                _check_read_length(chunk, length)
            return data

    @final
    def step(
        self,
        frames: int = 1,
        *,
        pressed: Iterable[Button] | None = None,
        reads: Sequence[tuple[str, int, int]] = (),
        screenshot: bool = False,
        watch: bool = False,
    ) -> StepResult:
        """Hold ``pressed`` (if given), advance ``frames``, then read and look.

        One round trip on a socket backend (T-822). Each part is checked only
        when it is used: a step with no screenshot works on an emulator that
        cannot capture its screen. With ``watch`` the backend leaves the pad
        to the person for those frames and names the buttons they held in
        ``StepResult.pressed`` (T-887); ``pressed`` cannot be given as well.
        While the game runs on its own (``free_run``, T-942) the frames are
        not driven but waited for: the step answers once that many have passed.
        """
        with self._lock:
            self._require_connected()
            if not self.free_run:
                self._require("frame_advance")
            _check_int(frames, "frame count", minimum=1)
            held: frozenset[Button] | None = None
            if pressed is not None:
                self._require("input")
                held = self._check_buttons(pressed)
            if watch:
                self._require("input")
                if held is not None:
                    raise ConfigurationError("A step cannot hold the pad and watch it at once.")
            wanted: tuple[tuple[str, int, int], ...] = ()
            if reads:
                self._require("memory_read")
                wanted = self._check_reads(reads)
            if screenshot:
                self._require("screen_capture")
            result = self._do_step(frames, held, wanted, screenshot, watch)
            if len(result.reads) != len(wanted):
                raise ConfigurationError(
                    f"Backend answered {len(result.reads)} reads for {len(wanted)} requested."
                )
            for chunk, (_, _, length) in zip(result.reads, wanted, strict=True):
                _check_read_length(chunk, length)
            if screenshot and result.frame is None:
                raise ConfigurationError("Backend stepped without the screenshot asked for.")
            if watch and result.pressed is None:
                raise ConfigurationError("Backend stepped without reporting the pad it watched.")
            return result

    @final
    def capture_frame(self) -> Frame:
        with self._lock:
            self._require_connected()
            self._require("screen_capture")
            return self._do_capture_frame()

    @final
    def set_input(self, pressed: Iterable[Button]) -> None:
        with self._lock:
            self._require_connected()
            self._require("input")
            self._do_set_input(self._check_buttons(pressed))

    @final
    def advance_frames(self, n: int = 1) -> None:
        with self._lock:
            self._require_connected()
            self._require("frame_advance")
            _check_int(n, "frame count", minimum=1)
            self._do_advance_frames(n)

    @final
    def set_speed(self, multiplier: float) -> None:
        with self._lock:
            self._require_connected()
            self._require("speed_control")
            if not isinstance(multiplier, int | float) or isinstance(multiplier, bool):
                raise ConfigurationError("Speed multiplier must be a number.")
            if multiplier <= 0:
                raise ConfigurationError("Speed multiplier must be greater than zero.")
            self._do_set_speed(float(multiplier))

    @final
    def save_state(self, slot: str) -> None:
        with self._lock:
            self._require_connected()
            self._require("save_states")
            _check_slot(slot)
            self._do_save_state(slot)

    @final
    def load_state(self, slot: str) -> None:
        with self._lock:
            self._require_connected()
            self._require("save_states")
            _check_slot(slot)
            self._do_load_state(slot)

    # -------------------------------------------------------------- helpers

    @final
    def _require_connected(self) -> None:
        if not self._connected:
            raise EmulatorDisconnectedError()

    @final
    def _require(self, capability: str) -> None:
        if not self._do_capabilities().supports(capability):
            raise UnsupportedCapabilityError(capability)

    @final
    def _check_buttons(self, pressed: Iterable[Button]) -> frozenset[Button]:
        pressed_set = frozenset(pressed)
        unknown = pressed_set - set(self._do_buttons())
        if unknown:
            raise ConfigurationError(
                f"Unknown buttons for this emulator: {sorted(b.name for b in unknown)}."
            )
        return pressed_set

    @final
    def _check_reads(
        self, reads: Sequence[tuple[str, int, int]]
    ) -> tuple[tuple[str, int, int], ...]:
        regions = {r.name: r for r in self._do_memory_regions()}
        checked = []
        for item in reads:
            if not isinstance(item, tuple) or len(item) != 3:
                raise ConfigurationError("Each read must be (region, offset, length).")
            region, offset, length = item
            if region not in regions:
                raise ConfigurationError(
                    f"Unknown memory region {region!r}; available: {sorted(regions)}."
                )
            _check_int(offset, "offset", minimum=0)
            _check_int(length, "length", minimum=0)
            if not regions[region].contains(offset, length):
                raise ConfigurationError(
                    f"Read of {length} bytes at {offset} is outside region "
                    f"{region!r} (size {regions[region].size})."
                )
            checked.append((region, offset, length))
        return tuple(checked)

    # ------------------------------------------------- backend hooks (abstract)

    @abstractmethod
    def _do_connect(self, timeout_s: float) -> None: ...

    @abstractmethod
    def _do_disconnect(self) -> None: ...

    @abstractmethod
    def _do_capabilities(self) -> Capabilities: ...

    @abstractmethod
    def _do_game_info(self) -> GameInfo: ...

    @abstractmethod
    def _do_buttons(self) -> Sequence[Button]: ...

    # The remaining hooks have safe defaults so an adapter that reports a
    # capability as unsupported need not implement it.

    def _do_memory_regions(self) -> Sequence[MemoryRegion]:
        raise UnsupportedCapabilityError("memory_read")

    def _do_read_memory(self, region: str, offset: int, length: int) -> bytes:
        raise UnsupportedCapabilityError("memory_read")

    def _do_capture_frame(self) -> Frame:
        raise UnsupportedCapabilityError("screen_capture")

    def _do_set_input(self, pressed: frozenset[Button]) -> None:
        raise UnsupportedCapabilityError("input")

    def _do_advance_frames(self, n: int) -> None:
        raise UnsupportedCapabilityError("frame_advance")

    def _do_set_speed(self, multiplier: float) -> None:
        raise UnsupportedCapabilityError("speed_control")

    def _do_save_state(self, slot: str) -> None:
        raise UnsupportedCapabilityError("save_states")

    def _do_load_state(self, slot: str) -> None:
        raise UnsupportedCapabilityError("save_states")

    def _do_read_many(self, reads: Sequence[tuple[str, int, int]]) -> Sequence[bytes]:
        """Hook: several reads. The default asks for each one in turn."""
        return [self._do_read_memory(region, offset, length) for region, offset, length in reads]

    def _do_step(
        self,
        frames: int,
        pressed: frozenset[Button] | None,
        reads: Sequence[tuple[str, int, int]],
        screenshot: bool,
        watch: bool = False,
    ) -> StepResult:
        """Hook: the bundled step. The default composes the primitive hooks.

        A backend whose transport can do it in one request overrides this.
        """
        if pressed is not None:
            self._do_set_input(pressed)
        self._do_advance_frames(frames)
        data = tuple(self._do_read_many(reads)) if reads else ()
        frame = self._do_capture_frame() if screenshot else None
        held = self._do_watch_pad() if watch else None
        return StepResult(frames, data, frame, held)

    def _do_watch_pad(self) -> frozenset[str]:
        """Hook: the names of the buttons the person's controller holds now (T-887).

        Only a backend that can leave the pad to the person supports this.
        """
        raise UnsupportedCapabilityError("The emulator cannot watch the controller.")


def _check_read_length(data: bytes, length: int) -> None:
    if len(data) != length:
        raise ConfigurationError(f"Backend returned {len(data)} bytes for a {length}-byte read.")


def _check_int(value: object, what: str, *, minimum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigurationError(f"The {what} must be an integer, got {value!r}.")
    if value < minimum:
        raise ConfigurationError(f"The {what} must be >= {minimum}, got {value}.")


def _check_slot(slot: object) -> None:
    if not isinstance(slot, str) or not slot or not slot.replace("_", "").isalnum():
        raise ConfigurationError(
            f"Save-state slot must be a non-empty name of letters, digits or underscores, "
            f"got {slot!r}."
        )


__all__ = ["DEFAULT_CONNECT_TIMEOUT_S", "EmulatorAdapter"]
