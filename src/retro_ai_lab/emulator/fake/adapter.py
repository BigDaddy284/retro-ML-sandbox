"""``FakeEmulatorAdapter``: the full adapter contract over :class:`ToyRacer`.

Besides being the test double for every upper layer, it is what the app's
demo mode will run, so it must behave like a real emulator would:

* inputs are *held* until changed, and the game only advances when frames
  are advanced;
* capabilities can be switched off to mimic a less capable emulator;
* faults can be injected (refused connection, slow connection that times
  out, a link that drops after N operations) so disconnection handling can
  be tested end to end.
"""

from __future__ import annotations

from collections.abc import Sequence

from retro_ai_lab.core import (
    Button,
    Capabilities,
    ConfigurationError,
    EmulatorDisconnectedError,
    EmulatorError,
    Frame,
    GameInfo,
    MemoryRegion,
)
from retro_ai_lab.emulator.base import EmulatorAdapter
from retro_ai_lab.emulator.fake.toy_racer import BUTTONS, RAM_REGION, ToyRacer


class FakeEmulatorAdapter(EmulatorAdapter):
    name = "fake"
    display_name = "Practice console (Toy Racer)"

    def __init__(
        self,
        *,
        seed: int = 0,
        capabilities: Capabilities | None = None,
        fail_connect: Exception | None = None,
        connect_delay_s: float = 0.0,
        drop_after_ops: int | None = None,
    ) -> None:
        super().__init__()
        self._game = ToyRacer(seed)
        self._caps = capabilities if capabilities is not None else Capabilities.all()
        self._fail_connect = fail_connect
        if connect_delay_s < 0:
            raise ConfigurationError("connect_delay_s must be >= 0.")
        self._connect_delay_s = connect_delay_s
        if drop_after_ops is not None and drop_after_ops < 0:
            raise ConfigurationError("drop_after_ops must be >= 0 or None.")
        self._drop_after_ops = drop_after_ops
        self._ops = 0
        self._held: frozenset[Button] = frozenset()
        self._speed = 1.0
        self._slots: dict[str, bytes] = {}

    # ------------------------------------------------------------ test aids

    @property
    def game(self) -> ToyRacer:
        """The underlying game, for tests that need ground truth."""
        return self._game

    @property
    def speed(self) -> float:
        return self._speed

    @property
    def held(self) -> frozenset[Button]:
        return self._held

    @property
    def slots(self) -> tuple[str, ...]:
        return tuple(sorted(self._slots))

    @property
    def operations(self) -> int:
        """Backend operations performed since connect (for fault injection)."""
        return self._ops

    def drop_link(self) -> None:
        """Simulate the emulator going away without a clean disconnect."""
        self.mark_disconnected()

    # -------------------------------------------------------------- hooks

    def _do_connect(self, timeout_s: float) -> None:
        if self._fail_connect is not None:
            raise self._fail_connect
        if self._connect_delay_s > timeout_s:
            raise EmulatorError(f"The emulator did not answer within {timeout_s:g} seconds.")
        self._ops = 0

    def _do_disconnect(self) -> None:
        self._held = frozenset()

    def _do_capabilities(self) -> Capabilities:
        return self._caps

    def _do_game_info(self) -> GameInfo:
        return GameInfo("TOY", "Toy Racer", rom_hash=f"toyracer-{self._game.seed}")

    def _do_buttons(self) -> Sequence[Button]:
        return BUTTONS

    def _do_memory_regions(self) -> Sequence[MemoryRegion]:
        return (RAM_REGION,)

    def _do_read_memory(self, region: str, offset: int, length: int) -> bytes:
        self._tick()
        return self._game.ram[offset : offset + length]

    def _do_capture_frame(self) -> Frame:
        self._tick()
        return self._game.render()

    def _do_set_input(self, pressed: frozenset[Button]) -> None:
        self._tick()
        self._held = pressed

    def _do_advance_frames(self, n: int) -> None:
        self._tick()
        names = tuple(b.name for b in self._held)
        for _ in range(n):
            self._game.step(names)

    def _do_watch_pad(self) -> frozenset[str]:
        """The toy game has no real controller: what is held is what was set."""
        self._tick()
        return frozenset(b.name for b in self._held)

    def _do_set_speed(self, multiplier: float) -> None:
        self._tick()
        self._speed = multiplier

    def _do_save_state(self, slot: str) -> None:
        self._tick()
        self._slots[slot] = self._game.snapshot()

    def _do_load_state(self, slot: str) -> None:
        self._tick()
        if slot not in self._slots:
            raise ConfigurationError(f"No save state in slot {slot!r}.")
        self._game.restore(self._slots[slot])

    # ------------------------------------------------------------ helpers

    def _tick(self) -> None:
        """Count an operation and drop the link if fault injection says so."""
        if self._drop_after_ops is not None and self._ops >= self._drop_after_ops:
            self.mark_disconnected()
            raise EmulatorDisconnectedError("The emulator stopped responding.")
        self._ops += 1


__all__ = ["FakeEmulatorAdapter"]
