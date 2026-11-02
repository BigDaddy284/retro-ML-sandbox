"""Toy Racer: a tiny deterministic racing game with "real" RAM and a screen.

It exists so every layer above the emulator adapter can be developed and
tested without an emulator or a ROM (ARCHITECTURE.md ADR-004). It behaves
like a very small console game:

* a loop track of ``TRACK_LENGTH`` units with three lanes and four
  checkpoints; completing all checkpoints then crossing the line counts a lap;
* fixed obstacles (seeded) that cost a life and stop the car;
* a 7-button pad where some buttons matter and some do not, so that the Game
  Understanding Engine has to *discover* which;
* a 32-byte RAM that holds the game state at fixed addresses, plus a few bytes
  of per-frame noise and a timer, so discovery has to tell signal from noise;
* a small greyscale screen rendered from the state.

Nothing outside this module may rely on the RAM layout: the whole point is
that the rest of the application learns it. Tests may import ``LAYOUT``.
"""

from __future__ import annotations

import struct
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Final

from retro_ai_lab.core import Button, ConfigurationError, Frame, MemoryRegion

# --------------------------------------------------------------- constants

TRACK_LENGTH: Final = 256
LANES: Final = 3
CHECKPOINTS: Final = (64, 128, 192)  # the start/finish line is at 0
MAX_SPEED: Final = 4
START_LIVES: Final = 3
MAX_SCORE: Final = 0xFFFF
OBSTACLE_COUNT: Final = 6
RAM_SIZE: Final = 32
SCREEN_WIDTH: Final = 64
SCREEN_HEIGHT: Final = 48

BUTTONS: Final = (
    Button("up", "Up"),
    Button("down", "Down"),
    Button("left", "Left"),
    Button("right", "Right"),
    Button("a", "A"),
    Button("b", "B"),
    Button("start", "Start"),
)
BUTTON_NAMES: Final = tuple(b.name for b in BUTTONS)
_BUTTON_BIT: Final = {name: 1 << i for i, name in enumerate(BUTTON_NAMES)}

RAM_REGION: Final = MemoryRegion("ram", RAM_SIZE, "Toy Racer work RAM")


@dataclass(frozen=True)
class _Layout:
    """Where each value lives in RAM (tests only; the app must discover it)."""

    frame: int = 0x00  # u16 little-endian
    position: int = 0x02  # u8, 0..TRACK_LENGTH-1
    lane: int = 0x03  # u8, 0..LANES-1
    speed: int = 0x04  # u8, 0..MAX_SPEED
    lap: int = 0x05  # u8
    checkpoint: int = 0x06  # u8, next checkpoint index 0..3
    lives: int = 0x07  # u8
    score: int = 0x08  # u16 little-endian
    game_over: int = 0x0A  # u8, 0/1
    input: int = 0x0B  # u8 bitmask of pressed buttons
    noise: int = 0x0C  # 4 bytes, pseudo-random each frame
    timer: int = 0x10  # u8, increments every frame regardless of state
    header: int = 0x11  # 15 bytes, constant


LAYOUT: Final = _Layout()
_HEADER: Final = b"TOYRACER\x01\x00\x00\x00\x00\x00\x00"
assert len(_HEADER) == RAM_SIZE - LAYOUT.header

# Pixel values used by the renderer (greyscale).
PX_BACKGROUND: Final = 16
PX_LANE_LINE: Final = 64
PX_CHECKPOINT: Final = 96
PX_OBSTACLE: Final = 160
PX_PLAYER: Final = 255
PX_HUD: Final = 200


def _lcg(state: int) -> int:
    return (state * 1103515245 + 12345) & 0x7FFFFFFF


@dataclass(frozen=True, slots=True)
class RacerState:
    """A read-only snapshot of the game, for tests and the adapter."""

    frame: int
    position: int
    lane: int
    speed: int
    lap: int
    checkpoint: int
    lives: int
    score: int
    game_over: bool


class ToyRacer:
    """The game. One instance is one running "console"."""

    def __init__(self, seed: int = 0) -> None:
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ConfigurationError("Toy Racer seed must be a non-negative integer.")
        self._seed = seed
        self._ram = bytearray(RAM_SIZE)
        self._obstacles: frozenset[tuple[int, int]] = self._make_obstacles(seed)
        self._noise = 0
        self.reset()

    # --------------------------------------------------------------- setup

    @staticmethod
    def _make_obstacles(seed: int) -> frozenset[tuple[int, int]]:
        """Fixed (position, lane) pairs, never on the line or a checkpoint."""
        forbidden = {0, *CHECKPOINTS}
        rng = _lcg(seed + 1)
        found: set[tuple[int, int]] = set()
        while len(found) < OBSTACLE_COUNT:
            rng = _lcg(rng)
            pos = 8 + (rng >> 8) % (TRACK_LENGTH - 16)
            rng = _lcg(rng)
            lane = (rng >> 8) % LANES
            if pos in forbidden or abs(pos - TRACK_LENGTH // 2) < 2:
                continue
            found.add((pos, lane))
        return frozenset(found)

    def reset(self) -> None:
        """Power-on state. Keeps the seed (and therefore the obstacles)."""
        ram = self._ram
        ram[:] = bytes(RAM_SIZE)
        ram[LAYOUT.lane] = LANES // 2
        ram[LAYOUT.lives] = START_LIVES
        ram[LAYOUT.header :] = _HEADER
        self._noise = _lcg(self._seed)
        self._write_noise()

    # ------------------------------------------------------------- access

    @property
    def seed(self) -> int:
        return self._seed

    @property
    def obstacles(self) -> frozenset[tuple[int, int]]:
        return self._obstacles

    @property
    def ram(self) -> bytes:
        return bytes(self._ram)

    @property
    def state(self) -> RacerState:
        r = self._ram
        return RacerState(
            frame=self._u16(LAYOUT.frame),
            position=r[LAYOUT.position],
            lane=r[LAYOUT.lane],
            speed=r[LAYOUT.speed],
            lap=r[LAYOUT.lap],
            checkpoint=r[LAYOUT.checkpoint],
            lives=r[LAYOUT.lives],
            score=self._u16(LAYOUT.score),
            game_over=bool(r[LAYOUT.game_over]),
        )

    def snapshot(self) -> bytes:
        """Everything needed to restore the game exactly (a "save state")."""
        return bytes(self._ram) + struct.pack("<I", self._noise)

    def restore(self, snapshot: bytes) -> None:
        if not isinstance(snapshot, bytes) or len(snapshot) != RAM_SIZE + 4:
            raise ConfigurationError("Not a Toy Racer snapshot.")
        if snapshot[LAYOUT.header : RAM_SIZE] != _HEADER:
            raise ConfigurationError("Not a Toy Racer snapshot.")
        self._ram[:] = snapshot[:RAM_SIZE]
        (self._noise,) = struct.unpack("<I", snapshot[RAM_SIZE:])

    # --------------------------------------------------------------- step

    def step(self, pressed: Iterable[str] = ()) -> None:
        """Advance one frame with the given buttons held."""
        held = frozenset(pressed)
        unknown = held - set(BUTTON_NAMES)
        if unknown:
            raise ConfigurationError(f"Unknown Toy Racer buttons: {sorted(unknown)}.")
        ram = self._ram
        mask = sum(_BUTTON_BIT[name] for name in held)
        ram[LAYOUT.input] = mask

        if ram[LAYOUT.game_over]:
            if "start" in held:
                self._restart_after_game_over()
        else:
            self._simulate(held)

        self._set_u16(LAYOUT.frame, (self._u16(LAYOUT.frame) + 1) & 0xFFFF)
        ram[LAYOUT.timer] = (ram[LAYOUT.timer] + 1) & 0xFF
        self._write_noise()

    def _restart_after_game_over(self) -> None:
        ram = self._ram
        frame = self._u16(LAYOUT.frame)
        timer = ram[LAYOUT.timer]
        noise = self._noise
        self.reset()
        self._set_u16(LAYOUT.frame, frame)
        ram[LAYOUT.timer] = timer
        self._noise = noise

    def _simulate(self, held: frozenset[str]) -> None:
        ram = self._ram
        speed = ram[LAYOUT.speed]
        if "b" in held:
            speed = 0
        elif "up" in held and "down" not in held:
            speed = min(MAX_SPEED, speed + 1)
        elif "down" in held and "up" not in held:
            speed = max(0, speed - 1)

        lane = ram[LAYOUT.lane]
        if "left" in held and "right" not in held:
            lane = max(0, lane - 1)
        elif "right" in held and "left" not in held:
            lane = min(LANES - 1, lane + 1)
        ram[LAYOUT.lane] = lane

        old_pos = ram[LAYOUT.position]
        score = self._u16(LAYOUT.score)

        # Move one unit at a time so obstacles and checkpoints cannot be skipped.
        pos = old_pos
        for _ in range(speed):
            pos = (pos + 1) % TRACK_LENGTH
            score = min(MAX_SCORE, score + 1)
            cp = ram[LAYOUT.checkpoint]
            if cp < len(CHECKPOINTS) and pos == CHECKPOINTS[cp]:
                ram[LAYOUT.checkpoint] = cp + 1
                score = min(MAX_SCORE, score + 10)
            if pos == 0:
                if ram[LAYOUT.checkpoint] == len(CHECKPOINTS):
                    ram[LAYOUT.lap] = min(0xFF, ram[LAYOUT.lap] + 1)
                    score = min(MAX_SCORE, score + 100)
                ram[LAYOUT.checkpoint] = 0
            if (pos, lane) in self._obstacles:
                speed = 0
                ram[LAYOUT.lives] -= 1
                if ram[LAYOUT.lives] == 0:
                    ram[LAYOUT.game_over] = 1
                break

        ram[LAYOUT.position] = pos
        ram[LAYOUT.speed] = speed
        self._set_u16(LAYOUT.score, score)

    # ------------------------------------------------------------- render

    def render(self) -> Frame:
        """Draw the current state as a greyscale frame (viewer looks down the track)."""
        w, h = SCREEN_WIDTH, SCREEN_HEIGHT
        px = bytearray([PX_BACKGROUND]) * (w * h)
        st = self.state

        # HUD: top 4 rows. Lives as blocks on the left, score as a bar on the right.
        for life in range(st.lives):
            for y in range(1, 3):
                for x in range(1 + life * 4, 3 + life * 4):
                    px[y * w + x] = PX_HUD
        bar = min(w // 2, st.score // 32)
        for x in range(w - 1 - bar, w - 1):
            px[1 * w + x] = PX_HUD

        # Track: rows 8..h-1, lanes are vertical bands; the car sits near the
        # bottom and the track scrolls toward it.
        top = 8
        lane_w = w // LANES
        for lane in range(1, LANES):
            x = lane * lane_w
            for y in range(top, h):
                px[y * w + x] = PX_LANE_LINE

        def row_for(track_pos: int) -> int | None:
            """Map a track position to a screen row ahead of the car, or None."""
            ahead = (track_pos - st.position) % TRACK_LENGTH
            if ahead >= h - top - 4:
                return None
            return h - 4 - ahead

        for cp in (0, *CHECKPOINTS):
            row = row_for(cp)
            if row is not None:
                for x in range(w):
                    px[row * w + x] = PX_CHECKPOINT
        for pos, lane in self._obstacles:
            row = row_for(pos)
            if row is not None:
                for x in range(lane * lane_w + 2, (lane + 1) * lane_w - 2):
                    px[row * w + x] = PX_OBSTACLE
                    if row + 1 < h:
                        px[(row + 1) * w + x] = PX_OBSTACLE

        cx = st.lane * lane_w + lane_w // 2
        for y in range(h - 4, h - 1):
            for x in range(cx - 1, cx + 2):
                px[y * w + x] = PX_PLAYER

        if st.game_over:
            # A thick bar across the middle so "game over" is visually obvious.
            for y in range(h // 2 - 2, h // 2 + 2):
                for x in range(4, w - 4):
                    px[y * w + x] = PX_HUD

        return Frame(w, h, 1, bytes(px), number=st.frame)

    # ------------------------------------------------------------ helpers

    def _u16(self, offset: int) -> int:
        return self._ram[offset] | (self._ram[offset + 1] << 8)

    def _set_u16(self, offset: int, value: int) -> None:
        self._ram[offset] = value & 0xFF
        self._ram[offset + 1] = (value >> 8) & 0xFF

    def _write_noise(self) -> None:
        self._noise = _lcg(self._noise)
        self._ram[LAYOUT.noise : LAYOUT.noise + 4] = struct.pack("<I", self._noise)


__all__ = [
    "BUTTONS",
    "BUTTON_NAMES",
    "CHECKPOINTS",
    "LANES",
    "LAYOUT",
    "MAX_SPEED",
    "RAM_REGION",
    "RAM_SIZE",
    "SCREEN_HEIGHT",
    "SCREEN_WIDTH",
    "START_LIVES",
    "TRACK_LENGTH",
    "RacerState",
    "ToyRacer",
]
