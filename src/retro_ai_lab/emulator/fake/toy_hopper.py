"""Toy Hopper: a tiny deterministic platform game with "real" RAM and a screen (T-984).

Toy Racer has no jumping, no scene of separate moving things and no falling to
a death, so nothing in the tests looked like the platform games people bring.
Toy Hopper is the smallest game that does:

* a side-scrolling level of 16-pixel tiles, laid out from the seed: ground
  with gaps of 2 to 4 tiles, one-tile walls, walkers pacing their stretch of
  ground, coins and an end flag;
* a player who walks with left and right and jumps with A, higher and further
  the longer A is held, so the widest gap is cleared only by a held jump at
  speed;
* falling out of the screen or touching a walker starts a dying pause, and
  only when it ends is the life taken and the player put back at the start,
  as console games do (the life counter moves well after the fatal touch);
* a 64-byte RAM with the state at fixed addresses, noise and a timer;
* a 256x240 colour screen drawn from the state.

Nothing outside this module may rely on the RAM layout or on how the level is
laid out: the rest of the application has to learn both. Tests may import
``LAYOUT`` and read the level off the game.
"""

from __future__ import annotations

import struct
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import NDArray

from retro_ai_lab.core import Button, ConfigurationError, Frame, MemoryRegion

# --------------------------------------------------------------- constants

TILE: Final = 16
SCREEN_WIDTH: Final = 256
SCREEN_HEIGHT: Final = 240
LEVEL_TILES: Final = 160
LEVEL_WIDTH: Final = LEVEL_TILES * TILE
GROUND_ROW: Final = 13  # the ground's top row of tiles; everything below is ground too
GROUND_Y: Final = GROUND_ROW * TILE
WALL_ROW: Final = GROUND_ROW - 1
COIN_ROW: Final = GROUND_ROW - 3  # a jump's reach above the ground
START_TILES: Final = 12  # plain ground the level opens with
END_TILES: Final = 8  # plain ground it closes with; the flag stands in it
FLAG_COL: Final = LEVEL_TILES - 4
WIDEST_GAP: Final = 4

PLAYER_W: Final = 12
PLAYER_H: Final = 16
START_X: Final = 32
WALKER_SIZE: Final = 14
WALKER_Y: Final = GROUND_Y - WALKER_SIZE
WALKER_STRETCH: Final = 4  # tiles of ground a walker paces
COIN_SIZE: Final = 8
CAMERA_X: Final = 80  # where on the screen the player is held once the level scrolls

# Speeds are sixteenths of a pixel a frame.
SUB: Final = 16
MAX_SPEED: Final = 32
ACCELERATION: Final = 4
JUMP_SPEED: Final = 72
GRAVITY: Final = 10
HELD_GRAVITY: Final = 3  # while A stays held on the way up
MAX_FALL: Final = 96

START_LIVES: Final = 3
DYING_FRAMES: Final = 48
COIN_SCORE: Final = 10
FLAG_SCORE: Final = 100
MAX_SCORE: Final = 0xFFFF
MAX_COINS: Final = 16
WALKER_SLOTS: Final = 4
RAM_SIZE: Final = 64

PLAYING: Final = 0
DYING: Final = 1
GAME_OVER: Final = 2

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

RAM_REGION: Final = MemoryRegion("ram", RAM_SIZE, "Toy Hopper work RAM")

Colour = tuple[int, int, int]
SKY: Final[Colour] = (92, 148, 252)
GROUND: Final[Colour] = (136, 84, 24)
GRASS: Final[Colour] = (0, 168, 0)
WALL: Final[Colour] = (188, 188, 188)
COIN: Final[Colour] = (252, 216, 0)
FLAG: Final[Colour] = (252, 252, 252)
WALKER: Final[Colour] = (168, 16, 0)
WALKER_FEET: Final[Colour] = (40, 0, 0)
PLAYER: Final[Colour] = (228, 0, 88)
PLAYER_HEAD: Final[Colour] = (252, 188, 176)
HUD: Final[Colour] = (0, 0, 0)


@dataclass(frozen=True)
class _Layout:
    """Where each value lives in RAM (tests only; the app must discover it)."""

    frame: int = 0x00  # u16 little-endian
    x_low: int = 0x02  # the player's left edge in the level, pixels: low byte ...
    x_page: int = 0x03  # ... and how many 256s
    x_sub: int = 0x04  # sixteenths of a pixel
    y: int = 0x05  # the player's top edge on the screen, pixels
    y_sub: int = 0x06
    vx: int = 0x07  # signed, sixteenths of a pixel a frame
    vy: int = 0x08  # signed; up is negative
    on_ground: int = 0x09  # 0/1
    lives: int = 0x0A
    score: int = 0x0B  # u16 little-endian
    coins: int = 0x0D  # collected this level
    coin_mask: int = 0x0E  # u16: which of the level's coins are gone
    level: int = 0x10
    mode: int = 0x11  # PLAYING, DYING or GAME_OVER
    dying: int = 0x12  # frames of the dying pause left
    scroll: int = 0x13  # u16: the level pixel at the screen's left edge
    clock: int = 0x15  # u16: frames since the level (re)started; walkers pace by it
    jumping: int = 0x17  # 1 while A has been held since the take-off and the player rises
    input: int = 0x18  # u8 bitmask of pressed buttons
    timer: int = 0x19  # u8, increments every frame regardless of state
    noise: int = 0x1A  # 4 bytes, pseudo-random each frame
    walkers: int = 0x1E  # WALKER_SLOTS x (on screen 0/1, screen x, screen y)
    seed: int = 0x2A  # u16: the level this memory belongs to
    header: int = 0x2C  # 20 bytes, constant


LAYOUT: Final = _Layout()
_HEADER: Final = b"TOYHOPPER\x01" + bytes(10)
assert len(_HEADER) == RAM_SIZE - LAYOUT.header


def _lcg(state: int) -> int:
    return (state * 1103515245 + 12345) & 0x7FFFFFFF


def _signed(byte: int) -> int:
    return byte - 256 if byte >= 128 else byte


Spans = tuple[tuple[int, int], ...]
_Level = tuple[Spans, tuple[int, ...], Spans, Spans]  # gaps, walls, walkers, coins


def _level(seed: int) -> _Level:
    """Lay the level out from the seed: every kind of thing, spaced so each can be passed."""
    kinds = ("gap2", "coin", "walker", "wall", "gap3", "coin", "walker", "gap4", "coin", "wall")
    rng = _lcg(seed + 1)
    turn = (rng >> 8) % len(kinds)
    gaps: list[tuple[int, int]] = []
    walls: list[int] = []
    walkers: list[tuple[int, int]] = []
    coins: list[tuple[int, int]] = []
    col = START_TILES
    while True:
        kind = kinds[turn % len(kinds)]
        turn += 1
        widths = {"gap2": 2, "gap3": 3, "gap4": 4, "walker": WALKER_STRETCH}
        width = widths.get(kind, 1)
        if col + width > LEVEL_TILES - END_TILES:
            break
        if kind.startswith("gap"):
            gaps.append((col, width))
        elif kind == "wall":
            walls.append(col)
        elif kind == "walker":
            walkers.append((col * TILE, (col + width) * TILE - WALKER_SIZE))
        elif len(coins) < MAX_COINS:
            coins.append((col, COIN_ROW))
        rng = _lcg(rng)
        col += width + 5 + (rng >> 8) % 4  # room to land, stop and run up again
    return tuple(gaps), tuple(walls), tuple(walkers), tuple(coins)


@dataclass(frozen=True)
class HopperState:
    """A readable view of the game state, for tests and the fake adapter."""

    frame: int
    x: int  # the player's left edge in the level, pixels
    y: int  # the player's top edge, pixels
    vx: int
    vy: int
    on_ground: bool
    lives: int
    score: int
    coins: int
    level: int
    mode: int
    dying: int
    scroll: int
    clock: int

    @property
    def game_over(self) -> bool:
        return self.mode == GAME_OVER


class ToyHopper:
    """The game. One instance is one running "console"."""

    def __init__(self, seed: int = 0) -> None:
        if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= 0xFFFF:
            raise ConfigurationError("Toy Hopper seed must be a whole number from 0 to 65535.")
        self._seed = seed
        self._ram = bytearray(RAM_SIZE)
        self._gaps, self._walls, self._walkers, self._coins = _level(seed)
        hole: set[int] = set()
        for first, wide in self._gaps:
            hole.update(range(first, first + wide))
        self._hole = frozenset(hole)
        self._wall: frozenset[int] = frozenset(self._walls)
        self._noise = 0
        self.reset()

    # --------------------------------------------------------------- setup

    def reset(self) -> None:
        """Power-on state. Keeps the seed (and therefore the level)."""
        ram = self._ram
        ram[:] = bytes(RAM_SIZE)
        ram[LAYOUT.lives] = START_LIVES
        self._set_u16(LAYOUT.seed, self._seed)
        ram[LAYOUT.header :] = _HEADER
        self._to_start()
        self._noise = _lcg(self._seed)
        self._write_noise()

    def _to_start(self) -> None:
        """The player back at the level's start, its coins back and its walkers too."""
        ram = self._ram
        ram[LAYOUT.x_low], ram[LAYOUT.x_page], ram[LAYOUT.x_sub] = START_X, 0, 0
        ram[LAYOUT.y], ram[LAYOUT.y_sub] = GROUND_Y - PLAYER_H, 0
        ram[LAYOUT.vx] = ram[LAYOUT.vy] = 0
        ram[LAYOUT.on_ground] = 1
        ram[LAYOUT.jumping] = 0
        ram[LAYOUT.coins] = 0
        self._set_u16(LAYOUT.coin_mask, 0)
        ram[LAYOUT.mode] = PLAYING
        ram[LAYOUT.dying] = 0
        self._set_u16(LAYOUT.scroll, 0)
        self._set_u16(LAYOUT.clock, 0)
        self._write_walkers()

    # ------------------------------------------------------------- access

    @property
    def seed(self) -> int:
        return self._seed

    @property
    def gaps(self) -> tuple[tuple[int, int], ...]:
        """(first column, width in tiles) of every gap in the ground."""
        return self._gaps

    @property
    def walls(self) -> tuple[int, ...]:
        """The columns with a one-tile wall standing on the ground."""
        return self._walls

    @property
    def walkers(self) -> tuple[tuple[int, int], ...]:
        """(leftmost, rightmost) level pixel of each walker's left edge."""
        return self._walkers

    @property
    def coins(self) -> tuple[tuple[int, int], ...]:
        """(column, row) of every coin the level starts with."""
        return self._coins

    @property
    def ram(self) -> bytes:
        return bytes(self._ram)

    @property
    def state(self) -> HopperState:
        r = self._ram
        return HopperState(
            frame=self._u16(LAYOUT.frame),
            x=r[LAYOUT.x_low] | (r[LAYOUT.x_page] << 8),
            y=r[LAYOUT.y],
            vx=_signed(r[LAYOUT.vx]),
            vy=_signed(r[LAYOUT.vy]),
            on_ground=bool(r[LAYOUT.on_ground]),
            lives=r[LAYOUT.lives],
            score=self._u16(LAYOUT.score),
            coins=r[LAYOUT.coins],
            level=r[LAYOUT.level],
            mode=r[LAYOUT.mode],
            dying=r[LAYOUT.dying],
            scroll=self._u16(LAYOUT.scroll),
            clock=self._u16(LAYOUT.clock),
        )

    def walker_at(self, index: int, clock: int | None = None) -> int:
        """The level pixel of a walker's left edge, now or at another clock."""
        low, high = self._walkers[index]
        span = high - low
        k = ((self._u16(LAYOUT.clock) if clock is None else clock) // 2) % (2 * span)
        return low + (k if k <= span else 2 * span - k)

    def snapshot(self) -> bytes:
        """Everything needed to restore the game exactly (a "save state")."""
        return bytes(self._ram) + struct.pack("<I", self._noise)

    def restore(self, snapshot: bytes) -> None:
        if not isinstance(snapshot, bytes) or len(snapshot) != RAM_SIZE + 4:
            raise ConfigurationError("Not a Toy Hopper snapshot.")
        if snapshot[LAYOUT.header : RAM_SIZE] != _HEADER:
            raise ConfigurationError("Not a Toy Hopper snapshot.")
        if snapshot[LAYOUT.seed] | (snapshot[LAYOUT.seed + 1] << 8) != self._seed:
            raise ConfigurationError("That Toy Hopper snapshot is from another level.")
        self._ram[:] = snapshot[:RAM_SIZE]
        (self._noise,) = struct.unpack("<I", snapshot[RAM_SIZE:])

    # --------------------------------------------------------------- step

    def step(self, pressed: Iterable[str] = ()) -> None:
        """Advance one frame with the given buttons held."""
        held = frozenset(pressed)
        unknown = held - set(BUTTON_NAMES)
        if unknown:
            raise ConfigurationError(f"Unknown Toy Hopper buttons: {sorted(unknown)}.")
        ram = self._ram
        held_before = ram[LAYOUT.input]
        ram[LAYOUT.input] = sum(_BUTTON_BIT[name] for name in held)

        mode = ram[LAYOUT.mode]
        if mode == GAME_OVER:
            if "start" in held:
                self._restart_after_game_over()
        elif mode == DYING:
            self._die_a_little()
        else:
            self._simulate(held, a_before=bool(held_before & _BUTTON_BIT["a"]))

        self._set_u16(LAYOUT.frame, (self._u16(LAYOUT.frame) + 1) & 0xFFFF)
        ram[LAYOUT.timer] = (ram[LAYOUT.timer] + 1) & 0xFF
        self._write_noise()

    def _restart_after_game_over(self) -> None:
        ram = self._ram
        frame, timer, noise = self._u16(LAYOUT.frame), ram[LAYOUT.timer], self._noise
        held = ram[LAYOUT.input]
        self.reset()
        self._set_u16(LAYOUT.frame, frame)
        ram[LAYOUT.timer] = timer
        ram[LAYOUT.input] = held
        self._noise = noise

    def _die_a_little(self) -> None:
        """The dying pause: nothing listens; when it ends the life goes."""
        ram = self._ram
        ram[LAYOUT.dying] -= 1
        if ram[LAYOUT.dying] > 0:
            return
        ram[LAYOUT.lives] -= 1
        if ram[LAYOUT.lives] == 0:
            ram[LAYOUT.mode] = GAME_OVER
            return
        self._to_start()

    def _solid(self, col: int, row: int) -> bool:
        if col < 0:
            return True  # the level's left edge
        if row >= GROUND_ROW:
            return col not in self._hole
        return row == WALL_ROW and col in self._wall

    def _simulate(self, held: frozenset[str], *, a_before: bool) -> None:
        ram = self._ram
        x = ((ram[LAYOUT.x_low] | (ram[LAYOUT.x_page] << 8)) * SUB) + ram[LAYOUT.x_sub]
        y = ram[LAYOUT.y] * SUB + ram[LAYOUT.y_sub]
        vx, vy = _signed(ram[LAYOUT.vx]), _signed(ram[LAYOUT.vy])
        on_ground, jumping = bool(ram[LAYOUT.on_ground]), bool(ram[LAYOUT.jumping])

        # Walking: speed up towards the side held, coast to a stop otherwise.
        want = ("right" in held) - ("left" in held)
        if want:
            vx = max(-MAX_SPEED, min(MAX_SPEED, vx + want * ACCELERATION))
        elif vx:
            vx -= ACCELERATION if vx > 0 else -ACCELERATION
            if abs(vx) < ACCELERATION:
                vx = 0

        # Jumping: a fresh press of A on the ground takes off; holding it keeps the rise long.
        a = "a" in held
        if a and not a_before and on_ground:
            vy, jumping, on_ground = -JUMP_SPEED, True, False
        if not (a and jumping and vy < 0):
            jumping = False
        vy = min(MAX_FALL, vy + (HELD_GRAVITY if jumping else GRAVITY))

        # Sideways first, stopped by whatever is solid at the player's height. Edges are
        # compared in sixteenths, so a player a fraction of a pixel into a tile is in it.
        cell = TILE * SUB
        x += vx
        rows = range(y // cell, (y + PLAYER_H * SUB - 1) // cell + 1)
        if vx > 0:
            col = (x + PLAYER_W * SUB - 1) // cell
            if any(self._solid(col, row) for row in rows):
                x, vx = col * cell - PLAYER_W * SUB, 0
        elif vx < 0:
            col = x // cell
            if any(self._solid(col, row) for row in rows):
                x, vx = (col + 1) * cell, 0

        # Then down or up.
        y += vy
        cols = range(x // cell, (x + PLAYER_W * SUB - 1) // cell + 1)
        on_ground = False
        if vy >= 0:
            row = (y + PLAYER_H * SUB - 1) // cell
            if any(self._solid(col, row) for col in cols):
                y, vy, on_ground, jumping = row * cell - PLAYER_H * SUB, 0, True, False
        else:
            row = y // cell
            if any(self._solid(col, row) for col in cols):
                y, vy, jumping = (row + 1) * cell, 0, False
        px, py = x // SUB, y // SUB

        clock = (self._u16(LAYOUT.clock) + 1) & 0xFFFF
        self._set_u16(LAYOUT.clock, clock)
        ram[LAYOUT.x_low], ram[LAYOUT.x_page], ram[LAYOUT.x_sub] = px & 0xFF, px >> 8, x % SUB
        ram[LAYOUT.y], ram[LAYOUT.y_sub] = min(py, 0xFF), y % SUB
        ram[LAYOUT.vx], ram[LAYOUT.vy] = vx & 0xFF, vy & 0xFF
        ram[LAYOUT.on_ground], ram[LAYOUT.jumping] = int(on_ground), int(jumping)
        self._set_u16(LAYOUT.scroll, max(0, min(LEVEL_WIDTH - SCREEN_WIDTH, px - CAMERA_X)))
        self._write_walkers()

        if py >= SCREEN_HEIGHT or self._touching_a_walker(px, py):
            ram[LAYOUT.mode], ram[LAYOUT.dying] = DYING, DYING_FRAMES
            ram[LAYOUT.vx] = ram[LAYOUT.vy] = 0
            return
        self._collect_coins(px, py)
        if px + PLAYER_W > FLAG_COL * TILE:
            self._set_u16(LAYOUT.score, min(MAX_SCORE, self._u16(LAYOUT.score) + FLAG_SCORE))
            ram[LAYOUT.level] = min(0xFF, ram[LAYOUT.level] + 1)
            self._to_start()

    def _touching_a_walker(self, px: int, py: int) -> bool:
        if py + PLAYER_H <= WALKER_Y or py >= GROUND_Y:
            return False
        for index in range(len(self._walkers)):
            wx = self.walker_at(index)
            if px < wx + WALKER_SIZE and wx < px + PLAYER_W:
                return True
        return False

    def _collect_coins(self, px: int, py: int) -> None:
        ram = self._ram
        mask = self._u16(LAYOUT.coin_mask)
        inset = (TILE - COIN_SIZE) // 2
        for index, (col, row) in enumerate(self._coins):
            if mask & (1 << index):
                continue
            cx, cy = col * TILE + inset, row * TILE + inset
            if (
                px < cx + COIN_SIZE
                and cx < px + PLAYER_W
                and py < cy + COIN_SIZE
                and cy < py + PLAYER_H
            ):
                mask |= 1 << index
                ram[LAYOUT.coins] += 1
                self._set_u16(LAYOUT.score, min(MAX_SCORE, self._u16(LAYOUT.score) + COIN_SCORE))
        self._set_u16(LAYOUT.coin_mask, mask)

    def _write_walkers(self) -> None:
        """The walkers on the screen, left to right, in their slots; the rest empty."""
        ram = self._ram
        scroll = self._u16(LAYOUT.scroll)
        slot = 0
        ram[LAYOUT.walkers : LAYOUT.walkers + 3 * WALKER_SLOTS] = bytes(3 * WALKER_SLOTS)
        for index in range(len(self._walkers)):
            sx = self.walker_at(index) - scroll
            if 0 <= sx <= SCREEN_WIDTH - WALKER_SIZE and slot < WALKER_SLOTS:
                at = LAYOUT.walkers + 3 * slot
                ram[at], ram[at + 1], ram[at + 2] = 1, sx, WALKER_Y
                slot += 1

    # ------------------------------------------------------------- render

    def render(self) -> Frame:
        """Draw the current state: the level side on, scrolled to the player."""
        st = self.state
        px: NDArray[np.uint8] = np.empty((SCREEN_HEIGHT, SCREEN_WIDTH, 3), dtype=np.uint8)
        px[:] = SKY

        def box(x: int, y: int, w: int, h: int, colour: Colour) -> None:
            x0, y0 = max(0, x), max(0, y)
            x1, y1 = min(SCREEN_WIDTH, x + w), min(SCREEN_HEIGHT, y + h)
            if x0 < x1 and y0 < y1:
                px[y0:y1, x0:x1] = colour

        scroll = st.scroll
        for col in range(scroll // TILE, (scroll + SCREEN_WIDTH - 1) // TILE + 1):
            sx = col * TILE - scroll
            if col not in self._hole:
                box(sx, GROUND_Y, TILE, SCREEN_HEIGHT - GROUND_Y, GROUND)
                box(sx, GROUND_Y, TILE, 4, GRASS)
            if col in self._wall:
                box(sx, WALL_ROW * TILE, TILE, TILE, WALL)
            if col == FLAG_COL:
                box(sx, GROUND_Y - 6 * TILE, 4, 6 * TILE, FLAG)
                box(sx + 4, GROUND_Y - 6 * TILE, 12, 10, FLAG)
        mask = self._u16(LAYOUT.coin_mask)
        inset = (TILE - COIN_SIZE) // 2
        for index, (col, row) in enumerate(self._coins):
            if not mask & (1 << index):
                box(col * TILE + inset - scroll, row * TILE + inset, COIN_SIZE, COIN_SIZE, COIN)
        for index in range(len(self._walkers)):
            sx = self.walker_at(index) - scroll
            box(sx, WALKER_Y, WALKER_SIZE, WALKER_SIZE, WALKER)
            box(sx, WALKER_Y + WALKER_SIZE - 3, WALKER_SIZE, 3, WALKER_FEET)
        box(st.x - scroll, st.y, PLAYER_W, PLAYER_H, PLAYER)
        box(st.x - scroll + 2, st.y + 2, PLAYER_W - 4, 5, PLAYER_HEAD)

        # The strip along the top: a square a life, and the score as a bar from the right.
        for life in range(st.lives):
            box(8 + life * 12, 8, 8, 8, HUD)
        bar = min(SCREEN_WIDTH // 2, st.score // 10)
        box(SCREEN_WIDTH - 8 - bar, 8, bar, 8, HUD)
        if st.game_over:
            box(32, SCREEN_HEIGHT // 2 - 8, SCREEN_WIDTH - 64, 16, HUD)
        return Frame(SCREEN_WIDTH, SCREEN_HEIGHT, 3, px.tobytes(), number=st.frame)

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
    "CAMERA_X",
    "COIN_ROW",
    "COIN_SCORE",
    "COIN_SIZE",
    "DYING",
    "DYING_FRAMES",
    "FLAG_COL",
    "FLAG_SCORE",
    "GAME_OVER",
    "GROUND_Y",
    "LAYOUT",
    "LEVEL_TILES",
    "LEVEL_WIDTH",
    "MAX_SPEED",
    "PLAYER_H",
    "PLAYER_W",
    "PLAYING",
    "RAM_REGION",
    "RAM_SIZE",
    "SCREEN_HEIGHT",
    "SCREEN_WIDTH",
    "START_LIVES",
    "START_X",
    "TILE",
    "WALKER_SIZE",
    "WALKER_SLOTS",
    "WALKER_Y",
    "WIDEST_GAP",
    "HopperState",
    "ToyHopper",
]
