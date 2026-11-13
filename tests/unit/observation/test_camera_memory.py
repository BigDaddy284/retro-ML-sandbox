"""T-1020: the byte of the game's memory that moves as the camera does, found from
the looks alone."""

from __future__ import annotations

import random

import pytest

from retro_ai_lab.core import ConfigurationError
from retro_ai_lab.observation.camera_memory import AGREES, NEEDS, REMEMBERS, CameraMemory

# A made-up game's memory, byte by byte.
CONST, CLOCK, NOISE, PLAYER, CAMERA, HALF, CAMERA_Y = range(7)
TICK = 8  # what the clock adds every look


class Game:
    """A level seen through a camera, and its memory after each look.

    A look either moves the camera (the player held still on the screen, so its own
    position moves as far) or leaves it still, when the player may walk by itself.
    """

    def __init__(self, seed: int = 0, *, without: tuple[int, ...] = ()) -> None:
        self.rng = random.Random(seed)
        self.camera, self.camera_y, self.player, self.clock = 300, 40, 380, 0
        self.without = without
        self.memory = self._memory()

    def _memory(self) -> bytes:
        cells = {
            CONST: 7,
            CLOCK: self.clock,
            NOISE: self.rng.randrange(256),
            PLAYER: self.player,
            CAMERA: self.camera,
            HALF: self.camera // 2,
            CAMERA_Y: self.camera_y,
        }
        return bytes(v % 256 for k, v in cells.items() if k not in self.without)

    def look(self, dx: int = 0, dy: int = 0, *, walks: int = 0) -> tuple[bytes, bytes]:
        """One look on: the memory before it and after it."""
        before = self.memory
        self.camera += dx
        self.camera_y += dy
        self.player += dx + walks
        self.clock += TICK
        self.memory = self._memory()
        return before, self.memory


def shown(
    memory: CameraMemory,
    game: Game,
    looks: int,
    *,
    wrong: float = 0.0,
    moves: tuple[int, ...] = (16, 16, 16, 12, 9, 2, -4, -16),
) -> None:
    """``looks`` looks, moving and still by turns. In ``wrong`` of the moving ones the
    picture misreads the move by a walker's 4 pixels or as none at all, and in
    ``wrong`` of the still ones the camera did move and the picture saw nothing; in
    three still looks of ten the player walks by itself."""
    rng = game.rng
    for i in range(looks):
        if i % 2 == 0:
            dx = rng.choice(moves)
            seen = dx
            if rng.random() < wrong:
                seen = rng.choice((dx + 4, dx - 4))
            memory.note(*game.look(dx), (seen, 0))
        elif rng.random() < wrong:
            memory.note(*game.look(16), (0, 0))
        else:
            memory.note(*game.look(walks=8 if rng.random() < 0.3 else 0), (0, 0))


def test_the_byte_that_moves_as_the_camera_is_found() -> None:
    assert NEEDS == 20 and AGREES == 0.6
    game, memory = Game(), CameraMemory()
    assert memory.found == (None, None) and memory.expect(*game.look(16)) == (None, None)
    # Nineteen looks of each kind are not enough; the twentieth of each is.
    # (On a still screen the player now and then walks by itself: that is what tells
    # its own position from the camera's.)
    for i in range(NEEDS - 1):
        memory.note(*game.look(16), (16, 0))
        memory.note(*game.look(walks=8 if i % 4 == 0 else 0), (0, 0))
    assert memory.found == (None, None)
    memory.note(*game.look(16), (16, 0))
    assert memory.found == (None, None)  # twenty moving, nineteen still
    memory.note(*game.look(), (0, 0))
    assert memory.found == (CAMERA, None)
    # Up and down it has seen no move at all: nothing is expected that way.
    assert memory.expect(*game.look(12, 5)) == (12, None)

    # Six in ten of each kind is the least that will do. One byte of memory that
    # changes by the move read in ``moving`` of 20 looks and stays put in ``still``.
    def agreeing(moving: int, still: int) -> CameraMemory:
        memory, value = CameraMemory(), 0
        for i in range(NEEDS):
            before, value = value, (value + (16 if i < moving else 3)) % 256
            memory.note(bytes([before]), bytes([value]), (16, 0))
            before, value = value, (value + (0 if i < still else 1)) % 256
            memory.note(bytes([before]), bytes([value]), (0, 0))
        return memory

    assert agreeing(12, 12).found == (0, None)
    assert agreeing(11, 20).found == (None, None)
    assert agreeing(20, 11).found == (None, None)
    # Of two that will do, the one that agrees more often; and the first of equals.
    both = CameraMemory()
    a = b = c = 0
    for i in range(NEEDS):
        was = bytes([a, b, c])
        a, b, c = (a + (16 if i < 14 else 0)) % 256, (b + 16) % 256, (c + 16) % 256
        both.note(was, bytes([a, b, c]), (16, 0))
        both.note(bytes([a, b, c]), bytes([a, b, c]), (0, 0))
    assert both.found == (1, None)


def test_it_stands_a_wrong_reading_and_is_not_fooled() -> None:
    """The picture misreads one look in five (a walker on bare ground). The camera's
    own byte still stands out, and nothing else is ever taken for it."""
    for seed in range(5):
        game, memory = Game(seed), CameraMemory()
        for _ in range(150):
            shown(memory, game, 2, wrong=0.2)
            assert memory.found[0] in (None, CAMERA) and memory.found[1] is None
        assert memory.found == (CAMERA, None)
    # A game with no such byte: the player's own position moves as the camera does
    # while the camera follows it, and will do in its place.
    game, memory = Game(1, without=(CAMERA, HALF, CAMERA_Y)), CameraMemory()
    shown(memory, game, 300)
    assert memory.found == (PLAYER, None)
    # A clock that adds, every look, just what the camera moves in a moving one agrees
    # in all of those and in none of the still ones: it is no camera.
    game, memory = Game(2, without=(PLAYER, CAMERA, HALF, CAMERA_Y)), CameraMemory()
    shown(memory, game, 300, moves=(TICK,))
    assert memory.found == (None, None)


def test_what_it_expects() -> None:
    game, memory = Game(3), CameraMemory()
    for i in range(2 * NEEDS):
        memory.note(*game.look(16, 6), (16, 6))
        memory.note(*game.look(walks=8 if i % 4 == 0 else 0), (0, 0))
    assert memory.found == (CAMERA, CAMERA_Y)
    assert memory.expect(*game.look(9, -3)) == (9, -3)
    assert memory.expect(*game.look()) == (0, 0)

    # A byte rolls over at 256: the short way round is the move.
    def memories(camera: int, camera_y: int) -> bytes:
        cells = [0] * 7
        cells[CAMERA], cells[CAMERA_Y] = camera, camera_y
        return bytes(cells)

    assert memory.expect(memories(250, 3), memories(6, 251)) == (12, -8)
    assert memory.expect(memories(0, 0), memories(127, 129)) == (127, -127)
    # Looks read as moving across only teach nothing about up and down, and a look in
    # which the move read is more than a byte can show teaches nothing either.
    across, game = CameraMemory(), Game(4)
    for i in range(NEEDS):
        across.note(*game.look(16), (16, 0))
        across.note(*game.look(walks=8 if i % 4 == 0 else 0), (0, 0))
        across.note(*game.look(), (200, 0))
    assert across.found == (CAMERA, None)


def test_it_starts_over_and_it_forgets() -> None:
    assert REMEMBERS >= 10 * NEEDS
    game, memory = Game(5), CameraMemory(remembers=80)
    # However long a byte has agreed (two thousand looks here), it is judged by the
    # last eighty or so of each kind: what came before is forgotten by halves.
    shown(memory, game, 2000)
    assert memory.found == (CAMERA, None)
    agreed = memory.agreed(0)
    assert agreed is not None and agreed[0] == agreed[1] <= 2 * 80
    # The camera's count moves to another byte (another part of the game keeps it
    # elsewhere): the old byte stops agreeing and is let go; the new one is found.
    other = Game(6, without=(CAMERA,))
    for i in range(400):
        dx = 16 if i % 2 == 0 else 0
        before, after = other.look(dx)
        # The same layout, the old camera byte now standing still.
        before = before[:CAMERA] + b"\x00" + before[CAMERA:]
        after = after[:CAMERA] + b"\x00" + after[CAMERA:]
        memory.note(before, after, (dx, 0))
        if memory.found[0] != CAMERA:
            break
    assert i < 200 and memory.found[0] in (None, PLAYER)
    # A memory of another size is another game, or the game read another way.
    memory = CameraMemory()
    shown(memory, Game(7), 60)
    assert memory.found == (CAMERA, None)
    small = Game(8, without=(CONST,))
    memory.note(*small.look(16), (16, 0))
    assert memory.found == (None, None)
    assert memory.expect(*game.look(16)) == (None, None)  # and the old size is not known
    shown(memory, small, 60)
    assert memory.found == (CAMERA - 1, None)


def test_nonsense_is_refused() -> None:
    memory = CameraMemory()
    for before, after in ((b"ab", b"abc"), ("ab", b"ab"), (b"ab", None), (b"", b"")):
        with pytest.raises(ConfigurationError, match="memory"):
            memory.note(before, after, (0, 0))  # type: ignore[arg-type]
        with pytest.raises(ConfigurationError, match="memory"):
            memory.expect(before, after)  # type: ignore[arg-type]
    for seen in (None, 16, (16,), (16, 0, 0), (1.5, 0), (True, 0), ("16", 0)):
        with pytest.raises(ConfigurationError, match="move"):
            memory.note(b"ab", b"ab", seen)  # type: ignore[arg-type]
    for kw in ({"needs": 0}, {"needs": True}, {"agrees": 0.0}, {"agrees": 1.5}, {"remembers": 3}):
        with pytest.raises(ConfigurationError):
            CameraMemory(**kw)  # type: ignore[arg-type]
    assert memory.found == (None, None)
