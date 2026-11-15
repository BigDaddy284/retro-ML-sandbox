"""T-1040: Toy Hopper can hold more than one level."""

from __future__ import annotations

import hashlib
from itertools import combinations

import numpy as np
import pytest

from retro_ai_lab.core import ConfigurationError, Frame
from retro_ai_lab.emulator.fake import HopperState, ToyHopper
from retro_ai_lab.emulator.fake import toy_hopper as th
from retro_ai_lab.emulator.fake.toy_hopper import LAYOUT, PALETTES
from tests.unit.emulator.fake.test_toy_hopper import GROUNDED_Y, _fall_in, pixel, place, run, script

# What a play of the one-level game came to before this task: its memory at every frame
# and its screen at every sixtieth, through the flag and on (see ``the_play``).
AS_IT_WAS = "4fcb03db95da726a8d09454a468232194682d66c8f4bd39510cfe9d417ce0d50"


def cartridge(seed: int, levels: int) -> ToyHopper:
    game = ToyHopper(seed)
    game.levels = levels
    return game


def layout(game: ToyHopper) -> tuple[object, ...]:
    return game.gaps, game.walls, game.walkers, game.coins


def to_the_flag(game: ToyHopper) -> HopperState:
    """Stand the player just short of the flag and walk through it: the frame it is passed."""
    level = game.state.level
    place(game, th.FLAG_COL * th.TILE - 60)
    for _ in range(80):
        game.step(["right"])
        if game.state.level != level:
            return game.state
    raise AssertionError("walking right from just short of the flag never passed it")


def stand(game: ToyHopper, x: int) -> Frame:
    """Stand the player at a level pixel and draw the screen as it then is."""
    place(game, x)
    game.step([])  # the camera follows at the next frame
    assert game.state.mode == th.PLAYING and game.state.on_ground
    return game.render()


def no_start(seed: int, frames: int) -> list[list[str]]:
    """The fixed, varied run of presses, with Start left out (it begins the game again)."""
    return [[name for name in held if name != "start"] for held in script(seed, frames)]


class Knowing:
    """A player who knows the level a game is on: runs right and holds a jump at each
    thing in the way."""

    def __init__(self) -> None:
        self.hold = 0

    def press(self, game: ToyHopper) -> list[str]:
        st = game.state
        front = st.x + th.PLAYER_W
        edges = [col * th.TILE for col, _ in game.gaps] + [col * th.TILE for col in game.walls]
        ahead = [game.walker_at(i) for i in range(len(game.walkers))]
        jump = any(0 <= edge - front <= 8 for edge in edges)
        jump = jump or any(0 <= wx - front <= 16 for wx in ahead)
        if st.on_ground and self.hold == 0 and jump:
            self.hold = 40
        if self.hold and st.on_ground and self.hold < 38:
            self.hold = 0  # landed: let go, so the next press is a fresh one
        held = ["right", "a"] if self.hold else ["right"]
        self.hold = max(0, self.hold - 1)
        return held


def finish(game: ToyHopper, frames: int = 4000) -> HopperState:
    """Play the level the game is on as one who knows it, until the flag is passed."""
    began = game.state.level
    player = Knowing()
    for _ in range(frames):
        st = game.state
        if st.level != began or st.mode != th.PLAYING:
            break
        game.step(player.press(game))
    return game.state


def drawn(frame: Frame) -> np.ndarray:
    return np.frombuffer(frame.pixels, dtype=np.uint8).reshape(frame.height, frame.width, 3)


def the_play(game: ToyHopper) -> str:
    """A long scripted play that passes the flag, as one number."""
    seen = hashlib.sha256()
    for n, held in enumerate(script(5, 600)):
        game.step(held)
        seen.update(game.ram)
        if n % 60 == 0:
            seen.update(game.render().pixels)
    to_the_flag(game)
    seen.update(game.ram)
    seen.update(game.render().pixels)
    for held in script(6, 200):
        game.step(held)
        seen.update(game.ram)
    seen.update(game.render().pixels)
    seen.update(game.snapshot())
    return seen.hexdigest()


def test_one_level_is_the_game_it_was() -> None:
    assert ToyHopper(3).levels == 1 and th.MOST_LEVELS == 8
    assert the_play(ToyHopper(3)) == AS_IT_WAS
    assert the_play(cartridge(3, 1)) == AS_IT_WAS
    # Its flag starts the same level again, as it always did.
    game = ToyHopper(3)
    start = game.render().pixels
    st = to_the_flag(game)
    assert st.level == 1 and layout(game) == layout(ToyHopper(3))
    assert pixel(game.render(), 4, 100) == th.SKY
    for _ in range(3):
        to_the_flag(game)
    assert game.state.level == 4 and layout(game) == layout(ToyHopper(3))
    game.reset()
    assert game.render().pixels == start


def test_the_flag_leads_to_the_next_level() -> None:
    game, first, second = cartridge(3, 2), ToyHopper(3), ToyHopper(4)
    assert game.levels == 2 and layout(game) == layout(first) != layout(second)
    assert pixel(game.render(), 4, 100) == PALETTES[0].sky == th.SKY
    st = to_the_flag(game)
    # The score and the counter as they always went; the player at the start, standing.
    assert (st.level, st.score, st.lives) == (1, th.FLAG_SCORE, th.START_LIVES)
    assert (st.x, st.y, st.vx, st.clock, st.scroll) == (th.START_X, GROUNDED_Y, 0, 0, 0)
    assert st.coins == 0 and game.ram[LAYOUT.coin_mask] == 0 and st.mode == th.PLAYING
    # The level is the next one: the practice level of the next seed.
    assert layout(game) == layout(second)
    # Its screen is that level's, in the second palette; the strip along the top apart
    # (the score is on it).
    theirs, ours = drawn(second.render())[24:], drawn(game.render())[24:]
    was, now = PALETTES[0], PALETTES[1]
    expected = theirs.copy()
    for old, new in zip(was, now, strict=True):
        expected[(theirs == old).all(axis=2)] = new
    assert (ours == expected).all() and not (ours == theirs).all()
    assert pixel(game.render(), 4, 100) == now.sky
    assert pixel(game.render(), 4, th.GROUND_Y + 1) == now.grass
    assert pixel(game.render(), 4, th.GROUND_Y + 8) == now.ground
    # And it plays as that level does from its start: the same ground, walkers and
    # coins. First as one who knows the level plays it, to its flag; then a play that
    # falls in.
    slots = slice(LAYOUT.walkers, LAYOUT.walkers + 3 * th.WALKER_SLOTS)

    def together(held: list[str]) -> HopperState:
        game.step(held)
        second.step(held)
        a, b = game.state, second.state
        assert (a.x, a.y, a.vx, a.vy, a.mode, a.scroll, a.clock, a.lives, a.coins) == (
            b.x,
            b.y,
            b.vx,
            b.vy,
            b.mode,
            b.scroll,
            b.clock,
            b.lives,
            b.coins,
        )
        assert game.ram[slots] == second.ram[slots]
        assert game.ram[LAYOUT.coin_mask] == second.ram[LAYOUT.coin_mask]
        return b

    player = Knowing()
    walkers = frames = 0
    while second.state.level == 0:
        st = together(player.press(second))
        assert st.level == 1 or game.state.level == 1
        walkers += any(game.ram[slots])
        frames += 1
        assert frames < 4000
    assert walkers and second.state.lives == th.START_LIVES
    assert (game.state.level, game.state.score) == (2, second.state.score + th.FLAG_SCORE)
    assert layout(game) == layout(first)  # and on to the level after it
    game, second = cartridge(3, 2), ToyHopper(4)
    to_the_flag(game)
    fell = 0
    for held in no_start(9, 900):
        fell += together(held).mode == th.DYING
    assert fell
    # A wall of it, in its colour.
    game = cartridge(3, 2)
    to_the_flag(game)
    wall = game.walls[0] * th.TILE
    frame = stand(game, wall - 40)
    assert pixel(frame, wall - game.state.scroll + 8, th.GROUND_Y - 8) == now.wall


def test_after_the_last_level_comes_the_first() -> None:
    game = cartridge(3, 2)
    seen = []
    for _ in range(5):
        seen.append((game.state.level, layout(game), pixel(game.render(), 4, 100)))
        to_the_flag(game)
    one, two = layout(ToyHopper(3)), layout(ToyHopper(4))
    assert seen == [
        (0, one, PALETTES[0].sky),
        (1, two, PALETTES[1].sky),
        (2, one, PALETTES[0].sky),
        (3, two, PALETTES[1].sky),
        (4, one, PALETTES[0].sky),
    ]
    # Three levels: all three differ, and then the first again.
    game = cartridge(3, 3)
    seen = []
    for _ in range(4):
        seen.append((layout(game), pixel(game.render(), 4, 100)))
        to_the_flag(game)
    three = layout(ToyHopper(5))
    assert seen == [
        (one, PALETTES[0].sky),
        (two, PALETTES[1].sky),
        (three, PALETTES[2].sky),
        (one, PALETTES[0].sky),
    ]
    assert len({one, two, three}) == 3
    # The most a cartridge holds: eight, each the practice level of the next seed, and
    # the colours come round again after three.
    game = cartridge(3, 8)
    for n in range(9):
        assert layout(game) == layout(ToyHopper(3 + n % 8)), n
        assert pixel(game.render(), 4, 100) == PALETTES[n % 8 % 3].sky, n
        to_the_flag(game)


def test_a_life_lost_starts_the_level_it_was_lost_on() -> None:
    game, first, second = cartridge(3, 2), ToyHopper(3), ToyHopper(4)
    to_the_flag(game)
    _fall_in(game)
    assert layout(game) == layout(second)  # while it dies, too
    st = run(game, [], th.DYING_FRAMES)
    assert (st.level, st.lives, st.mode) == (1, th.START_LIVES - 1, th.PLAYING)
    assert (st.x, st.clock, st.scroll) == (th.START_X, 0, 0)
    assert layout(game) == layout(second) and pixel(game.render(), 4, 100) == PALETTES[1].sky
    # Game over is on that level's screen, and Start begins at the first level.
    for _ in range(2):
        _fall_in(game)
        st = run(game, [], th.DYING_FRAMES)
    assert st.game_over and st.level == 1 and layout(game) == layout(second)
    assert pixel(game.render(), 4, 100) == PALETTES[1].sky
    game.step(["start"])
    st = game.state
    assert (st.level, st.lives, st.score, st.mode) == (0, th.START_LIVES, 0, th.PLAYING)
    assert layout(game) == layout(first) and pixel(game.render(), 4, 100) == PALETTES[0].sky
    assert game.levels == 2
    # And so does a reset.
    to_the_flag(game)
    assert layout(game) == layout(second)
    game.reset()
    assert game.state.level == 0 and layout(game) == layout(first) and game.levels == 2


def test_every_level_of_a_cartridge_can_be_finished() -> None:
    for seed in range(6):
        game = cartridge(seed, 3)
        for passed in (1, 2, 3):
            st = finish(game)
            assert (st.level, st.lives, st.mode) == (passed, th.START_LIVES, th.PLAYING), (
                seed,
                st,
            )
        assert st.score >= 3 * th.FLAG_SCORE


def test_a_snapshot_brings_back_its_level() -> None:
    game, first, second = cartridge(1, 2), ToyHopper(1), ToyHopper(2)
    at_start = game.snapshot()
    to_the_flag(game)
    for held in no_start(7, 150):
        game.step(held)
    saved, screen = game.snapshot(), game.render().pixels
    assert game.state.level == 1 and len(saved) == th.RAM_SIZE + 4
    presses = no_start(8, 300)
    after = []
    for held in presses:
        game.step(held)
        after.append(game.ram)
    # On to the level after it, which is the first again; then back.
    to_the_flag(game)
    assert game.state.level == 2 and layout(game) == layout(first)
    game.restore(saved)
    assert game.state.level == 1 and layout(game) == layout(second)
    assert game.render().pixels == screen and game.snapshot() == saved
    again = []
    for held in presses:
        game.step(held)
        again.append(game.ram)
    assert again == after
    # After a reset too; and a snapshot of the first level brings the first level back.
    game.reset()
    game.restore(saved)
    assert layout(game) == layout(second) and game.render().pixels == screen
    game.restore(at_start)
    assert game.state.level == 0 and layout(game) == layout(first)
    assert pixel(game.render(), 4, 100) == PALETTES[0].sky
    # A restore that fails leaves the level as it was.
    game.restore(saved)
    with pytest.raises(ConfigurationError, match="Not a Toy Hopper snapshot"):
        game.restore(saved[:-1])
    with pytest.raises(ConfigurationError, match="another level"):
        game.restore(cartridge(2, 2).snapshot())
    assert layout(game) == layout(second) and game.snapshot() == saved


def test_each_level_has_colours_of_its_own() -> None:
    assert len(PALETTES) == 3
    assert tuple(PALETTES[0]) == (th.SKY, th.GROUND, th.GRASS, th.WALL)
    fixed = (th.PLAYER, th.PLAYER_HEAD, th.WALKER, th.WALKER_FEET, th.COIN, th.FLAG, th.HUD)
    colours = [colour for palette in PALETTES for colour in palette] + list(fixed)
    assert len(colours) == 19 and len(set(colours)) == 19
    # Apart enough to be told apart by an eye that allows for a little: no two within
    # 24 on every channel.
    for a, b in combinations(colours, 2):
        assert max(abs(x - y) for x, y in zip(a, b, strict=True)) > 24, (a, b)
    # On every level the player, a coin, a walker, the flag and the lives are drawn in
    # the colours they always had.
    game = cartridge(3, 3)
    for n in range(3):
        palette = PALETTES[n]
        st = game.state
        frame = game.render()
        assert pixel(frame, 4, 100) == palette.sky
        assert pixel(frame, st.x + 1, st.y + 12) == th.PLAYER
        assert pixel(frame, st.x + 4, st.y + 4) == th.PLAYER_HEAD
        assert pixel(frame, 10, 10) == th.HUD
        col, row = game.coins[0]
        frame = stand(game, col * th.TILE - 60)
        scroll = game.state.scroll
        assert pixel(frame, col * th.TILE + 8 - scroll, row * th.TILE + 8) == th.COIN
        low, _ = game.walkers[0]
        frame = stand(game, low - 60)
        wx = game.walker_at(0) - game.state.scroll
        assert pixel(frame, wx + 4, th.WALKER_Y + 4) == th.WALKER
        assert pixel(frame, wx + 4, th.WALKER_Y + th.WALKER_SIZE - 1) == th.WALKER_FEET
        frame = stand(game, th.FLAG_COL * th.TILE - 60)
        flag = th.FLAG_COL * th.TILE - game.state.scroll
        assert pixel(frame, flag + 1, th.GROUND_Y - 40) == th.FLAG
        assert pixel(frame, flag - 8, th.GROUND_Y - 40) == palette.sky
        assert pixel(frame, flag - 8, th.GROUND_Y + 1) == palette.grass
        assert pixel(frame, flag - 8, th.GROUND_Y + 8) == palette.ground
        to_the_flag(game)


def test_levels_that_are_no_count_are_refused() -> None:
    game = cartridge(3, 2)
    to_the_flag(game)
    before = game.snapshot()
    for bad in (0, 9, -1, 1.5, "2", True, None):
        with pytest.raises(ConfigurationError, match="whole number of levels from 1 to 8"):
            game.levels = bad  # type: ignore[assignment]
        assert game.levels == 2 and game.snapshot() == before
        assert layout(game) == layout(ToyHopper(4))
    # It is the cartridge's: a reset keeps it, and each cartridge has its own.
    game.reset()
    assert game.levels == 2 and ToyHopper(3).levels == 1
    # Said during a game, the level played is the one the counter says under the new
    # count (it is meant to be said before the game starts).
    to_the_flag(game)
    assert layout(game) == layout(ToyHopper(4))
    game.levels = 1
    assert layout(game) == layout(ToyHopper(3)) and game.state.level == 1
    assert pixel(game.render(), 4, 100) == PALETTES[0].sky
    game.levels = 8
    assert layout(game) == layout(ToyHopper(4)) and game.levels == 8
    assert pixel(game.render(), 4, 100) == PALETTES[1].sky
