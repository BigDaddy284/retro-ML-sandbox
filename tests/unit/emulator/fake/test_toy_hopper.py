"""T-984: Toy Hopper, the practice platform game."""

from __future__ import annotations

from collections import Counter
from itertools import pairwise

import pytest

from retro_ai_lab.core import ConfigurationError, Frame
from retro_ai_lab.emulator.fake import HopperState, ToyHopper
from retro_ai_lab.emulator.fake import toy_hopper as th
from retro_ai_lab.emulator.fake.toy_hopper import LAYOUT

GROUNDED_Y = th.GROUND_Y - th.PLAYER_H


def run(game: ToyHopper, held: list[str], frames: int) -> HopperState:
    for _ in range(frames):
        game.step(held)
    return game.state


def place(game: ToyHopper, x: int, *, vx: int = 0) -> None:
    """Stand the player at a level pixel (tests may write the layout; the app may not)."""
    ram = bytearray(game.snapshot())
    ram[LAYOUT.x_low], ram[LAYOUT.x_page], ram[LAYOUT.x_sub] = x & 0xFF, x >> 8, 0
    ram[LAYOUT.y], ram[LAYOUT.y_sub] = GROUNDED_Y, 0
    ram[LAYOUT.vx], ram[LAYOUT.vy], ram[LAYOUT.on_ground] = vx & 0xFF, 0, 1
    game.restore(bytes(ram))


def script(seed: int, frames: int) -> list[list[str]]:
    """A fixed, varied run of button presses: changes every few frames, Start now and then."""
    out: list[list[str]] = []
    rng = seed
    held: list[str] = []
    for i in range(frames):
        if i % 6 == 0:
            rng = (rng * 1103515245 + 12345) & 0x7FFFFFFF
            bits = rng >> 8
            held = [name for n, name in enumerate(("right", "a", "left", "start")) if bits >> n & 1]
            if "right" in held and "left" in held and bits & 0x30:
                held.remove("left")  # mostly it heads right
        out.append(held)
    return out


def pixel(frame: Frame, x: int, y: int) -> tuple[int, ...]:
    at = (y * frame.width + x) * frame.channels
    return tuple(frame.pixels[at : at + frame.channels])


def test_the_same_seed_and_buttons_play_the_same_game() -> None:
    one, two = ToyHopper(3), ToyHopper(3)
    for i, held in enumerate(script(9, 900)):
        one.step(held)
        two.step(held)
        assert one.ram == two.ram
        if i % 150 == 0:
            assert one.render() == two.render()
    assert one.state.frame == 900 and one.state.x != th.START_X
    other = ToyHopper(4)
    assert (other.gaps, other.walls, other.walkers, other.coins) != (
        one.gaps,
        one.walls,
        one.walkers,
        one.coins,
    )
    for game in (one, other, ToyHopper(0)):  # every level has every kind of thing
        assert {width for _, width in game.gaps} == {2, 3, th.WIDEST_GAP}
        assert len(game.walls) >= 2 and len(game.walkers) >= 2 and len(game.coins) >= 3
        assert all(th.START_TILES <= col < th.LEVEL_TILES - th.END_TILES for col, _ in game.gaps)
    for bad in (-1, 70000, True, 1.5, "3"):
        with pytest.raises(ConfigurationError):
            ToyHopper(bad)  # type: ignore[arg-type]
    with pytest.raises(ConfigurationError, match="Unknown Toy Hopper buttons"):
        one.step(["select"])


def test_walking_speeds_up_coasts_and_stops_at_a_wall_and_the_edge() -> None:
    game = ToyHopper(0)
    speeds = []
    for _ in range(10):
        game.step(["right"])
        speeds.append(game.state.vx)
    assert speeds == [4, 8, 12, 16, 20, 24, 28, 32, 32, 32] and th.MAX_SPEED == 32
    before = game.state.x
    assert run(game, ["right"], 5).x == before + 10  # two pixels a frame at full speed
    for _ in range(8):
        game.step([])
    assert game.state.vx == 0
    rest = game.state.x
    assert run(game, [], 20).x == rest and game.state.on_ground
    # The level's left edge holds.
    assert run(game, ["left"], 200).x == 0 and game.state.vx == 0
    # A wall stops the player dead, on the ground, a pixel short of overlapping it.
    wall = game.walls[0]
    place(game, wall * th.TILE - 60)
    st = run(game, ["right"], 80)
    assert st.x == wall * th.TILE - th.PLAYER_W and st.vx == 0 and st.on_ground
    assert st.mode == th.PLAYING and st.lives == th.START_LIVES


def _apex(game: ToyHopper, hold: int) -> tuple[int, int]:
    """Rise in pixels and frames in the air for a jump with A held ``hold`` frames."""
    game.step([])
    top, frames = GROUNDED_Y, 0
    for i in range(200):
        game.step(["a"] if i < hold else [])
        frames += 1
        top = min(top, game.state.y)
        if game.state.on_ground:
            break
    return GROUNDED_Y - top, frames


def test_a_held_jump_goes_higher_and_only_it_clears_the_widest_gap() -> None:
    tap, tap_frames = _apex(ToyHopper(0), 4)  # one step of A
    held, held_frames = _apex(ToyHopper(0), 40)
    assert th.TILE < tap < held and tap_frames < held_frames  # a tap still clears a wall
    assert _apex(ToyHopper(0), 12)[0] > tap and _apex(ToyHopper(0), 12)[0] < held
    # Only from the ground, and only on a fresh press: A held through the landing does nothing.
    game = ToyHopper(0)
    game.step([])
    takeoffs, grounded = 0, True
    for _ in range(200):
        game.step(["a"])
        takeoffs += grounded and not game.state.on_ground
        grounded = game.state.on_ground
    assert takeoffs == 1 and game.state.on_ground
    game.step([])
    game.step(["a"])
    assert not game.state.on_ground and game.state.vy < 0  # let go and press again: a jump
    # The widest gap, taken at full speed from its edge.
    game = ToyHopper(0)
    gap = next(col for col, width in game.gaps if width == th.WIDEST_GAP)
    edge = gap * th.TILE - 8
    place(game, edge, vx=th.MAX_SPEED)
    st = run(game, ["a", "right"], 60)
    assert st.mode == th.PLAYING and st.on_ground and st.lives == th.START_LIVES
    assert st.x >= (gap + th.WIDEST_GAP) * th.TILE - th.PLAYER_W
    place(game, edge, vx=th.MAX_SPEED)
    game.step([])  # A let go, so the next press is fresh
    place(game, edge, vx=th.MAX_SPEED)
    run(game, ["a", "right"], 4)
    assert run(game, ["right"], 56).mode == th.DYING  # a tap from the same spot falls in
    # A two-tile gap needs no more than the tap.
    small = next(col for col, width in game.gaps if width == 2)
    fresh = ToyHopper(0)
    place(fresh, small * th.TILE - 8, vx=th.MAX_SPEED)
    run(fresh, ["a", "right"], 4)
    st = run(fresh, ["right"], 20)
    assert st.mode == th.PLAYING and st.on_ground and st.x >= (small + 2) * th.TILE - th.PLAYER_W


def _fall_in(game: ToyHopper) -> None:
    for _ in range(600):
        game.step(["right"])
        if game.state.mode == th.DYING:
            return
    raise AssertionError("walking right never fell into a gap")


def test_a_fall_pauses_then_takes_a_life_and_the_third_is_game_over() -> None:
    game = ToyHopper(0)
    for lives in (3, 2):
        _fall_in(game)
        st = game.state
        assert st.lives == lives and st.dying == th.DYING_FRAMES and st.y >= th.SCREEN_HEIGHT
        # The pause: the life count has not moved and no button is heard.
        for left in range(th.DYING_FRAMES - 1, 0, -1):
            game.step(["a", "left"])
            now = game.state
            assert (now.mode, now.lives, now.dying, now.x, now.y) == (
                th.DYING,
                lives,
                left,
                st.x,
                st.y,
            )
        game.step(["a", "left"])
        now = game.state
        assert now.lives == lives - 1 and now.mode == th.PLAYING and now.dying == 0
        assert (now.x, now.y, now.clock, now.scroll) == (th.START_X, GROUNDED_Y, 0, 0)
        assert now.on_ground and now.vx == 0
    _fall_in(game)
    st = run(game, [], th.DYING_FRAMES)
    assert st.game_over and st.lives == 0 and st.mode == th.GAME_OVER
    kept = (st.x, st.y, st.score, st.lives)
    for held in (["a"], ["right"], ["left", "a"], []):
        now = run(game, held, 10)
        assert now.game_over and (now.x, now.y, now.score, now.lives) == kept
    frame = game.state.frame
    game.step(["start"])
    now = game.state
    assert not now.game_over and now.lives == th.START_LIVES and now.score == 0
    assert now.x == th.START_X and now.frame == frame + 1  # the console's clock runs on


def _slots(game: ToyHopper) -> list[tuple[int, int, int]]:
    ram = game.ram
    return [
        (ram[at], ram[at + 1], ram[at + 2])
        for at in range(LAYOUT.walkers, LAYOUT.walkers + 3 * th.WALKER_SLOTS, 3)
    ]


def _walkers_on_screen(game: ToyHopper) -> list[tuple[int, int, int]]:
    scroll = game.state.scroll
    seen = [
        (1, game.walker_at(i) - scroll, th.WALKER_Y)
        for i in range(len(game.walkers))
        if 0 <= game.walker_at(i) - scroll <= th.SCREEN_WIDTH - th.WALKER_SIZE
    ][: th.WALKER_SLOTS]
    return seen + [(0, 0, 0)] * (th.WALKER_SLOTS - len(seen))


def test_a_walker_paces_its_ground_and_touching_it_is_a_death() -> None:
    game = ToyHopper(0)
    low, high = game.walkers[0]
    assert high - low == th.TILE * 4 - th.WALKER_SIZE
    path = [game.walker_at(0, clock) for clock in range(4 * (high - low) + 1)]
    assert min(path) == low and max(path) == high and path[0] == path[-1] == low
    assert all(abs(b - a) <= 1 for a, b in pairwise(path))
    # At the start no walker is in sight: every slot is empty.
    game.step([])
    assert _slots(game) == _walkers_on_screen(game) == [(0, 0, 0)] * th.WALKER_SLOTS
    # Standing where it turns round: fine until it arrives, then the dying pause.
    place(game, high + th.WALKER_SIZE - 4)
    game.step([])
    assert game.state.mode == th.PLAYING
    assert _slots(game) == _walkers_on_screen(game) and _slots(game)[0][0] == 1
    assert _slots(game)[0][1] == game.walker_at(0) - game.state.scroll
    for _ in range(4 * (high - low)):
        game.step([])
        assert _slots(game) == _walkers_on_screen(game)
        if game.state.mode == th.DYING:
            break
    st = game.state
    assert st.mode == th.DYING and st.lives == th.START_LIVES and st.on_ground
    touching = game.walker_at(0)
    assert touching < st.x + th.PLAYER_W and st.x < touching + th.WALKER_SIZE
    assert touching + th.WALKER_SIZE - st.x == 1  # caught the moment they overlapped
    st = run(game, [], th.DYING_FRAMES)
    assert st.lives == th.START_LIVES - 1 and st.x == th.START_X
    # Jumping over it is not a touch.
    place(game, low - 40, vx=th.MAX_SPEED)
    game.step([])
    while game.walker_at(0) - game.state.x > th.PLAYER_W + 14:
        game.step(["right"])
    st = run(game, ["a", "right"], 40)  # a held jump: over it and down well clear
    assert st.mode == th.PLAYING and st.on_ground and st.lives == th.START_LIVES - 1
    assert st.x > game.walker_at(0) + th.WALKER_SIZE


def test_a_coin_scores_once_and_the_flag_starts_the_level_again() -> None:
    game = ToyHopper(0)
    col, row = game.coins[0]
    assert row == th.COIN_ROW
    place(game, col * th.TILE + 2)
    game.step([])
    where = (col * th.TILE + th.TILE // 2 - game.state.scroll, row * th.TILE + th.TILE // 2)
    assert pixel(game.render(), *where) == th.COIN
    assert run(game, [], 30).score == 0  # standing under it is not enough
    run(game, ["a"], 4)
    st = run(game, [], 40)
    assert st.score == th.COIN_SCORE and st.coins == 1 and st.on_ground
    assert game.ram[LAYOUT.coin_mask] == 1
    assert pixel(game.render(), *where) == th.SKY
    run(game, ["a"], 4)
    assert run(game, [], 40).score == th.COIN_SCORE  # gone: a second jump finds nothing
    # The flag: 100 points, a level up, and the level starts again with its coins back.
    place(game, th.FLAG_COL * th.TILE - 60)
    st = run(game, ["right"], 80)
    assert st.level == 1 and st.score == th.COIN_SCORE + th.FLAG_SCORE
    assert st.coins == 0 and game.ram[LAYOUT.coin_mask] == 0 and st.lives == th.START_LIVES
    assert st.x < th.START_X + 2 * 80 and st.clock < 80  # back at the start, already walking


def test_memory_holds_the_state_at_fixed_addresses() -> None:
    game = ToyHopper(2)
    noise = set()
    for i, held in enumerate(script(5, 400)):
        timer = game.ram[LAYOUT.timer]
        game.step(held)
        ram, st = game.ram, game.state
        assert len(ram) == th.RAM_SIZE == th.RAM_REGION.size
        assert ram[LAYOUT.frame] | ram[LAYOUT.frame + 1] << 8 == st.frame == i + 1
        assert ram[LAYOUT.x_low] | ram[LAYOUT.x_page] << 8 == st.x
        assert ram[LAYOUT.y] == st.y and ram[LAYOUT.on_ground] == int(st.on_ground)
        assert ram[LAYOUT.vx] == st.vx & 0xFF and ram[LAYOUT.vy] == st.vy & 0xFF
        assert ram[LAYOUT.lives] == st.lives and ram[LAYOUT.level] == st.level
        assert ram[LAYOUT.score] | ram[LAYOUT.score + 1] << 8 == st.score
        assert ram[LAYOUT.coins] == st.coins == bin(ram[LAYOUT.coin_mask]).count("1")
        assert ram[LAYOUT.mode] == st.mode and ram[LAYOUT.dying] == st.dying
        assert ram[LAYOUT.scroll] | ram[LAYOUT.scroll + 1] << 8 == st.scroll
        assert ram[LAYOUT.clock] | ram[LAYOUT.clock + 1] << 8 == st.clock
        assert ram[LAYOUT.timer] == (timer + 1) & 0xFF
        expected = sum(1 << th.BUTTON_NAMES.index(name) for name in held)
        assert ram[LAYOUT.input] == expected
        assert ram[LAYOUT.seed] == 2 and ram[LAYOUT.header :].startswith(b"TOYHOPPER")
        if st.mode == th.PLAYING:
            assert st.scroll == max(0, min(th.LEVEL_WIDTH - th.SCREEN_WIDTH, st.x - th.CAMERA_X))
        noise.add(ram[LAYOUT.noise : LAYOUT.noise + 4])
    assert len(noise) == 400  # new noise every frame
    assert th.BUTTON_NAMES == ("up", "down", "left", "right", "a", "b", "start")


def test_the_screen_shows_the_level_where_the_state_says() -> None:
    game = ToyHopper(0)
    frame = game.render()
    assert (frame.width, frame.height, frame.channels) == (256, 240, 3)
    assert len(frame.pixels) == 256 * 240 * 3 and frame.number == 0
    counts = Counter(tuple(frame.pixels[i : i + 3]) for i in range(0, len(frame.pixels), 3 * 7))
    assert counts.most_common(1)[0][0] == th.SKY
    # At the start: the player where it stands, ground under it, three lives in the strip.
    assert pixel(frame, th.START_X + 1, GROUNDED_Y + 10) == th.PLAYER
    assert pixel(frame, th.START_X + 5, GROUNDED_Y + 4) == th.PLAYER_HEAD
    assert pixel(frame, 40, th.GROUND_Y + 1) == th.GRASS
    assert pixel(frame, 40, th.GROUND_Y + 10) == th.GROUND
    assert [pixel(frame, 10 + 12 * life, 10) for life in range(4)] == [th.HUD] * 3 + [th.SKY]
    # Once the level scrolls the player is held a third of the way across.
    wall = game.walls[0]
    place(game, wall * th.TILE - 60)
    game.step([])
    st, frame = game.state, game.render()
    assert st.scroll == st.x - th.CAMERA_X > 0 and frame.number == st.frame
    assert pixel(frame, th.CAMERA_X + 1, GROUNDED_Y + 10) == th.PLAYER
    assert pixel(frame, wall * th.TILE - st.scroll + 8, th.GROUND_Y - 8) == th.WALL
    assert pixel(frame, wall * th.TILE - st.scroll - 8, th.GROUND_Y - 8) == th.SKY
    # A gap is sky all the way down; the ground either side is ground.
    gap, width = game.gaps[1]
    place(game, gap * th.TILE - 40)
    game.step([])
    st, frame = game.state, game.render()
    for col in range(gap, gap + width):
        assert pixel(frame, col * th.TILE - st.scroll + 8, th.GROUND_Y + 12) == th.SKY
    assert pixel(frame, (gap - 1) * th.TILE - st.scroll + 8, th.GROUND_Y + 12) == th.GROUND
    assert pixel(frame, (gap + width) * th.TILE - st.scroll + 8, th.GROUND_Y + 12) == th.GROUND
    # A walker where it is now, feet darker than the body.
    low, _ = game.walkers[0]
    place(game, low - 60)
    game.step([])
    st, frame = game.state, game.render()
    wx = game.walker_at(0) - st.scroll
    assert pixel(frame, wx + 7, th.WALKER_Y + 3) == th.WALKER
    assert pixel(frame, wx + 7, th.WALKER_Y + th.WALKER_SIZE - 1) == th.WALKER_FEET
    # The flag, and the score bar growing from the right.
    place(game, th.FLAG_COL * th.TILE - 60)
    game.step([])
    st, frame = game.state, game.render()
    assert pixel(frame, th.FLAG_COL * th.TILE - st.scroll + 1, th.GROUND_Y - 40) == th.FLAG
    assert pixel(frame, th.SCREEN_WIDTH - 10, 10) == th.SKY
    run(game, ["right"], 80)  # through the flag: 100 points
    assert pixel(game.render(), th.SCREEN_WIDTH - 10, 10) == th.HUD


def test_a_snapshot_brings_back_the_exact_game() -> None:
    game = ToyHopper(1)
    presses = script(7, 700)
    for held in presses[:300]:
        game.step(held)
    saved = game.snapshot()
    assert isinstance(saved, bytes) and len(saved) == th.RAM_SIZE + 4
    after = []
    for held in presses[300:]:
        game.step(held)
        after.append(game.ram)
    assert game.snapshot() != saved
    game.restore(saved)
    assert game.snapshot() == saved
    again = []
    for held in presses[300:]:
        game.step(held)
        again.append(game.ram)
    assert again == after
    spoiled = bytearray(saved)
    spoiled[LAYOUT.header] ^= 0xFF
    for bad in (saved[:-1], saved + b"\0", bytes(spoiled), bytearray(saved), "save", None):
        with pytest.raises(ConfigurationError, match="Not a Toy Hopper snapshot"):
            game.restore(bad)  # type: ignore[arg-type]
    with pytest.raises(ConfigurationError, match="another level"):
        ToyHopper(2).restore(saved)
    # A restore that fails leaves the game as it was.
    assert game.ram == again[-1]


def _finish(game: ToyHopper, frames: int = 4000) -> HopperState:
    """A player who knows the level: runs right and holds a jump at each thing in the way."""
    hold = 0
    for _ in range(frames):
        st = game.state
        if st.level or st.mode != th.PLAYING:
            break
        front = st.x + th.PLAYER_W
        edges = [col * th.TILE for col, _ in game.gaps] + [col * th.TILE for col in game.walls]
        ahead = [game.walker_at(i) for i in range(len(game.walkers))]
        jump = any(0 <= edge - front <= 8 for edge in edges)
        jump = jump or any(0 <= wx - front <= 16 for wx in ahead)
        if st.on_ground and hold == 0 and jump:
            hold = 40
        if hold and st.on_ground and hold < 38:
            hold = 0  # landed: let go, so the next press is a fresh one
        game.step(["right", "a"] if hold else ["right"])
        hold = max(0, hold - 1)
    return game.state


def test_every_level_can_be_finished_without_losing_a_life() -> None:
    for seed in range(12):
        st = _finish(ToyHopper(seed))
        assert (st.level, st.lives, st.mode) == (1, th.START_LIVES, th.PLAYING), (seed, st)
        assert st.score >= th.FLAG_SCORE
