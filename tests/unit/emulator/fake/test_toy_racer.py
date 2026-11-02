"""T-004: the Toy Racer synthetic game."""

from __future__ import annotations

import time
from typing import Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from retro_ai_lab.core import ConfigurationError, Frame
from retro_ai_lab.emulator.fake import ToyRacer
from retro_ai_lab.emulator.fake.toy_racer import (
    BUTTON_NAMES,
    BUTTONS,
    CHECKPOINTS,
    LANES,
    LAYOUT,
    MAX_SPEED,
    PX_HUD,
    PX_OBSTACLE,
    PX_PLAYER,
    RAM_SIZE,
    SCREEN_HEIGHT,
    SCREEN_WIDTH,
    START_LIVES,
    TRACK_LENGTH,
)


def run(game: ToyRacer, buttons: tuple[str, ...], frames: int) -> None:
    for _ in range(frames):
        game.step(buttons)


def safe_lane(game: ToyRacer) -> int:
    """A lane with no obstacles at all, if one exists; otherwise the emptiest."""
    counts = {lane: 0 for lane in range(LANES)}
    for _, lane in game.obstacles:
        counts[lane] += 1
    return min(counts, key=lambda lane: counts[lane])


def drive_to_lane(game: ToyRacer, lane: int) -> None:
    while game.state.lane != lane:
        game.step(("left",) if game.state.lane > lane else ("right",))


# ---------------------------------------------------------- construction


def test_initial_state() -> None:
    g = ToyRacer()
    s = g.state
    assert s.frame == 0
    assert s.position == 0
    assert s.lane == LANES // 2
    assert s.speed == 0
    assert s.lap == 0
    assert s.checkpoint == 0
    assert s.lives == START_LIVES
    assert s.score == 0
    assert not s.game_over
    assert len(g.ram) == RAM_SIZE


@pytest.mark.parametrize("seed", [-1, 1.5, "0", True, None])
def test_rejects_bad_seed(seed: Any) -> None:
    with pytest.raises(ConfigurationError):
        ToyRacer(seed)


def test_buttons_are_well_formed_and_unique() -> None:
    assert len(BUTTONS) == 7
    assert len(set(BUTTON_NAMES)) == len(BUTTON_NAMES)
    assert set(BUTTON_NAMES) == {"up", "down", "left", "right", "a", "b", "start"}


def test_obstacles_are_seeded_and_off_special_positions() -> None:
    a, b, c = ToyRacer(1), ToyRacer(1), ToyRacer(2)
    assert a.obstacles == b.obstacles
    assert a.obstacles != c.obstacles
    for pos, lane in a.obstacles:
        assert 0 < pos < TRACK_LENGTH
        assert pos not in (0, *CHECKPOINTS)
        assert 0 <= lane < LANES


# ---------------------------------------------------------- determinism


@settings(max_examples=25, deadline=None)
@given(
    seed=st.integers(0, 50),
    script=st.lists(st.frozensets(st.sampled_from(BUTTON_NAMES)), min_size=1, max_size=200),
)
def test_same_seed_and_inputs_give_identical_ram_and_frames(
    seed: int, script: list[frozenset[str]]
) -> None:
    a, b = ToyRacer(seed), ToyRacer(seed)
    for held in script:
        a.step(held)
        b.step(held)
        assert a.ram == b.ram
    assert a.render() == b.render()


def test_step_rejects_unknown_button() -> None:
    g = ToyRacer()
    with pytest.raises(ConfigurationError, match="jump"):
        g.step(("up", "jump"))


# ------------------------------------------------------------ controls


def test_up_accelerates_to_max_and_moves() -> None:
    g = ToyRacer()
    for expected in range(1, MAX_SPEED + 1):
        g.step(("up",))
        assert g.state.speed == expected
    g.step(("up",))
    assert g.state.speed == MAX_SPEED
    assert g.state.position == sum(range(1, MAX_SPEED + 1)) + MAX_SPEED


def test_down_decelerates_and_stops_at_zero() -> None:
    g = ToyRacer()
    run(g, ("up",), 3)
    g.step(("down",))
    assert g.state.speed == 2
    run(g, ("down",), 5)
    assert g.state.speed == 0
    pos = g.state.position
    g.step(("down",))
    assert g.state.position == pos


def test_b_is_a_hard_brake() -> None:
    g = ToyRacer()
    run(g, ("up",), MAX_SPEED)
    g.step(("b",))
    assert g.state.speed == 0
    g.step(("b", "up"))
    assert g.state.speed == 0


def test_up_and_down_together_cancel() -> None:
    g = ToyRacer()
    g.step(("up", "down"))
    assert g.state.speed == 0


def test_left_right_change_lane_within_bounds() -> None:
    g = ToyRacer()
    assert g.state.lane == 1
    g.step(("left",))
    assert g.state.lane == 0
    g.step(("left",))
    assert g.state.lane == 0
    run(g, ("right",), 5)
    assert g.state.lane == LANES - 1
    g.step(("left", "right"))
    assert g.state.lane == LANES - 1


def test_a_button_has_no_effect_on_gameplay() -> None:
    """Deliberate: the understanding engine must discover that A does nothing."""
    a, b = ToyRacer(), ToyRacer()
    for _ in range(50):
        a.step(("up", "a"))
        b.step(("up",))
        sa, sb = a.state, b.state
        assert (sa.position, sa.speed, sa.lane, sa.score, sa.lives) == (
            sb.position,
            sb.speed,
            sb.lane,
            sb.score,
            sb.lives,
        )
    assert a.ram[LAYOUT.input] != b.ram[LAYOUT.input]


def test_speed_persists_when_nothing_is_pressed() -> None:
    g = ToyRacer()
    run(g, ("up",), 2)
    g.step()
    assert g.state.speed == 2


def test_input_byte_mirrors_held_buttons() -> None:
    g = ToyRacer()
    g.step(("up", "a"))
    assert g.ram[LAYOUT.input] == (1 << BUTTON_NAMES.index("up")) | (1 << BUTTON_NAMES.index("a"))
    g.step()
    assert g.ram[LAYOUT.input] == 0


# --------------------------------------------------------- progression


def test_checkpoints_then_lap_in_safe_lane() -> None:
    g = ToyRacer(seed=3)
    lane = safe_lane(g)
    drive_to_lane(g, lane)
    assert g.state.checkpoint == 0
    seen = []
    frames = 0
    while g.state.lap == 0 and frames < 2000:
        g.step(("up",))
        frames += 1
        if not seen or seen[-1] != g.state.checkpoint:
            seen.append(g.state.checkpoint)
    assert g.state.lap == 1, f"no lap after {frames} frames (lives={g.state.lives})"
    assert seen == [0, 1, 2, 3, 0]
    if first_obstacle_in_lane(g, lane) is None:
        assert g.state.lives == START_LIVES


def test_lap_requires_all_checkpoints() -> None:
    """Teleporting past the line without checkpoints must not count a lap."""
    g = ToyRacer(seed=3)
    drive_to_lane(g, safe_lane(g))
    # Drive almost to the line, then restore a snapshot whose checkpoint byte is 0.
    while g.state.position < TRACK_LENGTH - 2:
        g.step(("up",))
    snap = bytearray(g.snapshot())
    snap[LAYOUT.checkpoint] = 0
    g.restore(bytes(snap))
    run(g, ("up",), 3)
    assert g.state.position < 16
    assert g.state.lap == 0
    assert g.state.checkpoint == 0


def test_score_rewards_distance_checkpoints_and_laps() -> None:
    g = ToyRacer(seed=3)
    drive_to_lane(g, safe_lane(g))
    before = g.state.score
    g.step(("up",))
    assert g.state.score == before + 1  # moved 1 unit at speed 1
    while g.state.lap == 0:
        g.step(("up",))
    s = g.state
    # distance (one full loop) + 3 checkpoints * 10 + lap 100, minus nothing.
    assert s.score >= TRACK_LENGTH + 30 + 100


def test_score_and_lap_saturate() -> None:
    g = ToyRacer()
    snap = bytearray(g.snapshot())
    snap[LAYOUT.score] = 0xFF
    snap[LAYOUT.score + 1] = 0xFF
    snap[LAYOUT.lap] = 0xFF
    snap[LAYOUT.checkpoint] = len(CHECKPOINTS)
    snap[LAYOUT.position] = TRACK_LENGTH - 1
    snap[LAYOUT.lane] = safe_lane(g)
    g.restore(bytes(snap))
    g.step(("up",))
    assert g.state.score == 0xFFFF
    assert g.state.lap == 0xFF


# ---------------------------------------------------------- collisions


def first_obstacle_in_lane(g: ToyRacer, lane: int) -> int | None:
    positions = sorted(pos for pos, lane_ in g.obstacles if lane_ == lane)
    return positions[0] if positions else None


def test_hitting_an_obstacle_costs_a_life_and_stops_the_car() -> None:
    g = ToyRacer(seed=5)
    lane = next(lane for lane in range(LANES) if first_obstacle_in_lane(g, lane) is not None)
    target = first_obstacle_in_lane(g, lane)
    assert target is not None
    drive_to_lane(g, lane)
    frames = 0
    while g.state.lives == START_LIVES and frames < 1000:
        g.step(("up",))
        frames += 1
    s = g.state
    assert s.lives == START_LIVES - 1
    assert s.speed == 0
    assert s.position == target


def test_three_hits_end_the_game_and_inputs_are_ignored() -> None:
    g = ToyRacer(seed=5)
    lane = next(lane for lane in range(LANES) if first_obstacle_in_lane(g, lane) is not None)
    drive_to_lane(g, lane)
    frames = 0
    while not g.state.game_over and frames < 5000:
        g.step(("up",))
        frames += 1
    assert g.state.game_over
    assert g.state.lives == 0
    frozen = g.state
    run(g, ("up", "right"), 10)
    after = g.state
    assert (after.position, after.lane, after.speed, after.score) == (
        frozen.position,
        frozen.lane,
        frozen.speed,
        frozen.score,
    )
    assert after.frame == frozen.frame + 10  # the console still runs


def test_start_restarts_after_game_over_but_not_before() -> None:
    g = ToyRacer(seed=5)
    run(g, ("up",), 3)
    moving = g.state
    g.step(("start",))
    assert g.state.position > moving.position  # start did nothing mid-game
    lane = next(lane for lane in range(LANES) if first_obstacle_in_lane(g, lane) is not None)
    drive_to_lane(g, lane)
    while not g.state.game_over:
        g.step(("up",))
    frame_at_end = g.state.frame
    g.step(("start",))
    s = g.state
    assert not s.game_over
    assert s.lives == START_LIVES
    assert (s.position, s.speed, s.lap, s.score) == (0, 0, 0, 0)
    assert s.frame == frame_at_end + 1  # the frame counter is not reset


# ------------------------------------------------------- RAM integrity


@settings(max_examples=20, deadline=None)
@given(
    seed=st.integers(0, 20),
    script=st.lists(st.frozensets(st.sampled_from(BUTTON_NAMES)), max_size=300),
)
def test_ram_values_stay_in_range(seed: int, script: list[frozenset[str]]) -> None:
    g = ToyRacer(seed)
    for held in script:
        g.step(held)
        s = g.state
        assert 0 <= s.position < TRACK_LENGTH
        assert 0 <= s.lane < LANES
        assert 0 <= s.speed <= MAX_SPEED
        assert 0 <= s.checkpoint <= len(CHECKPOINTS)
        assert 0 <= s.lives <= START_LIVES
        assert s.game_over == (s.lives == 0)
        assert g.ram[LAYOUT.header :] == ToyRacer().ram[LAYOUT.header :]


def test_noise_and_timer_change_every_frame() -> None:
    g = ToyRacer()
    seen = set()
    for i in range(10):
        assert g.ram[LAYOUT.timer] == i
        seen.add(g.ram[LAYOUT.noise : LAYOUT.noise + 4])
        g.step()
    assert len(seen) == 10


def test_frame_counter_is_16_bit_little_endian_and_wraps() -> None:
    g = ToyRacer()
    snap = bytearray(g.snapshot())
    snap[LAYOUT.frame] = 0xFF
    snap[LAYOUT.frame + 1] = 0xFF
    g.restore(bytes(snap))
    g.step()
    assert g.state.frame == 0
    g.step()
    assert g.ram[LAYOUT.frame : LAYOUT.frame + 2] == b"\x01\x00"


# ---------------------------------------------------- snapshot / restore


def test_snapshot_restore_is_exact_and_continues_identically() -> None:
    g = ToyRacer(7)
    run(g, ("up",), 20)
    snap = g.snapshot()
    ram_then = g.ram
    run(g, ("up", "left"), 50)
    g.restore(snap)
    assert g.ram == ram_then
    h = ToyRacer(7)
    run(h, ("up",), 20)
    for _ in range(50):
        g.step(("up", "left"))
        h.step(("up", "left"))
    assert g.ram == h.ram


@pytest.mark.parametrize("bad", [b"", b"x" * 10, b"x" * 36, bytearray(36), None])
def test_restore_rejects_bad_snapshots(bad: Any) -> None:
    g = ToyRacer()
    with pytest.raises(ConfigurationError):
        g.restore(bad)


def test_reset_restores_power_on_state_but_keeps_obstacles() -> None:
    g = ToyRacer(9)
    obstacles = g.obstacles
    run(g, ("up",), 30)
    g.reset()
    assert g.state == ToyRacer(9).state
    assert g.ram == ToyRacer(9).ram
    assert g.obstacles == obstacles


# ------------------------------------------------------------- render


def test_render_dimensions_and_player_visible() -> None:
    g = ToyRacer()
    f = g.render()
    assert isinstance(f, Frame)
    assert f.size == (SCREEN_WIDTH, SCREEN_HEIGHT)
    assert f.channels == 1
    assert f.number == 0
    assert PX_PLAYER in f.pixels
    assert f.pixels.count(PX_HUD) > 0  # lives are drawn


def test_render_tracks_lane_and_lives_and_game_over() -> None:
    g = ToyRacer(seed=5)
    before = g.render()
    g.step(("left",))
    after = g.render()
    assert before != after
    assert after.number == 1

    def player_x(frame: Frame) -> int:
        idx = frame.pixels.index(PX_PLAYER)
        return idx % SCREEN_WIDTH

    assert player_x(after) < player_x(before)

    lane = next(lane for lane in range(LANES) if first_obstacle_in_lane(g, lane) is not None)
    drive_to_lane(g, lane)
    hud_before = g.render().pixels.count(PX_HUD)
    while not g.state.game_over:
        g.step(("up",))
    hud_after = g.render().pixels.count(PX_HUD)
    assert hud_after > hud_before  # the game-over bar is big


def test_render_shows_obstacles_as_they_approach() -> None:
    g = ToyRacer(seed=5)
    seen_obstacle = False
    for _ in range(TRACK_LENGTH // MAX_SPEED + 10):
        if PX_OBSTACLE in g.render().pixels:
            seen_obstacle = True
            break
        g.step(("up",))
    assert seen_obstacle


# -------------------------------------------------------- performance


@pytest.mark.slow
def test_headless_speed_is_at_least_2000_steps_per_second() -> None:
    g = ToyRacer()
    n = 4000
    start = time.perf_counter()
    for i in range(n):
        g.step(("up", "left") if i % 2 else ("up", "right"))
    elapsed = time.perf_counter() - start
    assert n / elapsed >= 2000, f"{n / elapsed:.0f} steps/s"
