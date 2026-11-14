"""T-1030: the practice game's dying pause is a setting.

On a real game the life counter drops long after the hit; the practice game's
pause was a fixed 48 frames, which hid what a long one does to going back.
"""

from __future__ import annotations

import pytest

from retro_ai_lab.core import ConfigurationError
from retro_ai_lab.emulator.fake import ToyHopper
from retro_ai_lab.emulator.fake import toy_hopper as th


def fall_in(game: ToyHopper) -> int:
    """Walk right until the dying pause starts. Returns the frames it took."""
    for frames in range(1, 601):
        game.step(["right"])
        if game.state.mode == th.DYING:
            return frames
    raise AssertionError("walking right never fell into a gap")


def test_the_pause_is_as_long_as_it_is_set() -> None:
    game = ToyHopper(0)
    assert game.dying_frames == th.DYING_FRAMES == 48
    game.dying_frames = 200
    assert game.dying_frames == 200
    fall_in(game)
    st = game.state
    assert (st.lives, st.dying) == (th.START_LIVES, 200)
    # The pause: the life count has not moved and no button is heard, 199 frames long.
    for left in range(199, 0, -1):
        game.step(["a", "left"])
        now = game.state
        assert (now.mode, now.lives, now.dying, now.x, now.y) == (
            th.DYING,
            th.START_LIVES,
            left,
            st.x,
            st.y,
        )
    game.step(["a", "left"])
    now = game.state
    assert (now.mode, now.lives, now.dying) == (th.PLAYING, th.START_LIVES - 1, 0)
    assert (now.x, now.scroll) == (th.START_X, 0)
    # The shortest: the life goes on the frame after the fall.
    quick = ToyHopper(0)
    quick.dying_frames = 1
    fall_in(quick)
    assert (quick.state.lives, quick.state.dying) == (th.START_LIVES, 1)
    quick.step([])
    assert (quick.state.mode, quick.state.lives) == (th.PLAYING, th.START_LIVES - 1)
    # The longest a byte counts.
    long = ToyHopper(0)
    long.dying_frames = 255
    fall_in(long)
    assert long.state.dying == 255
    for _ in range(254):
        long.step([])
    assert (long.state.mode, long.state.lives) == (th.DYING, th.START_LIVES)
    long.step([])
    assert (long.state.mode, long.state.lives) == (th.PLAYING, th.START_LIVES - 1)


def test_the_setting_is_the_cartridges() -> None:
    game = ToyHopper(0)
    game.dying_frames = 90
    # A reset is the console's: the cartridge is the same.
    game.reset()
    assert game.dying_frames == 90
    took = fall_in(game)
    assert game.state.dying == 90
    # Changed during a pause, the pause in hand runs out as it began.
    game.dying_frames = 20
    for _ in range(89):
        game.step([])
    assert (game.state.mode, game.state.lives) == (th.DYING, th.START_LIVES)
    game.step([])
    assert (game.state.mode, game.state.lives) == (th.PLAYING, th.START_LIVES - 1)
    # The next one has the new length, and so has every one after a game over.
    assert fall_in(game) == took and game.state.dying == 20
    for _ in range(20):
        game.step([])
    fall_in(game)
    for _ in range(20):
        game.step([])
    assert game.state.game_over
    game.step(["start"])
    assert not game.state.game_over and game.dying_frames == 20
    fall_in(game)
    assert game.state.dying == 20

    # A save state is the game's memory: it does not hold the setting.
    plain, set_long = ToyHopper(0), ToyHopper(0)
    set_long.dying_frames = 200
    set_long.restore(plain.snapshot())
    assert set_long.dying_frames == 200
    fall_in(set_long)
    assert set_long.state.dying == 200

    # The same seed, buttons and setting play the same game; another setting another.
    def played(pause: int | None) -> list[tuple[int, int, int, int]]:
        g = ToyHopper(3)
        if pause is not None:
            g.dying_frames = pause
        out = []
        for _ in range(900):
            g.step(["right"])
            out.append((g.state.x, g.state.lives, g.state.mode, g.state.dying))
        return out

    assert played(200) == played(200)
    assert played(None) == played(th.DYING_FRAMES)
    assert played(200) != played(None)


def test_a_pause_that_is_no_whole_number_from_1_to_255_is_refused() -> None:
    game = ToyHopper(0)
    game.dying_frames = 77
    for bad in (True, False, 0, -1, 256, 1.5, "200", None):
        with pytest.raises(
            ConfigurationError, match="dying pause must be a whole number of frames from 1 to 255"
        ):
            game.dying_frames = bad  # type: ignore[assignment]
        assert game.dying_frames == 77
