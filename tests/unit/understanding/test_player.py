"""T-973: the AI works out which thing on the screen is the player by pressing buttons."""

from __future__ import annotations

import numpy as np
import pytest

from retro_ai_lab.core import ConfigurationError, Frame
from retro_ai_lab.emulator.fake import ToyRacer
from retro_ai_lab.observation.objects import ObjectCatalogue
from retro_ai_lab.services import ConnectionService
from retro_ai_lab.understanding.knowledge import Confidence
from retro_ai_lab.understanding.player import PlayerFinding, find_player

SKY = (92, 148, 252)
BROWN = (200, 76, 12)
DARK = (120, 40, 0)
BLACK = (0, 0, 0)
WHITE = (252, 252, 252)
GOLD = (252, 188, 60)
GREEN = (0, 168, 0)
RED = (248, 56, 0)


def sprite(colour: tuple[int, int, int], w: int, h: int, mark: tuple[int, int, int]) -> np.ndarray:
    s = np.zeros((h, w, 3), np.uint8)
    s[:] = SKY
    s[1:, 1 : w - 1] = colour
    s[2, 2] = mark
    return s


HERO = sprite(RED, 12, 16, WHITE)
WALKER = sprite(DARK, 14, 14, WHITE)
COIN = sprite(GOLD, 6, 8, WHITE)


def world() -> np.ndarray:
    a = np.zeros((240, 1024, 3), np.uint8)
    a[:] = SKY
    for x0 in range(0, 1024, 97):
        a[150:208, x0 : x0 + 30] = GREEN
        a[150, x0 : x0 + 30] = BLACK
    for x0 in range(0, 1024, 16):
        a[208:240, x0 : x0 + 16] = BROWN
        a[208, x0 : x0 + 16] = BLACK
        a[208:240, x0] = BLACK
    return a


class Platformer:
    """A tiny made-up platformer: the hero walks and jumps, the camera follows him."""

    def __init__(self, hero_y: int = 192) -> None:
        self.world = world()
        self.hero = [300, hero_y]
        self.air = 0  # frames left going up
        self.walker = 420
        self.frame = 0

    @property
    def camera(self) -> int:
        return max(0, self.hero[0] - 120)

    def step(self, held: frozenset[str]) -> None:
        if "left" in held:
            self.hero[0] -= 2
        if "right" in held:
            self.hero[0] += 2
        if "a" in held and self.air == 0 and self.hero[1] == 192:
            self.air = 8
        if self.air:
            self.hero[1] -= 3
            self.air -= 1
        elif self.hero[1] < 192:
            self.hero[1] = min(192, self.hero[1] + 3)
        self.walker -= 1  # walks left by itself whatever is pressed
        self.frame += 1

    def render(self) -> Frame:
        cam = self.camera
        a = self.world[:, cam : cam + 256].copy()
        for lx, y, spr in (
            (self.hero[0], self.hero[1], HERO),
            (self.walker, 194, WALKER),
            (360, 120, COIN),
        ):
            x = lx - cam
            h, w = spr.shape[:2]
            solid = (spr != np.array(SKY, np.uint8)).any(axis=2)
            for yy in range(h):
                for xx in range(w):
                    if solid[yy, xx] and 0 <= x + xx < 256:
                        a[y + yy, x + xx] = spr[yy, xx]
        return Frame(256, 240, 3, a.tobytes(), number=self.frame)


def platformer_run(held: frozenset[str], frames: int, hero_y: int = 192) -> list[Frame]:
    game = Platformer(hero_y)  # the same start every run, like a save state
    out = [game.render()]
    for _ in range(frames - 1):
        game.step(held)
        out.append(game.render())
    return out


def test_the_hero_is_found_in_a_scrolling_platformer() -> None:
    found = find_player(platformer_run, ["left", "right", "a", "b"])
    assert isinstance(found, PlayerFinding)
    start = Platformer()
    # Standing on the ground, the hero is cut out by his motion: somewhere in his box.
    x0, y0 = start.hero[0] - start.camera, start.hero[1]
    assert x0 <= found.start[0] <= x0 + 12 and y0 <= found.start[1] <= y0 + 16
    assert found.buttons == {"left", "right", "a"}  # b does nothing
    assert found.tried == 4
    assert found.confidence is Confidence.HIGH
    assert "it is you" in found.describe()


def test_the_walker_and_the_coin_are_not_the_player() -> None:
    """The walker moves by itself the same way whatever is held; the coin never moves."""
    found = find_player(platformer_run, ["left", "right"])
    assert found is not None
    assert found.start[1] > 180  # the hero, on the ground, not the coin up at y=120
    assert found.confidence is Confidence.MEDIUM


def test_one_button_is_a_weak_answer() -> None:
    found = find_player(platformer_run, ["a", "b"])
    assert found is not None and found.buttons == {"a"}
    assert found.confidence is Confidence.LOW


def test_nothing_answers_when_the_game_ignores_the_buttons() -> None:
    def deaf(held: frozenset[str], frames: int) -> list[Frame]:
        return platformer_run(frozenset(), frames)

    assert find_player(deaf, ["left", "right"]) is None


def test_the_kind_is_named_when_the_player_was_seen_whole() -> None:
    catalogue = ObjectCatalogue()

    def in_the_air(held: frozenset[str], frames: int) -> list[Frame]:
        return platformer_run(held, frames, hero_y=120)  # clear of the ground: seen whole

    found = find_player(in_the_air, ["left", "right"], catalogue=catalogue)
    assert found is not None and found.kind is not None
    assert found.kind in {k.id for k in catalogue.kinds()}


def test_the_car_is_the_player_on_the_toy_racer() -> None:
    """Speeding up moves the whole track (one button); the car answers to left and right."""
    start = ToyRacer(seed=3)
    for _ in range(8):  # off the start line
        start.step(frozenset({"up"}))
    saved = start.snapshot()

    def run(held: frozenset[str], frames: int) -> list[Frame]:
        game = ToyRacer(seed=3)
        game.restore(saved)
        out = [game.render()]
        for _ in range(frames - 1):
            game.step(held)
            out.append(game.render())
        return out

    catalogue = ObjectCatalogue(min_pixels=2)
    found = find_player(run, ["left", "right", "up", "down"], catalogue=catalogue)
    assert found is not None
    assert found.start[1] >= 48 - 5  # the car sits at the bottom of the screen
    assert {"left", "right"} <= found.buttons


@pytest.mark.parametrize("buttons", [[], ()])
def test_needs_buttons(buttons: list[str]) -> None:
    with pytest.raises(ConfigurationError):
        find_player(platformer_run, buttons)


def test_needs_two_frames_or_more() -> None:
    with pytest.raises(ConfigurationError):
        find_player(platformer_run, ["left"], frames=1)


def test_a_run_that_gives_nothing_back_is_an_error() -> None:
    with pytest.raises(ConfigurationError):
        find_player(lambda held, n: [], ["left"])


# ------------------------------------------------------------ on the emulator (T-974)


def connected(**kw: object) -> ConnectionService:
    from retro_ai_lab.core.events import EventBus
    from retro_ai_lab.emulator.registry import default_registry

    bus = EventBus(on_error=lambda exc, event, handler: (_ for _ in ()).throw(exc))
    connection = ConnectionService(default_registry(), bus)
    connection.connect("fake", seed=3, **kw)  # type: ignore[arg-type]
    return connection


def off_the_start_line(connection: ConnectionService) -> None:
    from retro_ai_lab.emulator.fake import FakeEmulatorAdapter

    a = connection.adapter
    assert isinstance(a, FakeEmulatorAdapter)
    for _ in range(8):
        a.game.step(["up"])


def test_the_car_is_found_on_the_emulator_and_the_game_is_put_back() -> None:
    from retro_ai_lab.emulator.fake import FakeEmulatorAdapter
    from retro_ai_lab.understanding.player import find_player_in_game

    connection = connected()
    off_the_start_line(connection)
    a = connection.adapter
    assert isinstance(a, FakeEmulatorAdapter)
    before = a.game.snapshot()
    found = find_player_in_game(
        connection, ["left", "right", "up"], catalogue=ObjectCatalogue(min_pixels=2)
    )
    assert found is not None and {"left", "right"} <= found.buttons
    assert found.start[1] >= 48 - 5
    assert a.game.snapshot() == before  # where it was, pad let go


def test_no_screenshots_means_no_answer() -> None:
    from retro_ai_lab.core.types import Capabilities
    from retro_ai_lab.understanding.player import find_player_in_game

    caps = Capabilities(
        memory_read=True, input=True, frame_advance=True, save_states=True, speed_control=True
    )
    connection = connected(capabilities=caps)
    assert find_player_in_game(connection, ["left", "right"]) is None


def test_the_emulator_run_refuses_unknown_buttons() -> None:
    from retro_ai_lab.understanding.player import emulator_run

    connection = connected()
    connection.call(lambda a: a.save_state("x"))
    run = emulator_run(connection, "x")
    with pytest.raises(ConfigurationError):
        run(frozenset({"jump"}), 4)
    with pytest.raises(ConfigurationError):
        emulator_run(connection, "x", every=0)
    assert len(run(frozenset({"left"}), 4)) == 4


def test_a_finding_round_trips_and_broken_ones_are_refused() -> None:
    f = PlayerFinding("thing-2", (30, 45), frozenset({"left", "right"}), 4, Confidence.MEDIUM)
    assert PlayerFinding.from_dict(f.to_dict()) == f
    for broken in (
        {},
        {**f.to_dict(), "confidence": "SURE"},
        {**f.to_dict(), "start": [1]},
        {**f.to_dict(), "kind": 3},
        {**f.to_dict(), "buttons": "left"},
        {**f.to_dict(), "tried": "4"},
    ):
        with pytest.raises(ConfigurationError):
            PlayerFinding.from_dict(broken)


def test_the_knowledge_keeps_the_player() -> None:
    from retro_ai_lab.understanding.knowledge import Knowledge

    k = Knowledge("game")
    assert Knowledge.from_dict(k.to_dict()).player is None
    k.player = PlayerFinding(None, (30, 45), frozenset({"left"}), 2, Confidence.LOW)
    assert Knowledge.from_dict(k.to_dict()).player == k.player


def test_a_score_that_stays_put_on_the_screen_is_not_the_player() -> None:
    """T-977: a HUD stays on the screen while the camera follows the hero, so in the
    level it seems to move with the buttons; it never moves on the screen itself."""
    hud = sprite(WHITE, 24, 8, BLACK)

    def with_hud(held: frozenset[str], frames: int) -> list[Frame]:
        out = []
        for f in platformer_run(held, frames):
            a = np.frombuffer(f.pixels, np.uint8).reshape(240, 256, 3).copy()
            solid = (hud != np.array(SKY, np.uint8)).any(axis=2)
            a[10:18, 10:34][solid] = hud[solid]
            out.append(Frame(256, 240, 3, a.tobytes(), number=f.number))
        return out

    found = find_player(with_hud, ["left", "right", "a"])
    assert found is not None
    assert found.start[1] > 100  # the hero, not the score at the top


# ------------------------------------------------------------------ T-987: its colours


def pack(colour: tuple[int, int, int]) -> int:
    return (colour[0] << 16) | (colour[1] << 8) | colour[2]


def test_the_finding_carries_the_players_colours() -> None:
    """T-987: the screens on which the player was seen moving say what it looks like."""
    found = find_player(platformer_run, ["left", "right", "a", "b"])
    assert found is not None and found.colours is not None
    # The hero's red (10 by 15, less its one white mark); the mark is a single pixel
    # and white is the walker's and the coin's mark too.
    assert found.colours.colours == ((pack(RED), 149),)
    assert (found.colours.width, found.colours.height) == (10, 15)
    # With those colours it is picked out of any screen of the game, standing still.
    game = Platformer()
    for _ in range(20):  # (at 40 steps the walker has walked over him)
        game.step(frozenset({"right"}))
    box = found.colours.find(game.render())
    assert box == (game.hero[0] - game.camera + 1, game.hero[1] + 1, 10, 15)


def test_the_finding_carries_the_colours_on_toy_hopper() -> None:
    from retro_ai_lab.emulator.fake import ToyHopper, toy_hopper

    def run(held: frozenset[str], frames: int) -> list[Frame]:
        game = ToyHopper(0)  # the same start every run
        out = [game.render()]
        for _ in range(frames - 1):
            game.step(held)
            game.step(held)
            out.append(game.render())
        return out

    found = find_player(run, ["left", "right", "a", "b"])
    assert found is not None and found.colours is not None
    # The slivers of its edges that walking changes grow to the whole 12 by 16 player.
    assert {c for c, _ in found.colours.colours} == {
        pack(toy_hopper.PLAYER),
        pack(toy_hopper.PLAYER_HEAD),
    }
    assert (found.colours.width, found.colours.height) == (
        toy_hopper.PLAYER_W,
        toy_hopper.PLAYER_H,
    )
    assert found.colours.pixels == toy_hopper.PLAYER_W * toy_hopper.PLAYER_H


def test_a_finding_saved_before_colours_loads_without() -> None:
    f = PlayerFinding("thing-2", (30, 45), frozenset({"left", "right"}), 4, Confidence.MEDIUM)
    assert f.colours is None and f.to_dict()["colours"] is None
    old = {k: v for k, v in f.to_dict().items() if k != "colours"}
    assert PlayerFinding.from_dict(old) == f


def test_a_finding_with_colours_is_saved_and_read_back() -> None:
    from retro_ai_lab.observation.signature import ColourSignature
    from retro_ai_lab.understanding.knowledge import Knowledge

    sign = ColourSignature(((pack(RED), 149), (pack(WHITE), 3)), 10, 15)
    f = PlayerFinding(None, (30, 45), frozenset({"left", "right"}), 4, Confidence.MEDIUM, sign)
    assert f.to_dict()["colours"] == sign.to_dict()
    assert PlayerFinding.from_dict(f.to_dict()) == f
    k = Knowledge("game")
    k.player = f
    assert Knowledge.from_dict(k.to_dict()).player == f
    # It differs from the same finding without colours (so the watch is renewed).
    assert f != PlayerFinding(None, (30, 45), frozenset({"left", "right"}), 4, Confidence.MEDIUM)
    for broken in (
        {**f.to_dict(), "colours": "red"},
        {**f.to_dict(), "colours": {"colours": [], "width": 1, "height": 1, "channels": 3}},
        {**f.to_dict(), "colours": {"width": 1}},
    ):
        with pytest.raises(ConfigurationError):
            PlayerFinding.from_dict(broken)
