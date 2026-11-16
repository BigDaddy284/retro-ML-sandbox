"""T-1049: a put-back the screen cannot show is told by the game's own count of its
camera: back at what it read at the place's door, with a life lost about then and
the player seen somewhere else on the screen.

The practice game is put back here by loading a state saved at the door, and the
watch is told of a life lost by the test, so that each sign can be had without the
others. On most of this level the screen does show such a put-back (a jolt, T-993);
these tests take that away, so that the count is all there is to go by. Where the
screen really cannot show it is `tests/integration/test_a_life_lost_in_the_second_level.py`.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping

import pytest

from retro_ai_lab.emulator.fake import toy_hopper
from retro_ai_lab.training import vision
from retro_ai_lab.training.vision import BACK_LOOKS, LOOK_SHIFT, ScreenWatch
from tests.unit.training import test_trial as trial
from tests.unit.training.test_trial import FRAMES, Hopper
from tests.unit.training.test_vision_camera_memory import (
    BYTE,
    LEARN,
    ON,
    STAND,
    Move,
    a_watch,
    camera,
    frame_of,
    memory_of,
    walk,
)

hopper = trial.hopper  # the practice game behind a connection (pytest fixture)

LOGGER = "retro_ai_lab.training.vision"
SAID = (
    "A life was lost and the screen showed no jump, but the game's own count of its "
    "camera is back at what it was where this place began and the AI is somewhere else "
    "on the screen: the game put it back. Ground it walks again from there is not new."
)
DOOR, FAR = "door", "far"  # the names the game's moments are saved under
# Standing, to and fro, and on over the first gap and past the first walker, ending on
# a look (one comes every other step).
AFAR = LEARN + ON[:-1]
Watches = tuple[ScreenWatch, ScreenWatch]


@pytest.fixture
def no_jolt(monkeypatch: pytest.MonkeyPatch) -> None:
    """No change of the screen is a jolt: a put-back shows as none, as on plain ground."""
    monkeypatch.setattr(vision, "JOLT", 2.0)


def look(
    hopper: Hopper,
    watches: Watches,
    move: Move = STAND,
    *,
    steps: int = 2,
    lost: int = 0,
    memory: Mapping[str, bytes] | bool | None = True,
) -> None:
    """Play ``steps`` steps of ``move`` and have both watches look: the first is handed
    the game's memory (or ``memory``, or none), the second the screen alone."""
    if steps:
        step(hopper, move, steps)
    given, plain = watches
    frame = frame_of(hopper)
    handed = memory_of(hopper.game) if memory is True else (memory or None)
    given.look(frame, lost=lost, memory=handed)
    plain.look(frame, lost=lost)


def step(hopper: Hopper, move: Move = STAND, steps: int = 1) -> None:
    buttons = {b.name: b for b in hopper.adapter.buttons()}
    hopper.adapter.step(FRAMES * steps, pressed=[buttons[name] for name in move])


def begun(hopper: Hopper, *, memory: Mapping[str, bytes] | bool | None = True) -> Watches:
    """Two watches at the first look of a go, the game's moment saved as the door."""
    given = a_watch(hopper)
    plain = ScreenWatch(given.player)
    for watch in (given, plain):
        watch.begin_go()
    hopper.adapter.save_state(DOOR)
    look(hopper, (given, plain), steps=0, memory=memory)
    return given, plain


def far_off(hopper: Hopper, watches: Watches) -> int:
    """Walk on until the count is found and the camera is far from the door, with no
    life lost; that moment is saved. Returns how far the game's camera is."""
    walk(hopper, *watches, AFAR)
    assert watches[0].camera_bytes == (BYTE, None)
    assert hopper.game.state.lives == toy_hopper.START_LIVES
    hopper.adapter.save_state(FAR)
    far = camera(hopper.game)
    assert far > 400 and watches[0].ground.place == (far, 0)
    return far


def put_back(hopper: Hopper, watches: Watches, *, lost: int = 1) -> None:
    """The game puts the player back at the door, and the watches look."""
    hopper.adapter.load_state(DOOR)
    look(hopper, watches, steps=0, lost=lost)
    assert camera(hopper.game) == 0


def afar(watch: ScreenWatch, far: int) -> bool:
    """The watch still has its camera out where it was, give or take what the picture
    read into the put-back's look (a slide of a few pixels): it was not put at the door."""
    return far > 2 * LOOK_SHIFT and abs(watch.ground.place[0] - far) <= LOOK_SHIFT


def said(caplog: pytest.LogCaptureFixture) -> list[tuple[int, str]]:
    return [(r.levelno, r.getMessage()) for r in caplog.records if "put it back" in r.message]


# ------------------------------------------------------------ the rule


def test_put_back_by_the_count_where_the_screen_shows_none(
    hopper: Hopper, no_jolt: None, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.DEBUG, logger=LOGGER)
    given, plain = watches = begun(hopper)
    far = far_off(hopper, watches)
    reach, steps = given.ground.reach, given.ground.steps
    put_back(hopper, watches)
    # With the count: at the door, and what it reached stays reached.
    assert given.ground.place == (0, 0)
    assert (given.ground.reach, given.ground.steps) == (reach, steps)
    assert said(caplog) == [(logging.INFO, SAID)]
    # Without it the screen showed nothing, and the camera carries on from afar.
    assert plain.ground.place[0] >= far


def test_ground_walked_again_pays_nothing_until_the_furthest(hopper: Hopper, no_jolt: None) -> None:
    given, plain = watches = begun(hopper)
    far = far_off(hopper, watches)
    for watch in watches:
        watch.take_new_ground()
    counted = (given.ground.steps, plain.ground.steps)
    put_back(hopper, watches)
    # The same walk again, from the door to where it had got to.
    walk(hopper, given, plain, AFAR)
    assert camera(hopper.game) == far and hopper.game.state.lives == toy_hopper.START_LIVES
    assert given.ground.place == (far, 0)
    assert given.ground.steps == counted[0] and given.take_new_ground() == 0
    # The watch without the count took every step of it for new ground.
    assert plain.ground.steps - counted[1] >= far // toy_hopper.PLAYER_H - 1
    assert plain.take_new_ground() == plain.ground.steps - counted[1]
    # Past its furthest the ground is new again.
    look(hopper, watches, ("right",), steps=4)
    assert camera(hopper.game) == far + 32 and given.ground.place == (far + 32, 0)
    assert given.ground.steps - counted[0] == given.take_new_ground() == 2


@pytest.mark.parametrize(
    ("first", "apart", "back"),
    [
        ("count", BACK_LOOKS, True),
        ("count", BACK_LOOKS + 1, False),
        ("life", BACK_LOOKS, True),
        ("life", BACK_LOOKS + 1, False),
    ],
)
def test_the_life_and_the_count_may_be_a_few_looks_apart(
    hopper: Hopper, no_jolt: None, first: str, apart: int, back: bool
) -> None:
    assert BACK_LOOKS == 4
    given, _ = watches = begun(hopper)
    far = far_off(hopper, watches)
    if first == "count":
        # Put back, and the life is counted some looks later (it stands and waits).
        put_back(hopper, watches, lost=0)
        assert afar(given, far)  # nothing yet: no life has gone
        for _ in range(apart - 1):
            look(hopper, watches)
            assert afar(given, far)
        look(hopper, watches, lost=1)
    else:
        # The life is counted, and the game puts it back some looks later.
        look(hopper, watches, lost=1)
        for _ in range(apart - 1):
            look(hopper, watches)
        assert afar(given, far)
        assert hopper.game.state.lives == toy_hopper.START_LIVES  # (nothing touched it)
        put_back(hopper, watches, lost=0)
    assert given.ground.place == (0, 0) if back else afar(given, far)


def test_no_one_sign_is_enough(hopper: Hopper, no_jolt: None) -> None:
    given, plain = watches = begun(hopper)
    walk(hopper, given, plain, LEARN + ON[:10])
    assert given.camera_bytes == (BYTE, None)
    # A go that begins here, so that the count comes round to its door's value 256
    # pixels on.
    for watch in watches:
        watch.begin_go()
    hopper.adapter.save_state("here")
    look(hopper, watches, steps=0)
    here, value = camera(hopper.game), hopper.game.ram[BYTE[1]]
    # A life lost where the count reads the door's and the camera is far from the
    # door, but the player is where it was on the screen: a hit, not a put-back.
    hit = 0
    for move in ON[10:]:
        step(hopper, move)
        there = camera(hopper.game) == here + 256
        look(hopper, watches, steps=0, lost=int(there))
        if there:
            hit += 1
            assert hopper.game.ram[BYTE[1]] == value
            assert given.ground.place == (256, 0)
    assert hit == 1 and hopper.game.state.lives == toy_hopper.START_LIVES
    far = camera(hopper.game) - here
    assert far > 256 + LOOK_SHIFT and afar(given, far)


def test_a_count_one_off_the_doors_is_not_the_doors(
    hopper: Hopper, no_jolt: None, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.DEBUG, logger=LOGGER)
    given, _ = watches = begun(hopper)
    far = far_off(hopper, watches)
    # Everything of a put-back but the count, which reads one more than at the door.
    hopper.adapter.load_state(DOOR)
    ram = bytearray(hopper.game.ram)
    assert ram[BYTE[1]] == 0
    ram[BYTE[1]] = 1
    look(hopper, watches, steps=0, lost=1, memory={"ram": bytes(ram)})
    assert afar(given, far) and said(caplog) == []
    assert given.camera_bytes == (BYTE, None)


def test_far_from_the_door_is_far_either_way(hopper: Hopper, no_jolt: None) -> None:
    given, _ = watches = begun(hopper)
    walk(hopper, *watches, LEARN)
    assert given.camera_bytes == (BYTE, None)
    # A go that begins part-way along, and walks back to the level's start: the
    # camera is left of its door, and the player nearer the screen's edge.
    for watch in watches:
        watch.begin_go()
    hopper.adapter.save_state("here")
    look(hopper, watches, steps=0)
    here = camera(hopper.game)
    assert here > 2 * LOOK_SHIFT
    while hopper.game.state.x > toy_hopper.START_X:
        look(hopper, watches, ("left",))
    assert camera(hopper.game) == 0 and given.ground.place == (-here, 0)
    hopper.adapter.load_state("here")
    look(hopper, watches, steps=0, lost=1)
    assert given.ground.place == (0, 0)


def test_with_no_life_lost_it_is_no_put_back(
    hopper: Hopper, no_jolt: None, caplog: pytest.LogCaptureFixture
) -> None:
    given, _ = watches = begun(hopper)
    far = far_off(hopper, watches)
    # At the door with the count reading the door's and the player seen elsewhere, but
    # no life lost then or after: nothing is done.
    with caplog.at_level(logging.DEBUG, logger=LOGGER):
        put_back(hopper, watches, lost=0)
        for _ in range(BACK_LOOKS + 2):
            look(hopper, watches)
        assert afar(given, far) and said(caplog) == []
        # (With the life, the same put-back is one.)
        hopper.adapter.load_state(FAR)
        look(hopper, watches, steps=0)
        put_back(hopper, watches)
        assert given.ground.place == (0, 0) and len(said(caplog)) == 1


@pytest.mark.parametrize(("near", "back"), [(LOOK_SHIFT + 7, True), (LOOK_SHIFT - 1, False)])
def test_near_the_door_nothing_is_done(
    hopper: Hopper, no_jolt: None, caplog: pytest.LogCaptureFixture, near: int, back: bool
) -> None:
    given, plain = watches = begun(hopper)
    walk(hopper, *watches, LEARN)
    assert given.camera_bytes == (BYTE, None)
    # Back towards the door, until the camera is a look's slide from it and no more.
    while camera(hopper.game) > near:
        look(hopper, watches, ("left",), steps=1)
    assert camera(hopper.game) == near and given.ground.place == (near, 0)
    with caplog.at_level(logging.DEBUG, logger=LOGGER):
        put_back(hopper, watches)
    # Within a look's slide of the door there is nothing to mend (the picture can read
    # such a put-back as a slide, and then has it right by itself): the rule is not
    # used, and the watch is where the one without the count is. A step further out,
    # it is put at the door.
    assert len(said(caplog)) == int(back)
    assert given.ground.place == ((0, 0) if back else plain.ground.place)


# ------------------------------------------------------------ the door


def test_the_door_is_the_places_own(hopper: Hopper, no_jolt: None) -> None:
    given, plain = watches = begun(hopper)
    walk(hopper, *watches, LEARN)
    assert given.camera_bytes == (BYTE, None)
    # A go that begins part-way along the level: its door is here, and the count's
    # value at it is not nought.
    for watch in watches:
        watch.begin_go()
    hopper.adapter.save_state("here")
    look(hopper, watches, steps=0)
    here = camera(hopper.game)
    assert here == hopper.game.ram[BYTE[1]] == 73
    walk(hopper, given, plain, ON[:-1])
    far = camera(hopper.game) - here
    assert far > 300 and given.ground.place == (far, 0)
    # Put back to the level's start, which is not this go's door: the count reads
    # nought there, not what it read at the door. Not taken for a put-back to it.
    put_back(hopper, watches)
    assert afar(given, far)
    # Put back to the go's own door, it is.
    hopper.adapter.load_state("here")
    look(hopper, watches, steps=0, lost=1)
    assert given.ground.place == (0, 0)


def test_a_new_places_door_is_its_own(
    hopper: Hopper, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.DEBUG, logger=LOGGER)
    given, plain = watches = begun(hopper)
    walk(hopper, *watches, LEARN)
    assert given.camera_bytes == (BYTE, None)
    # A go that begins part-way along the level, by day: at its door the count is 73.
    for watch in watches:
        watch.begin_go()
    look(hopper, watches, steps=0)
    assert hopper.game.ram[BYTE[1]] == 73
    # The screen cuts to somewhere else: a level's start, in a cave. At the first look
    # that can be measured it is a new place, and there the count reads nought.
    monkeypatch.setattr(toy_hopper, "PALETTES", (toy_hopper.PALETTES[1],))
    hopper.adapter.load_state(DOOR)
    look(hopper, watches, steps=0)
    look(hopper, watches)
    assert (given.ground.place_number, given.ground.places) == (1, 2)
    assert hopper.game.ram[BYTE[1]] == 0
    # From here on the screen shows no put-back.
    monkeypatch.setattr(vision, "JOLT", 2.0)
    walk(hopper, given, plain, AFAR)
    far = camera(hopper.game)
    assert hopper.game.state.lives == toy_hopper.START_LIVES
    assert far > 400 and given.ground.place == (far, 0) and given.ground.place_number == 1
    # (A watch lent for a try knows the places and none of their doors' memory: handed
    # the memory all the same, it looks as it would without.)
    tried = given.trial(given.mark())
    tried.look(frame_of(hopper), memory=memory_of(hopper.game))
    assert tried.ground.place == (far, 0)
    # Put back to this place's door: told by this door's value, not the first's.
    hopper.adapter.load_state(DOOR)
    look(hopper, watches, steps=0, lost=1)
    assert given.ground.place == (0, 0) and given.ground.place_number == 1
    assert said(caplog) == [(logging.INFO, SAID)]


@pytest.mark.parametrize("door", ["none", "another", "renamed"])
def test_a_door_with_no_memory_has_no_value(hopper: Hopper, no_jolt: None, door: str) -> None:
    # The go's first look was handed no memory, or a memory laid out another way than
    # every look after it (more of it, or the same bytes under another name): nothing
    # is known of the count at the door.
    ram = bytes(hopper.game.ram)
    other = {"another": {"ram": ram, "more": b"\x00"}, "renamed": {"mem": ram}}
    given, _ = watches = begun(hopper, memory=other.get(door))
    far = far_off(hopper, watches)
    assert given.camera_bytes == (BYTE, None)
    put_back(hopper, watches)
    assert afar(given, far)


def test_without_the_count_it_is_the_watch_it_was(hopper: Hopper) -> None:
    # The screen's own jolt as it is, and a put-back before the count is found.
    given, plain = watches = begun(hopper)
    seen = []
    for at in range(12):
        if at == 8:
            put_back(hopper, watches)
        else:
            look(hopper, watches, ("right",))
        assert given.camera_bytes == (None, None)
        seen.append((given.ground.place, given.ground.reach, given.ground.steps))
        assert seen[-1] == (plain.ground.place, plain.ground.reach, plain.ground.steps)
        assert given.tracker.scroll == plain.tracker.scroll
    # It had got somewhere, and was put at the door by the screen's jolt.
    assert seen[7][0][0] > LOOK_SHIFT and seen[8][0] == (0, 0)


# ------------------------------------------------------------ what is said and kept


def test_the_log_says_it_once(
    hopper: Hopper, no_jolt: None, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.DEBUG, logger=LOGGER)
    given, plain = watches = begun(hopper)
    far_off(hopper, watches)
    put_back(hopper, watches)
    walk(hopper, given, plain, AFAR)
    put_back(hopper, watches)
    assert given.ground.place == (0, 0)
    assert said(caplog) == [(logging.INFO, SAID), (logging.DEBUG, SAID)]


def test_a_put_back_told_uses_up_its_signs(
    hopper: Hopper, no_jolt: None, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.DEBUG, logger=LOGGER)
    given, _ = watches = begun(hopper)
    far_off(hopper, watches)
    put_back(hopper, watches)
    assert len(said(caplog)) == 1
    # Its life does not answer for a count gone to the door again within a few looks
    # of it: far from the door once more, and at the door with no life lost.
    hopper.adapter.load_state(FAR)
    for _ in range(BACK_LOOKS - 1):
        look(hopper, watches, ("right",), steps=4)
    out = given.ground.place[0]
    assert out > 2 * LOOK_SHIFT
    put_back(hopper, watches, lost=0)
    assert afar(given, out) and len(said(caplog)) == 1
    # Nor does that count answer for a life lost after one it has answered for.
    hopper.adapter.load_state(FAR)
    look(hopper, watches, steps=0)
    for _ in range(BACK_LOOKS):
        look(hopper, watches, ("right",))
    put_back(hopper, watches)
    assert given.ground.place == (0, 0) and len(said(caplog)) == 2
    look(hopper, watches, lost=1)
    assert len(said(caplog)) == 2


@pytest.mark.parametrize("how", ["a new go", "going back"])
def test_a_new_go_and_a_going_back_forget_it(
    hopper: Hopper, no_jolt: None, caplog: pytest.LogCaptureFixture, how: str
) -> None:
    caplog.set_level(logging.DEBUG, logger=LOGGER)
    given, plain = watches = begun(hopper)
    far = far_off(hopper, watches)
    marks = (given.mark(), plain.mark())
    put_back(hopper, watches, lost=0)  # the count went to the door; no life yet
    if how == "a new go":
        for watch in watches:
            watch.begin_go()
        look(hopper, watches, steps=0, lost=1)
        assert given.ground.place == (0, 0)  # (where a go begins)
    else:
        hopper.adapter.load_state(FAR)
        for watch, mark in zip(watches, marks, strict=True):
            watch.back_to(mark)
        look(hopper, watches, steps=0, lost=1)
        assert given.ground.place == (far, 0)
    assert said(caplog) == []
