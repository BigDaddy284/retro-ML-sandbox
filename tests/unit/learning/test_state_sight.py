"""T-1055: a state may show what the screen showed, and is the state it was."""

from __future__ import annotations

from retro_ai_lab.learning import PICTURE_SIZE, Sight, State
from retro_ai_lab.learning.state import fine_bucket
from retro_ai_lab.understanding import VariableKind

DARK = bytes([20]) * PICTURE_SIZE
BRIGHT = bytes([200]) * PICTURE_SIZE
PARTS = (("ram:0002:u8", 3), ("ram:0007:u8", 1))


def test_a_state_shows_nothing_unless_given_a_sight() -> None:
    assert State().sight is None and State(PARTS).sight is None
    assert State.from_key(State(PARTS).key).sight is None
    seen = State(PARTS, Sight(BRIGHT, DARK, 2))
    assert seen.sight == Sight(BRIGHT, DARK, 2)
    assert State(PARTS).seeing(Sight(BRIGHT, DARK, 2)).sight == seen.sight
    assert seen.seeing(None).sight is None and seen.seeing(None).parts == PARTS


def test_what_it_saw_is_no_part_of_which_situation_it_is() -> None:
    plain = State(PARTS)
    bright = State(PARTS, Sight(BRIGHT, DARK, 2))
    dark = State(PARTS, Sight(DARK, DARK))
    assert plain == bright == dark
    assert hash(plain) == hash(bright) == hash(dark)
    assert plain.key == bright.key == dark.key == "ram:0002:u8=3;ram:0007:u8=1"
    assert len({plain, bright, dark}) == 1 and bright in {plain}
    assert {plain: "a"}[dark] == "a"
    assert State((("ram:0002:u8", 4),), Sight(BRIGHT, DARK, 2)) != bright  # other parts
    # And it is written as it was: the pictures are not printed.
    assert repr(bright) == repr(plain) == f"State(parts={PARTS!r})"


def test_read_coarsely_again_it_still_shows_what_it_saw() -> None:
    fine = fine_bucket(200, VariableKind.POSITION)
    state = State((("ram:0002:u8", fine),), Sight(BRIGHT, DARK, 1))
    assert state.is_fine
    coarse = state.coarse()
    assert not coarse.is_fine and coarse.sight is state.sight
    assert State((("ram:0002:u8", fine),)).coarse().sight is None
