"""T-1050: through the training service, each game over is the next go.

Boss, 16 Nov: "when it got game over the 'goes' number didn't go up". The practice
racer is made to count its lives as his game does: those in reserve, and below
nought (255 in its byte) when the last one goes.
"""

from __future__ import annotations

import logging

import pytest

from retro_ai_lab.emulator.fake import toy_racer
from retro_ai_lab.emulator.fake.toy_racer import LAYOUT, ToyRacer
from retro_ai_lab.services import ButtonsHeld, GoStarted
from tests.unit.training import test_service as the_service
from tests.unit.training.test_service import Lab

lab = the_service.lab  # the practice racer behind the training service (pytest fixture)

RESERVE = toy_racer.START_LIVES  # lives it may lose and play on: one more ends the game
BELOW = 0xFF
WHY = (
    f": Lives at ram[{LAYOUT.lives}] went from 0 to 255: "
    "it fell below nought, so every life is lost (frame "
)


@pytest.fixture
def counted_in_reserve(monkeypatch: pytest.MonkeyPatch) -> None:
    """The racer counting the lives in reserve: with none left it plays on, and the
    next crash takes the count below nought and ends the game."""
    simulate, render = ToyRacer._simulate, ToyRacer.render

    def simulated(game: ToyRacer, held: frozenset[str]) -> None:
        ram = game._ram
        last = ram[LAYOUT.lives] == 0
        if last:
            ram[LAYOUT.lives] = 1  # (the game's own rule ends it on reaching nought)
        simulate(game, held)
        if last:
            ram[LAYOUT.lives] = BELOW if ram[LAYOUT.game_over] else 0
        elif ram[LAYOUT.lives] == 0:
            ram[LAYOUT.game_over] = 0  # none in reserve: the one being played is left

    def rendered(game: ToyRacer) -> object:
        ram = game._ram
        count = ram[LAYOUT.lives]
        if count == BELOW:
            ram[LAYOUT.lives] = 0  # nothing to draw for it
        try:
            return render(game)
        finally:
            ram[LAYOUT.lives] = count

    monkeypatch.setattr(ToyRacer, "_simulate", simulated)
    monkeypatch.setattr(ToyRacer, "render", rendered)


def test_each_game_over_is_the_next_go(
    lab: Lab, counted_in_reserve: None, caplog: pytest.LogCaptureFixture
) -> None:
    lab.connect()
    t = lab.training
    t.set_go_until_over(True)
    game = lab.adapter().game
    fresh, began = [False], []

    def first_move(event: ButtonsHeld) -> None:
        if fresh[0]:
            fresh[0] = False
            began.append((game.state.lives, game.state.game_over))

    lab.bus.subscribe(GoStarted, lambda event: fresh.__setitem__(0, True))
    lab.bus.subscribe(ButtonsHeld, first_move)
    with caplog.at_level(logging.INFO, logger="retro_ai_lab.training.service"):
        records = t.train(12)
    results = [r.result for r in records]
    assert len(results) == 12 == t.episodes  # the number of goes went up with each
    # A go ends once every life is lost, and by nothing else sooner than before: the
    # others ran until they had been paid nothing for the time (T-1044).
    over = [r for r in results if r.setbacks > RESERVE]
    assert len(over) >= 3 and {r.setbacks for r in over} == {RESERVE + 1}
    assert {r.ended_by for r in over} == {"game_over"}
    assert {r.ended_by for r in results if r not in over} == {"no_gain"}
    assert results[-1] in over and game.state.game_over and game._ram[LAYOUT.lives] == BELOW
    # The restart is the next go: each begins from the checkpoint with every life,
    # and the count is read anew (the last six all ended this way, one after another).
    assert began == [(toy_racer.START_LIVES, False)] * 12
    assert [r in over for r in results[-6:]] == [True] * 6
    # The go's line in the log says what was read.
    said = [rec.getMessage() for rec in caplog.records if rec.getMessage().startswith("Go ")]
    assert len(said) == 12
    for line, result in zip(said, results, strict=True):
        told = WHY in line
        assert told == (result in over) == (" ended by game_over after " in line)
        assert not told or line.rpartition(WHY)[2].rstrip(")").isdigit()


def test_the_practice_game_as_it_is_ends_its_goes_as_before(lab: Lab) -> None:
    # The racer's own count stops at nought and stays there: nothing is read from it.
    lab.connect()
    t = lab.training
    t.set_go_until_over(True)
    results = [r.result for r in t.train(6)]
    assert {r.ended_by for r in results} == {"no_gain"}
    assert max(r.setbacks for r in results) == toy_racer.START_LIVES
