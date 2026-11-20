"""T-1063: what was found out about a pinned spot still holds at the next Start.

Whether the spot the goes start from loses a life by itself is checked once three
goes in a row lose their first life at the same early step (T-961). A Start saved
the spot again and began that from nothing. A pinned spot that is still the
checkpoint is the same spot: goes lost before the Stop count on, and a spot cleared
of a loss is not checked for it again.
"""

from __future__ import annotations

import logging

import pytest

from retro_ai_lab.training import Mode
from tests.unit.training import test_service as in_service
from tests.unit.training.test_service import Lab

lab = in_service.lab  # the practice racer behind the real services (pytest fixture)
lose_at = in_service._goes_lose_their_first_life_at
run_goes = in_service._run_goes

CLEARED = "is the AI's doing"


@pytest.mark.parametrize("pinned", [True, False])
def test_a_spot_cleared_of_a_loss_is_checked_again_only_when_it_is_another_spot(
    lab: Lab, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch, pinned: bool
) -> None:
    lab.connect()
    t = lab.training
    if pinned:
        t.pin_spot()
    caplog.set_level(logging.INFO)
    t.start(Mode.INDEPENDENT)
    harness = t.harness
    assert harness is not None and harness.taken == 1
    # Goes 1 to 3 lose a life six steps in; the check, as go 4 starts, finds the spot
    # keeps its lives holding nothing. The Stop ends go 4, which lost none.
    lose_at(monkeypatch, [6, 6, 6, None, 6, 6, 6, None])
    run_goes(lab, 3)
    assert t.step() and caplog.text.count(CLEARED) == 1
    t.stop()
    assert t.episodes == 4
    t.start(Mode.INDEPENDENT)
    assert harness.taken == (1 if pinned else 2)
    run_goes(lab, 7)  # three more goes losing it at step 6
    assert t.step()
    # The same spot was cleared of that loss; a spot saved anew has its own to check.
    assert caplog.text.count(CLEARED) == (1 if pinned else 2)
    t.stop()


@pytest.mark.parametrize("pinned", [True, False])
def test_goes_lost_before_the_stop_count_on_only_at_the_same_spot(
    lab: Lab, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch, pinned: bool
) -> None:
    lab.connect()
    t = lab.training
    if pinned:
        t.pin_spot()
    caplog.set_level(logging.INFO)
    t.start(Mode.INDEPENDENT)
    lose_at(monkeypatch, [6, 6, 6])
    run_goes(lab, 2)
    t.stop()  # between goes: two lost at step 6, and none cut short
    assert t.episodes == 2
    t.start(Mode.INDEPENDENT)
    run_goes(lab, 3)  # the third in a row from the same spot
    assert t.step()
    assert caplog.text.count(CLEARED) == (1 if pinned else 0)
    t.stop()
