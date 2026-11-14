"""T-1028: on the practice game, with the real runner, a go that was put back to a
saved moment and walked on past where it had been is counted like the same moves
played straight from the start."""

from __future__ import annotations

from retro_ai_lab.rewards import GROUND_ID
from retro_ai_lab.training.moments import Moments
from tests.unit.training import test_going_back as going_back
from tests.unit.training.test_going_back import SLOTS, Lab
from tests.unit.training.test_remembered_way import remembering

lab = going_back.lab  # the practice game behind the real services (pytest fixture)

STEPS = 26  # of walking right: short of the first gap


def test_a_go_that_went_back_is_counted_like_one_played_straight(lab: Lab) -> None:
    moments = Moments(SLOTS, every=10, lead=12)
    r, route, _ = remembering(lab, ways_per_go=0, moments=moments)
    walk = lab.move("right")

    def begin() -> None:
        lab.agent.script.clear()
        lab.agent.script.extend([walk] * 60)
        route.forget()  # every go here is its own: none begins with another's way
        r.begin()

    def walked(count: int) -> list[float]:
        """What the story had been paid after each of ``count`` more steps."""
        out = []
        for _ in range(count):
            assert r.step()
            out.append(route.counted)
        return out

    # A first go, which is paid differently until the screen is seen to scroll, and
    # then the walk played straight: what it is paid after each step.
    begin()
    walked(STEPS)
    r.finish()
    begin()
    straight = walked(STEPS)
    r.finish()
    assert straight[-1] >= 6.0 and straight[19] > straight[9] > 0.0

    # The same walk, put back ten steps after twenty, and walked on.
    begin()
    assert walked(20) == straight[:20]
    moment = moments.kept[-1]
    assert moments.steps - moment.step == 10
    r.go_back(moment)
    assert route.counted == straight[9] and len(route.story) == 10
    again = walked(STEPS - 10)
    # Until it is past where it had been, the ground it walks pays nothing ...
    past = next(at for at in range(20, STEPS) if straight[at] > straight[19])
    assert again[: past - 10] == [straight[9]] * (past - 10)
    # ... and from the first step of new ground on, the story is paid what the walk
    # played straight is paid: the ground gone back over is in it again.
    assert again[past - 10 :] == straight[past:]
    assert past < STEPS - 1 and again[-1] == straight[-1]
    assert lab.game.state.lives == 3 and r.times_back == 1
    # It becomes the way with that figure, every point of it new ground's.
    r.finish()
    assert route.paid == straight[-1] and len(route.moves) <= STEPS
    parts = [part for step in route.paid_by for part in step]
    assert {row for row, _ in parts} == {GROUND_ID}
    assert sum(amount for _, amount in parts) == route.paid
