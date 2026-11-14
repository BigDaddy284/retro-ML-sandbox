"""T-1028: the ground count keeps a second count for the story of the go.

When the game is put back to a saved moment, what the go has reached stays reached
(T-1000): ground walked a second time pays nothing. The story of the go, the moves
that lead from its start to where it is now, went back with the game, and the same
moves played straight from the start would be paid for that ground. The count now
keeps how far the story has got as well, puts that back with the game, and says at
every look how many steps the story covered that the go had counted before.
"""

from __future__ import annotations

from retro_ai_lab.observation.ground import MOST_A_LOOK, Ground, Kept

ME = (80, 200)  # where the camera holds the player
STILL = (0, 0)
STEP = (16, 0)  # a step to the right, the player's own size


def started() -> Ground:
    """A go under way: seen once, standing where the go began."""
    ground = Ground()
    ground.begin()
    assert ground.look(STILL, ME, 16) == 0
    return ground


def looks(ground: Ground, move: tuple[int, int], count: int) -> list[tuple[int, int]]:
    """(steps paid, steps the story covered again) for each of ``count`` looks."""
    return [(ground.look(move, ME, 16), ground.again) for _ in range(count)]


def test_ground_walked_again_is_told_and_not_paid() -> None:
    ground = started()
    assert ground.again == 0
    # Played straight, nothing is ever covered again.
    assert looks(ground, STEP, 3) == [(1, 0)] * 3
    kept = ground.keep()
    assert looks(ground, STEP, 3) == [(1, 0)] * 3
    assert looks(ground, (-16, 0), 2) == [(0, 0)] * 2  # back a little and on again:
    assert looks(ground, STEP, 2) == [(0, 0)] * 2  # the story has been there too
    # Put back three steps: the go has counted six, its story three.
    ground.put_back(kept)
    assert (ground.place, ground.steps, ground.again) == ((48, 0), 6, 0)
    # Three steps walked a second time: nothing paid, each one told.
    assert looks(ground, STEP, 3) == [(0, 1)] * 3
    # Past the furthest: paid as ever, and nothing covered again.
    assert looks(ground, STEP, 2) == [(1, 0)] * 2
    assert ground.steps == 8 and ground.reach == (128, 0, 0, 0)

    # Slower the second time, standing about and going back on the way: each step is
    # told where it is covered, once.
    ground.put_back(kept)
    assert looks(ground, (8, 0), 2) == [(0, 0), (0, 1)]
    assert looks(ground, STILL, 2) == [(0, 0)] * 2
    assert looks(ground, (-16, 0), 1) == [(0, 0)]
    assert looks(ground, STEP, 2) == [(0, 0), (0, 1)]
    # Quicker: two steps in one look, then one look that is half old ground, half new.
    assert looks(ground, (32, 0), 1) == [(0, 2)]
    assert looks(ground, (32, 0), 1) == [(1, 1)]
    assert ground.steps == 9
    # Another direction is new to both.
    ground.put_back(kept)
    assert looks(ground, (0, -16), 2) == [(1, 0)] * 2

    # No more than the most a look, for the story as for the go, and the rest is not
    # owed to either.
    ground = started()
    start = ground.keep()
    assert sum(paid for paid, _ in looks(ground, STEP, 6)) == 6
    ground.put_back(start)
    assert MOST_A_LOOK == 4
    assert looks(ground, (112, 0), 1) == [(1, 3)]  # seven steps for the story, one new
    assert looks(ground, STEP, 2) == [(1, 0)] * 2
    ground.put_back(start)
    assert looks(ground, (80, 0), 1) == [(0, 4)]  # five steps, all of them old
    assert looks(ground, STEP, 3) == [(0, 1)] * 3  # the fifth is not told later
    assert looks(ground, STEP, 2) == [(0, 1), (1, 0)]

    # After a cut to another place the story's reach is kept like the go's.
    ground = started()
    looks(ground, STEP, 3)
    kept = ground.keep()
    looks(ground, STEP, 3)
    ground.put_back(kept)
    ground.back()  # the place is taken to be where the go began
    assert looks(ground, STEP, 3) == [(0, 0)] * 3  # as far as the story had been
    assert looks(ground, STEP, 4) == [(0, 1)] * 3 + [(1, 0)]

    # Before there is a step to count in, nothing is counted for either; once there
    # is, the story and the go count the same.
    ground = Ground()
    ground.begin()
    assert (ground.look((32, 0)), ground.again) == (0, 0)
    assert (ground.look(STILL, ME, 16), ground.again) == (2, 0)

    # On a screen that never scrolls it is where the player stands that counts.
    ground = Ground()
    ground.begin()
    ground.look(STILL, (40, 200), 16)
    kept = ground.keep()
    assert (ground.look(STILL, (72, 200), 16), ground.again) == (2, 0)
    ground.put_back(kept)
    assert (ground.look(STILL, (56, 200), 16), ground.again) == (0, 1)
    assert (ground.look(STILL, (88, 200), 16), ground.again) == (1, 1)


def test_the_storys_count_is_kept_copied_and_cleared() -> None:
    ground = started()
    looks(ground, STEP, 3)
    kept = ground.keep()
    assert (kept.reach, kept.counted) == ((48, 0, 0, 0), (3, 0, 0, 0))
    looks(ground, STEP, 3)
    assert ground.keep().reach == ground.reach == (96, 0, 0, 0)
    # A moment kept after a going back holds the story's count of that moment, not
    # the go's.
    ground.put_back(kept)
    assert looks(ground, STEP, 1) == [(0, 1)]
    later = ground.keep()
    assert (later.reach, later.counted) == ((64, 0, 0, 0), (4, 0, 0, 0))
    assert ground.reach == (96, 0, 0, 0) and ground.steps == 6
    assert looks(ground, STEP, 1) == [(0, 1)]
    # Back to it: one step of the story's is gone again, the others are not.
    ground.put_back(later)
    assert ground.again == 0  # the last look's is not left standing
    assert looks(ground, STEP, 3) == [(0, 1), (0, 1), (1, 0)]
    # And back to the first: all of it again, to where the go has got by now.
    ground.put_back(kept)
    assert looks(ground, STEP, 5) == [(0, 1)] * 4 + [(1, 0)]

    # A copy counts on its own from the same state, and what it says stays its own.
    ground.put_back(kept)
    assert looks(ground, STEP, 1) == [(0, 1)]
    other = ground.copy()
    assert other.again == 1
    assert looks(other, STEP, 5) == [(0, 1)] * 4 + [(1, 0)]
    assert ground.again == 1 and ground.steps == 8
    assert looks(ground, STEP, 5) == [(0, 1)] * 4 + [(1, 0)]

    # A place kept the old way (no count with it) puts the story back to nothing.
    ground.put_back(Kept((0, 0), ME, (0, 0)))
    assert looks(ground, STEP, 2) == [(0, 1)] * 2

    # A new go: nothing reached and nothing told, for the story as for the go.
    ground.begin()
    assert ground.again == 0
    fresh = ground.keep()
    assert (fresh.reach, fresh.counted) == ((0, 0, 0, 0), (0, 0, 0, 0))
    assert ground.look(STILL, ME, 16) == 0
    assert looks(ground, STEP, 3) == [(1, 0)] * 3
    # A place kept in the last go is that go's. Put back all the same, the story is
    # never held to be ahead of the go: nothing is told that was not counted.
    ground.put_back(later)
    assert ground.place == (64, 0)
    assert looks(ground, STEP, 2) == [(2, 0), (1, 0)]
