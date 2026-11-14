"""T-1021: values seen to come home in mid-play cannot make a restart by themselves."""

from __future__ import annotations

import pytest

from retro_ai_lab.core import ConfigurationError
from retro_ai_lab.understanding import (
    Encoding,
    EventDetector,
    EventKind,
    Evidence,
    GameEvent,
    VariableKind,
)
from retro_ai_lab.understanding.events import (
    CYCLES,
    HOME_OFTEN,
    HOMES_KEPT,
    RESET_NEEDS,
    RESET_SHARE,
)
from tests.unit.understanding.test_events import RAM, SURE, knowledge_for, snap

SCORE, FLAG, PROGRESS, LIVES = (
    VariableKind.SCORE,
    VariableKind.FLAG,
    VariableKind.PROGRESS,
    VariableKind.LIVES,
)
# A fall speed taken for a score and "on the ground" taken for a switch, as the AI's
# own guessing has them on the practice game, and four values that move and stay away.
SIX = {0: SCORE, 1: FLAG, 2: PROGRESS, 3: PROGRESS, 4: PROGRESS, 5: LIVES}
AWAY = (1, 2, 3, 2)  # where the four others are once they have moved
UP = (5, 0, *AWAY)  # in the air: falling at 5, off the ground
DOWN = (0, 1, *AWAY)  # landed: both back where they began


def restarts(events: tuple[GameEvent, ...]) -> list[GameEvent]:
    return [e for e in events if e.kind is EventKind.RESET]


def started(layout: dict[int, VariableKind] = SIX) -> EventDetector:
    """A detector that has seen the game begin and the four others move away."""
    d = EventDetector(knowledge_for(layout))
    d.observe(snap(0, 0, 1, 0, 0, 0, 3))
    assert d.observe(snap(1, *DOWN)) != ()
    return d


def test_a_value_that_goes_home_in_play_is_a_home_goer() -> None:
    assert (HOME_OFTEN, HOMES_KEPT, RESET_NEEDS, RESET_SHARE) == (3, 64, 2, 0.5)
    d = started()
    assert d.moved == 4 and d.take_home_goers() == ()
    frame = 2
    for n in range(1, HOME_OFTEN + 1):
        assert not restarts(d.observe(snap(frame, *UP)))
        # Landed: the two are home together and the number dropped, but two of the
        # six that have moved is not the game starting over.
        assert not restarts(d.observe(snap(frame + 1, *DOWN)))
        frame += 2
        assert (d.homes("v1"), d.homes("v2")) == (n, n)
        # Given out at the third time, and not before.
        assert d.take_home_goers() == (("v1", "v2") if n == HOME_OFTEN else ())
    assert d.moved == 6
    # Once only, however often they go home after that.
    d.observe(snap(frame, *UP))
    d.observe(snap(frame + 1, *DOWN))
    assert d.homes("v1") == HOME_OFTEN + 1 and d.take_home_goers() == ()
    # The four that stayed away have never come home.
    assert [d.homes(f"v{i}") for i in range(3, 7)] == [0, 0, 0, 0]
    assert d.homes("v999") == 0


def test_what_is_not_counted() -> None:
    # A return at which a restart is called: the two values are all that ever moved,
    # so their going home together is, by the rule, the game starting over.
    d = EventDetector(knowledge_for({0: SCORE, 1: FLAG}))
    d.observe(snap(0, 0, 1))
    for frame in range(1, 11, 2):
        d.observe(snap(frame, 5, 0))
        called = restarts(d.observe(snap(frame + 1, 0, 1)))
        assert len(called) == 1 and called[0].variables == ("v1", "v2")
    assert (d.homes("v1"), d.homes("v2")) == (0, 0) and d.take_home_goers() == ()

    # A return in a snapshot the app made itself (a saved state loaded): the events
    # are what they would have been, and nothing is counted.
    told, plain = started(), started()
    for frame in range(2, 12, 2):
        assert told.observe(snap(frame, *UP)) == plain.observe(snap(frame, *UP))
        home = snap(frame + 1, *DOWN)
        assert told.observe(home, put_back=True) == plain.observe(home)
    assert (told.homes("v1"), told.homes("v2")) == (0, 0) and told.take_home_goers() == ()
    assert (plain.homes("v1"), plain.homes("v2")) == (5, 5)
    assert plain.take_home_goers() == ("v1", "v2")
    # ... and a restart the app made is still called a restart there.
    d = EventDetector(knowledge_for({0: SCORE, 1: FLAG}))
    d.observe(snap(0, 0, 1))
    d.observe(snap(1, 5, 0))
    assert len(restarts(d.observe(snap(2, 0, 1), put_back=True))) == 1

    # A value already marked as cycling is not counted again; the other still is.
    k = knowledge_for(SIX)
    k.support("v1", Evidence(CYCLES, "cycles", 0.0))
    d = EventDetector(k)
    d.observe(snap(0, 0, 1, 0, 0, 0, 3))
    d.observe(snap(1, *DOWN))
    for frame in range(2, 10, 2):
        d.observe(snap(frame, *UP))
        d.observe(snap(frame + 1, *DOWN))
    assert (d.homes("v1"), d.homes("v2")) == (0, 4) and d.take_home_goers() == ("v2",)
    assert d.moved == 6  # marked as cycling it still counts as moved, as before

    # A value that moves and comes back to anything but its first value is not home.
    d = started()
    for frame in range(2, 10, 2):
        d.observe(snap(frame, 5, 0, *AWAY))
        d.observe(snap(frame + 1, 2, 1, *AWAY))  # the number down to 2, not to its 0
    assert (d.homes("v1"), d.homes("v2")) == (0, 4)


def test_only_a_snapshot_in_which_little_else_changed_is_counted() -> None:
    # Two of the six home, as at a landing, while two of the others change as well:
    # four of six that have moved is most of them. That may be the game starting
    # over, whatever the values came to, and nothing in it is counted.
    d = started()
    d.observe(snap(2, *UP))
    for frame in range(3, 13, 2):
        assert not restarts(d.observe(snap(frame, 0, 1, 7, 7, 3, 2)))  # home, and two more
        d.observe(snap(frame + 1, 5, 0, 1, 2, 3, 2))
    assert d.moved == 6
    assert (d.homes("v1"), d.homes("v2")) == (0, 0) and d.take_home_goers() == ()
    # Exactly half of them changing is still not counted ...
    for frame in range(13, 23, 2):
        d.observe(snap(frame, 0, 1, 7, 2, 3, 2))  # home, and one more: three of six
        d.observe(snap(frame + 1, 5, 0, 1, 2, 3, 2))
    assert (d.homes("v1"), d.homes("v2")) == (0, 0)
    # ... and one fewer is: the two alone, of six.
    for frame in range(23, 23 + 2 * HOME_OFTEN, 2):
        d.observe(snap(frame, *DOWN))
        d.observe(snap(frame + 1, *UP))
    assert (d.homes("v1"), d.homes("v2")) == (HOME_OFTEN, HOME_OFTEN)
    assert d.take_home_goers() == ("v1", "v2")

    # The share is of what has moved, not of what is watched: one value changing
    # is half of the two that have moved (not counted), and a third of three.
    d = EventDetector(knowledge_for({0: FLAG, 1: PROGRESS, 2: PROGRESS, 3: PROGRESS}))
    d.observe(snap(0, 0, 0, 0, 0))
    d.observe(snap(1, 0, 4, 0, 0))
    for frame in range(2, 10, 2):
        d.observe(snap(frame, 1, 4, 0, 0))
        d.observe(snap(frame + 1, 0, 4, 0, 0))
    assert (d.moved, d.homes("v1")) == (2, 0)
    d.observe(snap(10, 0, 4, 9, 0))  # a third value moves
    d.observe(snap(11, 1, 4, 9, 0))
    d.observe(snap(12, 0, 4, 9, 0))
    assert (d.moved, d.homes("v1")) == (3, 1)


def test_it_is_counted_by_the_value_come_to() -> None:
    # A switch has two resting places. Which of them its watch calls its first value
    # changes when the AI learns something about the switch and the watch is made
    # anew; what was seen of the other resting place is there already.
    k = knowledge_for(SIX)
    d = EventDetector(k)
    d.observe(snap(0, 0, 1, 0, 0, 0, 3))
    d.observe(snap(1, *DOWN))
    for frame in range(2, 2 + 2 * HOME_OFTEN, 2):
        d.observe(snap(frame, *UP))
        d.observe(snap(frame + 1, *DOWN))
    assert d.take_home_goers() == ("v1", "v2")
    d.observe(snap(20, *UP))
    k.support("v2", Evidence("test", "seen again", 1.0))
    d.observe(snap(21, *UP))  # the new watch's first sight of it: off the ground
    assert d.homes("v2") == HOME_OFTEN + 1  # the times it has come to "off"

    # A life count watched from 1 comes to 1 in play, and to 3 only when the game
    # starts over. At 1 it is a home-goer; watched from 3 it is not.
    layout = {0: LIVES, 1: SCORE, 2: PROGRESS, 3: PROGRESS, 4: PROGRESS}
    k = knowledge_for(layout)
    d = EventDetector(k)
    d.observe(snap(0, 1, 0, 0, 0, 0))
    d.observe(snap(1, 1, 0, 1, 2, 3))
    for frame in range(2, 2 + 3 * HOME_OFTEN, 3):
        d.observe(snap(frame, 0, 0, 1, 2, 3))  # the last life goes
        d.observe(snap(frame + 1, 3, 9, 4, 4, 4))  # everything changes: a new game
        d.observe(snap(frame + 2, 2, 9, 4, 4, 4))  # a life lost ...
        d.observe(snap(frame + 2, 1, 9, 4, 4, 4))  # ... and another: at 1 again
        d.observe(snap(frame + 2, 1, 0, 1, 2, 3))
    assert d.homes("v1") == HOME_OFTEN and d.take_home_goers() == ("v1",)
    d.observe(snap(30, 0, 0, 1, 2, 3))
    d.observe(snap(31, 3, 9, 4, 4, 4))
    k.support("v1", Evidence("test", "seen again", 1.0))
    d.observe(snap(32, 3, 9, 4, 4, 4))  # watched from 3 now
    assert d.homes("v1") == 0


def test_a_home_goer_found_at_a_restart_that_is_not_called_is_given_out_there() -> None:
    # The number comes to 5 and to 7 in play, never to the 0 its watch began at, so
    # it is no home-goer and nothing is said. The switch is one, and is said.
    k = knowledge_for(SIX)
    d = EventDetector(k)
    d.observe(snap(0, 0, 1, 0, 0, 0, 3))
    d.observe(snap(1, *DOWN))
    for frame in range(2, 2 + 2 * HOME_OFTEN, 2):
        d.observe(snap(frame, 5, 0, *AWAY))
        d.observe(snap(frame + 1, 7, 1, *AWAY))
    assert (d.homes("v1"), d.homes("v2")) == (0, HOME_OFTEN)
    assert d.take_home_goers() == ("v2",)
    # The AI learns something about the number while it is at 7: watched from 7 it
    # is a home-goer at once, and nobody has been told.
    k.support("v1", Evidence("test", "seen again", 1.0))
    d.observe(snap(10, 7, 1, *AWAY))
    assert d.homes("v1") == HOME_OFTEN and d.take_home_goers() == ()
    # A new go in which the two are all that has moved: both home, the number down.
    # By the old rule a restart; now it is not, and the number is given out here.
    d.reset()
    k.support("v1", Evidence("test", "and again", 1.0))
    d.observe(snap(20, 7, 1, *AWAY))
    d.observe(snap(21, 9, 0, *AWAY))
    assert d.moved == 2
    assert not restarts(d.observe(snap(22, 7, 1, *AWAY)))
    assert d.take_home_goers() == ("v1",)
    # Once: the next time nothing is given out.
    d.observe(snap(23, 9, 0, *AWAY))
    assert not restarts(d.observe(snap(24, 7, 1, *AWAY)))
    assert d.take_home_goers() == ()


def test_a_roaming_number_has_only_its_first_values_counted() -> None:
    layout = {0: SCORE, 1: PROGRESS, 2: PROGRESS, 3: PROGRESS, 4: PROGRESS}
    k = knowledge_for(layout)
    d = EventDetector(k)
    d.observe(snap(0, 0, 0, 0, 0, 0))
    d.observe(snap(1, 0, 1, 2, 3, 4))

    def watched_from(value: int) -> int:
        """How often the number is known to have come to ``value`` in play: its
        watch is made anew there, as when the AI learns something about it."""
        d.observe(snap(50, value, 1, 2, 3, 4))
        k.support("v1", Evidence("test", f"at {value}", 0.0))
        d.observe(snap(51, value, 1, 2, 3, 4))
        return d.homes("v1")

    # It comes to as many different values as are kept, three times each ...
    for _ in range(HOME_OFTEN):
        for value in range(1, HOMES_KEPT + 1):
            d.observe(snap(2, value, 1, 2, 3, 4))
    # ... and then to two more, as often.
    for _ in range(HOME_OFTEN):
        d.observe(snap(3, 200, 1, 2, 3, 4))
        d.observe(snap(4, 201, 1, 2, 3, 4))
    # (Going there to look is one more coming to it.)
    assert watched_from(1) == HOME_OFTEN + 1
    assert watched_from(HOMES_KEPT) == HOME_OFTEN + 1
    assert watched_from(200) == 0 and watched_from(201) == 0
    # The kept ones are still counted.
    d.observe(snap(5, 9, 1, 2, 3, 4))
    d.observe(snap(6, 1, 1, 2, 3, 4))
    assert watched_from(1) == HOME_OFTEN + 2


def test_the_counts_outlive_a_reset_and_a_changed_record() -> None:
    k = knowledge_for(SIX)
    d = EventDetector(k)

    def land(frame: int) -> None:
        d.observe(snap(frame, *UP))
        assert not restarts(d.observe(snap(frame + 1, *DOWN)))

    d.observe(snap(0, 0, 1, 0, 0, 0, 3))
    d.observe(snap(1, *DOWN))
    land(2)
    assert d.homes("v1") == 1
    # A new go: the watches start again, what was seen of the values is kept.
    d.reset()
    assert d.moved == 0
    d.observe(snap(10, 0, 1, 0, 0, 0, 3))
    assert d.homes("v1") == 1
    d.observe(snap(11, *DOWN))
    land(12)
    assert (d.homes("v1"), d.homes("v2")) == (2, 2) and d.take_home_goers() == ()
    # The AI learns more about one of them (its record changes, its watch is made
    # anew): the count goes on from where it was.
    k.support("v1", Evidence("test", "seen again", 1.0))
    d.observe(snap(14, *DOWN))  # the new watch's first sight of it
    land(15)
    assert (d.homes("v1"), d.homes("v2")) == (3, 3)
    assert d.take_home_goers() == ("v1", "v2")
    # What was seen is kept by where the value is held, not by its id: a guess is
    # removed and the next guess, about another place, takes over its id. It has
    # none of what was seen of the old one ...
    k.remove("v2")
    other = k.propose(RAM, 6, kind=FLAG, name="Flag", evidence=SURE)
    assert other.id == "v2"
    d.observe(snap(17, *DOWN, 0))
    assert d.homes("v2") == 0 and d.homes("v1") == 3
    # ... and is found to be a home-goer by itself, and given out, though the id was
    # given out before.
    for frame in range(18, 18 + 2 * HOME_OFTEN, 2):
        d.observe(snap(frame, *DOWN, 1))
        d.observe(snap(frame + 1, *DOWN, 0))
    assert d.homes("v2") == HOME_OFTEN and d.take_home_goers() == ("v2",)
    # With nothing held at the old place any more, what was seen there is dropped:
    # a guess made about it later starts from nothing, and can be given out again.
    again = k.propose(RAM, 1, kind=FLAG, name="Flag", evidence=SURE)
    d.observe(snap(30, *DOWN, 0))
    assert d.homes(again.id) == 0
    for frame in range(31, 31 + 2 * HOME_OFTEN, 2):
        d.observe(snap(frame, 0, 0, *AWAY, 0))
        d.observe(snap(frame + 1, 0, 1, *AWAY, 0))
    assert d.homes(again.id) == HOME_OFTEN and d.take_home_goers() == (again.id,)


def test_a_value_read_another_way_is_another_value() -> None:
    # A number in one byte that comes to 0 in play. A person then says it is two
    # bytes long: what the one byte was seen to do says nothing about that number,
    # though it too reads 0 here (the second byte is empty).
    layout = {0: SCORE, 2: PROGRESS, 3: PROGRESS, 4: PROGRESS, 5: PROGRESS}
    k = knowledge_for(layout)
    d = EventDetector(k)
    d.observe(snap(0, 0, 0, 0, 0, 0, 0))
    d.observe(snap(1, 0, 0, 1, 2, 3, 4))
    for frame in range(2, 2 + 2 * HOME_OFTEN, 2):
        d.observe(snap(frame, 5, 0, 1, 2, 3, 4))
        d.observe(snap(frame + 1, 0, 0, 1, 2, 3, 4))
    assert d.homes("v1") == HOME_OFTEN and d.take_home_goers() == ("v1",)
    k.correct("v1", encoding=Encoding.U16LE)
    d.observe(snap(10, 0, 0, 1, 2, 3, 4))
    assert d.homes("v1") == 0
    # Read as two bytes it is seen to come home in its turn, and is given out again.
    for frame in range(11, 11 + 2 * HOME_OFTEN, 2):
        d.observe(snap(frame, 5, 0, 1, 2, 3, 4))
        d.observe(snap(frame + 1, 0, 0, 1, 2, 3, 4))
    assert d.homes("v1") == HOME_OFTEN and d.take_home_goers() == ("v1",)
    # Put back to one byte, what was seen of the one byte is still there.
    k.correct("v1", encoding=Encoding.U8)
    d.observe(snap(20, 0, 0, 1, 2, 3, 4))
    assert d.homes("v1") == HOME_OFTEN


def test_home_goers_alone_are_no_restart() -> None:
    # Five values. The first landings come while the three others are away from
    # home: two of five is no restart, and the two are seen to come home in play.
    layout = {0: SCORE, 1: FLAG, 2: PROGRESS, 3: PROGRESS, 4: LIVES}
    k = knowledge_for(layout)

    def seen(landings: int) -> EventDetector:
        """A new session that has seen so many landings in play, and then a new go
        in which only the two have moved."""
        d = EventDetector(k)
        d.observe(snap(0, 0, 1, 0, 0, 3))
        d.observe(snap(1, 0, 1, 1, 2, 2))
        frame = 2
        for _ in range(landings):
            d.observe(snap(frame, 5, 0, 1, 2, 2))
            assert not restarts(d.observe(snap(frame + 1, 0, 1, 1, 2, 2)))
            frame += 2
        d.reset()
        d.observe(snap(100, 0, 1, 1, 2, 2))
        d.observe(snap(101, 5, 0, 1, 2, 2))  # a jump: the two are all that moved
        assert d.moved == 2
        return d

    # Seen twice: not yet home-goers, and by the rule their landing is a restart.
    called = restarts(seen(HOME_OFTEN - 1).observe(snap(102, 0, 1, 1, 2, 2)))
    assert len(called) == 1 and called[0].variables == ("v1", "v2")
    # Seen three times: it is what they do, however often they do it.
    d = seen(HOME_OFTEN)
    assert d.take_home_goers() == ("v1", "v2")
    for frame in range(102, 122, 2):
        assert not restarts(d.observe(snap(frame, 0, 1, 1, 2, 2)))
        d.observe(snap(frame + 1, 5, 0, 1, 2, 2))
    # All that has moved changed each time: such a return is not counted.
    assert d.homes("v1") == HOME_OFTEN

    # With a value among them that does not come home in play, it is a restart as
    # before, and the home-goers are named with it.
    d = seen(HOME_OFTEN)
    d.observe(snap(102, 5, 0, 2, 2, 2))  # the third value moves on
    called = restarts(d.observe(snap(103, 0, 1, 1, 2, 2)))  # ... and is home with them
    assert len(called) == 1 and called[0].variables == ("v1", "v2", "v3")
    # The share is of everything that has moved, home-goers and all, as before: one
    # that is no home-goer and one that is, of five moved, is no restart...
    d = EventDetector(k)
    d.observe(snap(0, 0, 1, 0, 0, 3))
    d.observe(snap(1, 0, 1, 1, 2, 2))
    for frame in range(2, 8, 2):
        d.observe(snap(frame, 5, 0, 1, 2, 2))
        d.observe(snap(frame + 1, 0, 1, 1, 2, 2))
    assert d.take_home_goers() == ("v1", "v2") and d.moved == 5
    d.observe(snap(8, 5, 0, 1, 2, 2))
    assert not restarts(d.observe(snap(9, 0, 0, 0, 2, 2)))  # v1 and v3 home: two of five
    # ... and three of five, one of them no home-goer, is.
    d.observe(snap(10, 5, 0, 1, 2, 2))
    called = restarts(d.observe(snap(11, 0, 1, 0, 2, 2)))
    assert len(called) == 1 and called[0].variables == ("v1", "v2", "v3")
    # A restart that is called is not counted against the value it sent home.
    assert d.homes("v3") == 1  # the once it came home with no restart called


def test_what_a_restart_sends_home_stays_a_sign_of_one() -> None:
    # The life count, a "game over" switch and the place in the level go home when
    # the game starts over. Here that goes uncalled five times: the AI does not
    # follow the score yet, the lives go up and a switch is no number, so nothing
    # "dropped". Everything that has moved changed, and nothing is counted.
    layout = {0: SCORE, 1: LIVES, 2: FLAG, 3: FLAG, 4: PROGRESS}
    d = EventDetector(knowledge_for(layout))
    d.observe(snap(0, 0, 3, 0, 0, 0))
    for frame in range(1, 11, 2):
        d.observe(snap(frame, 0, 0, 1, 1, 0))  # the last life gone, game over
        assert not restarts(d.observe(snap(frame + 1, 0, 3, 0, 0, 0)))
    assert d.moved == 3
    assert [d.homes(f"v{i}") for i in range(1, 6)] == [0, 0, 0, 0, 0]
    assert d.take_home_goers() == ()
    # So when the score has moved too and goes back down with them, the game
    # starting over is called, and by all of them.
    d.observe(snap(11, 40, 0, 1, 1, 0))
    called = restarts(d.observe(snap(12, 0, 3, 0, 0, 0)))
    assert len(called) == 1 and called[0].variables == ("v1", "v2", "v3", "v4")

    # As on the racing game while the AI is still looking about: when the game
    # starts over nearly everything changes, but only the "game over" switch lands
    # on what its watch calls its first value (the others were first seen in the
    # middle of a game). One value home of five is far from most; it is not counted
    # all the same, because most of what has moved changed.
    layout = {0: FLAG, 1: LIVES, 2: SCORE, 3: PROGRESS, 4: PROGRESS}
    d = EventDetector(knowledge_for(layout))
    d.observe(snap(0, 0, 1, 90, 1, 5))  # first seen in mid-game
    d.observe(snap(1, 0, 0, 95, 2, 6))
    for frame in range(2, 12, 2):
        d.observe(snap(frame, 1, 0, 95, 2, 6))  # game over
        # A new game: called from the second on, once the values have been where it
        # puts them (T-1022; until then this line asserted that none was called).
        assert bool(restarts(d.observe(snap(frame + 1, 0, 3, 0, 0, 0)))) == (frame > 2)
        d.observe(snap(frame + 1, 0, 0, 95, 2, 6))  # ... played to its last life
    assert d.moved == 5
    assert d.homes("v1") == 0 and d.take_home_goers() == ()


def test_nonsense_is_refused() -> None:
    d = started()
    for nonsense in (1, 0, None, "yes"):
        with pytest.raises(ConfigurationError, match="put_back"):
            d.observe(snap(2, *DOWN), put_back=nonsense)  # type: ignore[arg-type]
    assert d.observed == 2  # nothing was taken from what was refused
