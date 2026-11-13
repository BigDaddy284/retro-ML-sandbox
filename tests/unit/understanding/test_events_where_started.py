"""T-1022: where a value started, and a life lost that is no restart."""

from __future__ import annotations

from retro_ai_lab.understanding import (
    Encoding,
    EventDetector,
    EventKind,
    Evidence,
    GameEvent,
    Knowledge,
    VariableKind,
)
from retro_ai_lab.understanding.events import HOME_OFTEN
from tests.unit.understanding.test_events import RAM, SURE, knowledge_for, snap

SCORE, FLAG, PROGRESS, LIVES, HEALTH = (
    VariableKind.SCORE,
    VariableKind.FLAG,
    VariableKind.PROGRESS,
    VariableKind.LIVES,
    VariableKind.HEALTH,
)
# A "game over" switch, the life count, the score and two numbers of progress. The
# byte between is never used (it lets the score be read as two bytes further down).
GAME = {0: FLAG, 1: LIVES, 2: SCORE, 4: PROGRESS, 5: PROGRESS}
MID = (0, 1, 90, 0, 1, 5)  # as the AI first sees them: in the middle of a game
LAST = (0, 0, 95, 0, 2, 6)  # the last life gone
OVER = (1, 0, 95, 0, 2, 6)  # "game over"
NEW = (0, 3, 0, 0, 0, 0)  # a new game
ALL = ("v1", "v2", "v3", "v4", "v5")


def restarts(events: tuple[GameEvent, ...]) -> list[GameEvent]:
    return [e for e in events if e.kind is EventKind.RESET]


def kinds(events: tuple[GameEvent, ...]) -> list[EventKind]:
    return [e.kind for e in events]


def seen_a_new_game(k: Knowledge) -> EventDetector:
    """A detector that first saw the game in its middle, saw it end and start over
    (not called: see the first test) and saw the next game end too."""
    d = EventDetector(k)
    d.observe(snap(0, *MID))
    d.observe(snap(1, *LAST))
    d.observe(snap(2, *OVER))
    assert not restarts(d.observe(snap(3, *NEW)))
    d.observe(snap(4, *LAST))
    d.observe(snap(5, *OVER))
    return d


def test_a_new_game_is_called_once_the_values_have_been_there() -> None:
    d = EventDetector(knowledge_for(GAME))
    d.observe(snap(0, *MID))
    d.observe(snap(1, *LAST))
    d.observe(snap(2, *OVER))
    # The first new game the AI sees: the lives are higher and the numbers lower
    # than it has ever seen them. Nothing has gone *back* anywhere but the switch.
    assert not restarts(d.observe(snap(3, *NEW)))
    # From then on it knows where they start, though no watch began there ...
    d.observe(snap(4, *LAST))
    assert not restarts(d.observe(snap(5, *OVER)))
    called = restarts(d.observe(snap(6, *NEW)))
    assert len(called) == 1 and called[0].variables == ALL
    assert called[0].detail == (
        "Flag, Lives, Score, Progress, Progress all went back to their starting values."
    )
    # ... and every time after.
    for frame in range(7, 37, 3):
        assert not restarts(d.observe(snap(frame, *LAST)))
        assert not restarts(d.observe(snap(frame + 1, *OVER)))
        called = restarts(d.observe(snap(frame + 2, *NEW)))
        assert len(called) == 1 and called[0].variables == ALL


def test_the_lowest_and_highest_belong_to_the_place() -> None:
    # The AI learns more about the lives and the score while the game is over: their
    # watches are made anew, at 0 and at 95. Where they start is still known.
    k = knowledge_for(GAME)
    d = seen_a_new_game(k)
    k.support("v2", Evidence("test", "seen again", 1.0))
    k.support("v3", Evidence("test", "seen again", 1.0))
    assert d.observe(snap(6, *OVER)) == ()  # the new watches' first sight
    called = restarts(d.observe(snap(7, *NEW)))
    assert len(called) == 1 and called[0].variables == ALL

    # A new go: every watch starts again, at "game over" this time. What was seen of
    # the numbers is kept; the switch has only its watch to go by, and is not home.
    d = seen_a_new_game(k)
    d.reset()
    assert d.observe(snap(6, *OVER)) == ()
    called = restarts(d.observe(snap(7, *NEW)))
    assert len(called) == 1 and called[0].variables == ("v2", "v3", "v4", "v5")

    # A person says the score is two bytes long. Read so it has never been seen at
    # 0, and it is not among the values that are back ...
    k = knowledge_for(GAME)
    d = seen_a_new_game(k)
    k.correct("v3", encoding=Encoding.U16LE)
    d.observe(snap(6, *OVER))
    called = restarts(d.observe(snap(7, *NEW)))
    assert len(called) == 1 and called[0].variables == ("v1", "v2", "v4", "v5")
    # ... until it has been there read that way,
    d.observe(snap(8, *LAST))
    d.observe(snap(9, *OVER))
    assert restarts(d.observe(snap(10, *NEW)))[0].variables == ALL
    # and read as one byte again, what the one byte was seen at is still known.
    d.observe(snap(11, *LAST))
    d.observe(snap(12, *OVER))
    k.correct("v3", encoding=Encoding.U8)
    d.observe(snap(13, *OVER))
    assert restarts(d.observe(snap(14, *NEW)))[0].variables == ALL

    # The guess about the lives is dropped. With nothing held at its place any more
    # what was seen there goes too: a guess made about it later starts from nothing.
    k = knowledge_for(GAME)
    d = seen_a_new_game(k)
    k.remove("v2")
    d.observe(snap(6, *OVER))
    again = k.propose(RAM, 1, kind=LIVES, name="Lives", evidence=SURE)
    assert again.id == "v2"
    d.observe(snap(7, *OVER))
    called = restarts(d.observe(snap(8, *NEW)))
    assert len(called) == 1 and called[0].variables == ("v1", "v3", "v4", "v5")
    d.observe(snap(9, *LAST))
    d.observe(snap(10, *OVER))
    assert set(restarts(d.observe(snap(11, *NEW)))[0].variables) == set(ALL)


def test_only_some_kinds_have_such_a_start() -> None:
    # A score is not at a start at the highest it has been. Here the progress goes
    # back to its lowest and the score, in the same snapshot, up to its highest.
    d = EventDetector(knowledge_for({0: SCORE, 1: PROGRESS}))
    d.observe(snap(0, 50, 5))
    d.observe(snap(1, 60, 3))
    d.observe(snap(2, 55, 6))
    assert not restarts(d.observe(snap(3, 60, 3)))
    # ... nor is progress: the same with the two the other way round.
    d = EventDetector(knowledge_for({0: PROGRESS, 1: SCORE}))
    d.observe(snap(0, 50, 5))
    d.observe(snap(1, 60, 3))
    d.observe(snap(2, 55, 6))
    assert not restarts(d.observe(snap(3, 60, 3)))
    # With both back at their lowest it is a restart.
    assert not restarts(d.observe(snap(4, 70, 6)))
    called = restarts(d.observe(snap(5, 50, 3)))
    assert len(called) == 1 and called[0].variables == ("v1", "v2")

    # Health is not at a start at the least it has been (the lives go up with it
    # here, or it would be a life lost and no restart for that reason).
    d = EventDetector(knowledge_for({0: HEALTH, 1: PROGRESS, 2: LIVES}))
    d.observe(snap(0, 5, 5, 2))
    d.observe(snap(1, 2, 3, 4))
    d.observe(snap(2, 9, 6, 1))
    events = d.observe(snap(3, 2, 3, 3))  # health at its least, progress at its lowest
    assert EventKind.SETBACK in kinds(events) and EventKind.RECOVERY in kinds(events)
    assert not restarts(events)
    # (A life count at its least cannot be asked the same way: when it goes down
    # there is no restart whatever else happens, see the next test.)
    # At their most both are at a start, and with the progress back it is a restart.
    d = EventDetector(knowledge_for({0: HEALTH, 1: PROGRESS, 2: LIVES}))
    d.observe(snap(0, 5, 5, 2))
    d.observe(snap(1, 9, 3, 3))
    d.observe(snap(2, 4, 6, 1))
    called = restarts(d.observe(snap(3, 9, 3, 3)))
    assert len(called) == 1 and called[0].variables == ("v1", "v2", "v3")

    # A switch is at a start only where its watch began: here it began "on", and
    # going off with the number that goes back to its lowest is no restart.
    d = EventDetector(knowledge_for({0: FLAG, 1: PROGRESS}))
    d.observe(snap(0, 1, 5))
    d.observe(snap(1, 0, 3))
    d.observe(snap(2, 1, 6))
    assert not restarts(d.observe(snap(3, 0, 3)))
    # ... and so is a value of a kind that has no such start, followed because a
    # person asked about it.
    k = knowledge_for({1: PROGRESS})
    k.propose(RAM, 0, kind=VariableKind.UNKNOWN, name="Something", evidence=SURE)
    d = EventDetector(k)
    d.ask(["v2"])
    d.observe(snap(0, 50, 5))
    d.observe(snap(1, 40, 3))
    d.observe(snap(2, 60, 6))
    events = d.observe(snap(3, 40, 3))
    assert EventKind.DOWN in kinds(events) and not restarts(events)
    # Where its watch began it is, as it always was.
    d.observe(snap(4, 60, 6))
    called = restarts(d.observe(snap(5, 50, 3)))
    assert len(called) == 1 and set(called[0].variables) == {"v1", "v2"}


def test_a_life_lost_is_no_restart() -> None:
    # The life count, the place in the level, "on the ground" and the score, all
    # seen from the game's start.
    layout = {0: LIVES, 1: PROGRESS, 2: FLAG, 3: SCORE}

    def playing(k: Knowledge | None = None) -> EventDetector:
        d = EventDetector(k or knowledge_for(layout))
        d.observe(snap(0, 3, 0, 1, 0))
        d.observe(snap(1, 3, 4, 0, 20))
        return d

    # A life lost, the player put back at the level's start: the place and the
    # switch are back where they began, and the place dropped. It was a "restart".
    d = playing()
    events = d.observe(snap(2, 2, 0, 1, 20))
    assert kinds(events) == [EventKind.SETBACK, EventKind.FLAG_ON]
    # The same with the life count left alone is one, as before.
    d = playing()
    called = restarts(d.observe(snap(2, 3, 0, 1, 20)))
    assert len(called) == 1 and called[0].variables == ("v2", "v3")
    # The game starting over gives the lives back: called, and the lives named.
    d = playing()
    d.observe(snap(2, 2, 0, 1, 20))
    d.observe(snap(3, 2, 4, 0, 30))
    d.observe(snap(4, 0, 4, 0, 30))  # the last two lives
    called = restarts(d.observe(snap(5, 3, 0, 1, 0)))
    assert len(called) == 1 and called[0].variables == ("v1", "v2", "v3", "v4")

    # Health that is filled up again for the next life does not make a life lost a
    # restart ...
    hearts = {**layout, 4: HEALTH}
    d = EventDetector(knowledge_for(hearts))
    d.observe(snap(0, 3, 0, 1, 0, 9))
    d.observe(snap(1, 3, 4, 0, 20, 1))
    events = d.observe(snap(2, 2, 0, 1, 20, 9))
    assert EventKind.RECOVERY in kinds(events) and not restarts(events)
    # ... and health going down while the place goes back is no restart either,
    d = EventDetector(knowledge_for(hearts))
    d.observe(snap(0, 3, 0, 1, 0, 9))
    d.observe(snap(1, 3, 4, 0, 20, 5))
    events = d.observe(snap(2, 3, 0, 1, 20, 4))
    assert EventKind.SETBACK in kinds(events) and not restarts(events)
    # unless something is given back with it: two health values, one down, one up.
    both = {**layout, 4: HEALTH, 5: HEALTH}
    d = EventDetector(knowledge_for(both))
    d.observe(snap(0, 3, 0, 1, 0, 9, 9))
    d.observe(snap(1, 3, 4, 0, 20, 5, 5))
    called = restarts(d.observe(snap(2, 3, 0, 1, 20, 4, 9)))
    assert len(called) == 1 and called[0].variables == ("v2", "v3", "v6")
    # A life count going down settles it whatever is given back.
    d = EventDetector(knowledge_for(both))
    d.observe(snap(0, 3, 0, 1, 0, 9, 9))
    d.observe(snap(1, 3, 4, 0, 20, 5, 5))
    assert not restarts(d.observe(snap(2, 2, 0, 1, 20, 9, 9)))


def test_a_number_that_keeps_coming_to_its_lowest() -> None:
    # A fall speed and a height above the ground, both taken for numbers that count
    # up: 0 at every landing. Three others have moved and stay away.
    layout = {0: SCORE, 1: PROGRESS, 2: PROGRESS, 3: PROGRESS, 4: LIVES}
    k = knowledge_for(layout)

    def seen(landings: int) -> EventDetector:
        """A session that has seen so many landings in play, and then a new go that
        the AI begins to watch in mid-air: no watch has 0 for its first value."""
        d = EventDetector(k)
        d.observe(snap(0, 0, 0, 0, 0, 3))
        d.observe(snap(1, 0, 0, 1, 2, 2))
        frame = 2
        for _ in range(landings):
            d.observe(snap(frame, 5, 7, 1, 2, 2))
            assert not restarts(d.observe(snap(frame + 1, 0, 0, 1, 2, 2)))
            frame += 2
        d.reset()
        d.observe(snap(100, 9, 9, 1, 2, 2))  # values they have not been seen to come to
        d.observe(snap(101, 6, 8, 1, 2, 2))  # the two are all that moved
        assert d.moved == 2 and (d.homes("v1"), d.homes("v2")) == (0, 0)
        return d

    # Seen to land twice: both are back at their lowest, one dropped, and they are
    # all that has moved. By the rule that is a restart.
    called = restarts(seen(HOME_OFTEN - 1).observe(snap(102, 0, 0, 1, 2, 2)))
    assert len(called) == 1 and called[0].variables == ("v1", "v2")
    # Seen three times: coming to 0 is what they do in play, wherever their watches
    # began, and they cannot make a restart by themselves.
    d = seen(HOME_OFTEN)
    for frame in range(102, 122, 2):
        assert not restarts(d.observe(snap(frame, 0, 0, 1, 2, 2)))
        d.observe(snap(frame + 1, 6, 8, 1, 2, 2))
    # With a value that does not come there in play it is a restart, and they are
    # named with it.
    d = seen(HOME_OFTEN)
    d.observe(snap(102, 6, 8, 2, 2, 2))  # the third value moves on
    called = restarts(d.observe(snap(103, 0, 0, 1, 2, 2)))  # ... and is back with them
    assert len(called) == 1 and called[0].variables == ("v1", "v2", "v3")
