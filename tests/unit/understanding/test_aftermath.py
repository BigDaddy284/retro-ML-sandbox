"""T-997: what goes up when a life is lost is not progress."""

from __future__ import annotations

import pytest

from retro_ai_lab.core import ConfigurationError
from retro_ai_lab.understanding.aftermath import AFTER, BEFORE, MIN_COUNT, Aftermath
from retro_ai_lab.understanding.events import EventKind, GameEvent


def rise(var: str, frame: int) -> GameEvent:
    return GameEvent(EventKind.PROGRESS, frame, var, "", 0, 1)


def loss(frame: int, kind: EventKind = EventKind.SETBACK) -> GameEvent:
    return GameEvent(kind, frame, "lives" if kind is EventKind.SETBACK else None, "", 3, 2)


def test_a_value_that_rises_as_lives_are_lost() -> None:
    seen = Aftermath()
    assert not seen.verdict("mode") and seen.counts("mode") == (0, 0, 0, 0)
    for n in range(6):
        at = 1000 * n
        seen.note((rise("mode", at),))  # the dying pause begins
        seen.note((loss(at + 40),))  # and then the life goes
    assert seen.counts("mode") == (6, 6, 6, 6)  # rises, at a loss; losses, with its rise
    assert seen.verdict("mode")
    # In the very same step is as good, and other kinds of event are not rises.
    same = Aftermath()
    for n in range(6):
        gain = GameEvent(EventKind.SCORE_GAIN, 500 * n, "score", "", 0, 50)
        on = GameEvent(EventKind.FLAG_ON, 500 * n, "flag", "", 0, 1)
        same.note((loss(500 * n), rise("over", 500 * n), gain, on))
    assert same.verdict("over") and same.counts("over") == (6, 6, 6, 6)
    assert same.counts("score") == (0, 0, 0, 0) and same.counts("flag") == (0, 0, 0, 0)
    # Two kinds of loss in one step (the life, and everything starting over) are one.
    both = Aftermath()
    for n in range(6):
        both.note((rise("mode", 900 * n),))
        both.note((loss(900 * n + 40), loss(900 * n + 40, EventKind.RESET)))
    assert both.counts("mode") == (6, 6, 6, 6)


def test_the_window_has_edges() -> None:
    assert (BEFORE, AFTER, MIN_COUNT) == (90, 30, 5)
    late = Aftermath()
    early = Aftermath()
    for n in range(6):
        at = 2000 * n
        late.note((loss(at),))
        late.note((rise("slow", at + AFTER + 1),))  # too long after the loss
        early.note((rise("far", at),))
        early.note((loss(at + BEFORE + 1),))  # too long before it
    assert late.counts("slow")[1] == 0 and not late.verdict("slow")
    assert early.counts("far") == (6, 0, 6, 0) and not early.verdict("far")
    edge = Aftermath()
    for n in range(6):
        at = 2000 * n
        edge.note((rise("first", at),))
        edge.note((loss(at + BEFORE),))
        edge.note((rise("last", at + BEFORE + AFTER),))
    assert edge.verdict("first") and edge.verdict("last")
    # Two rises before one loss both came at it, and the loss has its rise once.
    twice = Aftermath()
    twice.note((rise("mode", 10),))
    twice.note((rise("mode", 50),))
    twice.note((loss(60),))
    assert twice.counts("mode") == (2, 2, 1, 1)


def test_progress_followed_by_a_loss_now_and_then_is_left() -> None:
    # A checkpoint: six rises, two of them with a life lost soon after, among a
    # dozen lives lost elsewhere.
    seen = Aftermath()
    for n in range(6):
        at = 5000 * n
        seen.note((rise("checkpoint", at),))
        if n < 2:
            seen.note((loss(at + 60),))
        seen.note((loss(at + 2000),))
        seen.note((loss(at + 3500),))
    rises, met, losses, answered = seen.counts("checkpoint")
    assert (rises, met) == (6, 2) and answered == 2 and losses == 14
    assert not seen.verdict("checkpoint")
    # A page crossed just before a jump the AI cannot make yet: every rise is followed
    # by a life lost, but most lives are lost elsewhere with no such rise. Both ways
    # are needed, so this is left alone.
    page = Aftermath()
    for n in range(6):
        at = 9000 * n
        page.note((rise("page", at),))
        page.note((loss(at + 60),))
        for k in range(1, 4):
            page.note((loss(at + 2000 * k),))
    assert page.counts("page") == (6, 6, 24, 6) and not page.verdict("page")
    # And the other way round: at every loss it rises, but it mostly rises with no loss.
    busy = Aftermath()
    for n in range(6):
        at = 9000 * n
        busy.note((rise("lap", at),))
        busy.note((loss(at + 40),))
        for k in range(1, 4):
            busy.note((rise("lap", at + 2000 * k),))
    assert busy.counts("lap") == (24, 6, 6, 6) and not busy.verdict("lap")


def test_too_few_to_say_and_losses_before_it_rose() -> None:
    seen = Aftermath()
    for n in range(10):
        seen.note((loss(700 * n),))  # ten lives lost before the value was ever seen to rise
    for n in range(4):
        at = 10_000 + 1000 * n
        seen.note((rise("mode", at),))
        seen.note((loss(at + 40),))
    assert seen.counts("mode") == (4, 4, 4, 4) and not seen.verdict("mode")  # four is too few
    seen.note((rise("mode", 20_000),))
    seen.note((loss(20_040),))
    assert seen.counts("mode") == (5, 5, 5, 5) and seen.verdict("mode")
    assert seen.verdict("never seen") is False


def test_a_new_go_forgets_what_was_waiting() -> None:
    seen = Aftermath()
    seen.note((rise("mode", 100),))
    seen.begin()  # the game was put back to the start of the go
    seen.note((loss(120),))
    assert seen.counts("mode") == (1, 0, 1, 0)
    seen.note((loss(500),))
    seen.begin()
    seen.note((rise("mode", 510),))  # not "just after" a loss from the go before
    assert seen.counts("mode") == (2, 0, 2, 0)
    # Frames running backwards (a saved spot loaded by hand) do the same.
    seen.note((rise("mode", 900),))
    seen.note((loss(300),))
    assert seen.counts("mode") == (3, 0, 3, 0)
    # The counts themselves are kept from go to go, and so is the time watched.
    for n in range(5):
        seen.begin()
        seen.note((), 2000 * n)  # the go starts, and nothing happens for a while
        seen.note((rise("mode", 2000 * n + 900),))
        seen.note((loss(2000 * n + 940),))
    assert seen.counts("mode") == (8, 5, 8, 5) and seen.verdict("mode")  # 5 of 8 is 62%
    assert seen.watched == 5 * 940 + 380 + 390  # the frames each go spanned


def test_where_losses_come_thick_and_fast_nothing_is_said() -> None:
    """The toy racer at full speed: a checkpoint every 16 frames and a life every 20.
    Every rise is near a loss and every loss near a rise, by chance alone."""
    seen = Aftermath()
    for frame in range(0, 640, 4):
        events = []
        if frame % 16 == 0:
            events.append(rise("checkpoint", frame))
        if frame % 20 == 12:
            events.append(loss(frame))
        seen.note(tuple(events), frame)
    rises, met, losses, answered = seen.counts("checkpoint")
    assert rises == 40 and met >= 0.9 * rises and answered == losses >= MIN_COUNT
    assert not seen.verdict("checkpoint")
    # The same rises and losses with long quiet stretches of play around them would
    # be something: it is the crowding that says nothing.
    calm = Aftermath()
    for n in range(8):
        calm.note((), 3000 * n)
        calm.note((rise("mode", 3000 * n + 1500),))
        calm.note((loss(3000 * n + 1540),), 3000 * n + 1540)
    assert calm.verdict("mode")


def test_nonsense_is_refused() -> None:
    seen = Aftermath()
    for bad in (None, "events", [rise("a", 1)], (1, 2)):
        with pytest.raises(ConfigurationError):
            seen.note(bad)  # type: ignore[arg-type]
    for frame in (1.5, "7", True):
        with pytest.raises(ConfigurationError):
            seen.note((), frame)  # type: ignore[arg-type]
    seen.note(())  # nothing happened: fine
