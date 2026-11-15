"""T-1035: counting what the game said of a value's rises."""

from __future__ import annotations

import itertools

import pytest

from retro_ai_lab.core import ConfigurationError
from retro_ai_lab.understanding.unearned import ASK_UP_TO, SAY_AFTER, SHARE, Unearned


def told(*answers: bool) -> Unearned:
    seen = Unearned()
    for anyway in answers:
        seen.note("clock", anyway)
    return seen


def test_a_verdict_takes_two_and_half_of_them() -> None:
    assert (SAY_AFTER, SHARE, ASK_UP_TO) == (2, 0.5, 6)
    fresh = Unearned()
    assert fresh.counts("clock") == (0, 0) and not fresh.verdict("clock")
    # One rise that came anyway says nothing, whatever else was asked.
    assert not told(True).verdict("clock")
    assert not told(True, False).verdict("clock")
    # Two do, where they are half or more of the answers.
    assert told(True, True).verdict("clock")
    assert told(True, False, True).verdict("clock")
    assert told(False, False, True, True).verdict("clock")
    assert told(False, False, True, True).counts("clock") == (4, 2)
    # Two of five are fewer than half; a third makes it half of six.
    five = told(False, False, False, True, True)
    assert five.counts("clock") == (5, 2) and not five.verdict("clock")
    five.note("clock", True)
    assert five.counts("clock") == (6, 3) and five.verdict("clock")
    # Earned every time: never, however often.
    assert not told(*[False] * 20).verdict("clock")
    # Over every run of answers up to six, it is those two rules and nothing else.
    for length in range(ASK_UP_TO + 1):
        for answers in itertools.product((False, True), repeat=length):
            seen = told(*answers)
            assert seen.counts("clock") == (length, sum(answers))
            assert seen.verdict("clock") == (sum(answers) >= 2 and 2 * sum(answers) >= length)


def test_it_stops_wanting_answers() -> None:
    seen = Unearned()
    assert seen.wants("page")
    # A value whose rises are earned is asked about six times and then left alone.
    for asked in range(1, ASK_UP_TO + 1):
        seen.note("page", False)
        assert seen.wants("page") == (asked < ASK_UP_TO), asked
    assert not seen.verdict("page")
    # One with a verdict is not asked about again.
    clock = told(True)
    assert clock.wants("clock")
    clock.note("clock", True)
    assert clock.verdict("clock") and not clock.wants("clock")
    # Without a verdict after six answers it is left alone too.
    assert not told(False, False, False, False, True, False).wants("clock")
    assert told(False, False, False, True, True).wants("clock")


def test_values_are_counted_apart() -> None:
    seen = Unearned()
    seen.note("clock", True)
    seen.note("page", False)
    seen.note("clock", True)
    assert seen.counts("clock") == (2, 2) and seen.counts("page") == (1, 0)
    assert seen.verdict("clock") and not seen.verdict("page")
    assert not seen.wants("clock") and seen.wants("page") and seen.wants("coins")
    assert Unearned().counts("clock") == (0, 0)  # and each count is its own


def test_nonsense_is_refused() -> None:
    seen = Unearned()
    for var in ("", None, 7):
        with pytest.raises(ConfigurationError, match="named by its id"):
            seen.note(var, True)  # type: ignore[arg-type]
        with pytest.raises(ConfigurationError, match="named by its id"):
            seen.counts(var)  # type: ignore[arg-type]
        with pytest.raises(ConfigurationError, match="named by its id"):
            seen.verdict(var)  # type: ignore[arg-type]
        with pytest.raises(ConfigurationError, match="named by its id"):
            seen.wants(var)  # type: ignore[arg-type]
    for anyway in (1, 0, "yes", None):
        with pytest.raises(ConfigurationError, match="anyway must be true or false"):
            seen.note("clock", anyway)  # type: ignore[arg-type]
    assert seen.counts("clock") == (0, 0)  # nothing was counted
