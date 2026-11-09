"""T-894: the app's own timings, kept like the emulator link's."""

from __future__ import annotations

import pytest

from retro_ai_lab.core.worklog import CallStats, WorkLog, app_work

X = "\N{MULTIPLICATION SIGN}"


class Clock:
    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now


def test_timings_are_kept_by_name_and_summarised_slowest_first() -> None:
    clock = Clock()
    work = WorkLog(clock)
    assert work.summary() == "nothing" and work.window_s == 0
    work.note("picture", 0.002)
    with work.timed("learn"):
        clock.now += 0.04
    with pytest.raises(ZeroDivisionError), work.timed("learn"):  # timed even when it raises
        clock.now += 0.01
        _ = 1 / 0
    assert work.stats["learn"].count == 2 and work.stats["learn"].max_s == pytest.approx(0.04)
    assert work.summary() == f"learn {X}2 avg 25 ms max 40 ms; picture {X}1 avg 2 ms max 2 ms"
    clock.now += 4.95
    window, summary = work.take_window()
    assert window == pytest.approx(5.0) and summary.startswith(f"learn {X}2")
    assert work.summary() == "nothing" and work.window_s == 0  # a new window


def test_bad_timings_are_refused() -> None:
    work = WorkLog()
    with pytest.raises(ValueError, match="needs a name"):
        work.note(" ", 0.1)
    with pytest.raises(ValueError, match="cannot be negative"):
        work.note("learn", -0.1)
    stats = CallStats()
    assert stats.average_s == 0.0 and stats.describe("x") == f"x {X}0 avg 0 ms max 0 ms"


def test_the_app_shares_one_work_log() -> None:
    assert isinstance(app_work, WorkLog)


def test_quiet_work_is_left_out_of_the_line() -> None:
    """T-906: hundreds of free event deliveries must not bury the one slow handler."""
    work = WorkLog(Clock())
    for _ in range(300):
        work.note("on MemoryObserved", 0.0002)
    work.note("on DiscoveryUpdated", 0.1)
    assert work.summary(quiet_under_s=0.001) == f"on DiscoveryUpdated {X}1 avg 100 ms max 100 ms"
    assert work.summary().startswith(f"on DiscoveryUpdated {X}1")  # the default keeps all
    assert f"on MemoryObserved {X}300" in work.summary()
    assert work.summary(quiet_under_s=1.0) == "nothing"
    _, summary = work.take_window(quiet_under_s=0.001)
    assert summary == f"on DiscoveryUpdated {X}1 avg 100 ms max 100 ms"
    with pytest.raises(ValueError, match="cannot be negative"):
        work.summary(quiet_under_s=-1)


def test_notes_from_several_threads_are_all_counted() -> None:
    """T-956: the save writer is a third thread timing into the app's log."""
    import threading

    log = WorkLog()
    rounds = 2000

    def hammer(name: str) -> None:
        for _ in range(rounds):
            log.note(name, 0.001)
            log.note("shared", 0.001)

    threads = [threading.Thread(target=hammer, args=(f"t{i}",)) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    stats = log.stats
    assert stats["shared"].count == 4 * rounds
    assert all(stats[f"t{i}"].count == rounds for i in range(4))
