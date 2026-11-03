"""T-830: the profiling tool measures a run of the real window against an emulator."""

from __future__ import annotations

import socket

import pytest
from PySide6.QtWidgets import QApplication

from retro_ai_lab.core import ConfigurationError
from retro_ai_lab.emulator.bizhawk.protocol import CallStats
from retro_ai_lab.tools.profile_play import STUTTER_MS, Report, main, profile


def _closed_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def test_profiles_a_run_on_the_simulator(qapp: QApplication) -> None:
    report = profile("simulator", seconds=0.5)
    assert report.adapter == "simulator"
    assert 0.4 <= report.seconds <= 3.0
    assert report.interval_ms == 16  # the app's 60 fps clock
    assert report.steps > 0 and report.pictures > 0
    assert report.wanted >= report.steps
    assert report.failure is None
    assert report.link_summary.startswith("step \u00d7")  # the slowest command first
    assert "hello \u00d71" in report.link_summary
    # Every step the clock counted was timed on the worker; one more step may
    # have been in flight when time ran out (timed on the worker, never
    # delivered to the clock), which is what CI on Windows saw (T-834).
    assert report.steps <= report.step_work.count <= report.steps + 1
    assert report.pictures <= report.picture_work.count <= report.pictures + 1
    assert report.longest_stall_s > 0
    lines = report.lines()
    assert lines[0].startswith("Played 0.") and "no failures" in lines[0]
    assert lines[1] == f"Emulator round trips: {report.link_summary}."
    assert lines[2].startswith(
        f"App work per step (worker thread, snapshot bookkeeping): \u00d7{report.steps}"
    )
    assert lines[4].endswith(f"(a stall over {STUTTER_MS} ms shows as a stutter).")


def test_profiles_the_fake_emulator_without_a_link(qapp: QApplication) -> None:
    report = profile("fake", seconds=0.3)
    assert report.steps > 0 and report.link_summary == "not a bridge link"


def test_main_prints_the_report(qapp: QApplication, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--seconds", "0.3"]) == 0
    out = capsys.readouterr().out.splitlines()
    assert len(out) == 5 and out[0].startswith("Played 0.") and out[1].startswith("Emulator")


def test_main_reports_an_emulator_it_cannot_reach(
    qapp: QApplication, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["--adapter", "bizhawk", "--port", str(_closed_port()), "--seconds", "1"]) == 1
    assert capsys.readouterr().err.startswith("Could not play: Couldn't find BizHawk")


@pytest.mark.parametrize(
    "argv",
    [["--seconds", "0"], ["--seconds", "soon"], ["--adapter", "dolphin"], ["--port", "x"]],
)
def test_main_rejects_bad_arguments(argv: list[str], capsys: pytest.CaptureFixture[str]) -> None:
    assert main(argv) == 2
    assert "usage" in capsys.readouterr().err


def test_profile_rejects_bad_values() -> None:
    with pytest.raises(ConfigurationError, match="Unknown adapter"):
        profile("dolphin")
    with pytest.raises(ConfigurationError, match="above zero"):
        profile("fake", seconds=0)


def test_report_lines_without_work() -> None:
    report = Report(
        adapter="fake",
        seconds=1.0,
        interval_ms=16,
        steps=0,
        wanted=62,
        pictures=0,
        skipped=0,
        failure="The emulator did not answer.",
        link_summary="no calls",
    )
    lines = report.lines()
    assert "last failure: The emulator did not answer." in lines[0]
    assert lines[2].endswith(": none.") and lines[3].endswith(": none.")
    assert isinstance(report.step_work, CallStats)
