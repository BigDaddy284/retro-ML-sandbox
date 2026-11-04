"""Measure what playing costs: the emulator's round trips and the app's own work.

Run ``python -m retro_ai_lab.tools.profile_play`` against the built-in
simulator (the default), or ``--adapter bizhawk`` with EmuHawk running the
bridge script, to see how fast the play clock can run and where the time
goes. The report says how many frames the clock stepped against how many
it wanted, what each bridge command cost (the same numbers the log writes
every 5 s, T-819, but for the whole run), how long the app's own
bookkeeping took per step and per picture, and the longest stall of the
GUI thread.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from retro_ai_lab.core import ConfigurationError, EmulatorError
from retro_ai_lab.emulator.bizhawk.adapter import BizHawkAdapter
from retro_ai_lab.emulator.bizhawk.protocol import CallStats

PROG = "python -m retro_ai_lab.tools.profile_play"
ADAPTERS = ("simulator", "fake", "bizhawk", "mgba")
DEFAULT_SECONDS = 5.0
HEARTBEAT_MS = 1
# A stall of the GUI thread this long is what a person sees as a stutter.
STUTTER_MS = 50


@dataclass
class Report:
    """What one profiling run measured."""

    adapter: str
    seconds: float
    interval_ms: int
    steps: int
    wanted: int
    pictures: int
    skipped: int
    failure: str | None
    link_summary: str
    frames: int = 0  # the frames those steps carried (T-888: several per step when behind)
    forgiven: int = 0  # frames the clock owed beyond a step's cap and let go
    step_work: CallStats = field(default_factory=CallStats)
    picture_work: CallStats = field(default_factory=CallStats)
    longest_stall_s: float = 0.0

    def lines(self) -> list[str]:
        """The report as the lines the tool prints."""
        kept = (
            f"{self.frames} of the {self.wanted} frames the clock wanted in {self.steps} steps"
            f" ({self.forgiven} frames let go after stalls)"
        )
        pictures = f"{self.pictures} pictures"
        skipped = f"{self.skipped} pulses skipped while a step was still running"
        failed = f"last failure: {self.failure}" if self.failure else "no failures"
        return [
            f"Played {self.seconds:.1f} s, one frame every {self.interval_ms} ms, on "
            f"'{self.adapter}': {kept}, "
            f"{pictures}, {skipped}, {failed}.",
            f"Emulator round trips: {self.link_summary}.",
            "App work per step (worker thread, snapshot bookkeeping): "
            f"{_describe(self.step_work)}.",
            "App work per picture (GUI thread, screen diff and display): "
            f"{_describe(self.picture_work)}.",
            f"Longest GUI stall: {self.longest_stall_s * 1000:.0f} ms "
            f"(a stall over {STUTTER_MS} ms shows as a stutter).",
        ]


def _describe(stats: CallStats) -> str:
    if stats.count == 0:
        return "none"
    average, slowest = stats.average_s * 1000, stats.max_s * 1000
    return f"\u00d7{stats.count} avg {average:.2f} ms max {slowest:.2f} ms"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=PROG,
        description="Measure what playing costs: emulator round trips and the app's own work.",
    )
    parser.add_argument(
        "--adapter",
        choices=ADAPTERS,
        default="simulator",
        help="what to play on: the built-in simulator (default), the fake emulator, "
        "or a running BizHawk or mGBA with the bridge script",
    )
    parser.add_argument("--host", default=None, help="the emulator's host (bizhawk, mgba)")
    parser.add_argument("--port", type=int, default=None, help="the bridge script's port")
    parser.add_argument(
        "--seconds", type=_positive, default=DEFAULT_SECONDS, help="how long to play (default 5)"
    )
    return parser


def _positive(text: str) -> float:
    try:
        value = float(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"'{text}' is not a number.") from exc
    if not value > 0:
        raise argparse.ArgumentTypeError("must be above zero.")
    return value


class _Timed:
    """Wraps a callable, keeping :class:`CallStats` of every call."""

    def __init__(self, target: Callable[..., Any]) -> None:
        self._target = target
        self.stats = CallStats()

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        started = time.perf_counter()
        try:
            return self._target(*args, **kwargs)
        finally:
            self.stats.add(time.perf_counter() - started)


def profile(
    adapter: str = "simulator",
    *,
    seconds: float = DEFAULT_SECONDS,
    host: str | None = None,
    port: int | None = None,
) -> Report:
    """Play for ``seconds`` in the real main window and measure. Needs a QApplication."""
    if adapter not in ADAPTERS:
        raise ConfigurationError(f"Unknown adapter '{adapter}'. Choose from {', '.join(ADAPTERS)}.")
    if not seconds > 0:
        raise ConfigurationError("seconds must be above zero.")
    from PySide6.QtCore import QEventLoop, Qt, QTimer

    from retro_ai_lab.emulator.bizhawk.simulator import BizHawkSimulator
    from retro_ai_lab.ui.app import create_app
    from retro_ai_lab.ui.context import AppContext
    from retro_ai_lab.ui.main_window import MainWindow

    create_app()  # the process-wide QApplication, made if missing
    simulator = BizHawkSimulator() if adapter == "simulator" else None
    if simulator is not None:
        simulator.start()
    try:
        context = AppContext()
        window = MainWindow(context)
        window.show()
        clock = window.game_panel.clock
        step_work = _Timed(context.observation.advance)
        picture_work = _Timed(context.observation.observe_frame)
        context.observation.advance = step_work  # type: ignore[method-assign]
        context.observation.observe_frame = picture_work  # type: ignore[method-assign]

        options: dict[str, Any] = {}
        name = adapter
        if simulator is not None:
            name, options = "bizhawk", {"port": simulator.port}
        if host is not None:
            options["host"] = host
        if port is not None and simulator is None:
            options["port"] = port
        context.connection.connect(name, **options)
        clock.play()

        stalls = _Heartbeat()
        heartbeat = QTimer(window)
        heartbeat.setTimerType(Qt.TimerType.PreciseTimer)
        heartbeat.timeout.connect(stalls.beat)
        heartbeat.start(HEARTBEAT_MS)
        # A local loop, not app.exec(): QCoreApplication.quit() leaves the
        # thread's quitNow flag set, after which every later QEventLoop.exec()
        # (pytest-qt's waits included) returns at once without processing
        # anything (T-834).
        loop = QEventLoop()
        QTimer.singleShot(int(seconds * 1000), loop.quit)
        started = time.perf_counter()
        loop.exec()
        elapsed = time.perf_counter() - started
        heartbeat.stop()
        # A step still on the worker has started but delivered no frames yet,
        # so the report counts only landed steps against the frames (both read
        # here on the GUI thread, between which nothing can land).
        landed = clock.steps - (1 if clock.in_flight else 0)

        live = context.connection.adapter
        summary = live.link_summary if isinstance(live, BizHawkAdapter) else "not a bridge link"
        report = Report(
            adapter=adapter,
            seconds=elapsed,
            interval_ms=clock.interval_ms,
            steps=landed,
            wanted=int(elapsed * 1000 / clock.interval_ms),
            pictures=clock.frames_captured,
            skipped=clock.skipped,
            frames=clock.frames,
            forgiven=clock.forgiven,
            failure=clock.last_failure,
            link_summary=summary,
            step_work=step_work.stats,
            picture_work=picture_work.stats,
            longest_stall_s=stalls.longest_s,
        )
        if context.connection.is_connected:
            context.connection.disconnect()
        window.close()
        return report
    finally:
        if simulator is not None:
            simulator.stop()


class _Heartbeat:
    """Records the longest gap between beats of a 1 ms GUI timer."""

    def __init__(self) -> None:
        self._last: float | None = None
        self.longest_s = 0.0

    def beat(self) -> None:
        now = time.perf_counter()
        if self._last is not None:
            self.longest_s = max(self.longest_s, now - self._last)
        self._last = now


def main(argv: Sequence[str] | None = None) -> int:
    """Run the tool; prints the report and returns the exit code."""
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        return exc.code if isinstance(exc.code, int) else 1
    try:
        report = profile(args.adapter, seconds=args.seconds, host=args.host, port=args.port)
    except (EmulatorError, ConfigurationError) as exc:
        print(f"Could not play: {exc}", file=sys.stderr)
        return 1
    print("\n".join(report.lines()))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through main() in tests
    sys.exit(main())


__all__ = ["ADAPTERS", "DEFAULT_SECONDS", "STUTTER_MS", "Report", "build_parser", "main", "profile"]
