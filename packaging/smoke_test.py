"""Launch a built Retro AI Lab app and check it starts and exits cleanly.

Usage: python packaging/smoke_test.py <path-to-executable> [--starting-window]

Runs the executable with ``--exit-after-ms`` so the real window is created
and torn down, and then with ``--version`` to check the CLI survived
packaging. The window run writes its log to a fresh folder, and the log must
name every stage of the start in order and leave no trace of a start that got
stuck or crashed (T-995). With ``--starting-window`` the log must also say that
the built app's "starting" window was showing while it loaded and was closed
once the main window was up (T-1002). Exit code 0 means all of it passed.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

LAUNCH_TIMEOUT_S = 120
STAGES = (
    "starting",
    "Loading Qt",
    "Loading the app's own code",
    "Starting the app",
    "Creating the Qt application",
    "Qt application created",
    "Building the main window",
    "Main window shown",
    "Event loop finished",
)
STARTING_WINDOW = "--starting-window"
#: Where the starting window's two lines belong: each after the stage named with it.
WINDOW_LINES = (
    ("Loading Qt", "The starting window is showing"),
    ("Main window shown", "The starting window is closed"),
)


def stages(*, starting_window: bool = False) -> tuple[str, ...]:
    """The lines a start must log, in order."""
    after = dict(WINDOW_LINES) if starting_window else {}
    lines: list[str] = []
    for stage in STAGES:
        lines.append(stage)
        if stage in after:
            lines.append(after[stage])
    return tuple(lines)


def run(exe: Path, *args: str, data_dir: Path | None = None) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    if data_dir is not None:
        env["RETRO_AI_LAB_DATA_DIR"] = str(data_dir)
    if sys.platform != "win32":
        env.setdefault("QT_QPA_PLATFORM", "offscreen")
    return subprocess.run(
        [str(exe), *args],
        capture_output=True,
        text=True,
        timeout=LAUNCH_TIMEOUT_S,
        env=env,
        check=False,
    )


def check_log(logs: Path, *, starting_window: bool = False) -> str:
    """What is wrong with the log the window run left, or "" when nothing is."""
    log_file = logs / "retro-ai-lab.log"
    if not log_file.exists():
        return f"no log at {log_file}"
    text = log_file.read_text(encoding="utf-8", errors="replace")
    print(text)
    at = 0
    for stage in stages(starting_window=starting_window):
        found = text.find(stage, at)
        if found == -1:
            return f"the log does not reach {stage!r} in order"
        at = found + len(stage)
    traces = [p for p in logs.glob("start-*.txt") if p.read_text(errors="replace").strip()]
    if traces:
        return "a trace was left behind:\n" + traces[0].read_text(errors="replace")
    return ""


def main(argv: list[str]) -> int:
    starting_window = STARTING_WINDOW in argv[1:]
    paths = [arg for arg in argv[1:] if arg != STARTING_WINDOW]
    if len(paths) != 1:
        print(__doc__)
        return 2
    exe = Path(paths[0])
    if not exe.exists():
        print(f"not found: {exe}")
        return 2

    with tempfile.TemporaryDirectory() as folder:
        window = run(exe, "--exit-after-ms", "1500", data_dir=Path(folder))
        print(f"window run: exit {window.returncode}")
        if window.stderr.strip():
            print(window.stderr)
        problem = check_log(Path(folder) / "logs", starting_window=starting_window)
        if window.returncode != 0 or problem:
            print(problem)
            return 1

    version = run(exe, "--version")
    print(f"version run: exit {version.returncode}, stdout={version.stdout.strip()!r}")
    # A windowed build has no stdout on Windows, so only the exit code is checked there.
    if version.returncode != 0:
        return 1
    print("smoke test passed")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
