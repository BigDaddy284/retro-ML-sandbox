"""Launch a built Retro AI Lab app and check it starts and exits cleanly.

Usage: python packaging/smoke_test.py <path-to-executable>

Runs the executable with ``--exit-after-ms`` so the real window is created
and torn down, and then with ``--version`` to check the CLI survived
packaging. Exit code 0 means both passed.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

LAUNCH_TIMEOUT_S = 120


def run(exe: Path, *args: str) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
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


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__)
        return 2
    exe = Path(argv[1])
    if not exe.exists():
        print(f"not found: {exe}")
        return 2

    window = run(exe, "--exit-after-ms", "1500")
    print(f"window run: exit {window.returncode}")
    if window.stderr.strip():
        print(window.stderr)
    if window.returncode != 0:
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
