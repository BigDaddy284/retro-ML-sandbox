"""Packages must be importable in any order (no import cycles)."""

from __future__ import annotations

import subprocess
import sys

import pytest


@pytest.mark.parametrize(
    "module",
    [
        "retro_ai_lab.understanding",
        "retro_ai_lab.rewards",
        "retro_ai_lab.services",
        "retro_ai_lab.ui",
    ],
)
def test_package_imports_cleanly_on_its_own(module: str) -> None:
    result = subprocess.run(
        [sys.executable, "-c", f"import {module}"], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
