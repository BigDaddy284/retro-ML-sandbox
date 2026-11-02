"""T-001: package version is present, valid and consistent."""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import retro_ai_lab

# PEP 440 "canonical" public version (no local segment), enough for our use.
PEP440 = re.compile(
    r"^([1-9][0-9]*!)?(0|[1-9][0-9]*)(\.(0|[1-9][0-9]*))*"
    r"((a|b|rc)(0|[1-9][0-9]*))?(\.post(0|[1-9][0-9]*))?(\.dev(0|[1-9][0-9]*))?$"
)


def test_version_is_a_string() -> None:
    assert isinstance(retro_ai_lab.__version__, str)
    assert retro_ai_lab.__version__


def test_version_is_valid_pep440() -> None:
    assert PEP440.match(retro_ai_lab.__version__), retro_ai_lab.__version__


def test_version_matches_pyproject() -> None:
    root = Path(__file__).resolve().parents[2]
    with (root / "pyproject.toml").open("rb") as fh:
        data = tomllib.load(fh)
    assert data["project"]["version"] == retro_ai_lab.__version__


def test_package_exports_version_only() -> None:
    assert retro_ai_lab.__all__ == ["__version__"]
