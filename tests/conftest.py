"""Shared pytest configuration.

Real-emulator tests are only collected when the environment is configured for
them (see TESTING_STRATEGY.md §2.4); everywhere else they are deselected with a
recorded reason so the exclusion is visible in the run summary.
"""

from __future__ import annotations

import os

import pytest

REAL_EMULATOR_ENV = ("RETRO_AI_LAB_EMULATOR_PATH", "RETRO_AI_LAB_TEST_ROM")


def _real_emulator_configured() -> bool:
    return all(os.environ.get(name) for name in REAL_EMULATOR_ENV)


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if _real_emulator_configured():
        return
    reason = (
        "real emulator not configured: set "
        + " and ".join(REAL_EMULATOR_ENV)
        + " (see TESTING_STRATEGY.md §2.4)"
    )
    marker = pytest.mark.skip(reason=reason)
    for item in items:
        if "real_emulator" in item.keywords:
            item.add_marker(marker)
