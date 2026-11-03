"""Shared pytest configuration.

Real-emulator tests are only collected when the environment is configured for
them (see TESTING_STRATEGY.md §2.4); everywhere else they are deselected with a
recorded reason so the exclusion is visible in the run summary.
"""

from __future__ import annotations

import os

import pytest

REAL_EMULATOR_ENV = ("RETRO_AI_LAB_EMULATOR_PATH", "RETRO_AI_LAB_TEST_ROM")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    for item in items:
        marker = item.get_closest_marker("real_emulator")
        if marker is None:
            continue
        env = tuple(marker.kwargs.get("env", REAL_EMULATOR_ENV))
        if all(os.environ.get(name) for name in env):
            continue
        reason = (
            "real emulator not configured: set "
            + " and ".join(env)
            + " (see TESTING_STRATEGY.md §2.4)"
        )
        item.add_marker(pytest.mark.skip(reason=reason))
