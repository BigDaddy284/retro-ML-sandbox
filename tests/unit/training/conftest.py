"""The research tests run over the training tests' connected lab, and the search
tests over the trial tests' practice game."""

from tests.unit.training.test_service import lab  # noqa: F401  (pytest fixture)
from tests.unit.training.test_trial import hopper  # noqa: F401  (pytest fixture)
