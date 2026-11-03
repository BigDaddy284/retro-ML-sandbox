"""Reusable test support: adapter contract tests and fakes.

Importable by the test suite and by demo/diagnostic code; it must not pull in
Qt or heavy ML dependencies.
"""

from retro_ai_lab.testing.contract import AdapterContract

__all__ = ["AdapterContract"]
