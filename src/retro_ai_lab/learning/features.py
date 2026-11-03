"""A situation as numbers a neural network can read (DESIGN.md §6.1).

The state encoder gives a learner a list of (variable, bucket) pairs. A table
looks the pairs up as a key; a network needs a vector of fixed length whose
slots always mean the same thing. The feature map hands every variable two
slots the first time it is seen: "present" (0 or 1) and the bucket scaled
to [-1, 1], and keeps the slots for good, so a variable that is missing
from a state reads as absent rather than as bucket 0, and a network saved
with the map keeps seeing the game the same way next session.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
from numpy.typing import NDArray

from retro_ai_lab.core import ConfigurationError
from retro_ai_lab.learning.state import DEFAULT_BINS, State

SCHEMA_VERSION = 1
DEFAULT_WIDTH = 32  # believed variables the vector has room for
SLOTS_PER_VARIABLE = 2


class FeatureMap:
    def __init__(self, width: int = DEFAULT_WIDTH, bins: int = DEFAULT_BINS) -> None:
        if not isinstance(width, int) or width < 1:
            raise ConfigurationError("width must be at least 1.")
        if not isinstance(bins, int) or not 2 <= bins <= 64:
            raise ConfigurationError("bins must be between 2 and 64.")
        self._width = width
        self._bins = bins
        self._span = float(bins + 16)  # the largest bucket a 16-bit value can reach
        self._slots: dict[str, int] = {}
        self._dropped: set[str] = set()

    # ------------------------------------------------------------ queries

    @property
    def width(self) -> int:
        return self._width

    @property
    def bins(self) -> int:
        return self._bins

    @property
    def size(self) -> int:
        """Length of every vector, fixed from the start."""
        return SLOTS_PER_VARIABLE * self._width

    @property
    def ids(self) -> tuple[str, ...]:
        """Variables with a slot, in the order they were first seen."""
        return tuple(self._slots)

    @property
    def dropped(self) -> int:
        """Variables seen after the map was full; they are left out of every vector."""
        return len(self._dropped)

    def slot_of(self, variable_id: str) -> int:
        try:
            return self._slots[variable_id]
        except KeyError:
            raise ConfigurationError(f"No slot for variable {variable_id!r}.") from None

    # ------------------------------------------------------------ encoding

    def vector(self, state: State) -> NDArray[np.float32]:
        out = np.zeros(self.size, dtype=np.float32)
        self._fill(out, state)
        return out

    def batch(self, states: Iterable[State]) -> NDArray[np.float32]:
        rows = list(states)
        out = np.zeros((len(rows), self.size), dtype=np.float32)
        for row, state in zip(out, rows, strict=True):
            self._fill(row, state)
        return out

    def _fill(self, row: NDArray[np.float32], state: State) -> None:
        for vid, b in state.parts:
            slot = self._slots.get(vid)
            if slot is None:
                if len(self._slots) >= self._width:
                    self._dropped.add(vid)
                    continue
                slot = self._slots[vid] = len(self._slots)
            base = SLOTS_PER_VARIABLE * slot
            row[base] = 1.0
            row[base + 1] = max(-1.0, min(1.0, b / self._span))

    # ------------------------------------------------------------ persistence

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA_VERSION,
            "width": self._width,
            "bins": self._bins,
            "ids": list(self._slots),
            "dropped": sorted(self._dropped),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FeatureMap:
        if not isinstance(data, dict):
            raise ConfigurationError("Bad feature map data: not a mapping.")
        try:
            schema = int(data.get("schema", SCHEMA_VERSION))
            if schema > SCHEMA_VERSION:
                raise ConfigurationError(
                    f"This feature map is from a newer version (schema {schema})."
                )
            fm = cls(int(data.get("width", DEFAULT_WIDTH)), int(data.get("bins", DEFAULT_BINS)))
            ids = [str(i) for i in data.get("ids", [])]
            if len(set(ids)) != len(ids) or len(ids) > fm.width:
                raise ConfigurationError("Bad feature map data: slots are not unique or too many.")
            fm._slots = {vid: i for i, vid in enumerate(ids)}
            fm._dropped = {str(i) for i in data.get("dropped", [])}
            return fm
        except (TypeError, ValueError) as exc:
            raise ConfigurationError(f"Bad feature map data: {exc}") from exc


__all__ = ["DEFAULT_WIDTH", "SCHEMA_VERSION", "SLOTS_PER_VARIABLE", "FeatureMap"]
