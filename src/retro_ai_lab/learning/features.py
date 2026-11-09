"""A situation as numbers a neural network can read (DESIGN.md §6.1).

The state encoder gives a learner a list of (variable, bucket) pairs. A table
looks the pairs up as a key; a network needs a vector of fixed length whose
slots always mean the same thing. The feature map hands every variable two
slots the first time it is seen, "present" (0 or 1), the bucket scaled by
``bins`` (fine: lanes, lives and speeds stay well apart) and the bucket
scaled over the whole range a 16-bit value can reach (coarse: positions and
scores still differ), and keeps the slots for good, so a variable that is
missing from a state reads as absent rather than as bucket 0, and a network
saved with the map keeps seeing the game the same way next session.

A full map used to drop every newcomer for good (T-908): the user's brain
had its slots taken by the first flags it met and never saw a position
value found later. Now a newcomer takes the slot of a variable that is
absent from the state being encoded, the one unseen longest; the slot's
old meaning is forgotten and the network relearns it, which beats never
seeing the newcomer. Only when every slot's holder is present in the state
is the newcomer dropped (counted in :attr:`dropped`).
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
SLOTS_PER_VARIABLE = 3
# Finished rows kept by situation (T-959): a training step encodes 65 situations from
# the replay memory, nearly all seen before; past this many the cache starts over.
ROW_CACHE_MOST = 20_000


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
        self._seen: dict[str, int] = {}  # id -> the fill it was last present in (T-908)
        self._fills = 0
        self._evicted = 0
        # parts -> the finished row, valid while no slot changes hands (T-959)
        self._rows: dict[tuple[tuple[str, int], ...], NDArray[np.float32]] = {}
        self._built = 0

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
        """Variables with a slot, in slot order (first seen first, until a slot changes hands)."""
        return tuple(sorted(self._slots, key=self._slots.__getitem__))

    @property
    def dropped(self) -> int:
        """Variables that found the map full with every holder present; left out."""
        return len(self._dropped)

    @property
    def evicted(self) -> int:
        """Slots that changed hands to a newcomer (T-908)."""
        return self._evicted

    @property
    def built(self) -> int:
        """Rows worked out part by part; the rest were read back from the cache (T-959)."""
        return self._built

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
        self._fills += 1
        fills = self._fills
        parts = state.parts
        cached = self._rows.get(parts)
        if cached is not None:
            # The same situation with the same slots gives the same row (T-959); only
            # the record of when each variable was last seen moves on.
            for vid, _ in parts:
                self._seen[vid] = fills
            row[:] = cached
            return
        self._built += 1
        present = {vid for vid, _ in parts}
        for vid, b in parts:
            self._seen[vid] = fills
            slot = self._slots.get(vid)
            if slot is None:
                slot = self._place(vid, present)
                if slot is None:
                    self._dropped.add(vid)
                    continue
                self._dropped.discard(vid)
            base = SLOTS_PER_VARIABLE * slot
            row[base] = 1.0
            row[base + 1] = max(-1.0, min(1.0, b / self._bins))  # fine: exact small values
            row[base + 2] = max(-1.0, min(1.0, b / self._span))  # coarse: the whole range
        if ROW_CACHE_MOST > 0:
            if len(self._rows) >= ROW_CACHE_MOST:
                self._rows.clear()
            self._rows[parts] = row.copy()

    def _place(self, vid: str, present: set[str]) -> int | None:
        """A slot for a newcomer: a free one, else an absent holder's; None when none."""
        self._rows.clear()  # every cached row was laid out without this newcomer (T-959)
        if len(self._slots) < self._width:
            slot = self._slots[vid] = len(self._slots)
            return slot
        absent = [old for old in self._slots if old not in present]
        if not absent:
            return None
        # Unseen longest; the lowest slot on a tie, so a map loaded from disk (where
        # nothing has been seen yet) hands slots over the same way as the live one did.
        old = min(absent, key=lambda o: (self._seen.get(o, 0), self._slots[o]))
        slot = self._slots.pop(old)
        self._slots[vid] = slot
        self._evicted += 1
        return slot

    # ------------------------------------------------------------ persistence

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA_VERSION,
            "width": self._width,
            "bins": self._bins,
            "ids": list(self.ids),
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
