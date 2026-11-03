"""Screen change detection: which parts of the picture moved between frames.

Mirrors :mod:`retro_ai_lab.observation.diff` for pixels. A frame is split
into a coarse grid of cells; each diff reports how many pixels changed in
total, the bounding box of the change, and the changed-pixel count per
cell. A :class:`ScreenTracker` accumulates per-cell activity over a run so
the understanding engine can tell static HUD areas from the playfield.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from retro_ai_lab.core import ConfigurationError, Frame

DEFAULT_GRID = (8, 6)  # columns, rows


@dataclass(frozen=True, slots=True)
class FrameDiff:
    before_number: int
    after_number: int
    width: int
    height: int
    changed_pixels: int
    bbox: tuple[int, int, int, int] | None  # x0, y0, x1, y1 inclusive; None if nothing changed
    cells: tuple[tuple[int, ...], ...]  # rows x columns of changed-pixel counts

    @property
    def is_empty(self) -> bool:
        return self.changed_pixels == 0

    @property
    def fraction_changed(self) -> float:
        return self.changed_pixels / (self.width * self.height)

    @property
    def columns(self) -> int:
        return len(self.cells[0]) if self.cells else 0

    @property
    def rows(self) -> int:
        return len(self.cells)

    def changed_cells(self) -> tuple[tuple[int, int], ...]:
        """(column, row) of every cell with at least one changed pixel."""
        return tuple((c, r) for r, row in enumerate(self.cells) for c, n in enumerate(row) if n > 0)


def to_array(frame: Frame) -> NDArray[np.uint8]:
    """View a frame as a (height, width, channels) uint8 array (no copy)."""
    return np.frombuffer(frame.pixels, dtype=np.uint8).reshape(
        frame.height, frame.width, frame.channels
    )


def _check_grid(grid: tuple[int, int], frame: Frame) -> None:
    cols, rows = grid
    if cols <= 0 or rows <= 0:
        raise ConfigurationError("grid must have positive columns and rows.")
    if cols > frame.width or rows > frame.height:
        raise ConfigurationError(
            f"grid {grid} is finer than the {frame.width}x{frame.height} frame."
        )


def cell_bounds(size: int, parts: int) -> list[tuple[int, int]]:
    """Split ``size`` pixels into ``parts`` nearly equal [start, stop) spans."""
    edges = np.linspace(0, size, parts + 1).astype(int)
    return [(int(edges[i]), int(edges[i + 1])) for i in range(parts)]


def diff_frames(
    before: Frame,
    after: Frame,
    *,
    threshold: int = 0,
    grid: tuple[int, int] = DEFAULT_GRID,
) -> FrameDiff:
    """Compare two frames pixel by pixel.

    A pixel counts as changed when any channel differs by more than
    ``threshold`` (0 means any difference). Frames must share size and
    channels.
    """
    if (before.width, before.height, before.channels) != (
        after.width,
        after.height,
        after.channels,
    ):
        raise ConfigurationError(
            f"Frames differ in shape: {before.size}x{before.channels} "
            f"vs {after.size}x{after.channels}."
        )
    if not 0 <= threshold <= 255:
        raise ConfigurationError("threshold must be within 0..255.")
    _check_grid(grid, before)
    cols, rows = grid
    empty_cells = tuple(tuple(0 for _ in range(cols)) for _ in range(rows))
    if before.pixels == after.pixels:
        return FrameDiff(
            before.number, after.number, before.width, before.height, 0, None, empty_cells
        )
    a = to_array(before).astype(np.int16)
    b = to_array(after).astype(np.int16)
    changed = (np.abs(a - b) > threshold).any(axis=2)  # height x width bools
    total = int(changed.sum())
    if total == 0:
        return FrameDiff(
            before.number, after.number, before.width, before.height, 0, None, empty_cells
        )
    ys, xs = np.nonzero(changed)
    bbox = (int(np.min(xs)), int(np.min(ys)), int(np.max(xs)), int(np.max(ys)))
    cells = tuple(
        tuple(int(changed[y0:y1, x0:x1].sum()) for x0, x1 in cell_bounds(before.width, cols))
        for y0, y1 in cell_bounds(before.height, rows)
    )
    return FrameDiff(before.number, after.number, before.width, before.height, total, bbox, cells)


class ScreenTracker:
    """Accumulates how often each grid cell changes across a run."""

    def __init__(self, *, threshold: int = 0, grid: tuple[int, int] = DEFAULT_GRID) -> None:
        if not 0 <= threshold <= 255:
            raise ConfigurationError("threshold must be within 0..255.")
        if grid[0] <= 0 or grid[1] <= 0:
            raise ConfigurationError("grid must have positive columns and rows.")
        self._threshold = threshold
        self._grid = grid
        self._previous: Frame | None = None
        self._observed = 0
        self._activity: NDArray[np.int64] = np.zeros((grid[1], grid[0]), dtype=np.int64)
        self._pixels: NDArray[np.int64] = np.zeros((grid[1], grid[0]), dtype=np.int64)

    @property
    def grid(self) -> tuple[int, int]:
        return self._grid

    @property
    def observed(self) -> int:
        return self._observed

    @property
    def transitions(self) -> int:
        return max(0, self._observed - 1)

    def reset(self) -> None:
        self._previous = None
        self._observed = 0
        self._activity[:] = 0
        self._pixels[:] = 0

    def observe(self, frame: Frame) -> FrameDiff | None:
        """Feed the next frame; returns its diff from the previous one, if any."""
        if not isinstance(frame, Frame):
            raise ConfigurationError("ScreenTracker.observe needs a Frame.")
        if self._previous is not None and (
            self._previous.width,
            self._previous.height,
            self._previous.channels,
        ) != (frame.width, frame.height, frame.channels):
            # Resolution changed (new game, new mode): start over.
            self.reset()
        self._observed += 1
        if self._previous is None:
            self._previous = frame
            return None
        diff = diff_frames(self._previous, frame, threshold=self._threshold, grid=self._grid)
        cells = np.array(diff.cells, dtype=np.int64)
        self._activity += cells > 0
        self._pixels += cells
        self._previous = frame
        return diff

    def activity(self, column: int, row: int) -> float:
        """Fraction of transitions in which the cell changed."""
        cols, rows = self._grid
        if not (0 <= column < cols and 0 <= row < rows):
            raise ConfigurationError(f"cell ({column}, {row}) is outside the {self._grid} grid.")
        if self.transitions == 0:
            return 0.0
        return float(self._activity[row, column]) / self.transitions

    def activity_map(self) -> tuple[tuple[float, ...], ...]:
        """rows x columns of :meth:`activity`."""
        cols, rows = self._grid
        return tuple(tuple(self.activity(c, r) for c in range(cols)) for r in range(rows))

    def classify(
        self, *, static_below: float = 0.05, busy_above: float = 0.5
    ) -> dict[str, tuple[tuple[int, int], ...]]:
        """Split cells into ``static``, ``busy`` and ``active`` (in between)."""
        if not 0 <= static_below <= busy_above <= 1:
            raise ConfigurationError("need 0 <= static_below <= busy_above <= 1.")
        groups: dict[str, list[tuple[int, int]]] = {"static": [], "active": [], "busy": []}
        cols, rows = self._grid
        for r in range(rows):
            for c in range(cols):
                a = self.activity(c, r)
                key = "static" if a < static_below else "busy" if a >= busy_above else "active"
                groups[key].append((c, r))
        return {k: tuple(v) for k, v in groups.items()}


__all__ = [
    "DEFAULT_GRID",
    "FrameDiff",
    "ScreenTracker",
    "cell_bounds",
    "diff_frames",
    "to_array",
]
