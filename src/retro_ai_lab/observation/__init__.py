"""Game observation: frames, memory snapshots, diffs and input logs."""

from retro_ai_lab.observation.diff import (
    ByteChange,
    ByteStats,
    ChangeTracker,
    SnapshotDiff,
    diff_snapshots,
)
from retro_ai_lab.observation.memory import MemoryCapture, MemorySnapshot, SnapshotRing
from retro_ai_lab.observation.screen import FrameDiff, ScreenTracker, diff_frames

__all__ = [
    "ByteChange",
    "ByteStats",
    "ChangeTracker",
    "FrameDiff",
    "MemoryCapture",
    "MemorySnapshot",
    "ScreenTracker",
    "SnapshotDiff",
    "SnapshotRing",
    "diff_frames",
    "diff_snapshots",
]
