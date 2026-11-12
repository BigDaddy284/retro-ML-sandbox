"""Game observation: frames, memory snapshots, diffs and input logs."""

from retro_ai_lab.observation.diff import (
    ByteChange,
    ByteStats,
    ChangeTracker,
    SnapshotDiff,
    diff_snapshots,
)
from retro_ai_lab.observation.inputs import HeldSpan, InputEvent, InputLog
from retro_ai_lab.observation.memory import MemoryCapture, MemorySnapshot, SnapshotRing
from retro_ai_lab.observation.screen import FrameDiff, ScreenTracker, diff_frames
from retro_ai_lab.observation.search import Match, ScanKind, ValueSearch, parse_number
from retro_ai_lab.observation.sprites import SpriteFinder, SpriteMatch, SpriteTag, SpriteTemplate

__all__ = [
    "ByteChange",
    "ByteStats",
    "ChangeTracker",
    "FrameDiff",
    "HeldSpan",
    "InputEvent",
    "InputLog",
    "Match",
    "MemoryCapture",
    "MemorySnapshot",
    "ScanKind",
    "ScreenTracker",
    "SnapshotDiff",
    "SnapshotRing",
    "SpriteFinder",
    "SpriteMatch",
    "SpriteTag",
    "SpriteTemplate",
    "ValueSearch",
    "diff_frames",
    "diff_snapshots",
    "parse_number",
]
