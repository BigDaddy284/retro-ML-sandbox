"""The byte of the game's memory that moves as the camera does (T-1020).

The camera's move between two looks is measured off the screen: the shift under
which the new screen best matches the old one
(:func:`~retro_ai_lab.observation.tracking.estimate_scroll`). Where the screen
shows bare ground and one thing walking on it, that is one number for two
unknowns, and the measure takes the thing's own walk for part of the camera's
move; where it shows nothing that tells at all, or scenery that repeats, it
cannot say (T-1017). A game that scrolls keeps its own count of where its
camera is. Nobody tells the AI where: :class:`CameraMemory` is shown the memory
before and after each look together with the move the picture gave, and keeps
for every byte how often the byte changed by just that move in the looks read
as moving, and stayed put in the looks read as still. The camera's own byte
agrees in most of both, though the picture is wrong one look in five; a byte
that never changes agrees in none of the first, a clock in none of the second,
anything else in next to none of either. Once a byte has proved itself that way
it says what the camera's move was, for the tracker to take where the picture
cannot tell (:meth:`~retro_ai_lab.observation.tracking.ObjectTracker.update`).

Nothing here knows any game. Each way the camera can move (across, and up and
down) has its own byte, found by itself, or none.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from retro_ai_lab.core import ConfigurationError

NEEDS = 20  # looks read as moving, and as many read as still, before a byte is taken
AGREES = 0.6  # the share of each kind a byte must agree in
REMEMBERS = 400  # looks of a kind kept count of: past that, the counts are halved
HALF = 128  # a byte's change is counted the short way round: -127 to 128


def _memory(value: object, what: str) -> NDArray[np.int16]:
    if not isinstance(value, (bytes, bytearray)) or not value:
        raise ConfigurationError(f"The memory {what} a look must be bytes, at least one.")
    return np.frombuffer(bytes(value), np.uint8).astype(np.int16)


class CameraMemory:
    """Which byte of memory moves as the camera does, each way; and what it says."""

    def __init__(
        self, *, needs: int = NEEDS, agrees: float = AGREES, remembers: int = REMEMBERS
    ) -> None:
        if isinstance(needs, bool) or not isinstance(needs, int) or needs < 1:
            raise ConfigurationError("needs must be a whole number of looks, 1 or more.")
        if isinstance(agrees, bool) or not isinstance(agrees, (int, float)) or not 0 < agrees <= 1:
            raise ConfigurationError("agrees must be a share: above 0, at most 1.")
        if isinstance(remembers, bool) or not isinstance(remembers, int) or remembers < 2 * needs:
            raise ConfigurationError(
                "remembers must be a whole number of looks, at least twice what a byte needs."
            )
        self._needs = needs
        self._agrees = float(agrees)
        self._remembers = remembers
        self._size = 0
        # For each way (across, up and down): the looks read as moving and as still,
        # and for every byte how many of each it agreed in.
        self._looks = np.zeros((2, 2), np.int64)
        self._agreed: NDArray[np.int64] = np.zeros((2, 2, 0), np.int64)
        self._found: tuple[int | None, int | None] = (None, None)

    @property
    def found(self) -> tuple[int | None, int | None]:
        """The byte that moves as the camera does across, and the one for up and
        down: where in the memory it was shown, or None while none has proved itself."""
        return self._found

    def agreed(self, way: int) -> tuple[int, int] | None:
        """For the byte found for ``way`` (0 across, 1 up and down): the looks it
        agreed in and the looks it was judged over. None with no byte."""
        byte = self._found[way]
        if byte is None:
            return None
        return int(self._agreed[way, :, byte].sum()), int(self._looks[way].sum())

    def note(self, before: bytes, after: bytes, seen: tuple[int, int]) -> None:
        """One look: the memory before it and after it, and the camera's move as the
        picture alone had it. Only looks the picture could measure belong here: not
        one across which the game was put somewhere else."""
        change = self._change(before, after)
        if (
            not isinstance(seen, tuple)
            or len(seen) != 2
            or any(isinstance(v, bool) or not isinstance(v, int) for v in seen)
        ):
            raise ConfigurationError("The camera's move must be two whole numbers: across, down.")
        if change.size != self._size:
            # Another game, or this one read another way: what was learnt is void.
            self._size = change.size
            self._looks[:] = 0
            self._agreed = np.zeros((2, 2, change.size), np.int64)
        for way, moved in enumerate(seen):
            if not -HALF < moved <= HALF:
                continue  # more than a byte can show: nothing to compare
            kind = 0 if moved else 1
            self._looks[way, kind] += 1
            self._agreed[way, kind] += change == moved
            if self._looks[way, kind] > self._remembers:
                self._looks[way, kind] //= 2
                self._agreed[way, kind] //= 2
        self._found = (self._byte(0), self._byte(1))

    def expect(self, before: bytes, after: bytes) -> tuple[int | None, int | None]:
        """The camera's move from the memory before a look to the memory after it, by
        the bytes found: across and down, each None where no byte is known."""
        change = self._change(before, after)
        if change.size != self._size:
            return (None, None)
        x, y = self._found
        return (
            None if x is None else int(change[x]),
            None if y is None else int(change[y]),
        )

    def _change(self, before: bytes, after: bytes) -> NDArray[np.int16]:
        was, now = _memory(before, "before"), _memory(after, "after")
        if was.size != now.size:
            raise ConfigurationError(
                "The memory before a look and after it must be the same memory: "
                f"{was.size} bytes and {now.size}."
            )
        # The short way round a byte's roll-over: 250 to 6 is 12 on.
        return (now - was + HALF - 1) % 256 - (HALF - 1)

    def _byte(self, way: int) -> int | None:
        moving, still = (int(n) for n in self._looks[way])
        if moving < self._needs or still < self._needs:
            return None
        agreed = self._agreed[way]
        enough = (agreed[0] >= self._agrees * moving - 1e-9) & (
            agreed[1] >= self._agrees * still - 1e-9
        )
        if not enough.any():
            return None
        total = np.where(enough, agreed[0] + agreed[1], -1)
        return int(np.argmax(total))  # the first of equals


__all__ = ["AGREES", "NEEDS", "REMEMBERS", "CameraMemory"]
