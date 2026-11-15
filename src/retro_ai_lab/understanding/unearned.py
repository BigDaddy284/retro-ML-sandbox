"""What goes up whatever is pressed is not progress (T-1035).

The top byte of a game's clock, a count of seconds, a tune's place: each steps
up now and then and never comes down, which is everything the classifier asks
of progress. Paid as progress it pays best for standing still. Nothing in such
a value's own statistics tells it from a checkpoint; whether it rises with any
button held does, and the game can be asked
(:func:`retro_ai_lab.training.earned.rose_anyway`).

One answer is not enough. Another move can get the player killed and stop a
clock, and then a rise that was nobody's doing looks earned; on the practice
game three answers in thirty were wrong that way, and none the other way in
fifty-two. :class:`Unearned` counts the answers for each value and says
something only where it rose anyway at two rises or more and at half or more
of those asked about. It also says when to stop asking: a value whose rises
are earned costs a few plays and is then left alone.
"""

from __future__ import annotations

from retro_ai_lab.core import ConfigurationError

SAY_AFTER = 2  # rises that came whatever was pressed, before anything is said
SHARE = 0.5  # ... and of the rises asked about, this share or more
ASK_UP_TO = 6  # answers about one value; after that it is left alone


class Unearned:
    """Counts, for each value, the rises asked about and those that came anyway."""

    def __init__(self) -> None:
        self._asked: dict[str, int] = {}
        self._anyway: dict[str, int] = {}

    def note(self, var_id: str, anyway: bool) -> None:
        """Take in one answer of the game about a rise of this value."""
        _name(var_id)
        if not isinstance(anyway, bool):
            raise ConfigurationError("anyway must be true or false.")
        self._asked[var_id] = self._asked.get(var_id, 0) + 1
        if anyway:
            self._anyway[var_id] = self._anyway.get(var_id, 0) + 1

    def counts(self, var_id: str) -> tuple[int, int]:
        """(rises asked about, rises that came whatever was pressed)."""
        _name(var_id)
        return (self._asked.get(var_id, 0), self._anyway.get(var_id, 0))

    def verdict(self, var_id: str) -> bool:
        """It goes up whatever is pressed: at two rises or more, and at half or more
        of those the game was asked about."""
        asked, anyway = self.counts(var_id)
        return anyway >= SAY_AFTER and anyway >= SHARE * asked

    def wants(self, var_id: str) -> bool:
        """Whether another answer about this value would still tell something."""
        return not self.verdict(var_id) and self.counts(var_id)[0] < ASK_UP_TO


def _name(var_id: object) -> None:
    if not var_id or not isinstance(var_id, str):
        raise ConfigurationError("A value is named by its id.")


__all__ = ["ASK_UP_TO", "SAY_AFTER", "SHARE", "Unearned"]
