"""Variable candidate finder: turn change statistics into first guesses.

Given how every byte behaved while someone played (:class:`ChangeTracker`),
propose which bytes might be a score, a counter, lives, a position and so
on, each with plain-words evidence and a modest weight. Nothing here is
certain: candidates land in :class:`Knowledge` at Low or Medium confidence
and later tasks (input linking, event detection, a person's say-so) raise
or sink them.

The rules are deliberately simple and game-agnostic (DESIGN.md §3.6: no
pre-built game profiles):

* a byte that only ever goes up by one, every frame, is a counter;
* two neighbouring bytes where the upper one changes only when the lower
  one rolls over are one 16-bit value;
* a 16-bit value that only goes up, in uneven steps, looks like a score;
* a byte that nearly only goes down, rarely, through a handful of values,
  looks like lives (it jumps back up on a restart);
* exactly two values, rarely switching, looks like a flag;
* a handful of values, mostly climbing, rarely changing, looks like
  progress (level, lap, checkpoint);
* always climbing in small uneven steps (rolling over) looks like a position;
* a small set of values moving both ways looks like a speed or similar gauge;
* bytes that never change, or change almost every frame in no pattern, are
  skipped as constants or noise.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from retro_ai_lab.core import ConfigurationError
from retro_ai_lab.observation.diff import ByteStats, ChangeTracker
from retro_ai_lab.understanding.knowledge import (
    Encoding,
    Evidence,
    Knowledge,
    Variable,
    VariableKind,
)

MIN_TRANSITIONS = 30
NOISE_RATE = 0.9
RARE_RATE = 0.1
FEW_VALUES = 16
BOUNCE_UPS = 4  # a "lives" byte restored this often is judged on its ups against its downs
LIVES_SAMPLES = 100  # a "lives" byte is judged on how often it changes after this many steps


@dataclass(frozen=True, slots=True)
class Candidate:
    region: str
    offset: int
    encoding: Encoding
    kind: VariableKind
    evidence: tuple[Evidence, ...]

    @property
    def weight(self) -> float:
        return sum(e.weight for e in self.evidence)

    @property
    def width(self) -> int:
        return self.encoding.width

    def covers(self, offset: int) -> bool:
        return self.offset <= offset < self.offset + self.width


def _is_noise(st: ByteStats) -> bool:
    return st.change_rate >= NOISE_RATE and not st.mostly_up


def _is_counter(st: ByteStats) -> bool:
    return st.unit_steps >= 0.95 * st.changes and st.change_rate >= 0.95


def _is_pair(low: ByteStats, high: ByteStats) -> bool:
    """``high`` is the upper byte of a 16-bit value whose lower byte is ``low``.

    Every change of the upper byte should coincide with the lower byte
    rolling over, except for resets (both bytes dropping together).
    """
    if low.wraps < 1 or high.changes < 1 or _is_noise(low):
        return False
    stray = high.changes - high.carry_ins
    return high.carry_ins >= 0.6 * high.changes and stray <= high.resets


def lives_bounce(st: ByteStats) -> bool:
    """A life count does not bounce: it is lost one at a time and restored on a restart.

    True when the byte has gone up at least ``BOUNCE_UPS`` times and down fewer
    than twice that often (T-864); such a byte is not a life count whatever it
    looked like when first proposed.
    """
    return st.ups >= BOUNCE_UPS and st.downs < 2 * st.ups


def lives_busy(st: ByteStats) -> bool:
    """A life count changes rarely: more than one change in ten steps is no life count (T-865).

    Judged only after ``LIVES_SAMPLES`` observed steps, so a lively first few
    seconds do not condemn a byte. Progress (T-866) and flags (T-869) are held
    to the same rule.
    """
    return st.samples >= LIVES_SAMPLES and st.change_rate > RARE_RATE


def flag_doubt(st: ByteStats) -> str | None:
    """Why this byte is not a flag ("busy"), or None when it may be (T-869).

    A flag worth an event (game over, a menu, a pause) switches rarely; a
    two-value byte toggling every few steps is an animation or a "jumping" bit.
    """
    return "busy" if lives_busy(st) else None


def lives_doubt(st: ByteStats) -> str | None:
    """Why this byte is not a life count ("busy" or "bounces"), or None when it may be."""
    if lives_busy(st):
        return "busy"
    if lives_bounce(st):
        return "bounces"
    return None


def progress_doubt(st: ByteStats) -> str | None:
    """Why this byte is not progress ("busy" or "bounces"), or None when it may be (T-866).

    Progress (a level, lap or checkpoint) steps up rarely and only comes down
    on a restart, so a byte that changes more than once in ten steps, or has
    gone down at least as often as up, is not progress.
    """
    if lives_busy(st):
        return "busy"
    if st.ups >= BOUNCE_UPS and st.downs >= st.ups:
        return "bounces"
    return None


def _classify_single(st: ByteStats) -> tuple[VariableKind, tuple[Evidence, ...]] | None:
    """Kind and evidence for one 8-bit byte, or None to skip it."""
    if st.constant or _is_noise(st):
        return None
    values = len(st.distinct)
    rare = st.change_rate <= RARE_RATE
    if st.mostly_up:
        if _is_counter(st):
            return VariableKind.COUNTER, (
                Evidence("monotonic", "Only ever goes up.", 2.0),
                Evidence("every-frame", "Changes every frame.", 1.0),
                Evidence("unit-steps", "Always by exactly one.", 1.0),
            )
        if values == 2:
            if not rare:  # a blinking bit is no flag (T-869), and nothing else fits
                return None
            return VariableKind.FLAG, (
                Evidence("two-values", "Only ever two different values.", 1.5),
                Evidence("rare", "Hardly ever changes.", 0.5),
            )
        if values <= FEW_VALUES and rare:
            return VariableKind.PROGRESS, (
                Evidence("rare", "Hardly ever changes.", 0.5),
                Evidence("few-values", f"Only {values} different values seen.", 1.0),
                Evidence("mostly-up", "Nearly always goes up.", 1.0),
            )
        if st.max_step <= 8 and st.unit_steps < st.changes:
            return VariableKind.POSITION, (
                Evidence("monotonic", "Keeps climbing and rolls over.", 1.0),
                Evidence("varied-steps", "Moves by small, varying amounts.", 1.0),
                Evidence("frequent", "Changes often.", 0.5),
            )
        return VariableKind.SCORE, (
            Evidence("monotonic", "Only ever goes up.", 2.0),
            Evidence("irregular", "Goes up by different amounts.", 1.0),
        )
    if values == 2:
        if not rare:  # a blinking bit is no flag (T-869), and nothing else fits
            return None
        return VariableKind.FLAG, (
            Evidence("two-values", "Only ever two different values.", 1.5),
            Evidence("rare", "Switches now and then.", 0.5),
        )
    if st.downs > 0 and st.downs >= 2 * st.ups and values <= FEW_VALUES and rare:
        return VariableKind.LIVES, (
            Evidence("mostly-down", "Nearly always goes down.", 1.5),
            Evidence("few-values", f"Only {values} different values seen.", 1.0),
            Evidence("rare", "Changes rarely.", 0.5),
        )
    if values <= FEW_VALUES and rare and st.ups >= st.downs:
        return VariableKind.PROGRESS, (
            Evidence("rare", "Hardly ever changes.", 0.5),
            Evidence("few-values", f"Only {values} different values seen.", 1.0),
            Evidence("mostly-up", "Goes up more than down.", 1.0),
        )
    if values <= FEW_VALUES and not rare:
        return VariableKind.SPEED, (
            Evidence("both-ways", "Goes up and down.", 0.5),
            Evidence("few-values", f"Only {values} different values seen.", 1.0),
            Evidence("frequent", "Changes often.", 0.5),
        )
    return VariableKind.UNKNOWN, (
        Evidence("changes", "Changes sometimes, in no clear pattern yet.", 1.0),
    )


def _classify_pair(low: ByteStats, high: ByteStats) -> tuple[VariableKind, tuple[Evidence, ...]]:
    pair = Evidence("16-bit", "Two bytes count together as one number.", 1.0)
    if low.mostly_up:
        if _is_counter(low):
            return VariableKind.COUNTER, (
                Evidence("monotonic", "Only ever goes up.", 2.0),
                Evidence("every-frame", "Changes every frame.", 1.0),
                Evidence("unit-steps", "Always by exactly one.", 1.0),
                pair,
            )
        return VariableKind.SCORE, (
            Evidence("monotonic", "Only ever goes up.", 2.0),
            Evidence("irregular", "Goes up by different amounts.", 1.0),
            pair,
        )
    return VariableKind.UNKNOWN, (
        Evidence("changes", "A number that moves both ways.", 1.0),
        pair,
    )


def find_candidates(
    tracker: ChangeTracker, *, min_transitions: int = MIN_TRANSITIONS
) -> tuple[Candidate, ...]:
    """Propose variables from what the tracker has seen so far, lowest offset first."""
    if min_transitions < 1:
        raise ConfigurationError("min_transitions must be at least 1.")
    if tracker.transitions < min_transitions:
        return ()
    stats: Mapping[tuple[str, int], ByteStats] = tracker.all_stats()
    out: list[Candidate] = []
    taken: set[tuple[str, int]] = set()
    for (region, offset), st in sorted(stats.items()):
        if (region, offset) in taken:
            continue
        upper = stats.get((region, offset + 1))
        if upper is not None and (region, offset + 1) not in taken and _is_pair(st, upper):
            kind, evidence = _classify_pair(st, upper)
            out.append(Candidate(region, offset, Encoding.U16LE, kind, evidence))
            taken.update({(region, offset), (region, offset + 1)})
            continue
        single = _classify_single(st)
        if single is None:
            continue
        kind, evidence = single
        out.append(Candidate(region, offset, Encoding.U8, kind, evidence))
        taken.add((region, offset))
    return tuple(out)


def propose_candidates(
    knowledge: Knowledge, candidates: tuple[Candidate, ...]
) -> tuple[Variable, ...]:
    """Add candidates the knowledge has no opinion about yet. Returns what was added.

    A location already covered by any variable, confirmed, rejected or
    merely proposed, is left alone: the discovery engine never argues with
    a person and never duplicates itself.
    """
    added: list[Variable] = []
    for c in candidates:
        if any(_overlaps(v, c) for v in knowledge):
            continue
        added.append(
            knowledge.propose(
                c.region, c.offset, encoding=c.encoding, kind=c.kind, evidence=c.evidence
            )
        )
    return tuple(added)


def _overlaps(v: Variable, c: Candidate) -> bool:
    if v.region != c.region:
        return False
    return v.offset < c.offset + c.width and c.offset < v.offset + v.width


__all__ = [
    "BOUNCE_UPS",
    "FEW_VALUES",
    "LIVES_SAMPLES",
    "MIN_TRANSITIONS",
    "NOISE_RATE",
    "RARE_RATE",
    "Candidate",
    "find_candidates",
    "flag_doubt",
    "lives_bounce",
    "lives_busy",
    "lives_doubt",
    "progress_doubt",
    "propose_candidates",
]
