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
from retro_ai_lab.observation.diff import SMALL_HOP, ByteStats, ChangeTracker
from retro_ai_lab.understanding.knowledge import (
    INPUT_LINKED,
    MOVER,
    Encoding,
    Evidence,
    Knowledge,
    Variable,
    VariableKind,
    rejected_by_ai,
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
    changes: int = 0  # how many changes of the byte the verdict rests on (T-917)

    @property
    def weight(self) -> float:
        return sum(e.weight for e in self.evidence)

    @property
    def width(self) -> int:
        return self.encoding.width

    def covers(self, offset: int) -> bool:
        return self.offset <= offset < self.offset + self.width


def _is_noise(st: ByteStats) -> bool:
    # A walking enemy changes every step too, by small hops; noise jumps about (T-914).
    return st.change_rate >= NOISE_RATE and not st.mostly_up and not st.smooth


def _is_mover(st: ByteStats) -> bool:
    """Moves by small hops (one way or both), enough to be something on screen.

    A respawn or a slot taken by a new thing is one big hop; ``smooth`` allows a few.
    Every hop exactly one is a slow timer, not a thing moving (as the climbing
    position rule asks for varied steps), so at least one hop must be larger.
    A thing that is on screen only now and then (T-918: the user's enemies, a few
    seconds of every go) changes rarely over a session, so a rare byte counts
    when it has moved through more values than a life count or a level could
    have and changed enough times to be judged.
    """
    if not st.smooth or len(st.distinct) <= 2 or st.unit_hops >= st.changes:
        return False
    if st.change_rate > RARE_RATE:
        return True
    return st.changes >= MIN_TRANSITIONS and len(st.distinct) > FEW_VALUES


def _is_counter(st: ByteStats) -> bool:
    """Ticks by one nearly every observation, up or down (T-914 adds the down)."""
    ticks = max(st.unit_steps, st.unit_hops)
    return ticks >= 0.95 * st.changes and st.change_rate >= 0.95


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
        if st.min_hop == st.max_hop and st.change_rate >= NOISE_RATE:
            # A frame counter read every four frames steps by four every step (T-921):
            # the same hop every time is a clock, not something moving about.
            return VariableKind.COUNTER, (
                Evidence("monotonic", "Only ever goes up.", 2.0),
                Evidence("every-frame", "Changes every frame.", 1.0),
                Evidence("unit-steps", f"Always by exactly {st.max_hop}.", 1.0),
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
    if _is_counter(st):  # a timer ticking down every frame (T-914)
        return VariableKind.COUNTER, (
            Evidence("monotonic", "Only ever goes down.", 2.0),
            Evidence("every-frame", "Changes every frame.", 1.0),
            Evidence("unit-steps", "Always by exactly one.", 1.0),
        )
    if _is_mover(st):
        # An enemy, a platform, a ball: a number that moves in small steps, often. It is
        # a position of something, not of the player (T-911): the input link says whose
        # it is, and until then the brain still gets to watch it.
        return VariableKind.POSITION, (
            Evidence(MOVER, "Moves in small steps, often.", 1.5),
            Evidence("many-values", f"{values} different values seen.", 0.5),
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
            out.append(Candidate(region, offset, Encoding.U16LE, kind, evidence, st.changes))
            taken.update({(region, offset), (region, offset + 1)})
            continue
        single = _classify_single(st)
        if single is None:
            continue
        kind, evidence = single
        out.append(Candidate(region, offset, Encoding.U8, kind, evidence, st.changes))
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
                c.region,
                c.offset,
                encoding=c.encoding,
                kind=c.kind,
                evidence=c.evidence,
                judged=c.changes,
            )
        )
    return tuple(added)


# The evidence kinds the classifier itself writes; a rethought guess loses these and
# takes the fresh ones, keeping what other findings attached (T-916).
CLASSIFIER_KINDS = frozenset(
    {
        "16-bit",
        "both-ways",
        "changes",
        "every-frame",
        "few-values",
        "frequent",
        "irregular",
        "many-values",
        "monotonic",
        "mostly-down",
        "mostly-up",
        "rare",
        "two-values",
        "unit-steps",
        "varied-steps",
        MOVER,
    }
)
# Evidence that placed the guess by a person's word ("told", T-901) or by the AI's own
# restart finding ("restarts", T-904): both outrank the statistics, so the guess holds.
HELD_KINDS = frozenset({"told", "restarts"})


def rethink_candidates(
    knowledge: Knowledge, candidates: tuple[Candidate, ...]
) -> tuple[Variable, ...]:
    """Change the AI's mind about its own guesses whose statistics now say another kind.

    A byte is judged at the first review after it changes, often on a handful of
    changes: on the user's game an enemy slot became "a speed" on five values and
    stayed so for 77 goes, so the mover rule never ran on it (T-916). At every
    review a guess the AI made itself takes the classifier's current kind when
    that differs, with the classifier's fresh evidence; evidence other findings
    attached (the input link, "played", the long look's) is kept and the old
    kind's zero-weight doubts go. A person's word is never argued with: a
    confirmed, named or person-rejected guess holds, as does one placed by a
    person's "I just …" or by the restart finding, and a position the input
    link found. A guess the AI dropped stays dropped while the classifier
    repeats the kind it dropped, and comes back as the other kind. The same
    kind for other reasons counts as another verdict (T-922): a position by
    the climbing rule and one by the mover rule share a kind, but only the
    second is "something moving" to the brain. A guess is rethought only
    once the byte has changed at least twice as often as when it was last
    judged (T-917): on the user's game a byte on the edge of a rule flipped
    kind at every review, and with it the brain's seats; doubling bounds the
    rethinks of a byte to a handful over a session and lets each verdict
    rest on twice the evidence of the last. A guess with no record (an older
    file, a planted guess) is rethought on the first differing verdict.
    Returns the variables whose kind changed.
    """
    at: dict[tuple[str, int, Encoding], Variable] = {
        (v.region, v.offset, v.encoding): v for v in knowledge
    }
    changed: list[Variable] = []
    for c in candidates:
        v = at.get((c.region, c.offset, c.encoding))
        if v is None or c.changes < 2 * v.judged or not _rethinkable(v):
            continue
        if v.kind is c.kind and _reasons(v.evidence) == _reasons(c.evidence):
            continue  # the same verdict for the same reasons: nothing to rethink
        kept = tuple(e for e in v.evidence if e.kind not in CLASSIFIER_KINDS and e.weight > 0)
        changed.append(
            knowledge.update(
                v.id,
                kind=c.kind,
                rejected=False,
                evidence=(*c.evidence, *kept),
                judged=c.changes,
            )
        )
    return tuple(changed)


def _reasons(evidence: tuple[Evidence, ...]) -> frozenset[str]:
    """The classifier's grounds for a verdict (T-922): a position by the climbing rule
    and one by the mover rule share a kind but not a meaning."""
    return frozenset(e.kind for e in evidence if e.kind in CLASSIFIER_KINDS)


def _rethinkable(v: Variable) -> bool:
    if v.confirmed or v.name or (v.rejected and not rejected_by_ai(v)):
        return False
    kinds = {e.kind for e in v.evidence}
    if kinds & HELD_KINDS:
        return False
    return not (v.kind is VariableKind.POSITION and INPUT_LINKED in kinds)


def _overlaps(v: Variable, c: Candidate) -> bool:
    if v.region != c.region:
        return False
    return v.offset < c.offset + c.width and c.offset < v.offset + v.width


__all__ = [
    "BOUNCE_UPS",
    "CLASSIFIER_KINDS",
    "FEW_VALUES",
    "HELD_KINDS",
    "LIVES_SAMPLES",
    "MIN_TRANSITIONS",
    "MOVER",
    "NOISE_RATE",
    "RARE_RATE",
    "SMALL_HOP",
    "Candidate",
    "find_candidates",
    "flag_doubt",
    "lives_bounce",
    "lives_busy",
    "lives_doubt",
    "progress_doubt",
    "propose_candidates",
    "rethink_candidates",
]
