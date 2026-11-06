"""Input linking: find out which bytes each button moves.

The active-discovery loop from DESIGN.md §3.4, step by step: save the
game, hold one button for a few frames, look at what changed compared with
doing nothing, put the game back, repeat for the next button. Bytes that
differ between two *identical* idle trials are unstable (random numbers,
timers seeded from the clock) and are ignored.

The result feeds :func:`apply_input_links`, which adds "changes when Up is
held" evidence to the knowledge, relabels player-controlled values, spots
the byte that merely mirrors the controller, and proposes any linked byte
the knowledge had not noticed yet.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

from retro_ai_lab.core import ConfigurationError
from retro_ai_lab.understanding.knowledge import (
    INPUT_LINKED,
    Evidence,
    Knowledge,
    Variable,
    VariableKind,
)
from retro_ai_lab.understanding.longlook import LONG_LOOK_KINDS

if TYPE_CHECKING:
    # Imported lazily at run time: services already import this package, and
    # loading it first (as the rewards package does) must not go round in a circle.
    from retro_ai_lab.observation.memory import MemorySnapshot
    from retro_ai_lab.services.experiments import ExperimentHarness, Trial

Location = tuple[str, int]

DIRECTIONS = frozenset({"up", "down", "left", "right"})
DEFAULT_FRAMES = 8
DEFAULT_REPEATS = 2
# A button whose trial differs from idle in at least this share of what the game changes
# on its own (and at least SWEEP_LEAST bytes) stopped or replaced the game: a pause, a
# menu, Start on a title screen. It moves nothing in particular (T-930).
SWEEP_SHARE = 0.5
SWEEP_LEAST = 8


@dataclass(frozen=True, slots=True)
class InputLinks:
    buttons: tuple[str, ...]
    frames: int
    by_button: Mapping[str, frozenset[Location]]
    unstable: frozenset[Location]
    sweeping: tuple[str, ...] = ()  # buttons that stopped or replaced the game (T-930)

    def buttons_for(self, location: Location) -> tuple[str, ...]:
        return tuple(b for b in self.buttons if location in self.by_button[b])

    def active(self) -> tuple[str, ...]:
        """The buttons whose links say what each one moves: all but the sweeping ones."""
        return tuple(b for b in self.buttons if b not in self.sweeping)

    def locations(self) -> frozenset[Location]:
        out: set[Location] = set()
        for locs in self.by_button.values():
            out |= locs
        return frozenset(out)

    def dead_buttons(self) -> tuple[str, ...]:
        """Buttons that changed nothing in this situation."""
        return tuple(b for b in self.active() if not self.by_button[b])


def _changed(x: MemorySnapshot, y: MemorySnapshot) -> frozenset[Location]:
    out: set[Location] = set()
    for name in x.region_names:
        a = x.regions[name]
        b = y.regions[name]
        out.update((name, i) for i in range(min(len(a), len(b))) if a[i] != b[i])
    return frozenset(out)


def _differences(a: Trial, b: Trial) -> frozenset[Location]:
    return _changed(a.after, b.after)


def _sweeps(moved: frozenset[Location], drift: frozenset[Location]) -> bool:
    """Did holding the button stop or replace what the game was doing on its own?"""
    return len(moved & drift) >= max(SWEEP_LEAST, SWEEP_SHARE * len(drift))


def find_input_links(
    harness: ExperimentHarness,
    buttons: Iterable[str],
    *,
    frames: int = DEFAULT_FRAMES,
    repeats: int = DEFAULT_REPEATS,
) -> InputLinks:
    """Run one idle trial (``repeats`` times) and one trial per button from the checkpoint."""
    names = tuple(dict.fromkeys(buttons))
    if not names:
        raise ConfigurationError("find_input_links needs at least one button.")
    if frames < 1 or repeats < 1:
        raise ConfigurationError("frames and repeats must be positive.")
    if not harness.has_checkpoint:
        raise ConfigurationError("No checkpoint yet; call harness.checkpoint() first.")
    from retro_ai_lab.services.experiments import constant_plan

    idle = [harness.run(constant_plan((), frames), label=f"idle-{i}") for i in range(repeats)]
    unstable: set[Location] = set()
    for other in idle[1:]:
        unstable |= _differences(idle[0], other)
    drift = _changed(idle[0].before, idle[0].after) - unstable  # what moves by itself
    by_button: dict[str, frozenset[Location]] = {}
    sweeping: list[str] = []
    for name in names:
        trial = harness.run(constant_plan((name,), frames), label=name)
        moved = _differences(idle[0], trial) - unstable
        if _sweeps(moved, drift):
            # A pause button leaves every byte where idle moved it on; Start on a title
            # screen replaces the lot. Neither says what the button moves (T-930).
            sweeping.append(name)
            moved = frozenset()
        by_button[name] = moved
    return InputLinks(names, frames, by_button, frozenset(unstable), tuple(sweeping))


def _covering(knowledge: Knowledge, location: Location) -> Variable | None:
    region, offset = location
    for v in knowledge:
        if v.region == region and v.offset <= offset < v.offset + v.width:
            return v
    return None


def _describe(buttons: Iterable[str]) -> str:
    names = [b.capitalize() for b in buttons]
    if len(names) == 1:
        return names[0]
    return ", ".join(names[:-1]) + " or " + names[-1]


def _evidence(buttons: tuple[str, ...]) -> Evidence:
    return Evidence(
        INPUT_LINKED,
        f"Changes when {_describe(buttons)} is held.",
        min(2.0, 0.5 + 0.5 * len(buttons)),
    )


def apply_input_links(knowledge: Knowledge, links: InputLinks) -> tuple[Variable, ...]:
    """Fold input links into the knowledge. Returns every variable added or changed.

    * Linked variables gain "changes when X is held" evidence (once per
      distinct set of buttons, so re-running discovery does not inflate it).
    * A location moved by every tested button is the controller mirror.
    * An unknown or progress-looking value moved by a direction button is
      relabelled as a position (a person's confirmed label is never touched).
    * A linked location nobody has an opinion about is proposed.
    """
    touched: dict[str, Variable] = {}
    grouped: dict[str, set[Location]] = {}
    orphans: list[Location] = []
    for loc in links.locations():
        var = _covering(knowledge, loc)
        if var is None:
            orphans.append(loc)
        else:
            grouped.setdefault(var.id, set()).add(loc)
    active = links.active()  # a sweeping button links to nothing and counts for nothing (T-930)
    everything = len(active) >= 3

    def buttons_for(locs: Iterable[Location]) -> tuple[str, ...]:
        wanted = set(locs)
        return tuple(b for b in links.buttons if wanted & links.by_button[b])

    for loc in sorted(orphans):
        buttons = buttons_for([loc])
        mirror = everything and set(buttons) == set(active)
        kind = (
            VariableKind.INPUT
            if mirror
            else VariableKind.POSITION
            if set(buttons) & DIRECTIONS
            else VariableKind.UNKNOWN
        )
        proposed = knowledge.propose(
            loc[0],
            loc[1],
            kind=kind,
            name="Controller input" if mirror else "",
            evidence=[_evidence(buttons)],
        )
        touched[proposed.id] = proposed

    for var_id, locs in grouped.items():
        var = knowledge.get(var_id)
        if var.rejected:
            continue
        buttons = buttons_for(locs)
        proof = _evidence(buttons)
        mirror = everything and set(buttons) == set(active)
        changes: dict[str, object] = {}
        if not any(e.kind == proof.kind and e.detail == proof.detail for e in var.evidence):
            changes["evidence"] = (*var.evidence, proof)
        if not var.confirmed:
            if mirror and var.kind is not VariableKind.INPUT:
                changes["kind"] = VariableKind.INPUT
                changes["name"] = var.name or "Controller input"
            elif (
                var.kind in (VariableKind.UNKNOWN, VariableKind.PROGRESS)
                and set(buttons) & DIRECTIONS
                and not mirror
                and not any(e.kind in LONG_LOOK_KINDS for e in var.evidence)
            ):
                # A slow number that moves with the stick is probably a position, unless
                # the longer look (T-709) has already shown it counts progress.
                changes["kind"] = VariableKind.POSITION
        if changes:
            touched[var_id] = knowledge.update(var_id, **changes)
    return tuple(touched[k] for k in sorted(touched))


__all__ = [
    "DEFAULT_FRAMES",
    "DEFAULT_REPEATS",
    "DIRECTIONS",
    "SWEEP_LEAST",
    "SWEEP_SHARE",
    "InputLinks",
    "apply_input_links",
    "find_input_links",
]
