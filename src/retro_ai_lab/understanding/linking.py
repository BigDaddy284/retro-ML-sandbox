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

from retro_ai_lab.core import ConfigurationError
from retro_ai_lab.services.experiments import ExperimentHarness, Trial, constant_plan
from retro_ai_lab.understanding.knowledge import Evidence, Knowledge, Variable, VariableKind

Location = tuple[str, int]

DIRECTIONS = frozenset({"up", "down", "left", "right"})
DEFAULT_FRAMES = 8
DEFAULT_REPEATS = 2


@dataclass(frozen=True, slots=True)
class InputLinks:
    buttons: tuple[str, ...]
    frames: int
    by_button: Mapping[str, frozenset[Location]]
    unstable: frozenset[Location]

    def buttons_for(self, location: Location) -> tuple[str, ...]:
        return tuple(b for b in self.buttons if location in self.by_button[b])

    def locations(self) -> frozenset[Location]:
        out: set[Location] = set()
        for locs in self.by_button.values():
            out |= locs
        return frozenset(out)

    def dead_buttons(self) -> tuple[str, ...]:
        """Buttons that changed nothing in this situation."""
        return tuple(b for b in self.buttons if not self.by_button[b])


def _differences(a: Trial, b: Trial) -> frozenset[Location]:
    out: set[Location] = set()
    for name in a.after.region_names:
        x = a.after.regions[name]
        y = b.after.regions[name]
        out.update((name, i) for i in range(min(len(x), len(y))) if x[i] != y[i])
    return frozenset(out)


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
    idle = [harness.run(constant_plan((), frames), label=f"idle-{i}") for i in range(repeats)]
    unstable: set[Location] = set()
    for other in idle[1:]:
        unstable |= _differences(idle[0], other)
    by_button: dict[str, frozenset[Location]] = {}
    for name in names:
        trial = harness.run(constant_plan((name,), frames), label=name)
        by_button[name] = _differences(idle[0], trial) - unstable
    return InputLinks(names, frames, by_button, frozenset(unstable))


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
        "input-linked",
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
    everything = len(links.buttons) >= 3

    def buttons_for(locs: Iterable[Location]) -> tuple[str, ...]:
        wanted = set(locs)
        return tuple(b for b in links.buttons if wanted & links.by_button[b])

    for loc in sorted(orphans):
        buttons = buttons_for([loc])
        mirror = everything and set(buttons) == set(links.buttons)
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
        mirror = everything and set(buttons) == set(links.buttons)
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
            ):
                changes["kind"] = VariableKind.POSITION
        if changes:
            touched[var_id] = knowledge.update(var_id, **changes)
    return tuple(touched[k] for k in sorted(touched))


__all__ = [
    "DEFAULT_FRAMES",
    "DEFAULT_REPEATS",
    "DIRECTIONS",
    "InputLinks",
    "apply_input_links",
    "find_input_links",
]
