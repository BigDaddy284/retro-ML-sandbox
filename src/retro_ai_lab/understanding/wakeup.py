"""Working out which button gets a game going (T-844).

A title screen with a demo, a "press start" prompt or a game-over screen
all look alive to the AI (values move) while ignoring everything it
presses. DESIGN.md asks that nothing about a game be coded in, so the app
must find the way out itself: it tries every button the console has, one
at a time, and keeps the first one after which its buttons start to
matter. The answer is remembered with the game's knowledge.

Two questions, both answered by experiment from a save state:

* :func:`inputs_matter`: from here, does holding buttons change where the
  memory ends up, compared with holding nothing? Both runs end with a few
  idle frames so a controller mirror (a byte that just echoes the pad)
  settles to the same value in both.
* :func:`find_wake_button`: press one button, wait for the game to settle
  (a "WORLD 1-1" splash takes a few seconds), then ask the first question
  again. The first button that makes inputs matter is the answer, and the
  game is left running from there. A press that changed something lasting
  while the game stays deaf is the most promising kind (Start during a
  title demo only brings the title screen back; the next press starts the
  game), so such a button is pressed again, a few times at most (T-851).
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from retro_ai_lab.core import ConfigurationError
from retro_ai_lab.core.types import Button
from retro_ai_lab.emulator.base import EmulatorAdapter
from retro_ai_lab.observation.memory import MemoryCapture, MemorySnapshot

if TYPE_CHECKING:  # the services package imports this one; no cycle at run time
    from retro_ai_lab.services.connection import ConnectionService

log = logging.getLogger(__name__)

WAKE_SLOT = "wakeup"  # where the search keeps the spot it is probing from
START_SLOT = "wakeup_start"  # where the game was when the search began
PRESS_SLOT = "wakeup_press"  # where a promising button left the game, for the next press
DEFAULT_PER_BUTTON = 5  # frames each button is held, in turn, in the "busy" probe
DEFAULT_SETTLE = 10  # idle frames at the end of a probe, for controller mirrors
DEFAULT_HOLD = 8  # frames a candidate button is held
DEFAULT_WAIT = 300  # frames to let the game settle after the press (a splash screen)
DEFAULT_PRESSES = 3  # most presses of one button while each keeps changing something


@dataclass(frozen=True, slots=True)
class InputProbe:
    """What one probe from a save state showed (T-848, T-855)."""

    matters: bool  # holding the buttons changed where memory ended up
    self_moving: frozenset[tuple[str, int]]  # bytes that changed holding nothing, and
    # ended up the same whatever was held: a timer, an enemy, a demo; not a score


def inputs_matter(
    connection: ConnectionService,
    buttons: Iterable[str],
    *,
    per_button: int = DEFAULT_PER_BUTTON,
    settle: int = DEFAULT_SETTLE,
) -> bool:
    """From where the game is now: does holding buttons change where memory ends up?"""
    return probe_inputs(connection, buttons, per_button=per_button, settle=settle).matters


def probe_inputs(
    connection: ConnectionService,
    buttons: Iterable[str],
    *,
    per_button: int = DEFAULT_PER_BUTTON,
    settle: int = DEFAULT_SETTLE,
) -> InputProbe:
    """Probe from where the game is now: do the buttons matter, and what moves by itself?

    The busy probe holds each button in turn for ``per_button`` frames; the idle
    probe plays the same number of frames holding nothing. Leaves the game where
    it was. Locations that differ between two idle runs (noise) are ignored. The
    bytes that changed during the idle run yet ended the same after the busy run
    are ``self_moving`` (T-855): they move whether or not anyone plays.
    """
    names = tuple(dict.fromkeys(buttons))
    if not names:
        raise ConfigurationError("inputs_matter needs at least one button.")
    if per_button < 1 or settle < 0:
        raise ConfigurationError("per_button must be positive and settle non-negative.")
    known = _buttons(connection)
    _check_known(names, known)
    capture = MemoryCapture(connection)
    frames = per_button * len(names)
    connection.call(lambda a: a.save_state(WAKE_SLOT))
    try:
        before = _probe(connection, capture, [])
        idle = _probe(connection, capture, [((), frames + settle)])
        again = _probe(connection, capture, [((), frames + settle)])
        unstable = _differences(idle, again)
        plan: list[tuple[tuple[Button, ...], int]] = [
            ((known[name],), per_button) for name in names
        ]
        plan.append(((), settle))
        busy = _probe(connection, capture, plan)
    finally:
        connection.call(lambda a: a.load_state(WAKE_SLOT))
    driven = _differences(idle, busy) - unstable
    alone = _differences(before, idle) - unstable - driven
    return InputProbe(bool(driven), alone)


def find_wake_button(
    connection: ConnectionService,
    candidates: Iterable[str],
    playing: Iterable[str],
    *,
    first: str | None = None,
    hold: int = DEFAULT_HOLD,
    wait: int = DEFAULT_WAIT,
    per_button: int = DEFAULT_PER_BUTTON,
    settle: int = DEFAULT_SETTLE,
    presses: int = DEFAULT_PRESSES,
) -> str | None:
    """Try each candidate (``first`` first) until the playing buttons matter again.

    ``candidates`` are every button the console has; ``playing`` are the ones
    the AI plays with. A candidate counts when its press changes something
    lasting (the memory after pressing it and waiting differs from just
    waiting) and the playing buttons matter afterwards. When the press changed
    something lasting but the playing buttons still do not matter, the same
    button is pressed again from where it left the game, up to ``presses``
    times in all. Returns the button's name with the game left running after
    its press, or None with the game back where it was.
    """
    names = tuple(dict.fromkeys(candidates))
    plays = tuple(dict.fromkeys(playing))
    if not names or not plays:
        raise ConfigurationError("find_wake_button needs candidate and playing buttons.")
    if hold < 1 or wait < 0 or presses < 1:
        raise ConfigurationError("hold and presses must be positive and wait non-negative.")
    if first is not None and first in names:
        names = (first, *(n for n in names if n != first))
    known = _buttons(connection)
    _check_known(names + plays, known)
    capture = MemoryCapture(connection)
    connection.call(lambda a: a.save_state(START_SLOT))
    baseline = _play(connection, capture, START_SLOT, [((), hold + wait)])
    again = _play(connection, capture, START_SLOT, [((), hold + wait)])
    unstable = _differences(baseline, again)
    for name in names:
        press: list[tuple[tuple[Button, ...], int]] = [((known[name],), hold), ((), wait)]
        slot, before = START_SLOT, baseline
        for pressed in range(1, presses + 1):
            after = _play(connection, capture, slot, press)
            if not (_differences(before, after) - unstable):
                break  # the press changed nothing lasting
            if inputs_matter(connection, plays, per_button=per_button, settle=settle):
                label = known[name].label
                if pressed == 1:
                    log.info("Pressing %s gets the game going.", label)
                else:
                    log.info("Pressing %s %d times gets the game going.", label, pressed)
                return name
            # The game is where the press left it (inputs_matter put it back); press
            # again from here, compared with just waiting from here.
            connection.call(lambda a: a.save_state(PRESS_SLOT))
            slot = PRESS_SLOT
            before = _play(connection, capture, PRESS_SLOT, [((), hold + wait)])
    connection.call(lambda a: a.load_state(START_SLOT))
    log.info("No button got the game going: tried %s.", ", ".join(names))
    return None


# ------------------------------------------------------------------ helpers


def _buttons(connection: ConnectionService) -> dict[str, Button]:
    return {b.name: b for b in connection.call(lambda a: a.buttons())}


def _check_known(names: tuple[str, ...], known: dict[str, Button]) -> None:
    unknown = [n for n in names if n not in known]
    if unknown:
        raise ConfigurationError(f"Unknown buttons for this game: {unknown}.")


def _probe(
    connection: ConnectionService,
    capture: MemoryCapture,
    plan: list[tuple[tuple[Button, ...], int]],
) -> MemorySnapshot:
    """From the wake slot: play ``plan`` ((held buttons, frames) in turn), read memory."""
    return _play(connection, capture, WAKE_SLOT, plan)


def _play(
    connection: ConnectionService,
    capture: MemoryCapture,
    slot: str,
    plan: list[tuple[tuple[Button, ...], int]],
) -> MemorySnapshot:
    connection.call(lambda a: a.load_state(slot))
    played = 0
    for held, count in plan:
        if count <= 0:
            continue
        connection.call(_stepper(held, count))
        played += count
    connection.call(lambda a: a.set_input(()))
    return capture.capture(played)


def _stepper(held: tuple[Button, ...], count: int) -> Callable[[EmulatorAdapter], None]:
    def step(adapter: EmulatorAdapter) -> None:
        adapter.step(count, pressed=held)

    return step


def _differences(a: MemorySnapshot, b: MemorySnapshot) -> frozenset[tuple[str, int]]:
    out: set[tuple[str, int]] = set()
    for name in a.region_names:
        if name not in b.regions:
            continue
        x, y = a.regions[name], b.regions[name]
        out.update((name, i) for i in range(min(len(x), len(y))) if x[i] != y[i])
    return frozenset(out)


__all__ = [
    "DEFAULT_HOLD",
    "DEFAULT_PER_BUTTON",
    "DEFAULT_PRESSES",
    "DEFAULT_SETTLE",
    "DEFAULT_WAIT",
    "PRESS_SLOT",
    "START_SLOT",
    "WAKE_SLOT",
    "InputProbe",
    "find_wake_button",
    "inputs_matter",
    "probe_inputs",
]
