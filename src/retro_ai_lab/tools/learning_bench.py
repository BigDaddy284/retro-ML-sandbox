"""The learning bench (T-1054): what the learner alone learns on a practice game.

``python -m retro_ai_lab.tools.learning_bench racer`` connects the same services the
window drives to a practice game, plays a scripted first go so discovery has
something to learn from, and then lets the learner play go after go **with every
helper off**: no going back after a life lost, no way remembered, no pressing on,
no standing still looked into, nothing looked ahead. What is left is the learner and
its rewards, which is what has to carry over from one game to the next.

It reports block of goes by block, by the game's own account: the lives lost, how
far it got, and the goes in which no life was lost. A learner that learns shows it
from one block to the next. ``helpers=True`` trains as the app does, to compare.

Only this tool reads the game's truth. The app it measures learns everything
itself, as it must with a real game.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from retro_ai_lab.core import ConfigurationError
from retro_ai_lab.core.events import EventBus
from retro_ai_lab.emulator.fake import FakeEmulatorAdapter, ToyHopper, toy_hopper, toy_racer
from retro_ai_lab.emulator.fake.toy_racer import ToyRacer
from retro_ai_lab.emulator.registry import default_registry
from retro_ai_lab.learning import algorithm_choices, describe_algorithm
from retro_ai_lab.services import (
    ButtonsHeld,
    ConnectionService,
    DiscoveryService,
    GoalService,
    ObservationService,
    RewardService,
)
from retro_ai_lab.storage import GoalStore, KnowledgeStore, RewardStore
from retro_ai_lab.tools import practice_bench
from retro_ai_lab.training import Mode, TrainingService
from retro_ai_lab.training.episode import STAND_AFTER
from retro_ai_lab.training.service import WAYS_PER_GO

PROG = "python -m retro_ai_lab.tools.learning_bench"
GAMES = ("racer", "hopper")
TITLES = {"racer": "Toy Racer", "hopper": "Toy Hopper"}
# What "how far" is counted in, in the game's own units.
UNITS = {
    "racer": f"a lap is {toy_racer.TRACK_LENGTH:,}",
    "hopper": f"a level is {toy_hopper.LEVEL_WIDTH:,}",
}
DEFAULT_GOES = 100
DEFAULT_STEPS = 300
DEFAULT_BLOCK = 25
DEFAULT_ALGORITHM = "dqn"
PLAY_FRAMES = 3000  # frames of the scripted first go discovery learns from


@dataclass(frozen=True, slots=True)
class Go:
    """One go, by the game's own account."""

    lost: int  # lives that were lost and stayed lost
    far: int  # the furthest it got, in the game's own units
    steps: int
    ended_by: str


@dataclass(frozen=True, slots=True)
class Block:
    """Some goes one after another, added up."""

    first: int  # the number of its first go, from 1
    last: int
    lost: float  # lives lost a go
    far: float  # how far a go got, on average
    best: int  # the furthest any of them got
    clean: int  # goes in which no life was lost


@dataclass
class Report:
    """What one run of the bench measured."""

    game: str
    seed: int
    goes: int
    max_steps: int
    algorithm: str
    block: int = DEFAULT_BLOCK
    helpers: bool = False  # trained as the app does, going back and all
    played: list[Go] = field(default_factory=list)
    ways_sought: int = 0  # times it went back to look for a way (after a loss or before one)
    remembered: int = 0  # steps begun with from a way remembered
    looked_ahead: int = 0  # times it played ahead off the record

    def blocks(self) -> tuple[Block, ...]:
        """The goes in blocks of ``block``; those left over are a last, shorter one."""
        out: list[Block] = []
        for start in range(0, len(self.played), self.block):
            goes = self.played[start : start + self.block]
            out.append(
                Block(
                    start + 1,
                    start + len(goes),
                    statistics.fmean(go.lost for go in goes),
                    statistics.fmean(go.far for go in goes),
                    max(go.far for go in goes),
                    sum(go.lost == 0 for go in goes),
                )
            )
        return tuple(out)

    def lines(self) -> list[str]:
        """The report in plain words."""
        label, _ = describe_algorithm(self.algorithm)
        how = (
            "trained as the app does, with its helpers on"
            if self.helpers
            else "the learner alone, every helper off"
        )
        out = [
            f"Learning bench on {TITLES[self.game]}: {label.lower()}, {self.goes:,} "
            f"{'go' if self.goes == 1 else 'goes'} of up to {self.max_steps:,} steps, seed "
            f"{self.seed}, {how}. How far is in the game's own units ({UNITS[self.game]})."
        ]
        blocks = self.blocks()
        for b in blocks:
            goes = f"Go {b.first}" if b.first == b.last else f"Goes {b.first} to {b.last}"
            n = b.last - b.first + 1
            out.append(
                f"{goes}: {b.lost:.2f} lives lost a go; got {b.far:,.0f} far on average, "
                f"{b.best:,} at best; {b.clean} of {n} with no life lost."
            )
        if not blocks:
            out.append("No goes were played.")
        elif len(blocks) < 2:
            out.append("One block only: nothing to compare it with.")
        else:
            first, last = blocks[0], blocks[-1]
            lives = _compared(last.lost, first.lost, "fewer", "more", 2)
            far = _compared(last.far, first.far, "less far", "further", 0)
            out.append(
                f"From the first block to the last: lives lost a go {first.lost:.2f} to "
                f"{last.lost:.2f} ({lives}), how far {first.far:,.0f} to {last.far:,.0f} ({far})."
            )
        return out


def _compared(now: float, then: float, less: str, more: str, digits: int) -> str:
    """Which way a figure went, to the decimals the report shows it with."""
    now, then = round(now, digits), round(then, digits)
    return less if now < then else more if now > then else "the same"


def how_far(game: ToyRacer | ToyHopper) -> int:
    """How far the game has it: the racer's laps and its place on the track, the
    hopper's levels finished and its place in the level, as one number."""
    if isinstance(game, ToyRacer):
        racer = game.state
        return racer.lap * toy_racer.TRACK_LENGTH + racer.position
    hopper = game.state
    return hopper.level * toy_hopper.LEVEL_WIDTH + hopper.x


def first_drive(
    connection: ConnectionService, observation: ObservationService, frames: int
) -> None:
    """A child's first go at the racer: drives on, lifts off now and then, changes
    lane now and then, and starts again when the game is over."""
    adapter = connection.adapter
    assert isinstance(adapter, FakeEmulatorAdapter)
    by_name = {b.name: b for b in adapter.buttons()}
    for i in range(frames):
        if adapter.game.state.game_over:
            held = ["start"]
        else:
            held = ["up"] if i % 50 < 35 else []
            if i % 90 == 0:
                held.append("left")
            elif i % 90 == 45:
                held.append("right")
        adapter.set_input([by_name[name] for name in held])
        observation.record_input(held)
        adapter.advance_frames(1)
        observation.advance(1)


def run(
    game: str,
    goes: int = DEFAULT_GOES,
    *,
    seed: int = 0,
    algorithm: str = DEFAULT_ALGORITHM,
    max_steps: int = DEFAULT_STEPS,
    block: int = DEFAULT_BLOCK,
    helpers: bool = False,
) -> Report:
    """Run the bench and return what it measured. Seeded: the same arguments, the
    same report.

    ``helpers=False`` (as it is unless asked) trains the learner alone: it never goes
    back, remembers no way, does not press on or look into a standstill, and looks
    ahead at nothing. ``helpers=True`` trains as the app does.
    """
    if game not in GAMES:
        raise ConfigurationError(
            f"The learning bench has no practice game called {game!r} "
            f"(choices: {', '.join(GAMES)})."
        )
    if not isinstance(helpers, bool):
        raise ConfigurationError("The learning bench needs helpers to be True or False.")
    for value, what in ((goes, "goes"), (max_steps, "steps"), (block, "goes to a block")):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ConfigurationError(
                f"The learning bench needs a whole number of {what}, 1 or more."
            )
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= 0xFFFF:
        raise ConfigurationError("The learning bench needs a seed from 0 to 65535.")
    if algorithm not in algorithm_choices():
        raise ConfigurationError(
            f"No way of learning called {algorithm!r} (choices: {', '.join(algorithm_choices())})."
        )
    report = Report(game, seed, goes, max_steps, algorithm, block, helpers)
    with tempfile.TemporaryDirectory() as scratch:
        data = Path(scratch)
        bus = EventBus()
        connection = ConnectionService(default_registry(), bus)
        observation = ObservationService(connection, bus)
        discovery = DiscoveryService(
            connection, observation, bus, store=KnowledgeStore(data / "knowledge")
        )
        rewards = RewardService(connection, discovery, bus, store=RewardStore(data / "rewards"))
        goals = GoalService(connection, discovery, bus, store=GoalStore(data / "goals"))
        training = TrainingService(
            connection,
            observation,
            discovery,
            bus,
            max_steps=max_steps,
            seed=seed,
            goals=goals,
            rewards=rewards,
            way_seconds=None,  # a search cut short by the clock would not be repeatable
            # Every helper off: no going back takes the way remembered, the pressing
            # on, the standstill and the looking ahead with it; they are switched off
            # by name too. With the helpers on it is built as the app builds it.
            ways_per_go=WAYS_PER_GO if helpers else 0,
            remember=helpers,
            stand_after=STAND_AFTER if helpers else 0,
            look_ahead=helpers,
        )
        training.set_algorithm(algorithm)
        connection.connect("fake", game=game, seed=seed)
        try:
            adapter = connection.adapter
            assert isinstance(adapter, FakeEmulatorAdapter)
            played = adapter.game
            assert isinstance(played, ToyRacer | ToyHopper)
            if isinstance(played, ToyHopper):
                practice_bench.first_go(connection, observation, PLAY_FRAMES)
            else:
                first_drive(connection, observation, PLAY_FRAMES)
            played.reset()  # the AI's goes start from the game's start, every life in hand
            _train(training, bus, played, report)
        finally:
            training.stop()
            connection.disconnect()
    return report


def _train(
    training: TrainingService, bus: EventBus, game: ToyRacer | ToyHopper, report: Report
) -> None:
    # The lives as they stood when each step's buttons went down: a go's first step
    # comes after the game was put back to its start, which no one here is told of.
    lives: list[int] = []
    bus.subscribe(ButtonsHeld, lambda event: lives.append(game.state.lives))
    training.start(Mode.INDEPENDENT)
    while training.episodes < report.goes:
        go = training.episodes
        lost, far, looked = 0, 0, 0
        while training.episodes == go:
            lives.clear()
            training.step()
            if lives:
                # A life the game was put back for within the step is there again:
                # it was not lost.
                lost += max(0, lives[0] - game.state.lives)
                far = max(far, how_far(game))
            runner = training.episode
            if runner is not None:
                looked = max(looked, getattr(runner, "looked_ahead", 0))
        result = training.history[-1].result
        report.played.append(Go(lost, far, result.steps, result.ended_by))
        report.ways_sought += result.ways_sought + result.ways_ahead
        report.remembered += result.remembered
        report.looked_ahead += looked


# ---------------------------------------------------------------- command line


def _whole(text: str) -> int:
    try:
        number = int(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"'{text}' is not a whole number.") from exc
    if number < 1:
        raise argparse.ArgumentTypeError("must be 1 or more.")
    return number


def _seed(text: str) -> int:
    try:
        number = int(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"'{text}' is not a whole number.") from exc
    if not 0 <= number <= 0xFFFF:
        raise argparse.ArgumentTypeError("must be from 0 to 65535.")
    return number


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=PROG,
        description="Measure what the learner alone learns on a practice game.",
    )
    parser.add_argument("game", choices=GAMES, help="the practice game to play")
    parser.add_argument("--goes", type=_whole, default=DEFAULT_GOES, help="goes to train for")
    parser.add_argument("--steps", type=_whole, default=DEFAULT_STEPS, help="steps in a go")
    parser.add_argument("--seed", type=_seed, default=0, help="which track or level (0 to 65535)")
    parser.add_argument(
        "--algorithm",
        choices=algorithm_choices(),
        default=DEFAULT_ALGORITHM,
        help="how the AI learns",
    )
    parser.add_argument(
        "--block", type=_whole, default=DEFAULT_BLOCK, help="goes added up in each line"
    )
    parser.add_argument(
        "--helpers",
        action="store_true",
        help="train as the app does (going back, the way remembered, looking ahead)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = run(
        args.game,
        args.goes,
        seed=args.seed,
        algorithm=args.algorithm,
        max_steps=args.steps,
        block=args.block,
        helpers=args.helpers,
    )
    for line in report.lines():
        print(line)
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through main() in the tests
    sys.exit(main())


__all__ = [
    "DEFAULT_ALGORITHM",
    "DEFAULT_BLOCK",
    "DEFAULT_GOES",
    "DEFAULT_STEPS",
    "GAMES",
    "Block",
    "Go",
    "Report",
    "build_parser",
    "first_drive",
    "how_far",
    "main",
    "run",
]
