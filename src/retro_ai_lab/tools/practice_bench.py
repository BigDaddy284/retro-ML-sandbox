"""The practice bench (T-986): the whole app on Toy Hopper, measured against the truth.

``python -m retro_ai_lab.tools.practice_bench`` connects the same services the
window drives to the practice platform game, plays a first go the way a child
would so discovery has something to learn from, trains for a number of goes
with nobody watching, and reports how well the AI sees, what it reads from
memory and how it plays. Toy Hopper knows where the player really is and what
really killed it, so every number has the truth beside it.

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
from retro_ai_lab.emulator.fake import FakeEmulatorAdapter, ToyHopper, toy_hopper
from retro_ai_lab.emulator.registry import default_registry
from retro_ai_lab.learning import algorithm_choices, describe_algorithm
from retro_ai_lab.observation import surroundings
from retro_ai_lab.services import (
    ConnectionService,
    DiscoveryService,
    GoalService,
    ObservationService,
    RewardService,
)
from retro_ai_lab.storage import GoalStore, KnowledgeStore, RewardStore
from retro_ai_lab.training import Mode, TrainingService
from retro_ai_lab.understanding import VariableKind

PROG = "python -m retro_ai_lab.tools.practice_bench"
DEFAULT_GOES = 40
DEFAULT_STEPS = 300
DEFAULT_PLAY_FRAMES = 3000
DEFAULT_ALGORITHM = "dqn"
ON_PLAYER = 12  # pixels the watch's "me" may be from the real player's centre and still be it
WINDOW = 10  # goes in the first and last stretches the report compares
# The plan's targets for the eyes (Stage 2).
FOLLOWED_TARGET = 95
EXPLAINED_TARGET = 80
MOST_KINDS = 30
FLOOR_TARGET = 95  # the floor either side read as the game's level has it (T-989)
CAMERA_TARGET = 90  # the camera's move between two looks measured right (T-992)
MOST_CAMERA = 32  # further than this between two looks is a cut to another place
GROUND_TARGET = 90  # goes whose counted ground is the game's own furthest, to a step (T-993)


@dataclass
class Report:
    """What one run of the bench measured."""

    seed: int
    goes: int
    max_steps: int
    algorithm: str
    # Eyes.
    looks: int = 0
    followed: int = 0  # looks in which the watch had what it calls itself in sight
    right: int = 0  # ... and that thing really was the player
    kinds: int = 0
    around: int = 0  # looks in which it read what is around it (T-989)
    floor_right: int = 0  # ... and the floor either side was as the game's level has it
    hurts: int = 0
    items: int = 0
    # The camera's move between two looks of a go, against the game's (T-992).
    camera_moved: int = 0
    camera_moved_right: int = 0
    camera_still: int = 0
    camera_still_right: int = 0
    camera_cuts: int = 0  # the game put the camera somewhere else (a life lost)
    # The new ground the watch counted, against how far the game's own camera went (T-993).
    ground_steps: int = 0  # steps of new ground counted over the run
    ground_goes: int = 0  # goes in which the watch looked at the screen
    ground_right: int = 0  # ... and its furthest to the right was the game's, to a step
    # ... of which goes that began before the game's camera had moved since the AI
    # started: until it does the watch can only go by where the player stands on the
    # screen, and the game's own figure is how far the player itself went (T-998).
    ground_before: int = 0
    # Deaths, by the game's account, and what the watch made of them.
    falls: int = 0
    walkers: int = 0
    blamed: int = 0  # lives the watch put down to touching something
    fell: int = 0  # lives the watch put down to falling off the bottom of the screen (T-988)
    # Memory.
    own_position: str = "nothing"  # where the encoder measures distances from
    own_position_right: bool = False
    lives_found: bool = False
    score_found: bool = False
    # Brain, one entry per go.
    furthest: list[int] = field(default_factory=list)
    lost: list[int] = field(default_factory=list)
    finished: list[int] = field(default_factory=list)

    @property
    def deaths(self) -> int:
        return self.falls + self.walkers

    def lines(self) -> list[str]:
        """The report as the lines the tool prints."""
        label = describe_algorithm(self.algorithm)[0]
        out = [
            f"Practice bench: Toy Hopper level {self.seed}, {self.goes} goes of up to "
            f"{self.max_steps} steps, {label}."
        ]
        if self.looks:
            out.append(
                f"Eyes: {self.looks:,} looks; followed what it takes for itself in "
                f"{_share(self.followed, self.looks)}% of them (target {FOLLOWED_TARGET}%), and "
                f"that really was the player in {_share(self.right, self.looks)}%; "
                f"{self.kinds} kinds of thing (target under {MOST_KINDS}); the floor to its "
                f"right and left read as the level has it in "
                f"{_share(self.floor_right, self.around)}% of the {self.around:,} looks in which "
                f"it read what is around it (target {FLOOR_TARGET}%)."
            )
        else:
            out.append(
                "Eyes: the AI never looked at the screen (it did not find itself in memory)."
            )
        if self.camera_moved or self.camera_still:
            out.append(
                f"Camera: it moved between two looks {self.camera_moved:,} times, measured "
                f"right in {_share(self.camera_moved_right, self.camera_moved)}% (target "
                f"{CAMERA_TARGET}%); it stood still {self.camera_still:,} times, read as still "
                f"in {_share(self.camera_still_right, self.camera_still)}%; {self.camera_cuts} "
                f"cuts to another place not counted."
            )
        else:
            out.append("Camera: not followed (the AI never looked at the screen twice in a go).")
        if self.ground_goes:
            out.append(
                f"Ground: {self.ground_steps:,} steps of new ground counted for pay; the "
                f"furthest it counted to the right was as far as the game's own camera went, "
                f"to within a step, in "
                f"{_share(self.ground_right, self.ground_goes)}% of the {self.ground_goes} goes "
                f"(target {GROUND_TARGET}%)."
            )
        else:
            out.append("Ground: none counted (the AI never looked at the screen).")
        explained = (
            f"{_share(self.fell + self.blamed, self.deaths)}% explained"
            if self.deaths
            else "none to explain"
        )
        out.append(
            f"Deaths: {self.deaths} ({self.falls} falls, {self.walkers} walkers); the eyes put "
            f"{self.fell} down to a fall and {self.blamed} to touching something ({explained}, "
            f"target {EXPLAINED_TARGET}% and for the right reason); {self.hurts} kinds found to "
            f"hurt, {self.items} to pick up."
        )
        out.append(
            f"Memory: measures distances from {self.own_position} "
            f"({'the player' if self.own_position_right else 'not the player'}); "
            f"lives {'found' if self.lives_found else 'not found'}; "
            f"score {'found' if self.score_found else 'not found'}."
        )
        first, last = self.furthest[:WINDOW], self.furthest[-WINDOW:]
        flag = toy_hopper.FLAG_COL * toy_hopper.TILE
        if first:
            out.append(
                f"Brain: furthest reached {statistics.fmean(first):.0f} (best {max(first)}) in the "
                f"first {len(first)} goes and {statistics.fmean(last):.0f} (best {max(last)}) in "
                f"the last {len(last)}, of {flag:,} to the flag; lives lost a go "
                f"{statistics.fmean(self.lost[:WINDOW]):.2f} then "
                f"{statistics.fmean(self.lost[-WINDOW:]):.2f}; levels finished "
                f"{sum(self.finished)}."
            )
        return out


def _share(part: int, whole: int) -> int:
    return round(100 * part / whole) if whole else 0


def first_go(connection: ConnectionService, observation: ObservationService, frames: int) -> None:
    """A child's first go: runs right, jumps now and then, waits, turns back, starts again."""
    adapter = connection.adapter
    assert isinstance(adapter, FakeEmulatorAdapter)
    by_name = {b.name: b for b in adapter.buttons()}
    for i in range(frames):
        if adapter.game.state.game_over:
            held = ["start"]
        elif i % 300 in range(200, 215):
            held = ["left"]
        else:
            held = ["right"] if i % 120 < 100 else []
            if i % 40 < 14:
                held = [*held, "a"]
        adapter.set_input([by_name[name] for name in held])
        observation.record_input(held)
        adapter.advance_frames(1)
        observation.advance(1)


def run(
    goes: int = DEFAULT_GOES,
    *,
    seed: int = 0,
    algorithm: str = DEFAULT_ALGORITHM,
    max_steps: int = DEFAULT_STEPS,
    play_frames: int = DEFAULT_PLAY_FRAMES,
) -> Report:
    """Run the bench and return what it measured. Seeded: the same arguments, the same report."""
    for value, what in ((goes, "goes"), (max_steps, "steps"), (play_frames, "play frames")):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ConfigurationError(f"The bench needs a whole number of {what}, 1 or more.")
    if algorithm not in algorithm_choices():
        raise ConfigurationError(
            f"No way of learning called {algorithm!r} (choices: {', '.join(algorithm_choices())})."
        )
    report = Report(seed=seed, goes=goes, max_steps=max_steps, algorithm=algorithm)
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
        )
        training.set_algorithm(algorithm)
        connection.connect("fake", game="hopper", seed=seed)
        try:
            first_go(connection, observation, play_frames)
            adapter = connection.adapter
            assert isinstance(adapter, FakeEmulatorAdapter)
            game = adapter.game
            assert isinstance(game, ToyHopper)
            game.reset()  # the AI's goes start from the level's start, three lives in hand
            _train(training, game, report)
            _read_memory(training, discovery, report)
        finally:
            training.stop()
            connection.disconnect()
    return report


def _train(training: TrainingService, game: ToyHopper, report: Report) -> None:
    training.start(Mode.INDEPENDENT)
    looks = 0
    scrolled = False  # the game's camera has moved between two looks since the AI started
    while training.episodes < report.goes:
        go = training.episodes
        furthest, lost, level = 0, 0, game.state.level
        before = game.state
        camera: int | None = None  # where the game's camera was at the go's last look
        began: int | None = None  # where the game's camera was at the go's first look
        stood = 0  # where the player was at the go's first look
        fresh = not scrolled  # the go began before the camera had ever moved
        got, counted, steps = 0, 0, 0  # furthest at a look: the game's, the watch's; its steps
        while training.episodes == go:
            training.step()
            now = game.state
            furthest = max(furthest, now.x)
            if now.mode == toy_hopper.DYING and before.mode != toy_hopper.DYING:
                if now.y >= toy_hopper.SCREEN_HEIGHT:
                    report.falls += 1
                else:
                    report.walkers += 1
            lost += max(0, before.lives - now.lives)
            before = now
            watch = training.watch
            if watch is not None and watch.looks != looks:
                looks = watch.looks
                report.right += _on_the_player(training, game)
                if camera is not None:
                    _count_camera(report, now.scroll - camera, watch.tracker.scroll)
                first_move = camera is not None and now.scroll != camera
                camera = now.scroll
                if began is None:
                    began, stood = now.scroll, now.x
                # The camera's travel, once the camera has been seen to move. Up to the
                # look at which it first does, how far the player itself has gone:
                # nothing else has moved, and that is what the watch can count.
                got = max(got, now.scroll - began if scrolled else now.x - stood)
                scrolled = scrolled or first_move
                counted = max(counted, watch.ground.reach[0])
                steps = max(steps, watch.ground.steps)
                if watch.around is not None:
                    report.around += 1
                    floor = (watch.around.floor_right, watch.around.floor_left)
                    report.floor_right += floor == _floor_either_side(game)
        report.furthest.append(furthest)
        report.lost.append(lost)
        if began is not None and training.watch is not None:
            report.ground_goes += 1
            report.ground_before += fresh
            report.ground_steps += steps
            step = training.watch.ground.unit or toy_hopper.PLAYER_H
            report.ground_right += abs(counted - got) <= step
        report.finished.append(max(0, game.state.level - level))
    watch = training.watch
    if watch is None:
        return
    report.looks, report.followed = watch.looks, watch.seen_me
    report.blamed = watch.lives_after_touch
    report.fell = watch.lives_after_fall
    report.kinds = len(watch.tracker.catalogue.kinds())
    tags = [role.tag.name for role in watch.roles()]
    report.hurts = sum(tag in ("ENEMY", "HAZARD") for tag in tags)
    report.items = tags.count("ITEM")


def _on_the_player(training: TrainingService, game: ToyHopper) -> bool:
    """Whether what the watch follows as itself is, just now, on the real player."""
    watch = training.watch
    if watch is None or watch.me is None:
        return False
    left, top, width, height = watch.me
    st = game.state
    x = st.x - st.scroll + toy_hopper.PLAYER_W // 2
    y = st.y + toy_hopper.PLAYER_H // 2
    mx, my = left + width // 2, top + height // 2
    return abs(mx - x) <= ON_PLAYER and abs(my - y) <= ON_PLAYER


def _count_camera(report: Report, moved: int, measured: tuple[int, int]) -> None:
    """Set the tracker's measure of the camera's move since the last look against the
    game's (which only ever moves across)."""
    if abs(moved) > MOST_CAMERA:
        report.camera_cuts += 1
    elif moved:
        report.camera_moved += 1
        report.camera_moved_right += measured == (moved, 0)
    else:
        report.camera_still += 1
        report.camera_still_right += measured == (0, 0)


def _floor_either_side(game: ToyHopper) -> tuple[int, int]:
    """The steps of floor to the player's right and left before a gap, by the game's
    own level, counted as the reading counts them: from one pixel past its edge and
    outline, 8 pixels a step, as far as 48 pixels or the edge of the screen."""
    st = game.state
    holes = {col for first, wide in game.gaps for col in range(first, first + wide)}
    reach = surroundings.MOST * surroundings.STEP
    x = st.x - st.scroll

    def steps(first: int, way: int) -> int:
        for d in range(reach):
            column = first + way * d
            if not 0 <= column < toy_hopper.SCREEN_WIDTH:
                break
            if (column + st.scroll) // toy_hopper.TILE in holes:
                return d // surroundings.STEP
        return surroundings.MOST

    right = steps(x + toy_hopper.PLAYER_W + surroundings.MARGIN, 1)
    left = steps(x - surroundings.MARGIN - 1, -1)
    return right, left


def _read_memory(training: TrainingService, discovery: DiscoveryService, report: Report) -> None:
    layout = toy_hopper.LAYOUT
    encoder = training.encoder
    if encoder is not None and encoder.pairs:
        mine = encoder.pairs[0][1]
        report.own_position = f"{mine.region}[{mine.offset}]"
        report.own_position_right = mine.offset in (layout.x_low, layout.x_page)
    knowledge = discovery.knowledge
    if knowledge is None:
        return
    kinds = {v.offset: v.kind for v in knowledge if not v.rejected}
    report.lives_found = kinds.get(layout.lives) is VariableKind.LIVES
    report.score_found = kinds.get(layout.score) is VariableKind.SCORE


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=PROG,
        description="Run the whole app on the practice platform game and measure it.",
    )
    parser.add_argument("--goes", type=_whole, default=DEFAULT_GOES, help="goes to train for")
    parser.add_argument("--steps", type=_whole, default=DEFAULT_STEPS, help="steps in a go")
    parser.add_argument("--seed", type=_level, default=0, help="which level (0 to 65535)")
    parser.add_argument(
        "--algorithm",
        choices=algorithm_choices(),
        default=DEFAULT_ALGORITHM,
        help="how the AI learns",
    )
    parser.add_argument(
        "--play-frames",
        type=_whole,
        default=DEFAULT_PLAY_FRAMES,
        help="frames of the scripted first go discovery learns from",
    )
    return parser


def _whole(text: str) -> int:
    try:
        number = int(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"'{text}' is not a whole number.") from exc
    if number < 1:
        raise argparse.ArgumentTypeError("must be 1 or more.")
    return number


def _level(text: str) -> int:
    try:
        number = int(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"'{text}' is not a whole number.") from exc
    if not 0 <= number <= 0xFFFF:
        raise argparse.ArgumentTypeError("must be from 0 to 65535.")
    return number


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = run(
        args.goes,
        seed=args.seed,
        algorithm=args.algorithm,
        max_steps=args.steps,
        play_frames=args.play_frames,
    )
    for line in report.lines():
        print(line)
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through main() in the tests
    sys.exit(main())


__all__ = [
    "CAMERA_TARGET",
    "DEFAULT_ALGORITHM",
    "DEFAULT_GOES",
    "DEFAULT_PLAY_FRAMES",
    "DEFAULT_STEPS",
    "EXPLAINED_TARGET",
    "FLOOR_TARGET",
    "FOLLOWED_TARGET",
    "MOST_KINDS",
    "ON_PLAYER",
    "Report",
    "build_parser",
    "first_go",
    "main",
    "run",
]
