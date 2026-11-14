"""Discovery service: the automatic observe → think → experiment loop.

This is the Game Understanding Engine's conductor (DESIGN.md §3.4). While
a game is connected it:

* feeds every memory snapshot to the event detector and relationship
  finder, so the knowledge's story grows as the child plays;
* every ``review_every`` frames, turns the change statistics into new
  variable proposals (``find_candidates`` → ``propose_candidates``);
* when the emulator supports save states, every ``experiment_every``
  frames runs the button-linking experiments and folds them in, from
  whatever situation the game is in at that moment;
* with a :class:`KnowledgeStore`, loads what was learned about the game
  last time and saves whenever a belief is added or changed, on
  disconnect, and on request (``save``), plus ``export_to`` / ``import_from``
  for sharing files.

Everything it learns goes into :attr:`knowledge`, and every step is
announced on the bus (:class:`DiscoveryProgress`, :class:`DiscoveryUpdated`)
so the UI can show what the AI is doing.
"""

from __future__ import annotations

import contextlib
import logging
import threading
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from retro_ai_lab.core import ConfigurationError, EmulatorError, UnsupportedCapabilityError
from retro_ai_lab.core.events import EventBus
from retro_ai_lab.core.worklog import app_work
from retro_ai_lab.observation.diff import ByteStats
from retro_ai_lab.services.connection import (
    ConnectionService,
    EmulatorConnected,
    EmulatorDisconnected,
)
from retro_ai_lab.services.experiments import ExperimentHarness
from retro_ai_lab.services.observer import MemoryObserved, ObservationService
from retro_ai_lab.storage.knowledge import KnowledgeStore, export, load
from retro_ai_lab.understanding.aftermath import Aftermath
from retro_ai_lab.understanding.events import (
    CYCLES,
    HOME_OFTEN,
    EventDetector,
    EventKind,
    EventLog,
    GameEvent,
)
from retro_ai_lab.understanding.knowledge import (
    AI,
    Confidence,
    Encoding,
    Evidence,
    Knowledge,
    Relationship,
    Variable,
    VariableKind,
)
from retro_ai_lab.understanding.linking import InputLinks, apply_input_links, find_input_links
from retro_ai_lab.understanding.longlook import LongLook
from retro_ai_lab.understanding.moments import (
    WINDOW_FRAMES,
    Moment,
    MomentFinder,
    Narrowing,
    RestartWatch,
)
from retro_ai_lab.understanding.player import PlayerFinding
from retro_ai_lab.understanding.relationships import RelationshipFinder
from retro_ai_lab.understanding.variables import (
    RARE_RATE,
    clock_top,
    find_candidates,
    flag_doubt,
    lives_doubt,
    progress_doubt,
    propose_candidates,
    rethink_candidates,
)

log = logging.getLogger(__name__)

DEFAULT_REVIEW_EVERY = 120
DEFAULT_EXPERIMENT_EVERY = 600
DEFAULT_CENSUS_EVERY = 25  # reviews between census lines in the log (T-919)
SAVE_EVERY_S = 60.0  # the least time between two saves the AI makes on its own (T-949)
CENSUS_MOST = 16  # bytes a census names: the ones with the most changes


MOVES_ALONE = "moves-alone"  # evidence kind: changed while nobody pressed anything (T-855)
# The kinds a player earns by playing; one that moves by itself is not one. Lives and
# health are left out: standing still is a fine way to lose them.
DOUBTED_KINDS = frozenset({VariableKind.SCORE, VariableKind.PROGRESS})
# The byte above one that moves by itself, when it counts that byte's roll-overs and
# that byte ticks at one pace, is the top of a clock (T-996). It is caught moving once
# in 64 probes, so it is judged by the byte below it. A flag is one too until it shows
# a third value, and then "progress".
CLOCK_TOP = "clock-top"
CLOCK_KINDS = frozenset({VariableKind.SCORE, VariableKind.PROGRESS, VariableKind.FLAG})
_CLOCK_FIRST = (
    "Counts the roll-overs of the value below it, which ticks by itself "
    "at one pace: the top of a clock, not something the player earns. "
    "Once is a doubt; twice and the AI drops it."
)
_CLOCK_AGAIN = "Still the top of a clock that ticks by itself; not something the player earns."
_CLOCK_WHAT = "count the roll-overs of a clock"
BOUNCES = "bounces"  # evidence kind: a "lives" byte going up nearly as often as down (T-864)
BUSY = "busy"  # evidence kind: a "lives" byte changing more than once in ten steps (T-865)
AT_LOSS = "at-loss"  # evidence kind: a "progress" byte that goes up as lives are lost (T-997)
NOT_LIVES = frozenset({BOUNCES, BUSY, AT_LOSS})
# Kinds a game has one of: confirming one sets the AI's other guesses of it aside (T-939).
ONE_PER_GAME = frozenset({VariableKind.LIVES, VariableKind.HEALTH, VariableKind.SCORE})
# Kinds re-checked at every review against the byte's fresh statistics (T-864 to T-866, T-869):
# the doubt function says why the byte is not that kind, the words name the kind.
PLAYED = "played"  # evidence kind: the byte moved during a go, so it is worth a seat (T-870)
PLAYED_WEIGHT = 0.5
PLAYED_GOES = 3  # credit per go, this many goes at most (T-871)
KIND_DOUBTS: dict[VariableKind, tuple[Callable[[ByteStats], str | None], str]] = {
    VariableKind.LIVES: (lives_doubt, "a life count"),
    VariableKind.PROGRESS: (progress_doubt, "progress"),
    VariableKind.FLAG: (flag_doubt, "a flag"),
}


@dataclass(frozen=True, slots=True)
class DiscoveryProgress:
    """What the engine is doing right now, in plain words."""

    stage: str  # "idle", "watching", "thinking", "experimenting"
    detail: str


@dataclass(frozen=True, slots=True)
class Question:
    """A confirmed guess whose byte behaves unlike its kind (T-895).

    A person's word is kept, so the AI never drops a confirmed guess; it
    asks instead, in the Learning panel's insights and the log.
    """

    var_id: str
    name: str
    reason: str  # BOUNCES or BUSY
    text: str


@dataclass(frozen=True, slots=True)
class DiscoveryUpdated:
    """Something in the knowledge changed."""

    added: tuple[Variable, ...]
    changed: tuple[Variable, ...]
    events: tuple[GameEvent, ...]
    relationships: tuple[Relationship, ...]

    @property
    def is_empty(self) -> bool:
        return not (self.added or self.changed or self.events or self.relationships)


RESTARTS_WEIGHT = 9.0  # the AI's own restart finding outranks a planted or classified guess (T-904)


def _size_words(guesses: int, relationships: int, path: Path) -> str:
    """``"1,600 guesses, 20,000 relationships, 6.4 MB"`` for the log (T-955)."""
    try:
        size = path.stat().st_size
    except OSError:
        size = 0
    return (
        f"{guesses:,} guess{'es' if guesses != 1 else ''}, "
        f"{relationships:,} relationship{'s' if relationships != 1 else ''}, "
        f"{size / 1_000_000:.1f} MB"
    )


def _names(variables: list[Variable], most: int = 5) -> str:
    names = [v.display_name for v in variables[:most]]
    if len(variables) > most:
        names.append(f"and {len(variables) - most} more")
    return ", ".join(names) or "none"


def _say_doubts(lists: tuple[list[Variable], list[Variable]], what: str) -> None:
    """The log's line for guesses doubted and dropped for one reason, if there are any."""
    doubted, dropped = lists
    if doubted or dropped:
        log.info(
            "%d guesses %s: %d doubted (%s), %d dropped (%s).",
            len(doubted) + len(dropped),
            what,
            len(doubted),
            _names(doubted),
            len(dropped),
            _names(dropped),
        )


def game_id_for(system: str, title: str, rom_hash: str | None) -> str:
    """A stable id for a game: its hash when the emulator gives one."""
    return rom_hash or f"{system}:{title}"


def _experiment_words(k: Knowledge, links: InputLinks) -> str:
    """One line on what the buttons moved, for the log (T-929)."""
    moved: list[str] = []
    seen: set[str] = set()
    for loc in sorted(links.locations()):
        v = k.at(loc[0], loc[1])
        if v is None or v.id in seen:
            continue
        seen.add(v.id)
        moved.append(f"{v.display_name} ({', '.join(links.buttons_for(loc))})")
    dead = links.dead_buttons()
    words = (
        f"{len(moved)} values move with the buttons: {'; '.join(moved)}"
        if moved
        else "no value moves with the buttons here"
    )
    if dead:
        words += f". Buttons that moved nothing: {', '.join(dead)}"
    if links.sweeping:  # a pause or a menu (T-930)
        words += f". Buttons that stop or replace the whole game: {', '.join(links.sweeping)}"
    return words + "."


class DiscoveryService:
    def __init__(
        self,
        connection: ConnectionService,
        observation: ObservationService,
        bus: EventBus,
        *,
        review_every: int = DEFAULT_REVIEW_EVERY,
        experiment_every: int | None = DEFAULT_EXPERIMENT_EVERY,
        event_window: int = 5,
        store: KnowledgeStore | None = None,
        moment_window: int = WINDOW_FRAMES,
        census_every: int = DEFAULT_CENSUS_EVERY,
        save_every_s: float = SAVE_EVERY_S,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if review_every < 1:
            raise ValueError("review_every must be positive")
        if save_every_s < 0:
            raise ValueError("save_every_s cannot be negative")
        self._save_every_s = save_every_s
        self._clock = clock
        self._last_save: float | None = None
        self._dirty = False  # changes the store has not seen yet (T-949)
        self._writer: threading.Thread | None = None  # the save's file writer (T-956)
        if census_every < 1:
            raise ValueError("census_every must be positive")
        self._census_every = census_every
        if experiment_every is not None and experiment_every < 1:
            raise ValueError("experiment_every must be positive or None")
        self._connection = connection
        self._observation = observation
        self._bus = bus
        self._review_every = review_every
        self._experiment_every = experiment_every
        self._event_window = event_window
        self._store = store
        self._saves = 0
        self._knowledge: Knowledge | None = None
        self._detector: EventDetector | None = None
        self._asked: frozenset[str] = frozenset()  # values a person rewards up/down (T-860)
        self._look: LongLook | None = None
        self._aftermath = Aftermath()  # T-997; a new one with each knowledge
        # The bytes a probe has seen change with nothing pressed, and whether the tops
        # of clocks above them have been judged since the last go began (T-1025).
        self._moves_alone: frozenset[tuple[str, int]] = frozenset()
        self._clocks_judged = False
        self._finder: RelationshipFinder | None = None
        self._events = EventLog()
        # Snapshots arrive from the play clock's worker and from the GUI thread
        # (T-818); the detector, finder and long look are driven one at a time (T-847).
        self._lock = threading.RLock()
        self._harness: ExperimentHarness | None = None
        self._buttons: tuple[str, ...] = ()
        self._since_review = 0
        self._since_experiment = 0
        self._go_changes: dict[tuple[str, int], int] | None = None  # at the go's start (T-870)
        self._go_transitions = 0  # the tracker's count when the go began (T-871)
        self._experiments = 0
        self._reviews = 0
        self._questions: dict[str, Question] = {}  # confirmed guesses to check (T-895)
        # "I just lost a life" narrowing (T-901); the window is a setting for tests
        self._moments = MomentFinder(window_frames=moment_window)
        self._restarts = RestartWatch(window_frames=moment_window)  # the AI's own presses (T-904)
        self._stage = "idle"
        self._enabled = False
        self._experiments_allowed = True  # the training service holds them back (T-839)
        # The summary line, remembered with what it was built from (T-907): a review asks
        # for it on the worker and the panel twice more on the GUI thread.
        self._summary: tuple[tuple[int, int, str | None, PlayerFinding | None], str] | None = None
        bus.subscribe(EmulatorConnected, self._on_connected)
        bus.subscribe(EmulatorDisconnected, self._on_disconnected)
        bus.subscribe(MemoryObserved, self._on_memory)

    # ------------------------------------------------------------ queries

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def knowledge(self) -> Knowledge | None:
        """What has been learned about the connected game (None when idle)."""
        return self._knowledge

    @property
    def events(self) -> EventLog:
        return self._events

    @property
    def stage(self) -> str:
        return self._stage

    @property
    def reviews(self) -> int:
        return self._reviews

    @property
    def experiments(self) -> int:
        return self._experiments

    @property
    def store(self) -> KnowledgeStore | None:
        return self._store

    @property
    def saves(self) -> int:
        return self._saves

    @property
    def pending_save(self) -> bool:
        """Are there changes the store has not seen yet? (T-949)

        The AI's own changes are written at most once every ``save_every_s``
        while it plays; the rest wait here for the next due save, a stop, a
        disconnect or an explicit :meth:`save`.
        """
        return self._dirty

    def can_experiment(self) -> bool:
        return self._harness is not None and self._harness.available()

    @property
    def experiments_allowed(self) -> bool:
        """May the scheduled button experiments run now? (T-839)

        An experiment takes a save state, runs a trial per button and loads the
        state back: dozens of round trips that freeze the window and move the
        game under whoever is playing, so training holds them back.
        """
        return self._experiments_allowed

    def set_experiments_allowed(self, allowed: bool) -> None:
        self._experiments_allowed = bool(allowed)

    @property
    def experiment_due(self) -> bool:
        """Has the schedule come round while experiments were held back? (T-839)"""
        return (
            self._experiment_every is not None and self._since_experiment >= self._experiment_every
        )

    def run_due_experiment(self) -> bool:
        """Run the scheduled experiment now if one is due and possible (T-839).

        The training service calls this between goes, when the game is about to
        be put back to the checkpoint anyway. Returns True if one ran.
        """
        if not self._enabled or not self.experiment_due or not self.can_experiment():
            return False
        self.experiment()
        return True

    def _say_home_goers(self, variable_ids: tuple[str, ...]) -> None:
        """Values the detector has now seen come home in mid-play often enough
        (T-1021): said once, in plain words. Nothing in what the AI knows is changed."""
        k = self._knowledge
        if not variable_ids or k is None:
            return
        names = ", ".join(k.get(i).display_name for i in variable_ids if i in k)
        if names:
            log.info(
                "%s went back to where %s started %d times while most other values stayed "
                "where they were. That alone is no longer read as the game starting over.",
                names,
                "it" if len(variable_ids) == 1 else "they",
                HOME_OFTEN,
            )

    def doubt_reset(self, variable_ids: Iterable[str]) -> tuple[Variable, ...]:
        """These values went back to their start every few frames: not a restart (T-845).

        Each gets ``cycles`` evidence (weight 0, so how sure the AI is of the value
        itself does not change) and the detector leaves it out of the reset
        judgement from then on. Returns the variables changed.
        """
        with self._lock:
            k = self._require()
            changed: list[Variable] = []
            for var_id in dict.fromkeys(variable_ids):
                if var_id not in k or any(e.kind == CYCLES for e in k.get(var_id).evidence):
                    continue
                changed.append(
                    k.support(
                        var_id,
                        Evidence(
                            CYCLES,
                            "Goes back to its first value every few frames; that is not the "
                            "game starting over.",
                            0.0,
                        ),
                    )
                )
            if changed:
                names = ", ".join(v.display_name for v in changed)
                log.info(
                    "%s cycle back to their start; no longer reading that as a restart.", names
                )
                self._announce(DiscoveryUpdated((), tuple(changed), (), ()))
            return tuple(changed)

    def doubt_self_moving(
        self, locations: Iterable[tuple[str, int]], *, frame: int | None = None
    ) -> tuple[Variable, ...]:
        """These bytes changed while nobody pressed anything (T-855).

        A score or progress guess covering one of them is doubted the first time
        and rejected by the AI the second time; confirmed and already rejected
        guesses are left alone. So is a score, progress or flag guess on the byte
        above one of them when that byte is the top of a clock (T-996,
        :func:`clock_top`): it counts the roll-overs of something that ticks by
        itself. Returns the variables changed.

        What moved is remembered for the game (T-1025): the probe that calls this
        runs at few goes, and :meth:`begin_go` asks the clock's question of the same
        bytes at every one.
        """
        moved = frozenset(locations)
        if not moved:
            return ()
        with self._lock:
            k = self._require()
            self._moves_alone |= moved
            self._clocks_judged = True
            tops = self._clock_tops(moved)
            alone: tuple[list[Variable], list[Variable]] = ([], [])  # doubted, dropped
            clocks: tuple[list[Variable], list[Variable]] = ([], [])
            for v in k:
                if v.confirmed or v.rejected:
                    continue
                span = [(v.region, o) for o in range(v.offset, v.offset + v.width)]
                if v.kind in DOUBTED_KINDS and any(byte in moved for byte in span):
                    kind, lists = MOVES_ALONE, alone
                    first = (
                        "Changed while nobody pressed anything. Once is a doubt; "
                        "twice and the AI drops it."
                    )
                    again = (
                        "Changed again while nobody pressed anything; not something "
                        "the player earns."
                    )
                elif v.kind in CLOCK_KINDS and (v.region, v.offset) in tops:
                    kind, lists = CLOCK_TOP, clocks
                    first, again = _CLOCK_FIRST, _CLOCK_AGAIN
                else:
                    continue
                self._doubt_or_drop(k, v, kind, first, again, frame, lists)
            changed = (*alone[0], *alone[1], *clocks[0], *clocks[1])
            _say_doubts(alone, "changed while nobody pressed anything")
            _say_doubts(clocks, _CLOCK_WHAT)
            if changed:
                self._announce(DiscoveryUpdated((), changed, (), ()))
            return changed

    @staticmethod
    def _doubt_or_drop(
        k: Knowledge,
        v: Variable,
        kind: str,
        first: str,
        again: str,
        frame: int | None,
        lists: tuple[list[Variable], list[Variable]],
    ) -> None:
        """Once is a doubt; with a doubt of this kind already, the AI drops the guess."""
        if any(e.kind == kind for e in v.evidence):
            k.support(v.id, Evidence(kind, again, 0.0, frame))
            lists[1].append(k.reject(v.id, by=AI))
        else:
            lists[0].append(k.support(v.id, Evidence(kind, first, 0.0, frame)))

    def _judge_clocks(self) -> None:
        """A go starts: ask the clock's question of every byte a probe has seen move by
        itself (T-1025). The rule needs the upper byte to have changed three times,
        which no first probe has seen, and after that probe one comes only when a go
        ends stuck or by a reset: on most of the practice levels the top of the
        level's clock stayed "progress" for most of a session or all of it, 10 points
        every 256 frames for standing still. Exactly as at a probe: a doubt the first
        time, dropped the second, a person's word left alone."""
        k = self._knowledge
        if k is None or not self._moves_alone:
            return
        tops = self._clock_tops(self._moves_alone)
        clocks: tuple[list[Variable], list[Variable]] = ([], [])
        frame = self._observation.frame
        for v in k:
            if v.confirmed or v.rejected or v.kind not in CLOCK_KINDS:
                continue
            if (v.region, v.offset) in tops:
                self._doubt_or_drop(k, v, CLOCK_TOP, _CLOCK_FIRST, _CLOCK_AGAIN, frame, clocks)
        _say_doubts(clocks, _CLOCK_WHAT)
        changed = (*clocks[0], *clocks[1])
        if changed:
            self._announce(DiscoveryUpdated((), changed, (), ()))

    def _clock_tops(self, moved: frozenset[tuple[str, int]]) -> set[tuple[str, int]]:
        """The bytes just above ``moved`` ones that are the tops of clocks (T-996)."""
        tracker = self._observation.tracker
        tops: set[tuple[str, int]] = set()
        for region, offset in moved:
            try:
                low, high = tracker.stats(region, offset), tracker.stats(region, offset + 1)
            except ConfigurationError:
                continue  # the last byte of a region, or one never observed
            if clock_top(low, high):
                tops.add((region, offset + 1))
        return tops

    @property
    def aftermath(self) -> Aftermath:
        """Which values rise as something is lost (T-997); a new one with each game."""
        return self._aftermath

    @property
    def questions(self) -> tuple[Question, ...]:
        """Confirmed guesses whose bytes behave unlike their kind, for the person (T-895)."""
        with self._lock:
            return tuple(self._questions.values())

    def doubt_bouncing(self, *, frame: int | None = None) -> tuple[Variable, ...]:
        """A lives or progress guess that changes like no such value does is dropped.

        Read off the change tracker's fresh statistics (T-864 to T-866, T-869): a byte
        that goes up nearly as often as down ("bounces") or changes more than
        once in ten steps ("busy") is doubted the first time and rejected by the
        AI the next; already rejected guesses are left alone. A confirmed guess
        is a person's word, so it is kept and questioned instead (T-895): the
        question stays in :attr:`questions` while the byte behaves that way.
        Returns the variables doubted or dropped.
        """
        with self._lock:
            k = self._knowledge
            if k is None:
                return ()
            tracker = self._observation.tracker
            doubted: list[Variable] = []
            dropped: list[Variable] = []
            questioned: list[Variable] = []
            still: set[str] = set()
            for v in k:
                if v.rejected or v.kind not in KIND_DOUBTS:
                    continue
                judge, words = KIND_DOUBTS[v.kind]
                try:
                    stats = tracker.stats(v.region, v.offset)
                except ConfigurationError:
                    stats = None  # never observed: only when it rises can still be judged
                reason = judge(stats) if stats is not None else None
                rises_at_loss = v.kind is VariableKind.PROGRESS and self._aftermath.verdict(v.id)
                if reason is None and rises_at_loss:
                    reason = AT_LOSS  # T-997
                if reason is None or (reason != AT_LOSS and stats is None):
                    continue
                if reason == AT_LOSS:
                    rises, met, _, _ = self._aftermath.counts(v.id)
                    what = (
                        f"went up {rises} times, {met} of them as a life was lost; "
                        "what goes up when you lose is not progress"
                    )
                elif reason == BOUNCES:
                    assert stats is not None
                    what = (
                        f"went up {stats.ups} times and down {stats.downs}; {words} does not bounce"
                    )
                else:
                    assert stats is not None
                    what = (
                        f"changed {stats.changes} times in {stats.samples} steps; {words} "
                        "changes rarely"
                    )
                if v.confirmed:
                    still.add(v.id)
                    text = (
                        f'You confirmed "{v.name}", but it {what}. The AI keeps your word; '
                        f'if it is not {words}, pick it under "What the AI is doing" and press '
                        '"No, that\'s wrong".'
                    )
                    question = Question(v.id, v.name, reason, text)
                    before = self._questions.get(v.id)
                    self._questions[v.id] = question
                    if before is None or (before.name, before.reason) != (v.name, reason):
                        questioned.append(v)  # a new question; fresh counts are kept quietly
                        log.info("%s", text)
                    continue
                if any(e.kind in NOT_LIVES for e in v.evidence):
                    k.support(v.id, Evidence(reason, f"Still {what}.", 0.0, frame))
                    dropped.append(k.reject(v.id, by=AI))
                else:
                    doubted.append(
                        k.support(
                            v.id,
                            Evidence(
                                reason,
                                f"{what[0].upper()}{what[1:]}. Once is a doubt; twice and "
                                "the AI drops it.",
                                0.0,
                                frame,
                            ),
                        )
                    )
            for var_id in [i for i in self._questions if i not in still]:
                gone = self._questions.pop(var_id)
                log.info('"%s" behaves like its kind again; nothing to check.', gone.name)
                with contextlib.suppress(ConfigurationError):  # forgotten meanwhile
                    questioned.append(k.get(var_id))
            changed = (*doubted, *dropped)
            if changed:
                log.info(
                    "%d guesses change like no life count, progress or flag does: %d doubted (%s), "
                    "%d dropped (%s).",
                    len(changed),
                    len(doubted),
                    _names(doubted),
                    len(dropped),
                    _names(dropped),
                )
            if changed or questioned:
                self._announce(DiscoveryUpdated((), (*changed, *questioned), (), ()))
            return changed

    def begin_go(self) -> None:
        """A go starts: remember how often every byte had changed so far (T-870), and
        drop a guess that has turned out to be the top of a clock (T-1025)."""
        with self._lock:
            tracker = self._observation.tracker
            self._aftermath.begin()  # nothing is matched across the game being put back
            self._go_changes = {loc: st.changes for loc, st in tracker.all_stats().items()}
            self._go_transitions = tracker.transitions
            # The tops of clocks, unless this go's probe has just asked (T-1025): a
            # probe and the go it starts are one look at the same numbers.
            judged, self._clocks_judged = self._clocks_judged, False
            if not judged:
                self._judge_clocks()

    def credit_played(self, *, frame: int | None = None) -> tuple[Variable, ...]:
        """The go is over: credit every guess whose bytes moved during it (T-870).

        Seats in the Rewards panel go by evidence, and every "progress" guess the
        classifier makes carries the same evidence, so ties went to the lowest
        address: a coin count at a high address lost its seat to junk that never
        moved while anyone played. A guess that moved during a go gets ``PLAYED``
        evidence, which outranks a guess that never has. The first build credited
        everything that moved, up to 247 guesses a go (T-871): now only a guess
        that moved rarely during the go (at most once in ten steps, as a coin
        count, a level or a score does, where positions and timers move every
        step) is credited, once per go and for ``PLAYED_GOES`` goes at most, so a
        value that moves go after go outranks a one-off. Called only for a go the
        game listened to; returns the variables credited.
        """
        with self._lock:
            before = self._go_changes
            self._go_changes = None
            k = self._knowledge
            if k is None or before is None:
                return ()
            tracker = self._observation.tracker
            steps = tracker.transitions - self._go_transitions
            if steps <= 0:
                return ()
            credited: list[Variable] = []
            for v in k:
                if v.rejected or v.confirmed:
                    continue
                if sum(e.kind == PLAYED for e in v.evidence) >= PLAYED_GOES:
                    continue
                moved = 0
                for offset in range(v.offset, v.offset + v.width):
                    try:
                        now = tracker.stats(v.region, offset).changes
                    except ConfigurationError:
                        continue
                    moved = max(moved, now - before.get((v.region, offset), 0))
                if 0 < moved <= RARE_RATE * steps:
                    credited.append(
                        k.support(
                            v.id,
                            Evidence(
                                PLAYED,
                                f"Moved {moved} times in a go of {steps} steps.",
                                PLAYED_WEIGHT,
                                frame,
                            ),
                        )
                    )
            if credited:
                log.info(
                    "%d guesses moved a few times while the AI played: %s.",
                    len(credited),
                    _names(credited),
                )
                self._announce(DiscoveryUpdated((), tuple(credited), (), ()))
            return tuple(credited)

    def summary(self) -> str:
        """One line for the status bar / panel header.

        Built in one pass over the variables and remembered with the
        knowledge's version and wake button (T-907), so the handlers that ask
        for it after every review get the string back, not another walk.
        """
        k = self._knowledge
        if k is None or not self._enabled:
            return "Not watching a game."
        key = (id(k), k.version, k.wake_button, k.player)
        if self._summary is not None and self._summary[0] == key:
            return self._summary[1]
        beliefs = confident = asks = 0
        for v in k:
            if v.rejected:
                continue
            beliefs += 1
            if v.confidence >= Confidence.MEDIUM:
                confident += 1
            if not v.confirmed and v.confidence in (Confidence.LOW, Confidence.MEDIUM):
                asks += 1
        parts = [f"Watching {beliefs} value{'s' if beliefs != 1 else ''}"]
        if confident:
            parts.append(f"{confident} fairly sure")
        if asks:
            parts.append(f"{asks} could use your help")
        line = ", ".join(parts) + "."
        if k.wake_button:
            line += f" {k.wake_button.capitalize()} gets the game going."
        if k.player is not None and k.player.confidence >= Confidence.MEDIUM:  # T-974
            x, y = k.player.start
            line += f" It found itself on the screen, near ({x}, {y})."
        self._summary = (key, line)
        return line

    # ------------------------------------------------------------ actions

    def save(self) -> Path | None:
        """Write the knowledge to the store now (no store: nothing happens).

        A write still running from :meth:`save_in_background` finishes first, so
        the file always ends with the newest state.
        """
        if self._store is None or self._knowledge is None:
            return None
        self.wait_for_save()
        started = time.perf_counter()
        with app_work.timed("discovery: save"):
            data = self._snapshot()
            path = self._store.save_dict(self._knowledge.game_id, data)
        self._log_saved(self._knowledge.title or self._knowledge.game_id, data, path, started)
        self._note_saved()
        return path

    def save_in_background(self) -> bool:
        """Take the snapshot now and write the file on a worker thread (T-956).

        The user's knowledge is 27 MB and a save took 1.2 s on the AI's thread,
        20 s of a 16-minute run. Only the snapshot (``to_dict``) needs the
        knowledge; encoding and writing it need nothing from the AI. Returns
        False, leaving the changes pending, while an earlier write still runs.
        """
        if self._store is None or self._knowledge is None or self.writing:
            return False
        store, knowledge = self._store, self._knowledge
        started = time.perf_counter()
        with app_work.timed("discovery: save"):
            data = self._snapshot()
        title = knowledge.title or knowledge.game_id
        game_id = knowledge.game_id

        def write() -> None:
            try:
                with app_work.timed("discovery: save write"):
                    path = store.save_dict(game_id, data)
            except ConfigurationError as exc:
                log.warning("%s The AI will try again at the next save.", exc)
                self._dirty = True
                return
            self._log_saved(title, data, path, started)

        self._writer = threading.Thread(target=write, name="knowledge-save", daemon=True)
        self._writer.start()
        self._note_saved()
        return True

    @property
    def writing(self) -> bool:
        """Is a background save still writing its file? (T-956)"""
        writer = self._writer
        return writer is not None and writer.is_alive()

    def wait_for_save(self, timeout: float | None = None) -> None:
        """Wait for a background save's file to be on disk (T-956)."""
        writer = self._writer
        if writer is not None and writer is not threading.current_thread():
            writer.join(timeout)

    def _snapshot(self) -> dict[str, Any]:
        if self._finder is not None:
            self._finder.flush()  # rates the finder has not written yet (T-946)
        assert self._knowledge is not None
        return self._knowledge.to_dict()

    def _note_saved(self) -> None:
        self._saves += 1
        self._dirty = False
        self._last_save = self._clock()

    @staticmethod
    def _log_saved(title: str, data: dict[str, Any], path: Path, started: float) -> None:
        log.info(
            "Saved what the AI learned about %s: %s in %d ms.",
            title,
            _size_words(len(data["variables"]), len(data["relationships"]), path),
            round((time.perf_counter() - started) * 1000),
        )

    def _save_soon(self) -> None:
        """Note a change for the store; write it now if a save is due (T-949).

        The user's log showed a 1.1 s save every 9 s of play, each one holding
        the AI's step: every review and every changed guess wrote the whole
        knowledge at once. Now the first change after a save waits until
        ``save_every_s`` has passed since that save; a stop, a disconnect and
        the Save button still write at once.
        """
        self._dirty = True
        self._save_if_due()

    def _save_if_due(self) -> None:
        if not self._dirty or self._store is None or self._knowledge is None:
            return
        if self._last_save is None or self._clock() - self._last_save >= self._save_every_s:
            self.save_in_background()

    def export_to(self, path: str | Path) -> Path:
        """Write the current knowledge anywhere, for sharing."""
        if self._knowledge is None:
            raise ConfigurationError("Nothing to export yet; connect to a game first.")
        return export(self._knowledge, path)

    def import_from(self, path: str | Path) -> Knowledge:
        """Replace the current game's knowledge with a shared file for the same game."""
        current = self._knowledge
        if current is None:
            raise ConfigurationError("Connect to a game before importing knowledge for it.")
        imported = load(path)
        if imported.game_id != current.game_id:
            raise ConfigurationError(
                f"That file is about {imported.title or imported.game_id!r}, "
                f"not {current.title or current.game_id!r}."
            )
        self._adopt(imported)
        self.save()
        self._announce(DiscoveryUpdated(tuple(imported), (), (), imported.relationships()))
        return imported

    def name_value(
        self,
        region: str,
        offset: int,
        name: str,
        *,
        encoding: Encoding = Encoding.U8,
        length: int = 0,
    ) -> Variable:
        """A person names the value at an address (T-862), in a layout (T-941).

        The AI's guess there is confirmed under that name; where it had none, a
        named value of unknown kind is made and confirmed, so the lists use the name.
        A guess the AI had rejected comes back as a named value of unknown kind:
        the name is the person's, the old kind was wrong (T-895). A layout other
        than the guess's (a score over six digit bytes where the AI saw one byte)
        is the person's word too: the value takes it, and a kind the AI judged on
        the old layout is unknown again unless a person had confirmed it. The AI's
        unconfirmed guesses on the other bytes of the value are set aside: a byte
        that is part of a value a person named is not a value of its own.
        """
        name = name.strip()
        if not name:
            raise ConfigurationError("The value needs a name.")
        width = length if encoding.sized else encoding.width
        with self._lock:
            k = self._require()
            var = k.at(region, offset)
            if var is None:
                var = k.propose(
                    region,
                    offset,
                    name=name,
                    encoding=encoding,
                    length=length,
                    evidence=[Evidence("user", "Named by a person.", 0.0)],
                )
                k.correct(var.id, name=name)
                added: tuple[Variable, ...] = (k.get(var.id),)
                changed: tuple[Variable, ...] = ()
            else:
                relaid = (var.encoding, var.width) != (encoding, width)
                kind = None
                if var.rejected or (relaid and not var.confirmed):
                    kind = VariableKind.UNKNOWN
                if relaid:
                    k.correct(var.id, name=name, kind=kind, encoding=encoding, length=length)
                else:
                    k.correct(var.id, name=name, kind=kind)
                added, changed = (), (k.get(var.id),)
            named = k.get(var.id)
            aside = self._set_bytes_aside(k, named)
        self._announce(DiscoveryUpdated(added, (*changed, *aside), (), ()))
        return named

    def _set_bytes_aside(self, k: Knowledge, var: Variable) -> list[Variable]:
        """The AI's unconfirmed guesses on the other bytes of a value a person named (T-941)."""
        aside: list[Variable] = []
        if var.width < 2:
            return aside
        for other in tuple(k):
            if other.id == var.id or other.rejected or other.confirmed:
                continue
            if other.region != var.region:
                continue
            if other.offset >= var.offset + var.width or var.offset >= other.offset + other.width:
                continue
            k.support(
                other.id,
                Evidence(
                    "your-word",
                    f"You named {var.display_name} over {var.address}; this byte is part of it.",
                    0.0,
                    self._observation.frame,
                ),
            )
            aside.append(k.reject(other.id, by="your word"))
        if aside:
            log.info(
                "You named %s over %s: %d guess%s on its bytes set aside (%s).",
                var.display_name,
                var.address,
                len(aside),
                "" if len(aside) == 1 else "es",
                _names(aside),
            )
        return aside

    def confirm(self, var_id: str) -> Variable:
        """A person says a guess is right; everything listening hears it (T-902)."""
        with self._lock:
            k = self._require()
            var = k.confirm(var_id)
            aside = self._set_others_aside(k, var)
        self._announce(DiscoveryUpdated((), (var, *aside), (), ()))
        return var

    def _set_others_aside(self, k: Knowledge, var: Variable) -> list[Variable]:
        """A game has one life count, one health and one score (T-939).

        Once a person confirms one, the AI's own other guesses of that kind are set
        aside by the person's word: the user's 950-go log counted "45 lives lost" a go
        from three timer-like life counts beside the confirmed one, and they sat in
        what the AI sees and in its setback events every step.
        """
        if var.kind not in ONE_PER_GAME:
            return []
        label = var.kind.label.lower()
        aside: list[Variable] = []
        for other in k.by_kind(var.kind):
            if other.id == var.id or other.confirmed:
                continue
            k.support(
                other.id,
                Evidence(
                    "your-word",
                    f"You confirmed {var.display_name} as the game's {label}; a game has one.",
                    0.0,
                    self._observation.frame,
                ),
            )
            aside.append(k.reject(other.id, by="your word"))
        if aside:
            log.info(
                "You confirmed %s as the game's %s: %d other %s guess%s set aside (%s).",
                var.display_name,
                label,
                len(aside),
                label,
                "" if len(aside) == 1 else "es",
                _names(aside),
            )
        return aside

    def reject(self, var_id: str) -> Variable:
        """A person says a guess is wrong; the rewards and steps built on it go too (T-902)."""
        with self._lock:
            k = self._require()
            var = k.reject(var_id)
        self._announce(DiscoveryUpdated((), (var,), (), ()))
        return var

    def rename(self, var_id: str, name: str) -> Variable:
        """A person names a guess, which also confirms it; every list follows (T-902)."""
        name = name.strip()
        if not name:
            raise ConfigurationError("The value needs a name.")
        with self._lock:
            k = self._require()
            var = k.correct(var_id, name=name)
            aside = self._set_others_aside(k, var)
        self._announce(DiscoveryUpdated((), (var, *aside), (), ()))
        return var

    def tell(self, moment: Moment) -> Narrowing:
        """A person says what just happened; the byte that shows it is narrowed down (T-901).

        Each press keeps the bytes that moved the right way in the last few seconds
        and intersects them with the presses before. When one byte is left after
        enough presses it becomes the person's word: the guess there (or a new
        value) is confirmed with the moment's kind and name, and the AI's own
        guesses of that kind that did not move are rejected by the AI, so the
        rewards and the steps follow the right byte.
        """
        with self._lock:
            k = self._require()
            # A byte the person confirmed as something else cannot be this (T-905):
            # the score goes up whenever one gets further, but it is not progress.
            skip = {
                (v.region, v.offset)
                for v in k
                if v.confirmed
                and not v.rejected
                and v.kind not in (moment.kind, VariableKind.UNKNOWN)
            }
            narrowing = self._moments.tell(moment, self._observation.ring, skip=skip)
            log.info("%s", narrowing.text)
            if narrowing.settled is None:
                return narrowing
            region, offset = narrowing.settled
            told = Evidence(
                "told",
                f"You said you {moment.words} {narrowing.presses} times and this "
                f"{moment.movement} each time.",
                0.0,
                self._observation.frame,
            )
            var = k.at(region, offset)
            added: tuple[Variable, ...] = ()
            if var is None:
                var = k.propose(region, offset, kind=moment.kind, evidence=[told])
                added = (var,)
            else:
                k.support(var.id, told)
            keep_name = var.name if var.name and not var.rejected else moment.name
            k.correct(var.id, name=keep_name, kind=moment.kind, by="you")
            changed: list[Variable] = [] if added else [k.get(var.id)]
            for other in k.by_kind(moment.kind):
                if other.id == var.id or other.confirmed:
                    continue
                k.support(
                    other.id,
                    Evidence(
                        "told-otherwise",
                        f"You said you {moment.words}; this did not move like {moment.name} does.",
                        0.0,
                        self._observation.frame,
                    ),
                )
                changed.append(k.reject(other.id, by=AI))
            if len(changed) > (0 if added else 1):
                log.info(
                    "%d other %s guesses did not move when you %s: dropped (%s).",
                    len(changed) - (0 if added else 1),
                    moment.kind.label.lower(),
                    moment.words,
                    _names([c for c in changed if c.id != var.id]),
                )
            self._announce(DiscoveryUpdated(added, tuple(changed), (), ()))
            return narrowing

    def _narrow_at_restart(self) -> None:
        """A restart just happened: the AI presses "I just lost a life" for itself (T-904).

        Bytes the AI already believes are something else (fairly surely), and
        bytes anyone rejected, get no vote. When one byte has gone down by one
        before most restarts, it becomes the AI's lives guess with evidence
        strong enough to take the seat from its other lives guesses. Nothing is
        dropped: the AI's finding adds to what it knows, and the reviews keep
        judging every guess. The person's word stays above the AI's: a lives
        guess the person confirmed is kept whatever the restarts say.
        """
        k = self._knowledge
        if k is None:
            return
        skip: set[tuple[str, int]] = set()
        for v in k:
            other_kind = v.kind not in (VariableKind.LIVES, VariableKind.UNKNOWN)
            if v.rejected or (other_kind and v.confidence.value >= Confidence.MEDIUM.value):
                skip.add((v.region, v.offset))
        found = self._restarts.restart(self._observation.ring, skip=skip)
        if found is None:
            return
        region, offset = found
        best = k.best(VariableKind.LIVES)
        if best is not None and best.confirmed and (best.region, best.offset) != found:
            log.info(
                "%s[%d] went down by one before the last restarts, like a life counter, "
                'but you confirmed "%s"; the AI keeps your word.',
                region,
                offset,
                best.display_name,
            )
            return
        proof = Evidence(
            "restarts",
            "Went down by one just before each of the last restarts, the way a life counter does.",
            RESTARTS_WEIGHT,
            self._observation.frame,
        )
        var = k.at(region, offset)
        added: tuple[Variable, ...] = ()
        changed: tuple[Variable, ...] = ()
        if var is None:
            var = k.propose(region, offset, kind=VariableKind.LIVES, evidence=[proof])
            added = (var,)
        else:
            k.support(var.id, proof)
            if var.kind is not VariableKind.LIVES:
                k.update(var.id, kind=VariableKind.LIVES)
            changed = (k.get(var.id),)
        log.info(
            "The AI noticed %s[%d] went down by one before each of the last restarts: that "
            "looks like the life counter, so the rewards and steps follow it.",
            region,
            offset,
        )
        self._announce(DiscoveryUpdated(added, changed, (), ()))

    def ask_for(self, var_ids: Iterable[str]) -> None:
        """Have the detector report every move of these values as UP or DOWN (T-860)."""
        with self._lock:
            self._asked = frozenset(str(i) for i in var_ids)
            if self._detector is not None:
                self._detector.ask(self._asked)

    @property
    def asked(self) -> frozenset[str]:
        return self._asked

    def _adopt(self, knowledge: Knowledge) -> None:
        with self._lock:
            self._knowledge = knowledge
            self._detector = EventDetector(knowledge)
            self._detector.ask(self._asked)
            self._finder = RelationshipFinder(knowledge, window=self._event_window)
            self._look = LongLook(knowledge)
            self._aftermath = Aftermath()
            self._moves_alone = frozenset()  # another game: nothing seen to move yet

    def review(self) -> DiscoveryUpdated:
        """Turn the statistics gathered so far into proposals, right now."""
        with self._lock:
            k = self._require()
            self._set_stage(
                "thinking", "Looking for values that behave like scores, lives or counters."
            )
            candidates = find_candidates(self._observation.tracker)
            added = propose_candidates(k, candidates)
            rethought = list(rethink_candidates(k, candidates))
            if rethought:
                log.info(
                    "The AI changed its mind about %d of its own guesses now that it has seen "
                    "more: %s.",
                    len(rethought),
                    _names(rethought),
                )
            bounced = self.doubt_bouncing()
            changed = (
                *rethought,
                *bounced,
                *(self._look.review() if self._look is not None else ()),
            )
            self._reviews += 1
            self._since_review = 0
            if self._reviews % self._census_every == 0:
                log.info(
                    "Census of the busiest bytes after review %d:\n%s", self._reviews, self.census()
                )
            update = DiscoveryUpdated(added, changed, (), ())
            self._announce(update)
            self._set_stage("watching", self.summary())
            return update

    def census(self, most: int = CENSUS_MOST) -> str:
        """The busiest, smoothest and widest-ranging rare bytes, with the numbers the
        classifier reads, one line each (T-919, T-920).

        Three builds changed the mover rule on simulated bytes while the user's
        game kept showing no moving thing: the log named kinds but never the
        statistics behind them. Ranking by changes alone filled every line with
        random bytes and a frame counter (T-920), so a thing on screen for
        seconds of each go is looked for in the two lists that would hold it:
        the smooth bytes with the most changes, and the rare bytes with the
        most distinct values.
        """
        with self._lock:
            k = self._knowledge
            if k is None:
                return ""
            stats = self._observation.tracker.all_stats()
            moved = [(loc, st) for loc, st in stats.items() if st.changes > 0]
            lists = (
                ("Busiest", sorted(moved, key=lambda item: -item[1].changes)),
                (
                    "Smoothest",
                    sorted(
                        (item for item in moved if item[1].smooth),
                        key=lambda item: -item[1].changes,
                    ),
                ),
                (
                    "Widest-ranging rare",
                    sorted(
                        (item for item in moved if item[1].change_rate <= RARE_RATE),
                        key=lambda item: -len(item[1].distinct),
                    ),
                ),
            )
            lines: list[str] = []
            for heading, ranked in lists:
                lines.append(f"{heading}:")
                if not ranked:
                    lines.append("none")
                for (region, offset), st in ranked[:most]:
                    v = k.at(region, offset)
                    word = "nobody's guess" if v is None else v.display_name
                    if v is not None and v.rejected:
                        word += " (dropped)"
                    lines.append(
                        f"{region}[{offset}] {word}: {st.changes} changes in {st.samples} "
                        f"steps, {len(st.distinct)} values, hops small {st.small_hops} unit "
                        f"{st.unit_hops} largest {st.max_hop}, up {st.ups} down {st.downs} "
                        f"wraps {st.wraps}, repeat {st.same_hops} usual {st.usual_hop}"
                    )
            return "\n".join(lines)

    def experiment(self) -> DiscoveryUpdated:
        """Try every button from the current situation and learn what each does."""
        with self._lock:
            k = self._require()
            if self._harness is None or not self._harness.available():
                raise UnsupportedCapabilityError(
                    "save_states", "This emulator can't run experiments (no save states)."
                )
            self._set_stage("experimenting", "Trying each button to see what it changes.")
            self._harness.checkpoint()
            try:
                links = find_input_links(self._harness, self._buttons)
            finally:
                self._since_experiment = 0
            touched = apply_input_links(k, links)
            self._experiments += 1
            log.info("Experiment %d: %s", self._experiments, _experiment_words(k, links))
            added = tuple(
                v for v in touched if not any(e.kind != "input-linked" for e in v.evidence)
            )
            changed = tuple(v for v in touched if v not in added)
            update = DiscoveryUpdated(added, changed, (), ())
            self._announce(update)
            self._set_stage("watching", self.summary())
            return update

    # ------------------------------------------------------------- events

    def _on_connected(self, event: EmulatorConnected) -> None:
        info = event.game_info
        game_id = game_id_for(info.system, info.title, info.rom_hash)
        if self._store is not None:
            try:
                knowledge = self._store.load_or_new(game_id, title=info.title)
            except ConfigurationError:
                knowledge = Knowledge(game_id, title=info.title)  # a damaged file: start over
            if self._store.exists(game_id):  # how big it is tells how long a save takes (T-955)
                log.info(
                    "Picked up what the AI learned about %s: %s.",
                    info.title or game_id,
                    _size_words(
                        len(knowledge),
                        len(knowledge.relationships()),
                        self._store.path_for(game_id),
                    ),
                )
            else:
                log.info(
                    "Starting fresh on %s: nothing learned about it yet.", info.title or game_id
                )
        else:
            knowledge = Knowledge(game_id, title=info.title)
        # The change statistics start from zero with this connection, so the
        # "judged on N changes" records from an earlier run would never be reached (T-926).
        knowledge.forget_judged()
        self._adopt(knowledge)
        self._events.clear()
        self._buttons = tuple(b.name for b in event.buttons)
        self._harness = ExperimentHarness(
            self._connection, on_restore=self._observation.expect_jump
        )
        self._experiments_allowed = True
        self._since_review = 0
        self._since_experiment = 0
        self._experiments = 0
        self._reviews = 0
        self._questions = {}
        self._moments.forget()
        self._restarts.forget()
        self._enabled = event.capabilities.memory_read
        if self._enabled:
            self._set_stage("watching", f"Watching {info.title} to learn how it works.")
        else:
            self._set_stage(
                "idle", "This emulator can't share its memory, so there is nothing to learn from."
            )

    def _on_disconnected(self, event: EmulatorDisconnected) -> None:
        if self._enabled and (self._dirty or self._saves == 0):
            self.save()
        self._enabled = False
        self._harness = None
        self._set_stage("idle", "Not watching a game.")

    def _on_memory(self, event: MemoryObserved) -> None:
        with self._lock:
            self._observe_memory(event)

    def _observe_memory(self, event: MemoryObserved) -> None:
        if not self._enabled or self._knowledge is None:
            return
        assert self._detector is not None and self._finder is not None
        if event.went_back:
            # The app put the game back to an earlier moment of the go (T-1004). The
            # life that came back and the progress that went down are its doing: no
            # event, nothing paired or matched across the jump, nothing counted.
            with app_work.timed("discovery: events"):
                self._detector.rebase(event.snapshot)
            if self._look is not None:
                self._look.rebase(event.snapshot)
            self._finder.forget_recent()
            self._aftermath.begin()
            return
        # Each stage is timed for the "App work" log line (T-947).
        with app_work.timed("discovery: events"):
            events = self._detector.observe(event.snapshot, put_back=event.put_back)
            self._say_home_goers(self._detector.take_home_goers())
        if self._look is not None:
            with app_work.timed("discovery: long look"):
                self._look.observe(event.snapshot)
        relationships: tuple[Relationship, ...] = ()
        if events:
            self._events.extend(events)
            with app_work.timed("discovery: relationships"):
                relationships = self._finder.observe(events)
        self._aftermath.note(events, event.snapshot.frame)
        if events or relationships:
            with app_work.timed("discovery: announce"):
                self._announce(DiscoveryUpdated((), (), events, relationships))
        if any(e.kind is EventKind.RESET for e in events):
            self._narrow_at_restart()
        self._since_review += 1
        self._since_experiment += 1
        self._save_if_due()
        if self._since_review >= self._review_every:
            with app_work.timed("discovery: review"):
                self.review()
        if (
            self._experiment_every is not None
            and self._since_experiment >= self._experiment_every
            and self._experiments_allowed
            and self.can_experiment()
        ):
            try:
                with app_work.timed("discovery: experiment"):
                    self.experiment()
            except EmulatorError:
                # Link loss is announced by the connection service; carry on.
                self._since_experiment = 0

    # ------------------------------------------------------------ helpers

    def _require(self) -> Knowledge:
        if self._knowledge is None or not self._enabled:
            raise UnsupportedCapabilityError(
                "memory_read", "Connect to a game that shares its memory first."
            )
        return self._knowledge

    def _set_stage(self, stage: str, detail: str) -> None:
        self._stage = stage
        self._bus.publish(DiscoveryProgress(stage, detail))

    def _announce(self, update: DiscoveryUpdated) -> None:
        if update.is_empty:
            return
        if update.added or update.changed:
            self._save_soon()
        self._bus.publish(update)


__all__ = [
    "DEFAULT_EXPERIMENT_EVERY",
    "DEFAULT_REVIEW_EVERY",
    "SAVE_EVERY_S",
    "DiscoveryProgress",
    "DiscoveryService",
    "DiscoveryUpdated",
    "Question",
    "game_id_for",
]
