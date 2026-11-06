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
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

from retro_ai_lab.core import ConfigurationError, EmulatorError, UnsupportedCapabilityError
from retro_ai_lab.core.events import EventBus
from retro_ai_lab.observation.diff import ByteStats
from retro_ai_lab.services.connection import (
    ConnectionService,
    EmulatorConnected,
    EmulatorDisconnected,
)
from retro_ai_lab.services.experiments import ExperimentHarness
from retro_ai_lab.services.observer import MemoryObserved, ObservationService
from retro_ai_lab.storage.knowledge import KnowledgeStore, export, load
from retro_ai_lab.understanding.events import CYCLES, EventDetector, EventKind, EventLog, GameEvent
from retro_ai_lab.understanding.knowledge import (
    AI,
    Confidence,
    Evidence,
    Knowledge,
    Relationship,
    Variable,
    VariableKind,
)
from retro_ai_lab.understanding.linking import apply_input_links, find_input_links
from retro_ai_lab.understanding.longlook import LongLook
from retro_ai_lab.understanding.moments import (
    WINDOW_FRAMES,
    Moment,
    MomentFinder,
    Narrowing,
    RestartWatch,
)
from retro_ai_lab.understanding.relationships import RelationshipFinder
from retro_ai_lab.understanding.variables import (
    RARE_RATE,
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
CENSUS_MOST = 16  # bytes a census names: the ones with the most changes


MOVES_ALONE = "moves-alone"  # evidence kind: changed while nobody pressed anything (T-855)
# The kinds a player earns by playing; one that moves by itself is not one. Lives and
# health are left out: standing still is a fine way to lose them.
DOUBTED_KINDS = frozenset({VariableKind.SCORE, VariableKind.PROGRESS})
BOUNCES = "bounces"  # evidence kind: a "lives" byte going up nearly as often as down (T-864)
BUSY = "busy"  # evidence kind: a "lives" byte changing more than once in ten steps (T-865)
NOT_LIVES = frozenset({BOUNCES, BUSY})
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


def _names(variables: list[Variable], most: int = 5) -> str:
    names = [v.display_name for v in variables[:most]]
    if len(variables) > most:
        names.append(f"and {len(variables) - most} more")
    return ", ".join(names) or "none"


def game_id_for(system: str, title: str, rom_hash: str | None) -> str:
    """A stable id for a game: its hash when the emulator gives one."""
    return rom_hash or f"{system}:{title}"


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
    ) -> None:
        if review_every < 1:
            raise ValueError("review_every must be positive")
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
        self._summary: tuple[tuple[int, int, str | None], str] | None = None
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
        guesses are left alone. Returns the variables changed.
        """
        moved = frozenset(locations)
        if not moved:
            return ()
        with self._lock:
            k = self._require()
            doubted: list[Variable] = []
            dropped: list[Variable] = []
            for v in k:
                if v.confirmed or v.rejected or v.kind not in DOUBTED_KINDS:
                    continue
                if not any((v.region, o) in moved for o in range(v.offset, v.offset + v.width)):
                    continue
                if any(e.kind == MOVES_ALONE for e in v.evidence):
                    k.support(
                        v.id,
                        Evidence(
                            MOVES_ALONE,
                            "Changed again while nobody pressed anything; not something "
                            "the player earns.",
                            0.0,
                            frame,
                        ),
                    )
                    dropped.append(k.reject(v.id, by=AI))
                else:
                    doubted.append(
                        k.support(
                            v.id,
                            Evidence(
                                MOVES_ALONE,
                                "Changed while nobody pressed anything. Once is a doubt; "
                                "twice and the AI drops it.",
                                0.0,
                                frame,
                            ),
                        )
                    )
            changed = (*doubted, *dropped)
            if changed:
                log.info(
                    "%d guesses changed while nobody pressed anything: %d doubted (%s), "
                    "%d dropped (%s).",
                    len(changed),
                    len(doubted),
                    _names(doubted),
                    len(dropped),
                    _names(dropped),
                )
                self._announce(DiscoveryUpdated((), changed, (), ()))
            return changed

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
                try:
                    stats = tracker.stats(v.region, v.offset)
                except ConfigurationError:
                    continue
                judge, words = KIND_DOUBTS[v.kind]
                reason = judge(stats)
                if reason is None:
                    continue
                if reason == BOUNCES:
                    what = (
                        f"went up {stats.ups} times and down {stats.downs}; {words} does not bounce"
                    )
                else:
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
        """A go starts: remember how often every byte had changed so far (T-870)."""
        with self._lock:
            tracker = self._observation.tracker
            self._go_changes = {loc: st.changes for loc, st in tracker.all_stats().items()}
            self._go_transitions = tracker.transitions

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
        key = (id(k), k.version, k.wake_button)
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
        self._summary = (key, line)
        return line

    # ------------------------------------------------------------ actions

    def save(self) -> Path | None:
        """Write the knowledge to the store (no store: nothing happens)."""
        if self._store is None or self._knowledge is None:
            return None
        path = self._store.save(self._knowledge)
        self._saves += 1
        return path

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

    def name_value(self, region: str, offset: int, name: str) -> Variable:
        """A person names the value at an address (T-862).

        The AI's guess there is confirmed under that name; where it had none, a
        named value of unknown kind is made and confirmed, so the lists use the name.
        A guess the AI had rejected comes back as a named value of unknown kind:
        the name is the person's, the old kind was wrong (T-895).
        """
        name = name.strip()
        if not name:
            raise ConfigurationError("The value needs a name.")
        with self._lock:
            k = self._require()
            var = k.at(region, offset)
            if var is None:
                var = k.propose(
                    region,
                    offset,
                    name=name,
                    evidence=[Evidence("user", "Named by a person.", 0.0)],
                )
                k.correct(var.id, name=name)
                added: tuple[Variable, ...] = (k.get(var.id),)
                changed: tuple[Variable, ...] = ()
            elif var.rejected:
                k.correct(var.id, name=name, kind=VariableKind.UNKNOWN)
                added, changed = (), (k.get(var.id),)
            else:
                k.correct(var.id, name=name)
                added, changed = (), (k.get(var.id),)
        self._announce(DiscoveryUpdated(added, changed, (), ()))
        return k.get(var.id)

    def confirm(self, var_id: str) -> Variable:
        """A person says a guess is right; everything listening hears it (T-902)."""
        with self._lock:
            k = self._require()
            var = k.confirm(var_id)
        self._announce(DiscoveryUpdated((), (var,), (), ()))
        return var

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
        self._announce(DiscoveryUpdated((), (var,), (), ()))
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
        else:
            knowledge = Knowledge(game_id, title=info.title)
        # The change statistics start from zero with this connection, so the
        # "judged on N changes" records from an earlier run would never be reached (T-926).
        knowledge.forget_judged()
        self._adopt(knowledge)
        self._events.clear()
        self._buttons = tuple(b.name for b in event.buttons)
        self._harness = ExperimentHarness(self._connection)
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
        if self._enabled:
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
        events = self._detector.observe(event.snapshot)
        if self._look is not None:
            self._look.observe(event.snapshot)
        relationships: tuple[Relationship, ...] = ()
        if events:
            self._events.extend(events)
            relationships = self._finder.observe(events)
        if events or relationships:
            self._announce(DiscoveryUpdated((), (), events, relationships))
        if any(e.kind is EventKind.RESET for e in events):
            self._narrow_at_restart()
        self._since_review += 1
        self._since_experiment += 1
        if self._since_review >= self._review_every:
            self.review()
        if (
            self._experiment_every is not None
            and self._since_experiment >= self._experiment_every
            and self._experiments_allowed
            and self.can_experiment()
        ):
            try:
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
            self.save()
        self._bus.publish(update)


__all__ = [
    "DEFAULT_EXPERIMENT_EVERY",
    "DEFAULT_REVIEW_EVERY",
    "DiscoveryProgress",
    "DiscoveryService",
    "DiscoveryUpdated",
    "Question",
    "game_id_for",
]
