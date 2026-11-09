"""T-404: the reward service keeps a live, saved profile per game."""

from __future__ import annotations

from pathlib import Path

import pytest

from retro_ai_lab.core import Capabilities, ConfigurationError, UnsupportedCapabilityError
from retro_ai_lab.core.events import EventBus
from retro_ai_lab.emulator.fake.toy_racer import LAYOUT
from retro_ai_lab.emulator.registry import default_registry
from retro_ai_lab.rewards import Category, RewardProfile, is_button, press_id
from retro_ai_lab.rewards.suggestions import SuggestionReport, apply_suggestions
from retro_ai_lab.services import (
    ConnectionService,
    DiscoveryService,
    DiscoveryUpdated,
    GoStarted,
    ObservationService,
    RewardEarned,
    RewardProfileChanged,
    RewardService,
)
from retro_ai_lab.storage import RewardStore
from retro_ai_lab.understanding import (
    Encoding,
    EventKind,
    Evidence,
    GameEvent,
    Knowledge,
    Relationship,
    VariableKind,
)

RAM = "ram"
SURE = [Evidence("test", "", 6.0)]


def own_ids(profile: RewardProfile) -> tuple[str, ...]:
    """The profile's rows without the per-button rows every game gets (T-874)."""
    return tuple(i for i in profile.ids() if not is_button(i))


@pytest.fixture
def bus() -> EventBus:
    return EventBus(on_error=lambda exc, event, handler: (_ for _ in ()).throw(exc))


@pytest.fixture
def connection(bus: EventBus) -> ConnectionService:
    return ConnectionService(default_registry(), bus)


@pytest.fixture
def discovery(connection: ConnectionService, bus: EventBus) -> DiscoveryService:
    observation = ObservationService(connection, bus)
    return DiscoveryService(connection, observation, bus, experiment_every=None)


@pytest.fixture
def store(tmp_path: Path) -> RewardStore:
    return RewardStore(tmp_path / "rewards")


@pytest.fixture
def rewards(
    connection: ConnectionService, discovery: DiscoveryService, bus: EventBus, store: RewardStore
) -> RewardService:
    return RewardService(connection, discovery, bus, store=store)


@pytest.fixture
def earned(bus: EventBus) -> list[RewardEarned]:
    seen: list[RewardEarned] = []
    bus.subscribe(RewardEarned, seen.append)
    return seen


@pytest.fixture
def changes(bus: EventBus) -> list[RewardProfileChanged]:
    seen: list[RewardProfileChanged] = []
    bus.subscribe(RewardProfileChanged, seen.append)
    return seen


def teach(discovery: DiscoveryService, bus: EventBus) -> Knowledge:
    """Give the knowledge the toy racer's variables, the way a review would."""
    k = discovery.knowledge
    assert k is not None
    added = (
        k.propose(RAM, LAYOUT.lives, kind=VariableKind.LIVES, name="Lives", evidence=SURE),
        k.propose(
            RAM, LAYOUT.checkpoint, kind=VariableKind.PROGRESS, name="Checkpoint", evidence=SURE
        ),
        k.propose(
            RAM,
            LAYOUT.score,
            encoding=Encoding.U16LE,
            kind=VariableKind.SCORE,
            name="Score",
            evidence=SURE,
        ),
    )
    bus.publish(DiscoveryUpdated(added, (), (), ()))
    return k


def test_no_profile_until_a_memory_capable_game_connects(
    rewards: RewardService, connection: ConnectionService
) -> None:
    assert not rewards.enabled and rewards.profile is None
    assert rewards.summary() == "No game to reward."
    with pytest.raises(UnsupportedCapabilityError):
        rewards.increase("reset:*")
    connection.connect("fake", capabilities=Capabilities(screen_capture=True, frame_advance=True))
    assert not rewards.enabled and rewards.profile is None
    connection.disconnect()
    connection.connect("fake", seed=3)
    assert rewards.enabled
    assert rewards.profile is not None and rewards.profile.game_id == "toyracer-3"
    assert own_ids(rewards.profile) == ("reset:*",)  # nothing known yet but "Starting over"
    assert rewards.summary() == "1 reward switched on; +0.0 earned."


def test_new_beliefs_add_behaviours_and_announce(
    rewards: RewardService,
    connection: ConnectionService,
    discovery: DiscoveryService,
    bus: EventBus,
    changes: list[RewardProfileChanged],
) -> None:
    connection.connect("fake", seed=3)
    assert [c.reason for c in changes] == ["loaded", "suggestions"]
    teach(discovery, bus)
    assert changes[-1].reason == "suggestions" and len(changes[-1].added) == 4
    profile = rewards.profile
    assert profile is not None
    names = {b.name for b in profile}
    assert {"Losing Lives", "Gaining Lives", "Checkpoint goes up", "Score jumps up"} <= names
    assert rewards.refresh_suggestions().is_empty
    assert rewards.saves >= 2 and rewards.store is not None and rewards.store.exists("toyracer-3")


def test_a_rejected_guess_loses_its_rewards_at_once(
    rewards: RewardService,
    connection: ConnectionService,
    discovery: DiscoveryService,
    bus: EventBus,
    changes: list[RewardProfileChanged],
) -> None:
    """The user's rejected "Lifes" kept paying "Lifes goes up" (T-902)."""
    connection.connect("fake", seed=3)
    k = teach(discovery, bus)
    lives = k.at(RAM, LAYOUT.lives)
    assert lives is not None
    before = len(changes)
    discovery.reject(lives.id)  # the panel's No button
    profile = rewards.profile
    assert profile is not None
    names = {b.name for b in profile}
    assert "Losing Lives" not in names and "Gaining Lives" not in names
    assert len(changes) == before + 1 and changes[-1].removed
    discovery.confirm(lives.id)  # a change of mind brings them back
    assert "Losing Lives" in {b.name for b in rewards.profile or ()}


def test_a_person_adds_a_reward_on_a_value_going_up_or_down(
    rewards: RewardService,
    connection: ConnectionService,
    discovery: DiscoveryService,
    bus: EventBus,
    earned: list[RewardEarned],
    changes: list[RewardProfileChanged],
) -> None:
    """T-860: the row is added, the detector is asked, the points are earned, and it stays."""
    connection.connect("fake", seed=3)
    k = teach(discovery, bus)
    lane = k.propose(RAM, LAYOUT.lane, kind=VariableKind.POSITION, evidence=SURE)
    up = rewards.add_own(lane.id, goes_up=True, value=2.0)
    down = rewards.add_own(lane.id, goes_up=False, value=-2.0, name="Lane")
    assert up.name == "Your position goes up" and down.name == "Lane goes down"
    assert rewards.own() == (up, down) and discovery.asked == {lane.id}
    assert changes[-1].reason == "control" and changes[-1].added == (down.id,)
    bus.publish(DiscoveryUpdated((), (), (GameEvent(EventKind.UP, 9, lane.id, "", 1, 2),), ()))
    assert earned[-1].amount == 2.0 and earned[-1].names == (up.name,)
    assert rewards.refresh_suggestions().is_empty and rewards.own() == (up, down)
    with pytest.raises(ConfigurationError):
        rewards.add_own(lane.id, goes_up=True, value=1.0)  # already there
    with pytest.raises(ConfigurationError):
        rewards.add_own("nope", goes_up=True, value=1.0)
    with pytest.raises(ConfigurationError):
        rewards.remove_own("reset:*")
    # Saved: the rows and the asking come back after a reconnect.
    connection.disconnect()
    discovery.ask_for(())
    connection.connect("fake", seed=3)
    assert [b.id for b in rewards.own()] == [up.id, down.id] and discovery.asked == {lane.id}
    rewards.remove_own(up.id)
    rewards.remove_own(down.id)
    assert rewards.own() == () and discovery.asked == frozenset()
    assert changes[-1].removed == (down.id,)


def test_events_earn_rewards_into_the_tally(
    rewards: RewardService,
    connection: ConnectionService,
    discovery: DiscoveryService,
    bus: EventBus,
    earned: list[RewardEarned],
) -> None:
    connection.connect("fake", seed=3)
    k = teach(discovery, bus)
    lives = k.at(RAM, LAYOUT.lives)
    checkpoint = k.at(RAM, LAYOUT.checkpoint)
    assert lives and checkpoint
    crash = GameEvent(EventKind.SETBACK, 50, lives.id, "", 3, 2)
    step = GameEvent(EventKind.PROGRESS, 60, checkpoint.id, "", 0, 1)
    nothing = GameEvent(EventKind.FLAG_ON, 61, "unknown", "")
    bus.publish(DiscoveryUpdated((), (), (crash, step, nothing), ()))
    assert [e.amount for e in earned] == [-10.0, 10.0]
    assert earned[0].behaviour_ids == (f"setback:{lives.id}",) and earned[0].event is crash
    assert rewards.tally.total == 0.0 and rewards.tally.count == 2
    assert rewards.tally.by_category(rewards.profile or k)[Category.SURVIVAL] == -10.0  # type: ignore[arg-type]
    # Switched off: earns nothing and is not announced.
    rewards.set_enabled(f"progress:{checkpoint.id}", False)
    bus.publish(DiscoveryUpdated((), (), (step,), ()))
    assert len(earned) == 2 and rewards.tally.count == 2
    assert "+" in rewards.summary() or "-" in rewards.summary()


def test_progress_and_score_pay_only_past_the_best_of_the_go(
    rewards: RewardService,
    connection: ConnectionService,
    discovery: DiscoveryService,
    bus: EventBus,
    earned: list[RewardEarned],
) -> None:
    """T-878: a step back and forward again, or a value that resets on death, is not progress."""
    connection.connect("fake", seed=3)
    k = teach(discovery, bus)
    checkpoint = k.at(RAM, LAYOUT.checkpoint)
    score = k.at(RAM, LAYOUT.score)
    assert checkpoint and score
    bus.publish(GoStarted(0))

    def progress(frame: int, value: int) -> None:
        step = GameEvent(EventKind.PROGRESS, frame, checkpoint.id, "", 0, value)
        bus.publish(DiscoveryUpdated((), (), (step,), ()))

    for frame, value in ((10, 1), (20, 2), (30, 1), (40, 2), (50, 3)):
        progress(frame, value)
    assert [(e.frame, e.amount) for e in earned] == [(10, 10.0), (20, 10.0), (50, 10.0)]
    assert rewards.tally.count == 3  # the two capped steps are not tallied either
    jump = GameEvent(EventKind.SCORE_GAIN, 60, score.id, "", 0, 100)
    back = GameEvent(EventKind.SCORE_GAIN, 70, score.id, "", 20, 50)
    bus.publish(DiscoveryUpdated((), (), (jump, back), ()))
    assert [e.frame for e in earned[3:]] == [60]
    # A new go earns from scratch.
    bus.publish(GoStarted(100))
    progress(110, 1)
    assert earned[-1].frame == 110
    # A person's own "goes up" row is not capped: every step right pays.
    lane = k.propose(RAM, LAYOUT.lane, kind=VariableKind.POSITION, evidence=SURE)
    up = rewards.add_own(lane.id, goes_up=True, value=2.0)
    for frame in (120, 130):
        step_right = GameEvent(EventKind.UP, frame, lane.id, "", 1, 2)
        bus.publish(DiscoveryUpdated((), (), (step_right,), ()))
    assert [e.frame for e in earned if up.id in e.behaviour_ids] == [120, 130]
    # Without a value there is nothing to compare: it pays.
    bare = GameEvent(EventKind.PROGRESS, 140, checkpoint.id, "")
    bus.publish(DiscoveryUpdated((), (), (bare,), ()))
    assert earned[-1].frame == 140


def test_buttons_held_earn_their_points(
    rewards: RewardService,
    connection: ConnectionService,
    bus: EventBus,
    earned: list[RewardEarned],
) -> None:
    """T-874: a row per button and one for repeats, scored every step, silent at zero."""
    from retro_ai_lab.rewards import AGAIN_ID, Category, press_id
    from retro_ai_lab.services import ButtonsHeld

    bus.publish(ButtonsHeld(1, frozenset({"up"}), "ai"))  # nothing connected: ignored
    connection.connect("fake", seed=3)
    profile = rewards.profile
    assert profile is not None
    adapter = connection.adapter
    names = [b.name for b in adapter.buttons()]  # type: ignore[union-attr]
    rows = profile.by_category()[Category.CONTROLS]
    assert [b.id for b in rows] == [*(press_id(n) for n in names), AGAIN_ID]
    assert all(b.value == 0 for b in rows)
    bus.publish(ButtonsHeld(2, frozenset({"up"}), "ai"))
    bus.publish(ButtonsHeld(3, frozenset({"up"}), "ai"))
    assert earned == [] and rewards.tally.count == 0  # all at zero: nothing to say
    rewards.set_value(press_id("up"), 2)
    rewards.set_value(AGAIN_ID, -1)
    bus.publish(ButtonsHeld(4, frozenset(), "ai"))  # a step holding nothing earns nothing
    bus.publish(ButtonsHeld(5, frozenset({"up"}), "ai"))  # not held the step before: +2
    bus.publish(ButtonsHeld(6, frozenset({"up", "a"}), "ai"))  # Up again: +2 then -1
    bus.publish(ButtonsHeld(7, frozenset({"a"}), "ai"))  # A again, worth nothing itself: -1
    assert [(e.frame, e.amount) for e in earned] == [(5, 2.0), (6, 2.0), (6, -1.0), (7, -1.0)]
    assert earned[0].behaviour_ids == (press_id("up"),) and earned[2].behaviour_ids == (AGAIN_ID,)
    assert rewards.tally.by_category(profile)[Category.CONTROLS] == 2.0
    # A person's points come back with the game; the rows are not duplicated.
    connection.disconnect()
    connection.connect("fake", seed=3)
    profile = rewards.profile
    assert profile is not None and profile.get(press_id("up")).value == 2
    assert len(profile.by_category()[Category.CONTROLS]) == len(names) + 1


def test_controls_save_and_are_remembered(
    rewards: RewardService,
    connection: ConnectionService,
    discovery: DiscoveryService,
    bus: EventBus,
    changes: list[RewardProfileChanged],
    store: RewardStore,
) -> None:
    connection.connect("fake", seed=3)
    k = teach(discovery, bus)
    lives = k.at(RAM, LAYOUT.lives)
    assert lives
    hit = f"setback:{lives.id}"
    assert rewards.increase(hit).value == -9.0
    assert rewards.decrease(hit).value == -10.0
    assert rewards.set_value(hit, -20.0).value == -20.0
    assert not rewards.set_enabled("reset:*", False).enabled
    assert rewards.choose_personality("survivor").id == "survivor"
    assert [c.reason for c in changes[-5:]] == ["control"] * 4 + ["personality"]
    rewards.reset_to_suggested()
    assert changes[-1].reason == "control" and rewards.profile is not None
    assert rewards.profile.get(hit).value == -20.0  # survivor: -10 x 2
    rewards.set_value(hit, -7.0)
    before = rewards.saves
    connection.disconnect()
    assert rewards.saves == before + 1 and not rewards.enabled
    assert rewards.profile is not None  # readable after disconnect
    connection.connect("fake", seed=3)
    profile = rewards.profile
    assert profile is not None and profile.personality == "survivor"
    assert profile.get(hit).value == -7.0 and profile.get(hit).custom
    assert not profile.get("reset:*").enabled
    assert rewards.tally.count == 0
    # Another game starts fresh.
    connection.disconnect()
    connection.connect("fake", seed=4)
    assert rewards.profile is not None and rewards.profile.game_id == "toyracer-4"
    assert rewards.profile.personality == "custom" and own_ids(rewards.profile) == ("reset:*",)
    assert set(store.games()) == {"toyracer-3", "toyracer-4"}


def test_damaged_file_starts_fresh(
    rewards: RewardService, connection: ConnectionService, store: RewardStore
) -> None:
    store.path_for("toyracer-3").write_text("{nope", "utf-8")
    connection.connect("fake", seed=3)
    assert rewards.profile is not None and own_ids(rewards.profile) == ("reset:*",)
    connection.disconnect()
    assert own_ids(store.load("toyracer-3")) == ("reset:*",)


def test_without_a_store_nothing_is_saved(
    connection: ConnectionService, discovery: DiscoveryService, bus: EventBus
) -> None:
    rewards = RewardService(connection, discovery, bus)
    connection.connect("fake", seed=3)
    rewards.increase("reset:*")
    assert rewards.save() is None and rewards.saves == 0 and rewards.store is None


# ------------------------------------------------------------ T-405


def test_farming_is_announced_once_per_window_and_clears(
    connection: ConnectionService, discovery: DiscoveryService, bus: EventBus
) -> None:
    from retro_ai_lab.services import ExploitSuspected

    rewards = RewardService(connection, discovery, bus, exploit_window=100, exploit_min_reward=20)
    seen: list[ExploitSuspected] = []
    bus.subscribe(ExploitSuspected, seen.append)
    connection.connect("fake", seed=3)
    k = teach(discovery, bus)
    score = k.at(RAM, LAYOUT.score)
    checkpoint = k.at(RAM, LAYOUT.checkpoint)
    assert score and checkpoint

    def gain(frame: int) -> None:
        bus.publish(
            DiscoveryUpdated((), (), (GameEvent(EventKind.SCORE_GAIN, frame, score.id, ""),), ())
        )

    for f in range(10, 40, 10):
        gain(f)  # 15 points: not enough yet
    assert seen == [] and rewards.current_warning is None
    gain(40)
    assert len(seen) == 1 and seen[0].frame == 40 and seen[0].warning.kind == "farming"
    assert rewards.current_warning is seen[0] and rewards.warnings == (seen[0],)
    gain(50)
    assert len(seen) == 1  # same cause, same window: said once
    # A checkpoint inside the window means progress: no new warning, and after a window the
    # old one lapses.
    bus.publish(
        DiscoveryUpdated((), (), (GameEvent(EventKind.PROGRESS, 60, checkpoint.id, ""),), ())
    )
    assert len(seen) == 1 and rewards.current_warning is seen[0]
    bus.publish(
        DiscoveryUpdated((), (), (GameEvent(EventKind.PROGRESS, 150, checkpoint.id, ""),), ())
    )
    assert rewards.current_warning is None
    # Farming again much later is a fresh warning.
    for f in range(400, 450, 10):
        gain(f)
    assert len(seen) == 2 and seen[1].frame == 430
    connection.disconnect()
    connection.connect("fake", seed=3)
    assert rewards.warnings == () and rewards.current_warning is None


def test_relationship_updates_refresh_suggestions_sparingly(
    connection: ConnectionService,
    discovery: DiscoveryService,
    rewards: RewardService,
    bus: EventBus,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """T-842: the finder reports relationships on most steps; rebuild only on real change."""
    connection.connect("fake")
    k = teach(discovery, bus)
    calls: list[int] = []
    real = apply_suggestions

    def counted(profile: object, knowledge: object, **kw: object) -> SuggestionReport:
        calls.append(1)
        return real(profile, knowledge, **kw)  # type: ignore[arg-type]

    monkeypatch.setattr("retro_ai_lab.services.rewards.apply_suggestions", counted)
    weak = Relationship("score_gain:v3", "progress:v2", 1, 3, 3, 5)  # below LOW
    k.set_relationship(weak)
    for _ in range(10):
        bus.publish(DiscoveryUpdated((), (), (), (weak,)))
    assert calls == [1]  # the first look is taken once, then nothing moved
    sure = Relationship("score_gain:v3", "progress:v2", 6, 6, 6, 5)  # MEDIUM
    k.set_relationship(sure)
    bus.publish(DiscoveryUpdated((), (), (), (sure,)))
    bus.publish(DiscoveryUpdated((), (), (), (sure,)))
    assert len(calls) == 2
    assert rewards.profile is not None
    assert "This leads to progress" in rewards.profile.get("score_gain:v3").notes
    bus.publish(DiscoveryUpdated((k.get("v2"),), (), (), ()))  # a variable change: always
    assert len(calls) == 3


def test_dismiss_takes_a_row_out_for_good_and_suggest_again_undoes_it(
    rewards: RewardService,
    connection: ConnectionService,
    discovery: DiscoveryService,
    bus: EventBus,
    changes: list[RewardProfileChanged],
) -> None:
    """T-943: "every time i remove something from the rewards it auto readds it"."""
    connection.connect("fake", seed=3)
    k = teach(discovery, bus)
    lives = k.at(RAM, LAYOUT.lives)
    assert lives
    hit = f"setback:{lives.id}"
    assert rewards.dismiss(hit) == "setback:<lives>"
    profile = rewards.profile
    assert profile is not None and hit not in profile and f"recovery:{lives.id}" in profile
    assert changes[-1].removed == (hit,) and not k.get(lives.id).rejected
    assert rewards.refresh_suggestions().is_empty and hit not in profile
    # The AI's next life count brings nothing back; a row of one's own cannot be dismissed.
    hearts = k.propose(RAM, LAYOUT.lap, kind=VariableKind.LIVES, name="Hearts", evidence=SURE)
    k.update(lives.id, evidence=())
    rewards.refresh_suggestions()
    assert f"setback:{hearts.id}" not in profile and f"recovery:{hearts.id}" in profile
    with pytest.raises(ConfigurationError):
        rewards.dismiss(hit)
    # The reset row and a button row go the same way and stay out across a reconnect.
    assert rewards.dismiss("reset:*") == "reset:*"
    assert rewards.dismiss(press_id("up")) == press_id("up")
    connection.disconnect()
    connection.connect("fake", seed=3)
    profile = rewards.profile
    assert profile is not None
    assert "reset:*" not in profile and press_id("up") not in profile
    assert profile.dismissed == {"setback:<lives>", "reset:*", press_id("up")}
    report = rewards.suggest_again()
    assert profile.dismissed == frozenset() and "reset:*" in report.added
    assert press_id("up") in profile and "reset:*" in profile


def test_the_values_a_person_penalises_are_known(
    rewards: RewardService,
    connection: ConnectionService,
    discovery: DiscoveryService,
    bus: EventBus,
) -> None:
    """T-964: an own reward below zero names what a death is to the person."""
    assert rewards.penalised() == frozenset()  # no profile yet
    connection.connect("fake", seed=3)
    k = teach(discovery, bus)
    assert rewards.penalised() == frozenset()
    lane = k.propose(RAM, LAYOUT.lane, kind=VariableKind.POSITION, evidence=SURE)
    timer = k.propose(RAM, LAYOUT.timer, kind=VariableKind.COUNTER, evidence=SURE)
    rewards.add_own(lane.id, goes_up=True, value=2.0)  # a prize, not a penalty
    down = rewards.add_own(timer.id, goes_up=False, value=-3.0)
    up = rewards.add_own(lane.id, goes_up=False, value=-1.0)
    assert rewards.penalised() == frozenset({timer.id, lane.id})
    rewards.set_enabled(up.id, False)
    assert rewards.penalised() == frozenset({timer.id})
    rewards.remove_own(down.id)
    assert rewards.penalised() == frozenset()
