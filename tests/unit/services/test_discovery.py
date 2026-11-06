"""T-306: the discovery runner learns while the game is played."""

from __future__ import annotations

import logging
import re

import pytest

from retro_ai_lab.core import Capabilities, ConfigurationError, UnsupportedCapabilityError
from retro_ai_lab.core.events import EventBus
from retro_ai_lab.emulator.fake import FakeEmulatorAdapter
from retro_ai_lab.emulator.fake.toy_racer import BUTTONS, LAYOUT
from retro_ai_lab.emulator.registry import default_registry
from retro_ai_lab.services import (
    ConnectionService,
    DiscoveryProgress,
    DiscoveryService,
    DiscoveryUpdated,
    ObservationService,
)
from retro_ai_lab.services.discovery import BOUNCES, BUSY, MOVES_ALONE, PLAYED, game_id_for
from retro_ai_lab.understanding import (
    Confidence,
    Encoding,
    EventKind,
    Evidence,
    Moment,
    Narrowing,
    Variable,
    VariableKind,
    find_candidates,
)
from retro_ai_lab.understanding.events import CYCLES

BY_NAME = {b.name: b for b in BUTTONS}


@pytest.fixture
def bus() -> EventBus:
    return EventBus(on_error=lambda exc, event, handler: (_ for _ in ()).throw(exc))


@pytest.fixture
def connection(bus: EventBus) -> ConnectionService:
    return ConnectionService(default_registry(), bus)


@pytest.fixture
def observation(connection: ConnectionService, bus: EventBus) -> ObservationService:
    return ObservationService(connection, bus)


@pytest.fixture
def discovery(
    connection: ConnectionService, observation: ObservationService, bus: EventBus
) -> DiscoveryService:
    return DiscoveryService(connection, observation, bus, review_every=100, experiment_every=250)


@pytest.fixture
def progress(bus: EventBus) -> list[DiscoveryProgress]:
    seen: list[DiscoveryProgress] = []
    bus.subscribe(DiscoveryProgress, seen.append)
    return seen


@pytest.fixture
def updates(bus: EventBus) -> list[DiscoveryUpdated]:
    seen: list[DiscoveryUpdated] = []
    bus.subscribe(DiscoveryUpdated, seen.append)
    return seen


def play(connection: ConnectionService, observation: ObservationService, frames: int) -> None:
    """Drive like a child would: mostly accelerate, weave a bit, restart after crashes."""
    adapter = connection.adapter
    assert isinstance(adapter, FakeEmulatorAdapter)
    for i in range(frames):
        if adapter.game.state.game_over:
            held = [BY_NAME["start"]]
        else:
            held = [BY_NAME["up"]]
            if i % 90 in range(20, 24):
                held.append(BY_NAME["left"])
            if i % 130 in range(40, 44):
                held.append(BY_NAME["right"])
        connection.call(lambda a, held=held: a.set_input(held))
        connection.call(lambda a: a.advance_frames(1))
        observation.advance(1)


def test_idle_until_a_memory_capable_game_connects(
    discovery: DiscoveryService, connection: ConnectionService, progress: list[DiscoveryProgress]
) -> None:
    assert not discovery.enabled and discovery.knowledge is None
    assert discovery.summary() == "Not watching a game."
    with pytest.raises(UnsupportedCapabilityError):
        discovery.review()
    connection.connect("fake", capabilities=Capabilities(screen_capture=True, frame_advance=True))
    assert not discovery.enabled and discovery.stage == "idle"
    assert "memory" in progress[-1].detail
    connection.disconnect()
    connection.connect("fake", seed=3)
    assert discovery.enabled and discovery.stage == "watching"
    assert discovery.knowledge is not None
    assert discovery.knowledge.game_id == "toyracer-3" and discovery.knowledge.title == "Toy Racer"
    assert discovery.summary().startswith("Watching 0 values")


def test_game_id_prefers_the_rom_hash() -> None:
    assert game_id_for("NES", "Game", "abc") == "abc"
    assert game_id_for("NES", "Game", None) == "NES:Game"


def test_reviews_propose_variables_on_schedule(
    discovery: DiscoveryService,
    connection: ConnectionService,
    observation: ObservationService,
    updates: list[DiscoveryUpdated],
    progress: list[DiscoveryProgress],
) -> None:
    connection.connect("fake", seed=2)
    play(connection, observation, 99)
    assert discovery.reviews == 0
    play(connection, observation, 1)
    assert discovery.reviews == 1
    k = discovery.knowledge
    assert k is not None and len(k) >= 5
    assert any(u.added for u in updates)
    assert [p.stage for p in progress[-2:]] == ["thinking", "watching"]
    play(connection, observation, 100)
    assert discovery.reviews == 2
    assert "Watching" in discovery.summary()


def test_events_and_relationships_flow_while_playing(
    discovery: DiscoveryService,
    connection: ConnectionService,
    observation: ObservationService,
    updates: list[DiscoveryUpdated],
) -> None:
    connection.connect("fake", seed=2)
    play(connection, observation, 1200)
    assert discovery.events.total > 0
    kinds = {e.kind for e in discovery.events}
    assert EventKind.PROGRESS in kinds or EventKind.SCORE_GAIN in kinds
    k = discovery.knowledge
    assert k is not None and k.relationships()
    assert any(u.events for u in updates) and any(u.relationships for u in updates)


def test_experiments_run_on_schedule_and_on_demand(
    discovery: DiscoveryService,
    connection: ConnectionService,
    observation: ObservationService,
    progress: list[DiscoveryProgress],
) -> None:
    connection.connect("fake", seed=5)
    assert discovery.can_experiment()
    play(connection, observation, 249)
    assert discovery.experiments == 0
    play(connection, observation, 1)
    assert discovery.experiments == 1
    assert any(p.stage == "experimenting" for p in progress)
    k = discovery.knowledge
    assert k is not None
    mask = k.at("ram", LAYOUT.input)
    assert mask is not None and mask.kind is VariableKind.INPUT
    lane = k.at("ram", LAYOUT.lane)
    assert lane is not None and lane.kind is VariableKind.POSITION
    update = discovery.experiment()
    assert discovery.experiments == 2
    assert update.is_empty or update.changed or update.added  # nothing new is fine
    assert discovery.stage == "watching"


def test_scheduled_experiments_can_be_held_back(
    discovery: DiscoveryService, connection: ConnectionService, observation: ObservationService
) -> None:
    """T-839: training holds the scheduled experiments back; on demand still works."""
    connection.connect("fake", seed=5)
    assert discovery.experiments_allowed
    discovery.set_experiments_allowed(False)
    play(connection, observation, 300)
    assert discovery.experiments == 0
    discovery.experiment()  # the panel's "Try the buttons" is not held back
    assert discovery.experiments == 1
    play(connection, observation, 250)
    assert discovery.experiments == 1
    assert discovery.experiment_due
    assert discovery.run_due_experiment() and discovery.experiments == 2  # between goes
    assert not discovery.experiment_due and not discovery.run_due_experiment()
    play(connection, observation, 250)
    assert discovery.experiments == 2  # still held back
    discovery.set_experiments_allowed(True)
    play(connection, observation, 1)
    assert discovery.experiments == 3
    discovery.set_experiments_allowed(False)
    connection.disconnect()
    connection.connect("fake", seed=5)
    assert discovery.experiments_allowed  # a new link starts afresh


def test_experiments_unavailable_without_save_states(
    discovery: DiscoveryService, connection: ConnectionService, observation: ObservationService
) -> None:
    connection.connect(
        "fake",
        capabilities=Capabilities(
            memory_read=True, input=True, frame_advance=True, screen_capture=True
        ),
    )
    assert not discovery.can_experiment()
    with pytest.raises(UnsupportedCapabilityError):
        discovery.experiment()
    play(connection, observation, 300)  # schedule passes quietly
    assert discovery.experiments == 0 and discovery.reviews == 3


def test_disconnect_goes_idle_and_reconnect_starts_fresh(
    discovery: DiscoveryService, connection: ConnectionService, observation: ObservationService
) -> None:
    connection.connect("fake", seed=2)
    play(connection, observation, 150)
    first = discovery.knowledge
    assert first is not None and len(first) > 0
    connection.disconnect()
    assert not discovery.enabled and discovery.stage == "idle"
    assert discovery.knowledge is first  # kept for review until the next game
    connection.connect("fake", seed=2)
    assert discovery.knowledge is not first and discovery.reviews == 0
    assert discovery.events.total == 0


def test_link_loss_during_an_experiment_is_survived(
    discovery: DiscoveryService, connection: ConnectionService, observation: ObservationService
) -> None:
    # 250 frames of play cost ~750 adapter operations; the experiment that
    # follows needs ~200 more, so the link drops in the middle of it.
    connection.connect("fake", seed=2, drop_after_ops=850)
    play(connection, observation, 250)
    assert not connection.is_connected
    assert not discovery.enabled


def test_bad_schedules() -> None:
    bus = EventBus()
    c = ConnectionService(default_registry(), bus)
    o = ObservationService(c, bus)
    with pytest.raises(ValueError):
        DiscoveryService(c, o, bus, review_every=0)
    with pytest.raises(ValueError):
        DiscoveryService(c, o, bus, experiment_every=0)
    DiscoveryService(c, o, bus, experiment_every=None)  # experiments off is allowed


# ------------------------------------------------------- persistence (T-308)


def test_knowledge_is_remembered_between_connections(
    connection: ConnectionService, observation: ObservationService, bus: EventBus, tmp_path: object
) -> None:
    from pathlib import Path

    from retro_ai_lab.storage import KnowledgeStore
    from retro_ai_lab.understanding import Confidence

    store = KnowledgeStore(Path(str(tmp_path)) / "knowledge")
    d = DiscoveryService(
        connection, observation, bus, review_every=100, experiment_every=None, store=store
    )
    assert d.store is store and d.save() is None  # nothing to save yet
    connection.connect("fake", seed=2)
    play(connection, observation, 100)
    k = d.knowledge
    assert k is not None and len(k) > 0 and d.saves >= 1
    first = next(iter(k))
    k.confirm(first.id)
    d.save()
    connection.disconnect()
    saved = store.load("toyracer-2")
    assert saved.get(first.id).confidence is Confidence.CONFIRMED
    connection.connect("fake", seed=2)
    again = d.knowledge
    assert again is not None and again is not k
    assert again.get(first.id).confidence is Confidence.CONFIRMED
    assert len(again) == len(k)
    connection.disconnect()
    connection.connect("fake", seed=9)  # a different game starts empty
    assert d.knowledge is not None and len(d.knowledge) == 0


def test_damaged_file_starts_over(
    connection: ConnectionService, observation: ObservationService, bus: EventBus, tmp_path: object
) -> None:
    from pathlib import Path

    from retro_ai_lab.storage import KnowledgeStore

    store = KnowledgeStore(Path(str(tmp_path)) / "knowledge")
    store.path_for("toyracer-2").write_text("{broken", "utf-8")
    d = DiscoveryService(connection, observation, bus, store=store)
    connection.connect("fake", seed=2)
    assert d.knowledge is not None and len(d.knowledge) == 0


def test_export_and_import(
    connection: ConnectionService, observation: ObservationService, bus: EventBus, tmp_path: object
) -> None:
    from pathlib import Path

    from retro_ai_lab.core import ConfigurationError
    from retro_ai_lab.storage.knowledge import load
    from retro_ai_lab.understanding import Knowledge, VariableKind

    base = Path(str(tmp_path))
    d = DiscoveryService(connection, observation, bus, review_every=100, experiment_every=None)
    with pytest.raises(ConfigurationError):
        d.export_to(base / "x.json")
    with pytest.raises(ConfigurationError):
        d.import_from(base / "x.json")
    connection.connect("fake", seed=2)
    play(connection, observation, 100)
    out = d.export_to(base / "shared.json")
    assert load(out).game_id == "toyracer-2"
    shared = Knowledge("toyracer-2", title="Toy Racer")
    shared.propose("ram", LAYOUT.score, kind=VariableKind.SCORE, name="Points")
    shared.confirm("v1")
    from retro_ai_lab.storage.knowledge import export

    export(shared, base / "friend.json")
    imported = d.import_from(base / "friend.json")
    assert d.knowledge is imported and d.knowledge.get("v1").name == "Points"
    play(connection, observation, 100)  # discovery carries on with the imported beliefs
    assert len(d.knowledge) > 1
    other = Knowledge("someone-else")
    export(other, base / "other.json")
    with pytest.raises(ConfigurationError, match="not"):
        d.import_from(base / "other.json")


def test_the_long_look_explains_checkpoints_and_laps(
    discovery: DiscoveryService,
    connection: ConnectionService,
    observation: ObservationService,
    updates: list[DiscoveryUpdated],
) -> None:
    from retro_ai_lab.understanding.longlook import COUNTS_ROUNDS, SAME_PLACES

    connection.connect("fake", seed=2)
    play(connection, observation, 3000)
    k = discovery.knowledge
    assert k is not None
    checkpoint, lap = k.at("ram", LAYOUT.checkpoint), k.at("ram", LAYOUT.lap)
    assert checkpoint is not None and lap is not None
    assert checkpoint.kind is VariableKind.PROGRESS and lap.kind is VariableKind.PROGRESS
    assert any(e.kind == SAME_PLACES for e in checkpoint.evidence)
    assert any(e.kind == COUNTS_ROUNDS for e in lap.evidence)
    assert any(checkpoint in u.changed or lap in u.changed for u in updates)


def test_doubt_reset_adds_the_evidence_once_and_announces(
    discovery: DiscoveryService,
    connection: ConnectionService,
    bus: EventBus,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """T-845: values that cycle back to their start stop counting as a restart."""
    connection.connect("fake", seed=5)
    k = discovery.knowledge
    assert k is not None
    a = k.propose("ram", LAYOUT.score, kind=VariableKind.SCORE, name="Score")
    b = k.propose("ram", LAYOUT.lives, kind=VariableKind.LIVES, name="Lives")
    seen: list[DiscoveryUpdated] = []
    bus.subscribe(DiscoveryUpdated, seen.append)
    caplog.set_level(logging.INFO, logger="retro_ai_lab.services.discovery")
    changed = discovery.doubt_reset((a.id, b.id, "v999"))
    assert [v.id for v in changed] == [a.id, b.id]
    assert all(any(e.kind == CYCLES for e in k.get(v.id).evidence) for v in changed)
    assert len(seen) == 1 and seen[0].changed == changed
    assert "Score, Lives cycle back to their start" in caplog.text
    assert discovery.doubt_reset((a.id,)) == () and len(seen) == 1  # only once
    with pytest.raises(UnsupportedCapabilityError):
        connection.disconnect()
        discovery.doubt_reset((a.id,))


def test_self_moving_guesses_are_doubted_then_rejected(
    discovery: DiscoveryService,
    connection: ConnectionService,
    bus: EventBus,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """T-855: a score guess that changes while nobody presses anything is dropped."""
    connection.connect("fake", seed=5)
    k = discovery.knowledge
    assert k is not None
    real = k.propose("ram", LAYOUT.score, kind=VariableKind.SCORE, name="Score")
    junk = k.propose("ram", LAYOUT.timer, kind=VariableKind.SCORE, name="Timer as score")
    wide = k.propose(
        "ram", LAYOUT.frame, kind=VariableKind.PROGRESS, name="Frame as progress"
    )  # one byte; its second byte is what moves below
    kept = k.propose("ram", LAYOUT.noise, kind=VariableKind.SCORE, name="Noise as score")
    k.confirm(kept.id)
    speed = k.propose("ram", LAYOUT.speed, kind=VariableKind.SPEED, name="Speed")
    lives = k.propose("ram", LAYOUT.lives, kind=VariableKind.LIVES, name="Lives")
    seen: list[DiscoveryUpdated] = []
    bus.subscribe(DiscoveryUpdated, seen.append)
    caplog.set_level(logging.INFO, logger="retro_ai_lab.services.discovery")
    moved = {
        ("ram", LAYOUT.timer),
        ("ram", LAYOUT.frame),
        ("ram", LAYOUT.noise),
        ("ram", LAYOUT.speed),
        ("ram", LAYOUT.lives),  # lost by standing still: never doubted
    }
    changed = discovery.doubt_self_moving(moved, frame=7)
    assert {v.id for v in changed} == {junk.id, wide.id}
    assert all(not k.get(v.id).rejected for v in changed)  # a doubt, not yet a no
    assert [e.kind for e in k.get(junk.id).evidence][-1] == MOVES_ALONE
    assert len(seen) == 1 and set(seen[0].changed) == set(changed)
    assert "2 doubted (Timer as score, Frame as progress), 0 dropped (none)." in caplog.text
    changed = discovery.doubt_self_moving(moved)
    assert {v.id for v in changed} == {junk.id, wide.id}
    assert all(k.get(v.id).rejected for v in changed)
    assert "Rejected by the AI." in k.get(junk.id).evidence[-1].detail
    assert not k.get(real.id).rejected and k.get(kept.id).confirmed
    for other in (speed, lives):  # not doubted kinds
        assert not any(e.kind == MOVES_ALONE for e in k.get(other.id).evidence)
    assert "0 doubted (none), 2 dropped (Timer as score, Frame as progress)." in caplog.text
    assert discovery.doubt_self_moving(moved) == () and discovery.doubt_self_moving(()) == ()
    assert len(seen) == 2


def test_a_lives_guess_that_bounces_is_doubted_then_dropped(
    discovery: DiscoveryService,
    connection: ConnectionService,
    observation: ObservationService,
    bus: EventBus,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """T-864 to T-866: a lives or progress byte that bounces or is busy is neither."""
    assert discovery.doubt_bouncing() == ()  # nothing connected: nothing to doubt
    connection.connect("fake", seed=2)
    k = discovery.knowledge
    assert k is not None
    play(connection, observation, 600)  # the lane weaves left and right alike; lives go down

    def plant(offset: int, name: str, kind: VariableKind = VariableKind.LIVES) -> Variable:
        found = k.at("ram", offset)  # the reviews may have proposed the byte already
        if found is None:
            found = k.propose("ram", offset, kind=kind, name=name)
        return k.update(found.id, kind=kind, name=name)

    bogus = plant(LAYOUT.lane, "Lane as lives")
    busy = plant(LAYOUT.position, "Position as lives")  # changes nearly every step
    rushed = plant(LAYOUT.speed, "Speed as progress", VariableKind.PROGRESS)  # T-866: busy
    real = plant(LAYOUT.lives, "Lives")
    level = plant(LAYOUT.checkpoint, "Checkpoint", VariableKind.PROGRESS)  # rare, mostly up
    sure = plant(LAYOUT.lap, "Sworn lives")  # the lap also bounces, but a person swore to it
    k.confirm(sure.id)
    seen: list[DiscoveryUpdated] = []
    bus.subscribe(DiscoveryUpdated, seen.append)
    caplog.set_level(logging.INFO, logger="retro_ai_lab.services.discovery")
    changed = discovery.doubt_bouncing(frame=9)
    assert {v.id for v in changed} == {bogus.id, busy.id, rushed.id}
    assert not any(k.get(v.id).rejected for v in changed)
    assert "progress changes rarely" in k.get(rushed.id).evidence[-1].detail
    last = k.get(bogus.id).evidence[-1]
    assert last.kind == BOUNCES and last.frame == 9 and "does not bounce" in last.detail
    last = k.get(busy.id).evidence[-1]
    assert last.kind == BUSY and "changes rarely" in last.detail
    # T-895: the confirmed guess is kept and questioned, in the same announcement
    assert len(seen) == 1 and seen[0].changed == (*changed, k.get(sure.id))
    assert "3 guesses change like no life count, progress or flag does: 3 doubted (" in caplog.text
    (question,) = discovery.questions
    assert question.var_id == sure.id and question.name == "Sworn lives"
    assert question.reason == BOUNCES and question.text.startswith('You confirmed "Sworn lives"')
    assert (
        "does not bounce. The AI keeps your word; if it is not a life count, pick it under "
        '"What the AI is doing" and press "No, that\'s wrong".'
    ) in question.text  # T-897: the panel's real title, not a name the app never shows
    assert question.text in caplog.text
    # The next review drops all three; the real and the confirmed guesses are untouched.
    update = discovery.review()
    assert {
        v.id for v in update.changed if v.kind in (VariableKind.LIVES, VariableKind.PROGRESS)
    } == {
        bogus.id,
        busy.id,
        rushed.id,
    }
    for gone in (bogus, busy, rushed):
        assert k.get(gone.id).rejected
        assert "Rejected by the AI." in k.get(gone.id).evidence[-1].detail
    assert "0 doubted (none), 3 dropped (" in caplog.text
    for other in (real, level, sure):
        got = k.get(other.id)
        assert not got.rejected and not any(e.kind in (BOUNCES, BUSY) for e in got.evidence)
    assert k.get(sure.id).confirmed
    announced = len(seen)
    assert discovery.doubt_bouncing() == ()
    assert len(seen) == announced and discovery.questions == (question,)  # asked once, kept
    # T-896: more play changes the counts in the question; that is neither news nor noise.
    play(connection, observation, 200)  # (its steps announce events of their own)
    caplog.clear()
    announced = len(seen)
    assert discovery.doubt_bouncing() == ()
    (fresh,) = discovery.questions
    assert fresh != question and (fresh.var_id, fresh.reason) == (sure.id, BOUNCES)
    assert len(seen) == announced and "Sworn lives" not in caplog.text
    # Rejecting the sworn guess ends the question; the change is announced once.
    k.reject(sure.id)
    before = len(seen)
    assert discovery.doubt_bouncing() == () and discovery.questions == ()
    assert len(seen) == before + 1 and seen[-1].changed == (k.get(sure.id),)
    assert '"Sworn lives" behaves like its kind again' in caplog.text


def test_a_busy_flag_guess_is_doubted_then_dropped(
    discovery: DiscoveryService,
    connection: ConnectionService,
    observation: ObservationService,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """T-869: a flag switching more than once in ten steps is no event worth rewarding."""
    connection.connect("fake", seed=2)
    k = discovery.knowledge
    assert k is not None
    play(connection, observation, 600)

    def plant(offset: int, name: str) -> Variable:
        found = k.at("ram", offset)
        if found is None:
            found = k.propose("ram", offset, kind=VariableKind.FLAG, name=name)
        return k.update(found.id, kind=VariableKind.FLAG, name=name)

    blink = plant(LAYOUT.timer, "Timer as flag")  # changes every step
    over = plant(LAYOUT.game_over, "Game over")  # switches rarely, if at all
    caplog.set_level(logging.INFO, logger="retro_ai_lab.services.discovery")
    changed = discovery.doubt_bouncing(frame=4)
    assert [v.id for v in changed] == [blink.id]
    last = k.get(blink.id).evidence[-1]
    assert last.kind == BUSY and last.frame == 4 and "a flag changes rarely" in last.detail
    assert not k.get(blink.id).rejected
    assert "1 guesses change like no life count, progress or flag does: 1 doubted (" in caplog.text
    discovery.review()
    assert k.get(blink.id).rejected
    assert "Rejected by the AI." in k.get(blink.id).evidence[-1].detail
    assert "0 doubted (none), 1 dropped (Timer as flag)" in caplog.text
    got = k.get(over.id)
    assert not got.rejected and not any(e.kind == BUSY for e in got.evidence)


def test_the_census_names_the_busiest_bytes_and_their_numbers(
    discovery: DiscoveryService, connection: ConnectionService, observation: ObservationService
) -> None:
    """T-919/T-920: the log can say why a byte is or is not taken for something moving."""
    assert discovery.census() == ""  # nothing connected
    connection.connect("fake", seed=2)
    play(connection, observation, 300)
    text = discovery.census()
    stats = observation.tracker.all_stats()
    moved = [(loc, st) for loc, st in stats.items() if st.changes > 0]
    heads = [i for i, line in enumerate(text.split("\n")) if line.endswith(":")]
    assert [text.split("\n")[i] for i in heads] == [
        "Busiest:",
        "Smoothest:",
        "Widest-ranging rare:",
    ]
    busy = text.split("\n")[heads[0] + 1 : heads[1]]
    smooth = text.split("\n")[heads[1] + 1 : heads[2]]
    rare = text.split("\n")[heads[2] + 1 :]
    busiest = sorted(moved, key=lambda item: -item[1].changes)
    assert 1 <= len(busy) <= 16 and f"{busiest[0][1].changes} changes in " in busy[0]
    assert busy[0].startswith(f"ram[{busiest[0][0][1]}] ")
    for word in ("values", "hops small", "unit", "largest", "up", "down", "wraps"):
        assert word in busy[0]
    k = discovery.knowledge
    assert k is not None
    offset = int(busy[0].split("]")[0].split("[")[1])
    v = k.at("ram", offset)
    assert (v.display_name if v is not None else "nobody's guess") in busy[0]

    # the smooth list holds only smooth bytes, most changes first; the rare list only
    # rare bytes, most values first; both say "none" when empty
    def offsets(lines: list[str]) -> list[int]:
        return [int(line.split("]")[0].split("[")[1]) for line in lines if line != "none"]

    for off in offsets(smooth):
        assert stats[("ram", off)].smooth
    assert (
        offsets(smooth)
        == [
            loc[1]
            for loc, _ in sorted((i for i in moved if i[1].smooth), key=lambda i: -i[1].changes)
        ][:16]
    )
    for off in offsets(rare):
        assert stats[("ram", off)].change_rate <= 0.1
    assert (
        offsets(rare)
        == [
            loc[1]
            for loc, _ in sorted(
                (i for i in moved if i[1].change_rate <= 0.1), key=lambda i: -len(i[1].distinct)
            )
        ][:16]
    )
    assert smooth == ["none"] or len(smooth) >= 1
    narrow = discovery.census(most=2).split("\n")
    assert narrow.count("Busiest:") == 1 and len(narrow) <= 3 * 3
    wrong = k.propose("ram", 2000, kind=VariableKind.LIVES, name="Nothing there")  # never changes
    assert wrong.display_name not in discovery.census()


def test_the_review_logs_the_census_on_schedule(
    connection: ConnectionService,
    observation: ObservationService,
    bus: EventBus,
    caplog: pytest.LogCaptureFixture,
) -> None:
    with pytest.raises(ValueError):
        DiscoveryService(connection, observation, bus, census_every=0)
    discovery = DiscoveryService(
        connection, observation, bus, review_every=100, experiment_every=None, census_every=2
    )
    connection.connect("fake", seed=2)
    caplog.set_level(logging.INFO, logger="retro_ai_lab.services.discovery")
    play(connection, observation, 100)
    assert discovery.reviews == 1 and "Census of the busiest bytes" not in caplog.text
    play(connection, observation, 100)
    assert discovery.reviews == 2
    (line,) = [m for m in caplog.messages if m.startswith("Census of the busiest bytes")]
    assert "after review 2:" in line and "changes in" in line


def test_a_review_rethinks_the_ais_own_guesses(
    discovery: DiscoveryService,
    connection: ConnectionService,
    observation: ObservationService,
    updates: list[DiscoveryUpdated],
    caplog: pytest.LogCaptureFixture,
) -> None:
    """T-916: a guess judged on its first few changes is judged again as more is seen."""
    connection.connect("fake", seed=2)
    play(connection, observation, 100)  # the first review proposes the toy racer's bytes
    k = discovery.knowledge
    assert k is not None and discovery.reviews == 1
    own = k.at("ram", LAYOUT.speed)
    assert own is not None and own.kind is not VariableKind.LIVES
    wrong = k.update(own.id, kind=VariableKind.LIVES)  # as if judged on a handful of values
    sworn = k.at("ram", LAYOUT.lives)
    assert sworn is not None
    k.update(sworn.id, kind=VariableKind.FLAG)
    k.confirm(sworn.id)  # a person's word, however odd, holds
    updates.clear()
    caplog.set_level(logging.INFO, logger="retro_ai_lab.services.discovery")
    play(connection, observation, 120)
    assert discovery.reviews == 2
    (now,) = [c for c in find_candidates(observation.tracker) if c.offset == LAYOUT.speed]
    assert k.get(wrong.id).kind is now.kind and now.kind is not VariableKind.LIVES
    assert k.get(sworn.id).kind is VariableKind.FLAG and k.get(sworn.id).confirmed
    assert any(k.get(wrong.id) in u.changed for u in updates)
    line = next(m for m in caplog.messages if "changed its mind about" in m)
    assert f"ram[{LAYOUT.speed}]" in line and "of its own guesses" in line


def test_a_person_names_a_value_by_address(
    discovery: DiscoveryService,
    connection: ConnectionService,
    updates: list[DiscoveryUpdated],
) -> None:
    """T-862: naming an address confirms the AI's guess there, or makes a named value."""
    with pytest.raises(UnsupportedCapabilityError):
        discovery.name_value("ram", 0, "Lives")
    connection.connect("fake", seed=3)
    k = discovery.knowledge
    assert k is not None
    guess = k.propose("ram", LAYOUT.lives, kind=VariableKind.LIVES)
    updates.clear()
    named = discovery.name_value("ram", LAYOUT.lives, "  Hearts ")
    assert named.id == guess.id and named.name == "Hearts" and named.confirmed
    assert updates[-1].changed == (named,) and updates[-1].added == ()
    fresh = discovery.name_value("ram", LAYOUT.lane, "Lane")
    assert fresh.id != guess.id and fresh.name == "Lane" and fresh.confirmed
    assert fresh.kind is VariableKind.UNKNOWN and k.at("ram", LAYOUT.lane) == fresh
    assert updates[-1].added == (fresh,) and updates[-1].changed == ()
    with pytest.raises(ConfigurationError):
        discovery.name_value("ram", LAYOUT.lane, "   ")
    # T-895: a guess the AI rejected comes back under the person's name, its kind unknown
    # (the Research window shows a rejected guess as a bare address).
    wrong = k.propose("ram", LAYOUT.speed, kind=VariableKind.PROGRESS)
    k.reject(wrong.id, by="the AI")
    mystery = discovery.name_value("ram", LAYOUT.speed, "Mystery")
    assert mystery.id == wrong.id and mystery.name == "Mystery" and mystery.confirmed
    assert not mystery.rejected and mystery.kind is VariableKind.UNKNOWN
    assert updates[-1].changed == (mystery,) and updates[-1].added == ()


def test_memory_from_several_threads_is_observed_one_at_a_time(
    connection: ConnectionService,
    observation: ObservationService,
    bus: EventBus,
) -> None:
    """T-847: snapshots come from the play clock's worker and the GUI thread at once."""
    import sys
    import threading

    discovery = DiscoveryService(connection, observation, bus, review_every=50)
    connection.connect("fake", seed=5)
    k = discovery.knowledge
    assert k is not None
    k.propose("ram", LAYOUT.score, kind=VariableKind.SCORE, name="Score")
    k.propose("ram", LAYOUT.lives, kind=VariableKind.LIVES, name="Lives")
    play(connection, observation, 60)
    events = [e for e in (observation.observe() for _ in range(8)) if e is not None]
    assert len(events) == 8
    failures: list[BaseException] = []

    def publish(times: int) -> None:
        try:
            for i in range(times):
                bus.publish(events[i % len(events)])
        except BaseException as exc:
            failures.append(exc)

    before = discovery.reviews
    interval = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)
    threads = [threading.Thread(target=publish, args=(100,), daemon=True) for _ in range(4)]
    try:
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)
    finally:
        sys.setswitchinterval(interval)
    assert not any(t.is_alive() for t in threads) and failures == []
    assert discovery.reviews == before + 400 // 50  # every snapshot counted, none lost


def test_a_value_that_moves_while_the_ai_plays_earns_its_seat(
    connection: ConnectionService, observation: ObservationService, bus: EventBus
) -> None:
    """T-870/T-871: among look-alike guesses, the one that moved a few times a go gets the seat."""
    discovery = DiscoveryService(
        connection, observation, bus, review_every=10**6, experiment_every=None
    )
    assert discovery.credit_played() == ()  # nothing connected, no go begun
    connection.connect("fake", seed=2)
    k = discovery.knowledge
    assert k is not None
    discovery.begin_go()
    assert discovery.credit_played() == ()  # no step taken: nothing to credit
    weight = lambda w: (Evidence("test", "Planted.", w),)  # noqa: E731
    dead = [
        k.propose("ram", o, kind=VariableKind.PROGRESS, evidence=weight(2.5))
        for o in (LAYOUT.header, LAYOUT.header + 1, LAYOUT.header + 2)  # constant bytes
    ]
    moving = k.propose("ram", LAYOUT.checkpoint, kind=VariableKind.PROGRESS, evidence=weight(2.0))
    busy = k.propose("ram", LAYOUT.position, kind=VariableKind.PROGRESS, evidence=weight(2.0))
    assert moving.id not in {v.id for v in k.offered()}  # the weaker guess has no seat
    seen: list[DiscoveryUpdated] = []
    bus.subscribe(DiscoveryUpdated, seen.append)
    discovery.begin_go()
    play(connection, observation, 300)  # the checkpoint moves ~18 times, the position 300
    credited = discovery.credit_played(frame=7)
    assert [v.id for v in credited] == [moving.id]
    last = k.get(moving.id).evidence[-1]
    assert last.kind == PLAYED and last.weight == 0.5 and last.frame == 7
    assert re.fullmatch(r"Moved \d+ times in a go of (299|300) steps\.", last.detail)
    assert moving.id in {v.id for v in k.offered()}  # a tie now, and it moved
    for other in (*dead, busy):
        assert not any(e.kind == PLAYED for e in k.get(other.id).evidence)
    assert [u for u in seen if u.changed] == [DiscoveryUpdated((), credited, (), ())]
    # Credit comes once a go, three goes at most; and never without a go begun.
    for _ in range(3):
        discovery.begin_go()
        play(connection, observation, 300)
        discovery.credit_played()
    assert sum(e.kind == PLAYED for e in k.get(moving.id).evidence) == 3
    play(connection, observation, 10)
    assert discovery.credit_played() == ()


# ------------------------------------------- T-901: "I just lost a life"


SURE_LIVES = [Evidence("test", "Planted.", 6.0)]


def _play_until_a_life_is_lost(
    connection: ConnectionService, observation: ObservationService, most: int = 6000
) -> None:
    adapter = connection.adapter
    assert isinstance(adapter, FakeEmulatorAdapter)
    lives = adapter.game.state.lives
    for _ in range(most):
        held = [BY_NAME["start"]] if adapter.game.state.game_over else [BY_NAME["up"]]
        connection.call(lambda a, held=held: a.set_input(held))
        connection.call(lambda a: a.advance_frames(1))
        observation.advance(1)
        now = adapter.game.state.lives
        if now < lives:
            return
        lives = now
    raise AssertionError("no life lost")


def _telling(
    connection: ConnectionService, observation: ObservationService, bus: EventBus
) -> DiscoveryService:
    """A service whose narrowing window is shorter than the toy racer's crashes are apart."""
    return DiscoveryService(connection, observation, bus, review_every=100, moment_window=30)


def _tell_until_settled(
    discovery: DiscoveryService, connection: ConnectionService, observation: ObservationService
) -> Narrowing:
    for _ in range(6):
        _play_until_a_life_is_lost(connection, observation)
        n = discovery.tell(Moment.LOST_LIFE)
        if n.settled is not None:
            return n
    raise AssertionError(f"never settled: {n.text}")


def test_telling_the_ai_about_a_lost_life_finds_the_counter(
    connection: ConnectionService,
    observation: ObservationService,
    bus: EventBus,
    updates: list[DiscoveryUpdated],
    caplog: pytest.LogCaptureFixture,
) -> None:
    from retro_ai_lab.understanding import Confidence

    discovery = _telling(connection, observation, bus)
    with pytest.raises(UnsupportedCapabilityError):
        discovery.tell(Moment.LOST_LIFE)  # nothing connected
    connection.connect("fake", seed=2)
    k = discovery.knowledge
    assert k is not None
    wrong = k.propose("ram", LAYOUT.lane, kind=VariableKind.LIVES, name="", evidence=SURE_LIVES)
    assert k.best(VariableKind.LIVES) is not None and k.best(VariableKind.LIVES).id == wrong.id
    with pytest.raises(ConfigurationError, match="Play for a few seconds"):
        discovery.tell(Moment.LOST_LIFE)
    caplog.set_level(logging.INFO, logger="retro_ai_lab.services.discovery")
    _play_until_a_life_is_lost(connection, observation)
    first = discovery.tell(Moment.LOST_LIFE)
    assert ("ram", LAYOUT.lives) in first.candidates and first.settled is None
    assert first.text.startswith("You lost a life:") and first.text in caplog.text
    before = len(updates)
    second = _tell_until_settled(discovery, connection, observation)
    assert second.settled == ("ram", LAYOUT.lives) and second.text.startswith('Got it: "Lives"')
    found = k.at("ram", LAYOUT.lives)
    assert found is not None and found.confirmed and found.kind is VariableKind.LIVES
    assert found.name == "Lives" and found.confidence is Confidence.CONFIRMED
    assert any(e.kind == "told" and f"{second.presses} times" in e.detail for e in found.evidence)
    assert k.best(VariableKind.LIVES) is not None and k.best(VariableKind.LIVES).id == found.id
    # The AI's own wrong guess of that kind did not move: dropped by the AI, with a reason.
    dropped = k.get(wrong.id)
    assert dropped.rejected and any(e.kind == "told-otherwise" for e in dropped.evidence)
    assert "1 other lives guesses did not move when you lost a life: dropped" in caplog.text
    # One update carried both verdicts (the reviews in between carry their own).
    assert any(
        {v.id for v in u.added} | {v.id for v in u.changed} >= {found.id, wrong.id}
        for u in updates[before:]
    )
    # Settled: the next press starts a fresh search.
    _play_until_a_life_is_lost(connection, observation)
    again = discovery.tell(Moment.LOST_LIFE)
    assert again.presses == 1 and again.settled is None


def test_telling_about_a_named_guess_keeps_its_name(
    connection: ConnectionService, observation: ObservationService, bus: EventBus
) -> None:
    discovery = _telling(connection, observation, bus)
    connection.connect("fake", seed=2)
    k = discovery.knowledge
    assert k is not None
    mine = k.propose("ram", LAYOUT.lives, kind=VariableKind.PROGRESS, name="Cars left")
    k.reject(mine.id)  # a wrong "no" is taken back by the person's own word
    n = _tell_until_settled(discovery, connection, observation)
    assert n.settled == ("ram", LAYOUT.lives)
    got = k.get(mine.id)
    assert got.confirmed and not got.rejected and got.kind is VariableKind.LIVES
    assert got.name == "Lives"  # a rejected guess's name was part of what was wrong
    k.correct(mine.id, name="Cars left")
    n = _tell_until_settled(discovery, connection, observation)
    assert n.settled == ("ram", LAYOUT.lives) and k.get(mine.id).name == "Cars left"


def _play_until_a_restart(
    connection: ConnectionService,
    observation: ObservationService,
    updates: list[DiscoveryUpdated],
    most: int = 6000,
) -> None:
    """Hold up until the game is over, press Start, and return once a restart was announced."""
    adapter = connection.adapter
    assert isinstance(adapter, FakeEmulatorAdapter)
    seen = sum(1 for u in updates for e in u.events if e.kind is EventKind.RESET)
    for _ in range(most):
        held = [BY_NAME["start"]] if adapter.game.state.game_over else [BY_NAME["up"]]
        connection.call(lambda a, held=held: a.set_input(held))
        connection.call(lambda a: a.advance_frames(1))
        observation.advance(1)
        if sum(1 for u in updates for e in u.events if e.kind is EventKind.RESET) > seen:
            return
    raise AssertionError("no restart")


def test_the_ai_finds_the_life_counter_from_restarts(
    connection: ConnectionService,
    observation: ObservationService,
    bus: EventBus,
    updates: list[DiscoveryUpdated],
    caplog: pytest.LogCaptureFixture,
) -> None:
    """T-904: the AI presses "I just lost a life" for itself at each restart."""
    discovery = DiscoveryService(
        connection, observation, bus, review_every=10_000, experiment_every=None, moment_window=30
    )
    connection.connect("fake", seed=2)
    k = discovery.knowledge
    assert k is not None
    for offset, kind, name in (
        (LAYOUT.lane, VariableKind.POSITION, "Lane"),
        (LAYOUT.speed, VariableKind.SPEED, "Speed"),
        (LAYOUT.checkpoint, VariableKind.PROGRESS, "Checkpoint"),
        (LAYOUT.game_over, VariableKind.FLAG, "Game over"),
    ):
        k.propose("ram", offset, kind=kind, name=name, evidence=SURE_LIVES)
    k.propose(
        "ram",
        LAYOUT.score,
        encoding=Encoding.U16LE,
        kind=VariableKind.SCORE,
        name="Score",
        evidence=SURE_LIVES,
    )
    wrong = k.propose("ram", LAYOUT.noise, kind=VariableKind.LIVES, name="", evidence=SURE_LIVES)
    bus.publish(DiscoveryUpdated((), tuple(k), (), ()))
    caplog.set_level(logging.INFO, logger="retro_ai_lab.services.discovery")
    for _ in range(3):
        _play_until_a_restart(connection, observation, updates)
    found = k.at("ram", LAYOUT.lives)
    assert found is not None and found.kind is VariableKind.LIVES and not found.confirmed
    assert any(e.kind == "restarts" for e in found.evidence)
    assert found.confidence is Confidence.HIGH
    assert k.best(VariableKind.LIVES) is not None and k.best(VariableKind.LIVES).id == found.id
    assert not k.get(wrong.id).rejected  # the AI's finding adds; it never drops on its own
    assert (
        f"The AI noticed ram[{LAYOUT.lives}] went down by one before each of the last restarts"
        in caplog.text
    )
    assert any(found.id in {v.id for v in u.added} for u in updates)
    # The person's word stays above the AI's: a confirmed guess elsewhere is kept.
    k.reject(found.id)
    k.correct(wrong.id, name="Lifes", kind=VariableKind.LIVES, by="you")
    caplog.clear()
    for _ in range(3):
        _play_until_a_restart(connection, observation, updates)
    assert k.get(wrong.id).confirmed and not k.get(wrong.id).rejected
    assert (
        'but you confirmed "Lifes"; the AI keeps your word' in caplog.text
        or k.get(found.id).rejected
    )


def test_a_confirmed_score_is_never_the_progress_byte(
    connection: ConnectionService, observation: ObservationService, bus: EventBus
) -> None:
    """T-905: the user's "I just got further" settled on the byte he had confirmed as Score."""
    discovery = _telling(connection, observation, bus)
    connection.connect("fake", seed=2)
    k = discovery.knowledge
    assert k is not None
    score = k.propose("ram", LAYOUT.score, kind=VariableKind.SCORE, name="Score")
    k.correct(score.id, name="Score", kind=VariableKind.SCORE, by="you")
    play(connection, observation, 200)
    n = discovery.tell(Moment.GOT_FURTHER)
    assert ("ram", LAYOUT.score) not in n.candidates
    assert ("ram", LAYOUT.position) in n.candidates or ("ram", LAYOUT.checkpoint) in n.candidates
    # Said to be a score instead, the score byte is a candidate again.
    m = discovery.tell(Moment.SCORED)
    assert ("ram", LAYOUT.score) in m.candidates


# ------------------------------------------------------------ T-907: the summary cache


def test_the_summary_is_computed_once_per_change(
    discovery: DiscoveryService,
    connection: ConnectionService,
    observation: ObservationService,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The panel asks for the line after every review; a walk over thousands of values each time."""
    from retro_ai_lab.understanding.knowledge import Knowledge as Kn

    walks = 0
    original = Kn.__iter__

    def counted(self: Kn) -> object:
        nonlocal walks
        walks += 1
        return original(self)

    monkeypatch.setattr(Kn, "__iter__", counted)
    connection.connect("fake", seed=2)
    play(connection, observation, 100)  # one review: guesses proposed
    k = discovery.knowledge
    assert k is not None and len(k) >= 5
    before = walks
    first = discovery.summary()  # the review built it already: no walk
    assert first.startswith("Watching") and walks == before
    assert discovery.summary() == first and walks == before  # remembered
    # A verdict is a change: a fresh line, one walk.
    v = next(iter(k))
    before = walks
    discovery.reject(v.id)
    second = discovery.summary()
    assert second != first and walks == before + 1
    assert discovery.summary() == second and walks == before + 1
    # The wake button is in the line but not in the version: it still refreshes.
    k.wake_button = "start"
    assert discovery.summary().endswith("Start gets the game going.") and walks == before + 2
    connection.disconnect()
    assert discovery.summary() == "Not watching a game." and walks == before + 2
    connection.connect("fake", seed=2)
    before = walks
    assert discovery.summary().startswith("Watching") and walks == before + 1
