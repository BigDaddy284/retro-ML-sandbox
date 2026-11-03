"""T-604: what the AI is doing right now."""

from __future__ import annotations

from pathlib import Path

import pytest
from PySide6.QtWidgets import QApplication
from pytestqt.qtbot import QtBot

from retro_ai_lab.emulator.fake.toy_racer import BUTTONS
from retro_ai_lab.learning import ActionSpace, Decision, QLearningAgent, Settings, State
from retro_ai_lab.training import Mode, Status, TrainingStepped
from retro_ai_lab.ui.activity_strip import (
    GOAL_DONE,
    LOOKING,
    NO_GOAL,
    NO_MOVES,
    PAUSED,
    RESTING,
    ActivityStrip,
    buttons_text,
    now_line,
)
from retro_ai_lab.ui.context import AppContext
from retro_ai_lab.ui.discovery_panel import DiscoveryPanel
from tests.ui.test_learning_panel import connect


@pytest.fixture
def context(tmp_path: Path) -> AppContext:
    return AppContext(data_dir=tmp_path)


@pytest.fixture
def strip(qtbot: QtBot, context: AppContext) -> ActivityStrip:
    s = ActivityStrip(context)
    qtbot.addWidget(s)
    s.show()
    return s


def settle() -> None:
    QApplication.processEvents()  # the strip redraws once per event-loop turn


def test_agent_records_how_it_chose() -> None:
    space = ActionSpace.from_buttons(BUTTONS)
    s = State((("v1", 1),))
    explorer = QLearningAgent(space, Settings(epsilon=1.0, epsilon_min=1.0), seed=1)
    assert explorer.last_decision is None
    explorer.choose(s)
    d = explorer.last_decision
    assert d is not None and d.explored and d.value == 0.0 and 0 <= d.action < len(space)
    careful = QLearningAgent(space, Settings(epsilon=0.0, epsilon_min=0.0), seed=1)
    careful.learn(s, 3, 2.0, s, True)
    assert careful.choose(s) == 3
    d = careful.last_decision
    assert d is not None and not d.explored and d.action == 3 and d.value == pytest.approx(0.2)
    careful.reset()
    assert careful.last_decision is None


def test_now_line_wording() -> None:
    up = frozenset({"up"})
    assert buttons_text(frozenset()) == "do nothing"
    assert buttons_text(frozenset({"a", "up"})) == "up + a"
    assert now_line(Status.IDLE, True, up, None) == RESTING
    assert now_line(Status.PAUSED, True, up, None) == PAUSED
    assert now_line(Status.RUNNING, True, None, None) == LOOKING
    assert now_line(Status.RUNNING, False, up, None) == "Watching you press up."
    assert now_line(Status.RUNNING, True, frozenset({"start"}), None) == (
        "Now: pressing start to get the game going again."
    )
    assert now_line(Status.RUNNING, True, up, Decision(1, True, 0.0)) == (
        "Now: pressing up, trying something new."
    )
    assert now_line(Status.RUNNING, True, frozenset(), Decision(0, False, 3.25)) == (
        "Now: pressing do nothing, its best known move (worth +3.2)."
    )


def test_strip_follows_an_ai_go(strip: ActivityStrip, context: AppContext) -> None:
    assert (strip.now_text, strip.before_text, strip.goal_text) == (RESTING, NO_MOVES, NO_GOAL)
    connect(context)
    assert strip.now_text == RESTING
    assert strip.goal_text.startswith("Working on: Get further in Checkpoint")
    t = context.training
    stepped: list[TrainingStepped] = []
    context.bus.subscribe(TrainingStepped, stepped.append)
    t.start(Mode.INDEPENDENT)
    assert strip.now_text == LOOKING
    t.step()
    settle()
    agent = t.agent
    assert agent is not None and agent.last_decision is not None
    assert len(stepped) == 1 and stepped[0].decision == agent.last_decision
    assert stepped[0].frame == context.observation.frame
    pressed = buttons_text(strip.moves[0])
    assert strip.now_text.startswith(f"Now: pressing {pressed}, ")
    assert strip.before_text == NO_MOVES  # only one move so far
    for _ in range(12):
        t.step()
    settle()
    assert len(stepped) == 13 and len(strip.moves) == 8  # the last eight, newest first
    assert strip.before_text.startswith("Before: ") and strip.before_text.count(",") == 6
    assert strip.moves[0] == stepped[-1].buttons and strip.moves[1] == stepped[-2].buttons
    t.pause()
    assert strip.now_text == PAUSED
    t.resume()
    t.stop()
    assert strip.now_text == RESTING
    context.connection.disconnect()
    assert strip.moves == () and strip.before_text == NO_MOVES and strip.goal_text == NO_GOAL


def test_strip_reports_the_childs_presses_in_manual_mode(
    strip: ActivityStrip, context: AppContext
) -> None:
    connect(context)
    t = context.training

    def play(buttons: list[str]) -> None:
        context.observation.record_input(buttons)
        context.connection.call(
            lambda a: a.set_input([b for b in a.buttons() if b.name in buttons])
        )
        for _ in range(t.frames_per_step):
            context.connection.call(lambda a: a.advance_frames(1))
            context.observation.advance(1)
        settle()

    t.start(Mode.MANUAL)
    assert strip.now_text == LOOKING
    play(["up"])
    assert strip.now_text == "Watching you press up."
    play(["up", "a"])
    assert strip.before_text == "Before: up."
    assert strip.now_text == "Watching you press up + a."
    t.stop()
    assert strip.now_text == RESTING and strip.moves[0] == frozenset({"up", "a"})


def test_goal_line_follows_the_goal(strip: ActivityStrip, context: AppContext) -> None:
    connect(context)
    goals = context.goals
    assert goals.goal is not None
    first = goals.goal.next()
    assert first is not None and strip.goal_text == f"Working on: {first.description}."
    goals.mark_complete(first.id)
    if goals.goal.is_complete:
        assert strip.goal_text == GOAL_DONE
    else:
        nxt = goals.goal.next()
        assert nxt is not None and strip.goal_text == f"Working on: {nxt.description}."
    for o in list(goals.goal.ids()):
        goals.remove_objective(o)
    assert strip.goal_text == NO_GOAL


def test_discovery_panel_hosts_the_strip_and_it_unsubscribes(
    qtbot: QtBot, context: AppContext
) -> None:
    panel = DiscoveryPanel(context)
    assert isinstance(panel.activity, ActivityStrip)
    assert panel.activity.parent() is panel
    before = context.bridge.handler_count()
    panel.deleteLater()
    qtbot.waitUntil(lambda: context.bridge.handler_count() < before)
    connect(context)
    context.connection.disconnect()
