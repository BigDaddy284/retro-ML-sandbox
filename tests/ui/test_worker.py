"""T-829: a worker whose signal owner is gone drops its report instead of crashing."""

from __future__ import annotations

import logging
import threading

import pytest
from pytestqt.qtbot import QtBot
from shiboken6 import delete

from retro_ai_lab.ui.context import AppContext
from retro_ai_lab.ui.controller_pad import _SendTask
from retro_ai_lab.ui.game_panel import _CaptureTask
from retro_ai_lab.ui.play_clock import _StepSignals, _StepTask
from retro_ai_lab.ui.worker import report


def test_report_emits_when_the_owner_is_alive(qapp: object) -> None:
    signals = _StepSignals()
    seen: list[tuple[int, object]] = []
    signals.done.connect(lambda g, r: seen.append((g, r)))
    assert report(signals.done, 3, "ok") is True
    assert seen == [(3, "ok")]


def test_report_is_quiet_when_the_owner_is_gone(
    qapp: object, caplog: pytest.LogCaptureFixture
) -> None:
    signals = _StepSignals()
    done = signals.done
    delete(signals)
    with caplog.at_level(logging.DEBUG, logger="retro_ai_lab.ui.worker"):
        assert report(done, 1, "late") is False
    assert "nobody left to report to" in caplog.text


@pytest.mark.parametrize("which", ["step", "capture", "send"])
def test_a_task_finishing_after_its_owner_is_gone_does_not_raise(qtbot: QtBot, which: str) -> None:
    context = AppContext()
    context.connection.connect("fake")
    if which == "step":
        task = _StepTask(
            context.connection, context.observation, 1, 0, pressed=None, screenshot=False
        )
    elif which == "capture":
        task = _CaptureTask(context.connection, 0)  # type: ignore[assignment]
    else:
        task = _SendTask(context, frozenset())  # type: ignore[assignment]
    task.setAutoDelete(False)
    delete(task.signals)
    failures: list[BaseException] = []
    worker = threading.Thread(target=lambda: _run(task, failures))
    worker.start()
    worker.join(5)
    assert not worker.is_alive() and failures == []
    context.connection.disconnect()


def _run(task: object, failures: list[BaseException]) -> None:
    try:
        task.run()  # type: ignore[attr-defined]
    except BaseException as exc:
        failures.append(exc)
