"""T-1036: in training, a go that can go back asks the game about the rises it is
paid for, and tells the discovery service."""

from __future__ import annotations

import pytest

from retro_ai_lab.training import service as module
from retro_ai_lab.training.records import Mode
from tests.unit.training import test_going_back_in_training as in_training
from tests.unit.training.test_going_back_in_training import NO_SAVES, Lab

lab = in_training.lab  # the services, with no game connected yet (pytest fixture)


def test_the_service_hands_the_runner_the_asking(lab: Lab, monkeypatch: pytest.MonkeyPatch) -> None:
    made: list[dict[str, object]] = []
    real = module.EpisodeRunner

    def building(*args: object, **kw: object) -> object:
        made.append(kw)
        return real(*args, **kw)  # type: ignore[arg-type]

    monkeypatch.setattr(module, "EpisodeRunner", building)
    discovery = lab.discovery
    # Where a go can go back, the discovery service is asked and told.
    t = lab.training()
    lab.connect()
    t.start(Mode.INDEPENDENT)
    assert t.step()
    assert len(made) == 1 and made[0]["moments"] is not None
    assert made[0]["rise_worth_asking"] == discovery.rise_worth_asking
    assert made[0]["rise_answered"] == discovery.rise_answered
    t.stop()
    # Going back switched off, or a console that cannot save: nobody.
    for kw, connect in (({"ways_per_go": 0}, {}), ({}, {"capabilities": NO_SAVES})):
        lab.connection.disconnect()
        other = lab.training(**kw)
        lab.connect(**connect)
        other.start(Mode.INDEPENDENT)
        assert other.step()
        assert made[-1]["moments"] is None
        assert made[-1]["rise_worth_asking"] is None and made[-1]["rise_answered"] is None
        other.stop()
    assert len(made) == 3
