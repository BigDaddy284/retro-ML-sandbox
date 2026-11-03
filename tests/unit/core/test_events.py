"""T-006: the event bus."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from retro_ai_lab.core.events import EventBus


@dataclass(frozen=True)
class Ping:
    n: int


@dataclass(frozen=True)
class LoudPing(Ping):
    pass


@dataclass(frozen=True)
class Other:
    pass


def test_publish_delivers_in_subscription_order() -> None:
    bus = EventBus()
    seen: list[str] = []
    bus.subscribe(Ping, lambda e: seen.append(f"a{e.n}"))
    bus.subscribe(Ping, lambda e: seen.append(f"b{e.n}"))
    assert bus.publish(Ping(1)) == 2
    assert seen == ["a1", "b1"]


def test_type_filtering_includes_subclasses_only() -> None:
    bus = EventBus()
    pings: list[Ping] = []
    others: list[Other] = []
    bus.subscribe(Ping, pings.append)
    bus.subscribe(Other, others.append)
    bus.publish(LoudPing(2))
    bus.publish(Other())
    assert pings == [LoudPing(2)]
    assert others == [Other()]
    assert bus.publish(object()) == 0


def test_unsubscribe_via_returned_callable_and_method() -> None:
    bus = EventBus()
    seen: list[int] = []
    handler = lambda e: seen.append(e.n)  # noqa: E731
    off = bus.subscribe(Ping, handler)
    bus.publish(Ping(1))
    off()
    bus.publish(Ping(2))
    bus.unsubscribe(Ping, handler)  # already removed: no error
    bus.unsubscribe(Other, handler)  # never registered: no error
    assert seen == [1]
    assert bus.handler_count() == 0


def test_handler_errors_are_isolated_and_reported() -> None:
    reported: list[tuple[str, Any]] = []
    bus = EventBus(on_error=lambda exc, event, handler: reported.append((str(exc), event)))
    seen: list[int] = []

    def bad(e: Ping) -> None:
        raise ValueError("boom")

    bus.subscribe(Ping, bad)
    bus.subscribe(Ping, lambda e: seen.append(e.n))
    assert bus.publish(Ping(3)) == 2
    assert seen == [3]
    assert reported == [("boom", Ping(3))]


def test_default_error_hook_logs(caplog: pytest.LogCaptureFixture) -> None:
    bus = EventBus()
    bus.subscribe(Ping, lambda e: 1 / 0)
    with caplog.at_level("ERROR"):
        bus.publish(Ping(0))
    assert "failed" in caplog.text


def test_handler_may_unsubscribe_itself_during_publish() -> None:
    bus = EventBus()
    seen: list[int] = []

    def once(e: Ping) -> None:
        seen.append(e.n)
        bus.unsubscribe(Ping, once)

    bus.subscribe(Ping, once)
    bus.publish(Ping(1))
    bus.publish(Ping(2))
    assert seen == [1]


def test_handler_count_by_type() -> None:
    bus = EventBus()
    bus.subscribe(Ping, lambda e: None)
    bus.subscribe(Other, lambda e: None)
    assert bus.handler_count(Ping) == 1
    assert bus.handler_count(LoudPing) == 0
    assert bus.handler_count() == 2


@pytest.mark.parametrize(("event_type", "handler"), [("Ping", lambda e: None), (Ping, 42)])
def test_subscribe_validates_arguments(event_type: Any, handler: Any) -> None:
    with pytest.raises(TypeError):
        EventBus().subscribe(event_type, handler)
