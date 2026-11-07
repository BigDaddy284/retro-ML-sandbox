"""T-003: the EmulatorAdapter contract enforced by the base class."""

from __future__ import annotations

import threading
import time
from collections.abc import Sequence
from typing import Any

import pytest

from retro_ai_lab.core import (
    Button,
    Capabilities,
    ConfigurationError,
    EmulatorDisconnectedError,
    Frame,
    GameInfo,
    MemoryRegion,
    StepResult,
    UnsupportedCapabilityError,
)
from retro_ai_lab.emulator import DEFAULT_CONNECT_TIMEOUT_S, EmulatorAdapter

A = Button("a", "A")
B = Button("b", "B")
X = Button("x", "X")  # not offered by the test adapter


class RecordingAdapter(EmulatorAdapter):
    """Minimal adapter that records every backend call."""

    name = "recording"
    display_name = "Recording test adapter"

    def __init__(self, caps: Capabilities | None = None, ram_size: int = 16) -> None:
        super().__init__()
        self.caps = caps if caps is not None else Capabilities.all()
        self.calls: list[tuple[str, Any]] = []
        self.ram = bytes(range(ram_size))
        self.fail_connect: Exception | None = None

    def _do_connect(self, timeout_s: float) -> None:
        self.calls.append(("connect", timeout_s))
        if self.fail_connect:
            raise self.fail_connect

    def _do_disconnect(self) -> None:
        self.calls.append(("disconnect", None))

    def _do_capabilities(self) -> Capabilities:
        return self.caps

    def _do_game_info(self) -> GameInfo:
        return GameInfo("TEST", "Recorder")

    def _do_buttons(self) -> Sequence[Button]:
        return [A, B]

    def _do_memory_regions(self) -> Sequence[MemoryRegion]:
        return [MemoryRegion("ram", len(self.ram))]

    def _do_read_memory(self, region: str, offset: int, length: int) -> bytes:
        self.calls.append(("read", (region, offset, length)))
        return self.ram[offset : offset + length]

    def _do_capture_frame(self) -> Frame:
        self.calls.append(("frame", None))
        return Frame(1, 1, 1, b"\x00")

    def _do_set_input(self, pressed: frozenset[Button]) -> None:
        self.calls.append(("input", pressed))

    def _do_advance_frames(self, n: int) -> None:
        self.calls.append(("advance", n))

    def _do_watch_pad(self) -> frozenset[str]:
        self.calls.append(("watch", None))
        return frozenset({"a"})

    def _do_set_speed(self, multiplier: float) -> None:
        self.calls.append(("speed", multiplier))

    def _do_save_state(self, slot: str) -> None:
        self.calls.append(("save", slot))

    def _do_load_state(self, slot: str) -> None:
        self.calls.append(("load", slot))


class ShortReadAdapter(RecordingAdapter):
    def _do_read_memory(self, region: str, offset: int, length: int) -> bytes:
        return b""


class MinimalAdapter(EmulatorAdapter):
    """Implements only the abstract hooks and reports no capabilities."""

    name = "minimal"
    display_name = "Minimal"

    def _do_connect(self, timeout_s: float) -> None: ...

    def _do_disconnect(self) -> None: ...

    def _do_capabilities(self) -> Capabilities:
        return Capabilities()

    def _do_game_info(self) -> GameInfo:
        return GameInfo("TEST", "Minimal")

    def _do_buttons(self) -> Sequence[Button]:
        return []


class LyingAdapter(MinimalAdapter):
    """Claims every capability but implements none: defaults must still raise."""

    def _do_capabilities(self) -> Capabilities:
        return Capabilities.all()


@pytest.fixture
def adapter() -> RecordingAdapter:
    a = RecordingAdapter()
    a.connect()
    return a


# -------------------------------------------------------------- lifecycle


def test_cannot_instantiate_abstract_base() -> None:
    with pytest.raises(TypeError):
        EmulatorAdapter()  # type: ignore[abstract]


def test_connect_and_disconnect_lifecycle() -> None:
    a = RecordingAdapter()
    assert not a.is_connected
    a.connect()
    assert a.is_connected
    assert a.calls == [("connect", DEFAULT_CONNECT_TIMEOUT_S)]
    a.disconnect()
    assert not a.is_connected
    assert a.calls[-1] == ("disconnect", None)


def test_connect_is_idempotent_and_disconnect_is_idempotent() -> None:
    a = RecordingAdapter()
    a.disconnect()  # never connected: no-op, no backend call
    assert a.calls == []
    a.connect(1.5)
    a.connect(2.5)
    assert a.calls == [("connect", 1.5)]
    a.disconnect()
    a.disconnect()
    assert a.calls.count(("disconnect", None)) == 1


def test_connect_passes_timeout_as_float() -> None:
    a = RecordingAdapter()
    a.connect(3)
    assert a.calls == [("connect", 3.0)]
    assert isinstance(a.calls[0][1], float)


@pytest.mark.parametrize("timeout", [0, -1, "5", None, True])
def test_connect_rejects_bad_timeout(timeout: Any) -> None:
    a = RecordingAdapter()
    with pytest.raises(ConfigurationError):
        a.connect(timeout)
    assert not a.is_connected


def test_failed_connect_leaves_adapter_disconnected() -> None:
    a = RecordingAdapter()
    a.fail_connect = OSError("refused")
    with pytest.raises(OSError, match="refused"):
        a.connect()
    assert not a.is_connected


def test_mark_disconnected_drops_connection_without_backend_call() -> None:
    a = RecordingAdapter()
    a.connect()
    a.mark_disconnected()
    assert not a.is_connected
    assert ("disconnect", None) not in a.calls
    with pytest.raises(EmulatorDisconnectedError):
        a.capture_frame()


OPERATIONS: list[tuple[str, tuple[Any, ...]]] = [
    ("capabilities", ()),
    ("game_info", ()),
    ("buttons", ()),
    ("memory_regions", ()),
    ("read_memory", ("ram", 0, 1)),
    ("capture_frame", ()),
    ("set_input", ([A],)),
    ("advance_frames", (1,)),
    ("set_speed", (2.0,)),
    ("save_state", ("s1",)),
    ("load_state", ("s1",)),
]


@pytest.mark.parametrize(("op", "args"), OPERATIONS)
def test_every_operation_requires_connection(op: str, args: tuple[Any, ...]) -> None:
    a = RecordingAdapter()
    with pytest.raises(EmulatorDisconnectedError):
        getattr(a, op)(*args)
    assert a.calls == []


# ----------------------------------------------------------- capabilities

CAPABILITY_OPS: list[tuple[str, str, tuple[Any, ...]]] = [
    ("memory_read", "memory_regions", ()),
    ("memory_read", "read_memory", ("ram", 0, 1)),
    ("screen_capture", "capture_frame", ()),
    ("input", "set_input", ([A],)),
    ("frame_advance", "advance_frames", (1,)),
    ("speed_control", "set_speed", (2.0,)),
    ("save_states", "save_state", ("s1",)),
    ("save_states", "load_state", ("s1",)),
]


@pytest.mark.parametrize(("cap", "op", "args"), CAPABILITY_OPS)
def test_missing_capability_raises_before_backend(cap: str, op: str, args: tuple[Any, ...]) -> None:
    caps = Capabilities.from_dict({**Capabilities.all().to_dict(), cap: False})
    a = RecordingAdapter(caps)
    a.connect()
    with pytest.raises(UnsupportedCapabilityError) as info:
        getattr(a, op)(*args)
    assert info.value.capability == cap
    assert a.calls == [("connect", DEFAULT_CONNECT_TIMEOUT_S)]


@pytest.mark.parametrize(("cap", "op", "args"), CAPABILITY_OPS)
def test_default_hooks_raise_if_adapter_claims_but_does_not_implement(
    cap: str, op: str, args: tuple[Any, ...]
) -> None:
    a = LyingAdapter()
    a.connect()
    with pytest.raises(UnsupportedCapabilityError) as info:
        if op == "set_input":
            a.set_input([])  # LyingAdapter offers no buttons
        else:
            getattr(a, op)(*args)
    assert info.value.capability == cap


def test_minimal_adapter_describes_itself() -> None:
    a = MinimalAdapter()
    a.connect()
    assert a.capabilities() == Capabilities()
    assert a.game_info() == GameInfo("TEST", "Minimal")
    assert a.buttons() == ()


# ------------------------------------------------------------- read_memory


def test_read_memory_happy_path(adapter: RecordingAdapter) -> None:
    assert adapter.read_memory("ram", 2, 3) == bytes([2, 3, 4])
    assert adapter.read_memory("ram", 0, 0) == b""
    assert adapter.read_memory("ram", 15, 1) == bytes([15])
    assert adapter.memory_regions() == (MemoryRegion("ram", 16),)


@pytest.mark.parametrize(
    ("region", "offset", "length"),
    [
        ("vram", 0, 1),
        ("ram", -1, 1),
        ("ram", 0, -1),
        ("ram", 16, 1),
        ("ram", 0, 17),
        ("ram", 10, 7),
        ("ram", 0.0, 1),
        ("ram", 0, True),
    ],
)
def test_read_memory_rejects_bad_requests(
    adapter: RecordingAdapter, region: str, offset: Any, length: Any
) -> None:
    with pytest.raises(ConfigurationError):
        adapter.read_memory(region, offset, length)
    assert not any(c[0] == "read" for c in adapter.calls)


def test_read_memory_detects_short_backend_reads() -> None:
    a = ShortReadAdapter()
    a.connect()
    with pytest.raises(ConfigurationError, match="0 bytes"):
        a.read_memory("ram", 0, 4)


# ------------------------------------------------------------------ input


def test_set_input_accepts_known_buttons_and_dedupes(adapter: RecordingAdapter) -> None:
    adapter.set_input([A, B, A])
    assert adapter.calls[-1] == ("input", frozenset({A, B}))
    adapter.set_input(())
    assert adapter.calls[-1] == ("input", frozenset())


def test_set_input_rejects_unknown_button(adapter: RecordingAdapter) -> None:
    with pytest.raises(ConfigurationError, match="x"):
        adapter.set_input([A, X])
    assert not any(c[0] == "input" for c in adapter.calls)


# --------------------------------------------------------- frames & speed


def test_capture_frame_returns_backend_frame(adapter: RecordingAdapter) -> None:
    assert adapter.capture_frame() == Frame(1, 1, 1, b"\x00")


def test_advance_frames(adapter: RecordingAdapter) -> None:
    adapter.advance_frames()
    adapter.advance_frames(10)
    assert [c for c in adapter.calls if c[0] == "advance"] == [("advance", 1), ("advance", 10)]


@pytest.mark.parametrize("n", [0, -1, 1.0, True, "1"])
def test_advance_frames_rejects_bad_counts(adapter: RecordingAdapter, n: Any) -> None:
    with pytest.raises(ConfigurationError):
        adapter.advance_frames(n)


def test_set_speed(adapter: RecordingAdapter) -> None:
    adapter.set_speed(4)
    adapter.set_speed(0.5)
    assert [c for c in adapter.calls if c[0] == "speed"] == [("speed", 4.0), ("speed", 0.5)]


@pytest.mark.parametrize("m", [0, -2, "fast", None, True])
def test_set_speed_rejects_bad_values(adapter: RecordingAdapter, m: Any) -> None:
    with pytest.raises(ConfigurationError):
        adapter.set_speed(m)


# ------------------------------------------------------------ save states


def test_save_and_load_state(adapter: RecordingAdapter) -> None:
    adapter.save_state("boss_1")
    adapter.load_state("boss_1")
    assert adapter.calls[-2:] == [("save", "boss_1"), ("load", "boss_1")]


@pytest.mark.parametrize("slot", ["", "a b", "a/b", "../x", None, 1])
def test_save_and_load_reject_bad_slots(adapter: RecordingAdapter, slot: Any) -> None:
    with pytest.raises(ConfigurationError):
        adapter.save_state(slot)
    with pytest.raises(ConfigurationError):
        adapter.load_state(slot)


# ------------------------------------------------------------- sequences


def test_buttons_and_regions_are_returned_as_tuples(adapter: RecordingAdapter) -> None:
    assert isinstance(adapter.buttons(), tuple)
    assert isinstance(adapter.memory_regions(), tuple)
    assert adapter.buttons() == (A, B)


# ------------------------------------------------------------- T-817 threads


class _OverlapAdapter(RecordingAdapter):
    """Counts how many backend calls run at once; the base must keep it at one."""

    def __init__(self) -> None:
        super().__init__()
        self.active = 0
        self.max_active = 0
        self.entered = threading.Event()
        self.release = threading.Event()

    def _do_capture_frame(self) -> Frame:
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        self.entered.set()
        self.release.wait(5)
        self.active -= 1
        return super()._do_capture_frame()

    def _do_advance_frames(self, n: int) -> None:
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        self.active -= 1
        super()._do_advance_frames(n)


def test_operations_from_two_threads_never_overlap() -> None:
    adapter = _OverlapAdapter()
    adapter.connect()
    adapter.release.set()
    stop = time.monotonic() + 0.3

    def hammer(op: object) -> None:
        while time.monotonic() < stop:
            op()  # type: ignore[operator]

    threads = [
        threading.Thread(target=hammer, args=(adapter.capture_frame,)),
        threading.Thread(target=hammer, args=(lambda: adapter.advance_frames(1),)),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    names = {name for name, _ in adapter.calls}
    assert adapter.max_active == 1
    assert {"frame", "advance"} <= names


def test_check_link_answers_at_once_while_another_thread_holds_the_adapter() -> None:
    adapter = _OverlapAdapter()
    adapter.connect()
    worker = threading.Thread(target=adapter.capture_frame)
    worker.start()
    assert adapter.entered.wait(5)
    started = time.perf_counter()
    assert adapter.check_link() is True
    assert time.perf_counter() - started < 0.5  # did not wait for the capture
    adapter.release.set()
    worker.join(5)
    assert adapter.check_link() is True
    adapter.mark_disconnected()
    assert adapter.check_link() is False


# ------------------------------------------------------------ T-822 bundles


def test_step_composes_input_advance_reads_and_screenshot_in_order(
    adapter: RecordingAdapter,
) -> None:
    result = adapter.step(3, pressed={A}, reads=[("ram", 0, 4), ("ram", 4, 2)], screenshot=True)
    assert result == StepResult(3, (bytes([0, 1, 2, 3]), bytes([4, 5])), Frame(1, 1, 1, b"\x00"))
    assert [name for name, _ in adapter.calls[1:]] == ["input", "advance", "read", "read", "frame"]
    assert adapter.calls[1] == ("input", frozenset({A}))
    assert adapter.calls[2] == ("advance", 3)


def test_step_defaults_to_one_plain_frame(adapter: RecordingAdapter) -> None:
    assert adapter.step() == StepResult(1)
    assert [name for name, _ in adapter.calls[1:]] == ["advance"]


def test_step_checks_each_capability_only_when_used() -> None:
    a = RecordingAdapter(Capabilities(frame_advance=True))
    a.connect()
    assert a.step(2) == StepResult(2)
    with pytest.raises(UnsupportedCapabilityError, match="input"):
        a.step(1, pressed={A})
    with pytest.raises(UnsupportedCapabilityError, match="memory_read"):
        a.step(1, reads=[("ram", 0, 1)])
    with pytest.raises(UnsupportedCapabilityError, match="screen_capture"):
        a.step(1, screenshot=True)
    assert [name for name, _ in a.calls[1:]] == ["advance"]  # nothing reached the backend
    a2 = RecordingAdapter(Capabilities(memory_read=True))
    a2.connect()
    with pytest.raises(UnsupportedCapabilityError, match="frame_advance"):
        a2.step()


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"frames": 0}, "frame count"),
        ({"pressed": {Button("turbo", "Turbo")}}, "Unknown buttons"),
        ({"reads": [("vram", 0, 1)]}, "Unknown memory region"),
        ({"reads": [("ram", 10, 10)]}, "outside region"),
        ({"reads": [("ram", -1, 1)]}, "offset"),
        ({"reads": [("ram", 0)]}, "Each read must be"),
    ],
)
def test_step_rejects_bad_requests_before_the_backend(
    adapter: RecordingAdapter, kwargs: dict[str, Any], match: str
) -> None:
    frames = kwargs.pop("frames", 1)
    with pytest.raises(ConfigurationError, match=match):
        adapter.step(frames, **kwargs)
    assert adapter.calls[1:] == []


def test_step_refuses_a_short_read_and_a_missing_screenshot() -> None:
    class Short(RecordingAdapter):
        def _do_read_memory(self, region: str, offset: int, length: int) -> bytes:
            return b"x"

    a = Short()
    a.connect()
    with pytest.raises(ConfigurationError, match="1 bytes for a 4-byte read"):
        a.step(1, reads=[("ram", 0, 4)])

    class Blind(RecordingAdapter):
        def _do_step(
            self, frames: int, pressed: Any, reads: Any, screenshot: bool, watch: bool = False
        ) -> StepResult:
            return StepResult(frames)

    b = Blind()
    b.connect()
    with pytest.raises(ConfigurationError, match="without the screenshot"):
        b.step(1, screenshot=True)
    with pytest.raises(ConfigurationError, match="answered 0 reads for 1"):
        b.step(1, reads=[("ram", 0, 1)])
    with pytest.raises(ConfigurationError, match="without reporting the pad"):
        b.step(1, watch=True)


def test_a_watched_step_leaves_the_pad_to_the_person(adapter: RecordingAdapter) -> None:
    """T-887: ``watch`` sets nothing and answers what the person held."""
    result = adapter.step(2, watch=True, reads=[("ram", 0, 1)])
    assert result == StepResult(2, (bytes([0]),), None, frozenset({"a"}))
    assert [name for name, _ in adapter.calls[1:]] == ["advance", "read", "watch"]
    assert adapter.step(1).pressed is None
    with pytest.raises(ConfigurationError, match="hold the pad and watch"):
        adapter.step(1, pressed={A}, watch=True)
    blind = RecordingAdapter(Capabilities(frame_advance=True))
    blind.connect()
    with pytest.raises(UnsupportedCapabilityError, match="input"):
        blind.step(1, watch=True)

    class Deaf(RecordingAdapter):
        def _do_watch_pad(self) -> frozenset[str]:
            return EmulatorAdapter._do_watch_pad(self)

    deaf = Deaf()
    deaf.connect()
    with pytest.raises(UnsupportedCapabilityError, match="cannot watch"):
        deaf.step(1, watch=True)


def test_step_requires_a_connection() -> None:
    with pytest.raises(EmulatorDisconnectedError):
        RecordingAdapter().step()


def test_read_many_returns_bytes_in_order(adapter: RecordingAdapter) -> None:
    assert adapter.read_many([("ram", 2, 3), ("ram", 0, 1)]) == (bytes([2, 3, 4]), bytes([0]))
    assert adapter.read_many([]) == ()
    assert [c for c in adapter.calls if c[0] == "read"] == [
        ("read", ("ram", 2, 3)),
        ("read", ("ram", 0, 1)),
    ]
    with pytest.raises(ConfigurationError, match="Unknown memory region"):
        adapter.read_many([("ram", 0, 1), ("rom", 0, 1)])
    blind = RecordingAdapter(Capabilities(frame_advance=True))
    blind.connect()
    with pytest.raises(UnsupportedCapabilityError):
        blind.read_many([("ram", 0, 1)])


def test_the_base_adapter_holds_and_cannot_free_run() -> None:
    """T-942: an adapter says whether its game stands still between requests and whether
    it can be let run; the base holds and cannot switch."""
    a = RecordingAdapter()
    assert a.hold and not a.free_run and not a.supports_free_run
    with pytest.raises(ConfigurationError):
        a.set_free_run("yes")  # type: ignore[arg-type]
    with pytest.raises(UnsupportedCapabilityError):
        a.set_free_run(True)
