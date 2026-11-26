"""T-1069: the emulator is not left at the app's speed when the script ends.

The app runs the game at up to 64 times its speed, and the bridge script puts
100% back when the app leaves (T-836). Where EmuHawk was closed first, nothing
did: EmuHawk keeps its speed in its settings and saves them as it closes, before
it ends its scripts, and started at that speed the next time, with no app there
(the user's recording of 25 Nov: the title screen flicking by). The script now
puts 100% back as it ends, and leaves a note in the temp folder for as long as
the speed is the app's, so the script that starts next can put it back too.
"""

from __future__ import annotations

import socket
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from retro_ai_lab.emulator.bizhawk import BizHawkAdapter
from tests.unit.emulator.bizhawk.lua_host import LuaScriptHost

BACK = b"The emulator's speed is back at 100%."
NOT_BACK = b"could not be put back"


class Closing(LuaScriptHost):
    """The host with the script's end: what it gave ``event.onexit`` is run.

    ``speed`` is the speed EmuHawk starts at, the one in its settings.
    """

    def __init__(self, *, port: int = 0, speed: float | None = None) -> None:
        super().__init__(port=port)
        self.speed = speed
        self.speed_fails = False  # client.speedmode raises, as it may in a closing EmuHawk
        self.cannot_listen = False
        self._at_exit: list[Any] = []

    def _onexit(self, fn: Any) -> None:
        super()._onexit(fn)
        self._at_exit.append(fn)

    def _speedmode(self, percent: float) -> None:
        if self.speed_fails:
            self._record("speedmode")
            raise RuntimeError("EmuHawk is closing.")
        super()._speedmode(percent)

    def _bind(self, host: bytes, port: int) -> int:
        if self.cannot_listen:
            raise OSError("The port is taken.")
        return super()._bind(host, port)

    def end(self) -> None:
        """EmuHawk closes, or the script is stopped: its loop is run no more, and
        the functions it gave ``event.onexit`` are called."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
        for fn in self._at_exit:
            fn()
        self.stop()

    def said(self, words: bytes) -> bool:
        return any(words in line for line in self.log)


@pytest.fixture
def temp(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """The temp folder the script finds, empty."""
    folder = tmp_path / "temp"
    folder.mkdir()
    for name in ("TEMP", "TMP", "TMPDIR"):
        monkeypatch.setenv(name, str(folder))
    return folder


def note(temp: Path, port: int = 0) -> Path:
    """The note for the script asked to listen on ``port``."""
    return temp / f"retro_ai_lab_speed_{port}.txt"


def wait_until(check: Callable[[], bool], *, timeout_s: float = 5.0) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if check():
            return
        time.sleep(0.01)
    raise AssertionError("condition not met in time")


def app(host: LuaScriptHost) -> BizHawkAdapter:
    adapter = BizHawkAdapter(port=host.port, call_timeout_s=5.0)
    adapter.connect(timeout_s=5.0)
    return adapter


def started(**settings: Any) -> Closing:
    host = Closing(**settings)
    host.start()
    return host


def times(host: LuaScriptHost) -> int:
    """How often the script has set the emulator's speed."""
    return host.calls.count("speedmode")


# ------------------------------------------------------- while the app is there


def test_the_note_is_there_for_as_long_as_the_speed_is_the_apps(temp: Path) -> None:
    host = started()
    try:
        adapter = app(host)
        assert not note(temp).exists() and host.speed is None
        adapter.set_speed(8.0)
        assert host.speed == 800 and note(temp).is_file()
        said = note(temp).read_text(encoding="utf-8")
        assert "Retro AI Lab" in said and "speed" in said
        adapter.set_speed(16.0)
        assert host.speed == 1600 and note(temp).read_text(encoding="utf-8") == said
        adapter.disconnect()
        wait_until(lambda: host.speed == 100)
        wait_until(lambda: not note(temp).exists())
        assert list(temp.iterdir()) == [] and host.error is None
        # The app again, and its speed: the note is there again.
        adapter = app(host)
        adapter.set_speed(4.0)
        assert host.speed == 400 and note(temp).is_file()
        adapter.disconnect()
        wait_until(lambda: not note(temp).exists())
        assert host.speed == 100
    finally:
        host.stop()


def test_a_speed_refused_leaves_no_note(temp: Path) -> None:
    host = started()
    try:
        adapter = app(host)
        for bad in ("fast", 0, -1, True):
            with pytest.raises(Exception, match="number above zero"):
                adapter._call("speed", percent=bad)
        assert not note(temp).exists() and times(host) == 0
        adapter.disconnect()
        wait_until(lambda: not host.paused)
        assert host.speed is None and times(host) == 0
    finally:
        host.stop()


# --------------------------------------------------------- the script's end


def test_the_script_puts_the_speed_back_as_it_ends(temp: Path) -> None:
    """Stopped in the Lua console with the app's speed in force, EmuHawk runs on."""
    host = started()
    adapter = app(host)
    adapter.set_speed(64.0)
    assert host.speed == 6400
    host.end()
    assert host.speed == 100 and times(host) == 2 and host.error is None
    assert host.exit_hooks == 1


def test_emuhawk_closed_at_the_apps_speed_starts_at_100_the_next_time(temp: Path) -> None:
    host = started()
    adapter = app(host)  # the app is still there as EmuHawk closes
    adapter.set_speed(64.0)
    host.end()
    # EmuHawk saved its settings before it ended the script: the note has to stay.
    assert note(temp).is_file()
    again = started(speed=6400)
    assert again.speed == 100 and times(again) == 1 and again.said(BACK)
    assert not note(temp).exists() and again.error is None
    again.end()
    assert times(again) == 1  # nothing to put back as this one ends
    # And from then on a speed is the person's own: no script touches it.
    later = started(speed=200)
    assert later.speed == 200 and times(later) == 0 and not later.said(BACK)
    adapter = app(later)
    adapter.disconnect()
    wait_until(lambda: not later.paused)
    later.end()
    assert later.speed == 200 and times(later) == 0 and list(temp.iterdir()) == []


def test_the_app_still_works_after_a_note_was_found(temp: Path) -> None:
    note(temp).write_text("left from last time\n", encoding="utf-8")
    host = started(speed=6400)
    try:
        assert host.speed == 100 and not note(temp).exists()
        adapter = app(host)
        adapter.set_speed(2.0)
        assert host.speed == 200 and note(temp).is_file()
        adapter.advance_frames(3)
        adapter.disconnect()
        wait_until(lambda: host.speed == 100 and not note(temp).exists())
    finally:
        host.stop()


def test_a_speed_that_cannot_be_set_at_the_end_does_not_stop_the_rest(temp: Path) -> None:
    host = started()
    adapter = app(host)
    adapter.set_speed(64.0)
    host.speed_fails = True
    host.end()  # no error out of the exit function
    assert host.speed == 6400 and times(host) == 2 and note(temp).is_file()
    assert host._clients == {} and host._servers == {}
    with pytest.raises(Exception, match=r"."):
        adapter.advance_frames(1)  # the link was closed all the same
    again = started(speed=6400)
    assert again.speed == 100 and not note(temp).exists()
    again.end()


def test_a_speed_that_cannot_be_set_at_the_start_does_not_stop_the_script(temp: Path) -> None:
    note(temp).write_text("left from last time\n", encoding="utf-8")
    host = Closing(speed=6400)
    host.speed_fails = True
    host.start()
    assert host.speed == 6400 and host.said(NOT_BACK) and not host.said(BACK)
    assert note(temp).is_file()  # the next start tries again
    host.speed_fails = False
    adapter = app(host)
    adapter.advance_frames(2)
    adapter.disconnect()
    wait_until(lambda: not host.paused)
    assert host.speed == 6400  # the app set none, so none is put back as it leaves
    host.end()
    again = started(speed=6400)
    assert again.speed == 100 and again.said(BACK) and not note(temp).exists()
    again.end()


def test_a_speed_the_app_never_set_is_left_alone_at_the_end(temp: Path) -> None:
    host = started(speed=150)
    adapter = app(host)
    adapter.advance_frames(2)
    host.end()
    assert host.speed == 150 and times(host) == 0 and list(temp.iterdir()) == []
    again = started(speed=150)
    assert again.speed == 150 and times(again) == 0 and not again.said(BACK)
    again.end()


def test_the_app_gone_first_leaves_nothing_for_the_end(temp: Path) -> None:
    host = started()
    adapter = app(host)
    adapter.set_speed(32.0)
    adapter.disconnect()
    wait_until(lambda: host.speed == 100 and not note(temp).exists())
    assert times(host) == 2
    host.speed = 150  # the person's own choice, after the app left
    host.end()
    assert host.speed == 150 and times(host) == 2 and list(temp.iterdir()) == []
    again = started(speed=150)
    assert again.speed == 150 and times(again) == 0
    again.end()


# ------------------------------------------------------------ the temp folder


def test_with_no_temp_folder_the_speed_is_still_put_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name in ("TEMP", "TMP", "TMPDIR"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.chdir(tmp_path)
    host = started(speed=300)
    assert host.speed == 300 and times(host) == 0
    adapter = app(host)
    adapter.set_speed(8.0)
    assert host.speed == 800
    host.end()
    assert host.speed == 100 and host.error is None
    assert list(tmp_path.iterdir()) == []  # and no note anywhere else


@pytest.mark.parametrize("empty", [False, True], ids=["not there", "named as nothing"])
def test_a_temp_folder_that_cannot_be_written_to_does_not_stop_the_speed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, empty: bool
) -> None:
    for name in ("TEMP", "TMP", "TMPDIR"):
        monkeypatch.setenv(name, "" if empty else str(tmp_path / "gone"))
    monkeypatch.chdir(tmp_path)
    host = started()
    try:
        adapter = app(host)
        adapter.set_speed(8.0)
        assert host.speed == 800
        adapter.disconnect()
        wait_until(lambda: host.speed == 100)
        assert host.error is None and list(tmp_path.iterdir()) == []
    finally:
        host.stop()


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def test_the_note_is_for_the_port_the_script_listens_on(temp: Path) -> None:
    """Two EmuHawks on one PC, each with its script on its own port."""
    port = free_port()
    one = started(port=port)
    adapter = app(one)
    adapter.set_speed(64.0)
    assert note(temp, port).is_file() and not note(temp).exists()
    other = started(speed=200)  # on another port: this note is not for it
    assert other.speed == 200 and times(other) == 0 and note(temp, port).is_file()
    other.end()
    one.end()
    assert note(temp, port).is_file()
    again = started(port=port, speed=6400)
    assert again.speed == 100 and not note(temp, port).exists()
    again.end()


def test_a_script_that_cannot_listen_leaves_the_note(temp: Path) -> None:
    """Another script has the port, and may be the one whose speed it is."""
    note(temp).write_text("left from last time\n", encoding="utf-8")
    host = Closing(speed=6400)
    host.cannot_listen = True
    with pytest.raises(RuntimeError):
        host.start()
    assert host.speed == 6400 and times(host) == 0 and note(temp).is_file()
