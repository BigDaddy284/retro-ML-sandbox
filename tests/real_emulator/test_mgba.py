"""T-811: the mGBA bridge on a real mGBA (opt-in, ADR-007).

Collected only when both environment variables are set (see docs/MGBA.md):

* ``RETRO_AI_LAB_MGBA_PATH`` is the full path to the mGBA executable
  (``mGBA.exe`` on Windows, ``mgba-qt`` elsewhere).
* ``RETRO_AI_LAB_TEST_ROM`` is a Game Boy or Game Boy Advance ROM.

The test launches mGBA with ``--script`` pointing at the shipped script. If
your mGBA has no ``--script`` option, start mGBA yourself, load the script
from Tools > Scripting, and set ``RETRO_AI_LAB_MGBA_RUNNING=1`` so the test
connects to it instead of launching. It prints what it measured so the numbers
and the answers to ARCHITECTURE.md §6.4's questions can be copied into
TASK_TRACKER.md. Nothing here runs in CI.
"""

from __future__ import annotations

import os
import socket
import subprocess
import time
from collections.abc import Iterator
from pathlib import Path

import pytest

from retro_ai_lab.core import Button, ConfigurationError
from retro_ai_lab.emulator.bizhawk import DEFAULT_PORT
from retro_ai_lab.emulator.mgba import SCRIPT_PATH, MgbaAdapter

MGBA_ENV = ("RETRO_AI_LAB_MGBA_PATH", "RETRO_AI_LAB_TEST_ROM")
pytestmark = pytest.mark.real_emulator(env=MGBA_ENV)

PORT = int(os.environ.get("RETRO_AI_LAB_MGBA_PORT", DEFAULT_PORT))
LAUNCH_TIMEOUT_S = 90.0


def _port_open(port: int) -> bool:
    with socket.socket() as probe:
        probe.settimeout(0.2)
        try:
            probe.connect(("127.0.0.1", port))
        except OSError:
            return False
        return True


@pytest.fixture(scope="module")
def mgba() -> Iterator[subprocess.Popen[bytes] | None]:
    if os.environ.get("RETRO_AI_LAB_MGBA_RUNNING"):
        assert _port_open(PORT), f"Nothing listens on port {PORT}: load the script in mGBA first."
        yield None
        return
    exe = Path(os.environ["RETRO_AI_LAB_MGBA_PATH"])
    rom = Path(os.environ["RETRO_AI_LAB_TEST_ROM"])
    assert exe.is_file(), f"mGBA not found at {exe}"
    assert rom.is_file(), f"ROM not found at {rom}"
    assert not _port_open(PORT), f"Something already listens on port {PORT}; close mGBA first."
    env = {**os.environ, "RETRO_AI_LAB_PORT": str(PORT)}
    process = subprocess.Popen([str(exe), "--script", str(SCRIPT_PATH), str(rom)], env=env)
    deadline = time.monotonic() + LAUNCH_TIMEOUT_S
    while not _port_open(PORT):
        if process.poll() is not None:
            pytest.fail(f"mGBA exited with code {process.returncode} before the script was up.")
        if time.monotonic() > deadline:
            process.kill()
            pytest.fail(f"mGBA did not open port {PORT} within {LAUNCH_TIMEOUT_S:g} seconds.")
        time.sleep(0.5)
    try:
        yield process
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(10)
            except subprocess.TimeoutExpired:
                process.kill()


@pytest.fixture
def adapter(mgba: subprocess.Popen[bytes] | None) -> Iterator[MgbaAdapter]:
    a = MgbaAdapter(port=PORT, call_timeout_s=10.0)
    a.connect(timeout_s=10.0)
    try:
        yield a
    finally:
        a.disconnect()


def test_handshake_describes_the_game(adapter: MgbaAdapter) -> None:
    info = adapter.game_info()
    print(f"\n{adapter.description.bizhawk}: {info.system} / {info.title} / {info.rom_hash}")
    print("hello:", {k: v for k, v in adapter.hello.items() if k not in ("domains", "buttons")})
    print("buttons:", [b.label for b in adapter.buttons()])
    print("regions:", [(r.name, r.size) for r in adapter.description.regions])
    print("offered:", [(r.name, r.size) for r in adapter.memory_regions()])
    print("capabilities:", adapter.capabilities())
    assert info.system and info.title
    assert not adapter.hold
    assert adapter.buttons()
    assert adapter.memory_regions()
    assert adapter.capabilities().supports("frame_advance")


def test_free_run_step_counts_frames(adapter: MgbaAdapter) -> None:
    """§6.4: the game runs on between requests; a step waits for its frames."""
    before = adapter._call("ping")["frame"]
    time.sleep(0.5)
    after = adapter._call("ping")["frame"]
    print(f"\n{after - before} frames passed in 0.5 s while idle (free run)")
    assert after > before, "the game stood still: mGBA should run between requests"
    started = time.perf_counter()
    answered = adapter._call("step", frames=60)["frame"]
    elapsed = time.perf_counter() - started
    print(f"a 60-frame step answered after {elapsed:.3f}s at frame {answered}")
    assert answered >= after + 60


def test_memory_reads_and_screenshots(adapter: MgbaAdapter) -> None:
    """§6.4: readRange addressing (bus address = base + offset) and the screenshot file."""
    region = adapter.memory_regions()[0]
    started = time.perf_counter()
    data = adapter.read_memory(region.name, 0, region.size)
    elapsed = time.perf_counter() - started
    print(f"\n{region.name}: {region.size} bytes in {elapsed * 1000:.1f} ms")
    assert len(data) == region.size
    started = time.perf_counter()
    frame = adapter.capture_frame()
    elapsed = time.perf_counter() - started
    print(f"screenshot {frame.width}x{frame.height} in {elapsed * 1000:.1f} ms")
    assert frame.width > 0 and frame.height > 0 and frame.channels == 3


def test_input_and_save_states(adapter: MgbaAdapter) -> None:
    """§6.4: saveStateBuffer/loadStateBuffer without flags."""
    start = next(b for b in adapter.buttons() if b.name == "start")
    adapter.set_input({Button(start.name, start.label)})
    adapter.advance_frames(10)
    adapter.set_input(())
    adapter.save_state("t811")
    snapshot = adapter.read_memory(adapter.memory_regions()[0].name, 0, 64)
    adapter.advance_frames(120)
    adapter.load_state("t811")
    assert adapter.read_memory(adapter.memory_regions()[0].name, 0, 64) == snapshot
    with pytest.raises(ConfigurationError, match="No save state"):
        adapter.load_state("t811_missing")


def test_disconnect_and_reconnect(adapter: MgbaAdapter) -> None:
    """§6.4: what `receive` reports when the app closes the socket."""
    adapter.disconnect()
    time.sleep(1.0)
    adapter.connect(timeout_s=10.0)
    assert adapter.is_connected
