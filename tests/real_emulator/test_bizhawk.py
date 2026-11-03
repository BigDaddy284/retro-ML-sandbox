"""T-117: the BizHawk bridge on a real EmuHawk (opt-in, ADR-007).

Collected only when both environment variables are set (see conftest.py and
docs/BIZHAWK.md):

* ``RETRO_AI_LAB_EMULATOR_PATH`` is the full path to ``EmuHawk.exe``.
* ``RETRO_AI_LAB_TEST_ROM`` is a ROM EmuHawk can open.

The test launches EmuHawk with the shipped script, connects the adapter and
exercises every operation. It prints what it measured so the numbers can be
copied into TASK_TRACKER.md. Nothing here runs in CI.
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
from retro_ai_lab.emulator.bizhawk import DEFAULT_PORT, SCRIPT_PATH, BizHawkAdapter

pytestmark = pytest.mark.real_emulator

PORT = int(os.environ.get("RETRO_AI_LAB_BIZHAWK_PORT", DEFAULT_PORT))
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
def emuhawk() -> Iterator[subprocess.Popen[bytes]]:
    exe = Path(os.environ["RETRO_AI_LAB_EMULATOR_PATH"])
    rom = Path(os.environ["RETRO_AI_LAB_TEST_ROM"])
    assert exe.is_file(), f"EmuHawk not found at {exe}"
    assert rom.is_file(), f"ROM not found at {rom}"
    assert not _port_open(PORT), f"Something already listens on port {PORT}; close EmuHawk first."
    env = {**os.environ, "RETRO_AI_LAB_PORT": str(PORT)}
    process = subprocess.Popen(
        [str(exe), f"--lua={SCRIPT_PATH}", str(rom)], cwd=exe.parent, env=env
    )
    deadline = time.monotonic() + LAUNCH_TIMEOUT_S
    while not _port_open(PORT):
        if process.poll() is not None:
            pytest.fail(f"EmuHawk exited with code {process.returncode} before the script was up.")
        if time.monotonic() > deadline:
            process.kill()
            pytest.fail(f"EmuHawk did not open port {PORT} within {LAUNCH_TIMEOUT_S:g} seconds.")
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
def adapter(emuhawk: subprocess.Popen[bytes]) -> Iterator[BizHawkAdapter]:
    a = BizHawkAdapter(port=PORT, call_timeout_s=10.0)
    a.connect(timeout_s=10.0)
    try:
        yield a
    finally:
        a.disconnect()


def test_handshake_describes_the_game(adapter: BizHawkAdapter) -> None:
    info = adapter.game_info()
    print(
        f"\nEmuHawk {adapter.description.bizhawk}: {info.system} / {info.title} / {info.rom_hash}"
    )
    print("buttons:", [b.label for b in adapter.buttons()])
    print("regions:", [(r.name, r.size) for r in adapter.description.regions])
    print("offered:", [(r.name, r.size) for r in adapter.memory_regions()])
    print("capabilities:", adapter.capabilities())
    assert info.system and info.title
    assert adapter.buttons()
    assert adapter.memory_regions()
    assert adapter.capabilities().supports("frame_advance")


def test_hold_mode_frame_advance_moves_exactly(adapter: BizHawkAdapter) -> None:
    """ADR-005's open question: does emu.frameadvance() move a paused EmuHawk?"""
    before = adapter._call("ping")["frame"]
    time.sleep(0.5)
    assert adapter._call("ping")["frame"] == before, "the game ran while it should be held"
    adapter.advance_frames(5)
    assert adapter._call("ping")["frame"] == before + 5
    time.sleep(0.5)
    assert adapter._call("ping")["frame"] == before + 5
    started = time.perf_counter()
    adapter.advance_frames(60)
    elapsed = time.perf_counter() - started
    print(f"\n60 frames stepped in {elapsed:.3f}s ({60 / elapsed:.0f} frames/s)")
    assert adapter._call("ping")["frame"] == before + 65


def test_memory_reads_and_screenshots(adapter: BizHawkAdapter) -> None:
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


def test_input_speed_and_save_states(adapter: BizHawkAdapter) -> None:
    start = next(b for b in adapter.buttons() if b.name in ("start", "p1_start"))
    adapter.set_input({Button(start.name, start.label)})
    adapter.advance_frames(10)
    adapter.set_input(())
    adapter.set_speed(2.0)
    adapter.set_speed(1.0)
    adapter.save_state("t117")
    snapshot = adapter.read_memory(adapter.memory_regions()[0].name, 0, 64)
    adapter.advance_frames(120)
    adapter.load_state("t117")
    assert adapter.read_memory(adapter.memory_regions()[0].name, 0, 64) == snapshot
    with pytest.raises(ConfigurationError, match="No save state"):
        adapter.load_state("t117_missing")


def test_disconnect_frees_the_emulator(adapter: BizHawkAdapter) -> None:
    before = adapter._call("ping")["frame"]
    adapter.disconnect()
    time.sleep(1.0)
    adapter.connect(timeout_s=10.0)
    assert adapter._call("ping")["frame"] > before, "the game stayed paused after a disconnect"
