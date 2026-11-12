"""T-012: packaging helpers are sound even where we cannot build the Windows app."""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SPEC = ROOT / "packaging" / "retro_ai_lab.spec"
SMOKE = ROOT / "packaging" / "smoke_test.py"
WORKFLOW = ROOT / ".github" / "workflows" / "package.yml"


def test_spec_file_is_valid_python_and_points_at_the_entry_point() -> None:
    tree = ast.parse(SPEC.read_text(), filename=str(SPEC))
    assert tree.body
    text = SPEC.read_text()
    assert "../src/retro_ai_lab/__main__.py" in text
    assert 'name="RetroAILab"' in text
    assert "console=False" in text


def test_spec_entry_point_exists() -> None:
    assert (ROOT / "src" / "retro_ai_lab" / "__main__.py").exists()


def test_workflow_builds_and_smoke_tests_on_windows() -> None:
    text = WORKFLOW.read_text()
    assert "runs-on: windows-latest" in text
    assert "pyinstaller packaging/retro_ai_lab.spec" in text
    assert "packaging/smoke_test.py dist/RetroAILab/RetroAILab.exe" in text
    assert "upload-artifact" in text


def test_smoke_test_usage_and_missing_path() -> None:
    usage = subprocess.run(
        [sys.executable, str(SMOKE)], capture_output=True, text=True, check=False
    )
    assert usage.returncode == 2
    assert "Usage" in usage.stdout
    missing = subprocess.run(
        [sys.executable, str(SMOKE), str(ROOT / "nope.exe")],
        capture_output=True,
        text=True,
        check=False,
    )
    assert missing.returncode == 2
    assert "not found" in missing.stdout


@pytest.mark.slow
def test_smoke_test_passes_against_the_unpackaged_entry_point(tmp_path: Path) -> None:
    """The smoke script's checks hold for the real entry point run via a launcher."""
    launcher = tmp_path / "retro-ai-lab-launcher.py"
    launcher.write_text(
        "import runpy, sys\n"
        "sys.argv = ['retro-ai-lab', *sys.argv[1:]]\n"
        "runpy.run_module('retro_ai_lab', run_name='__main__')\n"
    )
    if sys.platform == "win32":
        shim = tmp_path / "launcher.cmd"
        shim.write_text(f'@"{sys.executable}" "{launcher}" %*\n')
    else:
        shim = tmp_path / "launcher.sh"
        shim.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{launcher}" "$@"\n')
        shim.chmod(0o755)
    result = subprocess.run(
        [sys.executable, str(SMOKE), str(shim)],
        capture_output=True,
        text=True,
        check=False,
        timeout=300,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "smoke test passed" in result.stdout


def test_spec_and_package_data_ship_the_luasocket_loader() -> None:
    """T-816: EmuHawk has no socket library, so the loader and its folder travel with the script."""
    text = SPEC.read_text(encoding="utf-8")
    assert "emulator/bizhawk/socket.lua" in text
    assert "emulator/bizhawk/luasocket" in text
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert '"luasocket/x64/*"' in pyproject


def test_the_starting_picture_and_the_spec() -> None:
    """T-1002: the built app's "starting" window has a picture the build tool takes as it is."""
    import re

    from PySide6.QtGui import QImage

    picture = ROOT / "packaging" / "splash.png"
    assert picture.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    image = QImage(str(picture))
    assert not image.isNull()
    # No larger than the build tool shows without resizing (which would need Pillow).
    assert 0 < image.width() <= 760 and 0 < image.height() <= 480
    # Pure magenta is see-through on Windows.
    colours = {
        image.pixel(x, y) & 0xFFFFFF for x in range(image.width()) for y in range(image.height())
    }
    assert 0xFF00FF not in colours
    assert len(colours) > 8  # something is drawn on it

    text = SPEC.read_text(encoding="utf-8")
    found = re.search(r"TEXT_POS = \((\d+), (\d+)\)", text)
    assert found is not None
    x, y = int(found[1]), int(found[2])
    # The step is written with its lower left corner here: inside the picture, with room.
    assert 0 < x < image.width() // 4 and image.height() // 2 < y < image.height() - 8
    # The spec makes the window from the picture, gives it to the program and ships its files.
    assert 'Splash(\n        os.path.join(SPECPATH, "splash.png")' in text
    assert "*([splash] if splash is not None else [])" in text
    assert "*([splash.binaries] if splash is not None else [])" in text
    # The CI build is told to expect it.
    assert "RetroAILab.exe --starting-window" in WORKFLOW.read_text(encoding="utf-8")


def _smoke():  # type: ignore[no-untyped-def]
    import importlib.util

    spec = importlib.util.spec_from_file_location("retro_ai_lab_smoke_test", SMOKE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_smoke_test_can_expect_the_starting_window(tmp_path: Path) -> None:
    """T-1002: told to, the smoke test wants the starting window's two lines, each in its place."""
    smoke = _smoke()
    plain = "\n".join(smoke.STAGES)
    whole = "\n".join(smoke.stages(starting_window=True))
    assert smoke.stages() == smoke.STAGES
    assert len(smoke.stages(starting_window=True)) == len(smoke.STAGES) + 2
    log = tmp_path / "retro-ai-lab.log"

    def problem(text: str, *, starting_window: bool) -> str:
        log.write_text(text, encoding="utf-8")
        return str(smoke.check_log(tmp_path, starting_window=starting_window))

    assert problem(plain, starting_window=False) == ""
    assert problem(whole, starting_window=False) == ""
    assert problem(whole, starting_window=True) == ""
    # Built without the window: the log never says it was showing.
    assert "The starting window is showing" in problem(plain, starting_window=True)
    # Shown but never taken down.
    never_closed = whole.replace("The starting window is closed\n", "")
    assert "The starting window is closed" in problem(never_closed, starting_window=True)
    # Taken down before the main window was up: the line is there, but not in its place.
    early = never_closed.replace(
        "Building the main window", "The starting window is closed\nBuilding the main window"
    )
    assert "The starting window is closed" in problem(early, starting_window=True)
