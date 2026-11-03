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
