"""T-308: where the app keeps its files."""

from __future__ import annotations

from pathlib import Path

import pytest

from retro_ai_lab.core.paths import APP_FOLDER, ENV_VAR, default_data_dir


def test_env_override_wins(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv(ENV_VAR, str(tmp_path / "here"))
    assert default_data_dir() == tmp_path / "here"


def test_platform_default_is_under_the_user_profile(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(ENV_VAR, raising=False)
    path = default_data_dir()
    assert path.is_absolute()
    assert path.name.lower() == APP_FOLDER.lower()


@pytest.mark.parametrize(
    ("platform", "env", "expected_parts"),
    [
        (
            "win32",
            {"LOCALAPPDATA": "C:/Users/kid/AppData/Local"},
            ("AppData", "Local", "RetroAILab"),
        ),
        ("darwin", {}, ("Library", "Application Support", "RetroAILab")),
        ("linux", {"XDG_DATA_HOME": "/data"}, ("data", "retroailab")),
        ("linux", {}, (".local", "share", "retroailab")),
    ],
)
def test_each_platform(
    monkeypatch: pytest.MonkeyPatch,
    platform: str,
    env: dict[str, str],
    expected_parts: tuple[str, ...],
) -> None:
    monkeypatch.delenv(ENV_VAR, raising=False)
    for var in ("LOCALAPPDATA", "APPDATA", "XDG_DATA_HOME"):
        monkeypatch.delenv(var, raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr("retro_ai_lab.core.paths.sys.platform", platform)
    path = default_data_dir()
    assert path.parts[-len(expected_parts) :] == expected_parts
