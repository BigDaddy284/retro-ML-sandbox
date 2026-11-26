"""T-1068: save states are kept in a folder the app names, not the temp folder.

The bridge script wrote every save state to the temp folder. Since T-1064 one
of them, the spot a remembered way starts from, has to be there at the next
sitting, and the system empties that folder: the user's log of 25 Nov has "No
save state in slot 'way_…'" at the first Start in five days. The app now names
a folder of its own, and the script saves and loads there.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from pathlib import Path

import pytest

from retro_ai_lab.core import ConfigurationError, EmulatorError, ProtocolError
from retro_ai_lab.core.events import EventBus
from retro_ai_lab.emulator.bizhawk import PROTOCOL_VERSION, BizHawkAdapter, BizHawkSimulator
from retro_ai_lab.emulator.bizhawk.adapter import SCRIPT_PATH
from retro_ai_lab.emulator.fake import FakeEmulatorAdapter
from retro_ai_lab.emulator.mgba import MgbaAdapter
from retro_ai_lab.emulator.mgba.adapter import SCRIPT_PATH as MGBA_SCRIPT_PATH
from retro_ai_lab.emulator.registry import AdapterRegistry, default_registry
from retro_ai_lab.services.connection import ConnectionService
from tests.unit.emulator.bizhawk.lua_host import LuaScriptHost
from tests.unit.emulator.mgba.lua_host import MgbaScriptHost

CANNOT = "Couldn't make the folder for save states"


@pytest.fixture
def temp(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """The temp folder the script finds, empty."""
    folder = tmp_path / "temp"
    folder.mkdir()
    for name in ("TEMP", "TMP", "TMPDIR"):
        monkeypatch.setenv(name, str(folder))
    return folder


@pytest.fixture
def script() -> Iterator[LuaScriptHost]:
    """The real BizHawk script, running against stubs."""
    with LuaScriptHost() as host:
        yield host


def named(host: LuaScriptHost, folder: Path | None) -> BizHawkAdapter:
    """An adapter connected to the script, told where save states are kept."""
    adapter = BizHawkAdapter(port=host.port, call_timeout_s=5.0)
    if folder is not None:
        adapter.keep_states_in(folder)
    adapter.connect(timeout_s=5.0)
    return adapter


def kept(host: LuaScriptHost) -> list[Path]:
    """Where the emulator was told to write its save states."""
    return [Path(path.decode()) for path in host.states]


def state_file(folder: Path, slot: str) -> Path:
    return folder / f"retro_ai_lab_{slot}.State"


# ------------------------------------------------------------ the real script


def test_states_are_written_to_the_folder_named(
    script: LuaScriptHost, temp: Path, tmp_path: Path
) -> None:
    folder = tmp_path / "data" / "states"
    adapter = named(script, folder)
    assert folder.is_dir()
    adapter.advance_frames(3)
    adapter.save_state("lap_1")
    snapshot = script.game.snapshot()
    assert kept(script) == [state_file(folder, "lap_1")]  # and nothing in the temp folder
    adapter.advance_frames(10)
    assert script.game.snapshot() != snapshot
    adapter.load_state("lap_1")
    assert script.game.snapshot() == snapshot
    with pytest.raises(ConfigurationError, match=r"No save state in slot 'lap_2'"):
        adapter.load_state("lap_2")
    assert script.game.snapshot() == snapshot


def test_a_state_from_before_is_still_found_in_the_temp_folder(
    script: LuaScriptHost, temp: Path, tmp_path: Path
) -> None:
    """An older build saved the spot a way starts from where the script then put it."""
    folder = tmp_path / "states"
    adapter = named(script, folder)
    adapter.advance_frames(3)
    adapter._call("save", slot="way_1")  # as the script was asked before: no folder named
    before = script.game.snapshot()
    assert kept(script) == [state_file(temp, "way_1")]
    adapter.advance_frames(10)
    loads = script.calls.count("load")
    adapter.load_state("way_1")
    assert script.game.snapshot() == before
    assert script.calls.count("load") == loads + 2  # the folder first, then the temp folder
    # Saved again it is in the folder, and from then on that one is loaded.
    adapter.advance_frames(10)
    adapter.save_state("way_1")
    now = script.game.snapshot()
    assert now != before
    assert kept(script) == [state_file(temp, "way_1"), state_file(folder, "way_1")]
    adapter.advance_frames(10)
    adapter.load_state("way_1")
    assert script.game.snapshot() == now
    assert script.calls.count("load") == loads + 3


def test_with_no_folder_named_the_temp_folder_is_looked_in_once(
    script: LuaScriptHost, temp: Path
) -> None:
    adapter = named(script, None)
    adapter.advance_frames(3)
    adapter.save_state("lap_1")
    assert kept(script) == [state_file(temp, "lap_1")]
    loads = script.calls.count("load")
    with pytest.raises(ConfigurationError, match=r"No save state in slot 'lap_2'"):
        adapter.load_state("lap_2")
    assert script.calls.count("load") == loads + 1
    # A folder given as nothing is no folder named.
    adapter._call("save", slot="lap_3", dir=None)
    assert kept(script) == [state_file(temp, "lap_1"), state_file(temp, "lap_3")]
    assert adapter._call("load", slot="lap_3", dir=None)["loaded"] is True
    assert script.calls.count("load") == loads + 2


def test_what_is_no_folder_and_no_slot_is_refused(
    script: LuaScriptHost, temp: Path, tmp_path: Path
) -> None:
    folder = tmp_path / "states"
    adapter = named(script, folder)
    for bad in ("", 7, True, False, ["states"], {"dir": "states"}):
        for command in ("save", "load"):
            with pytest.raises(EmulatorError, match="'dir' must be the path of a folder"):
                adapter._call(command, slot="lap_1", dir=bad)
    for bad_slot in ("", "no/slashes", "..", 7, None):
        for command in ("save", "load"):
            with pytest.raises(EmulatorError, match="letters, digits and underscores"):
                adapter._call(command, slot=bad_slot, dir=str(folder))
    assert kept(script) == [] and "load" not in script.calls
    assert adapter.is_connected


# ----------------------------------------------------------------- the adapter


def asked(sim: BizHawkSimulator, command: str) -> list[dict[str, object]]:
    return [dict(r) for r in sim.requests if r.get("cmd") == command]


def test_the_folder_goes_with_every_save_and_load_and_nothing_else(tmp_path: Path) -> None:
    folder = tmp_path / "a" / "b" / "states"
    with BizHawkSimulator() as sim:
        adapter = BizHawkAdapter(port=sim.port)
        adapter.keep_states_in(folder)
        assert folder.is_dir()  # parents and all
        adapter.keep_states_in(folder)  # there already: no harm
        adapter.connect(timeout_s=5.0)
        adapter.advance_frames(2)
        adapter.save_state("a")
        adapter.load_state("a")
        assert [r.get("dir") for r in asked(sim, "save")] == [str(folder)]
        assert [r.get("dir") for r in asked(sim, "load")] == [str(folder)]
        assert all("dir" not in r for r in sim.requests if r.get("cmd") not in ("save", "load"))


def test_with_no_folder_named_none_is_sent() -> None:
    with BizHawkSimulator() as sim:
        adapter = BizHawkAdapter(port=sim.port)
        adapter.connect(timeout_s=5.0)
        adapter.save_state("a")
        adapter.load_state("a")
        assert asked(sim, "save") and asked(sim, "load")
        assert all("dir" not in r for r in sim.requests)


def test_a_folder_that_cannot_be_made_leaves_the_temp_folder(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    good = tmp_path / "states"
    in_the_way = tmp_path / "a_file"
    in_the_way.write_text("not a folder")
    bad = in_the_way / "states"
    with BizHawkSimulator() as sim:
        adapter = BizHawkAdapter(port=sim.port)
        adapter.keep_states_in(good)
        with caplog.at_level(logging.INFO):
            assert CANNOT not in caplog.text
            adapter.keep_states_in(bad)  # the one named before is not kept on with
        assert f"{CANNOT}, {bad} (" in caplog.text and "They stay in the temp folder" in caplog.text
        assert [r.levelno for r in caplog.records if CANNOT in r.getMessage()] == [logging.WARNING]
        adapter.connect(timeout_s=5.0)
        adapter.save_state("a")
        adapter.load_state("a")
        assert all("dir" not in r for r in sim.requests)


@pytest.mark.parametrize(
    ("host", "local"),
    [
        ("127.0.0.1", True),
        ("localhost", True),
        (" LocalHost ", True),
        ("::1", True),
        ("192.0.2.7", False),
        ("emulator.example", False),
    ],
)
def test_only_an_emulator_on_this_pc_is_given_the_folder(
    tmp_path: Path, host: str, local: bool
) -> None:
    """The folder is on this PC; an emulator somewhere else could not write to it."""
    folder = tmp_path / "states"
    adapter = BizHawkAdapter(host=host)
    adapter.keep_states_in(folder)
    assert adapter._state_fields() == ({"dir": str(folder)} if local else {})
    assert folder.exists() is local


def test_naming_a_folder_on_another_pc_forgets_the_one_named_before(tmp_path: Path) -> None:
    adapter = BizHawkAdapter()
    adapter.keep_states_in(tmp_path / "states")
    adapter._host = "192.0.2.7"
    adapter.keep_states_in(tmp_path / "other")
    assert adapter._state_fields() == {} and not (tmp_path / "other").exists()


def test_mgba_is_given_no_folder_and_keeps_its_states_as_before(tmp_path: Path) -> None:
    """Its script holds save states in memory: there are no files to keep."""
    folder = tmp_path / "states"
    with MgbaScriptHost() as host:
        adapter = MgbaAdapter(port=host.port, call_timeout_s=5.0)
        adapter.keep_states_in(folder)
        assert not folder.exists() and adapter._state_fields() == {}
        adapter.connect(timeout_s=5.0)
        adapter.advance_frames(3)
        adapter.save_state("lap_1")
        adapter.load_state("lap_1")
        assert host.calls.count("saveStateBuffer") == 1
        assert host.calls.count("loadStateBuffer") == 1
        assert not folder.exists()


def test_the_scripts_are_version_three_and_an_older_one_is_refused() -> None:
    """BizHawk keeps the old build's script open: without the new one the states
    would go on being written to the temp folder, and nothing would say so."""
    assert PROTOCOL_VERSION == 3
    for path in (SCRIPT_PATH, MGBA_SCRIPT_PATH):
        assert "local SCRIPT_VERSION = 3 " in path.read_text(encoding="utf-8")
    with BizHawkSimulator(script_version=2) as sim:
        adapter = BizHawkAdapter(port=sim.port)
        with pytest.raises(ProtocolError, match="an older bridge script") as refused:
            adapter.connect(timeout_s=1.0)
        assert "(version 2; this app needs 3)" in str(refused.value)
        assert "open the one that came with this app" in str(refused.value)


# ---------------------------------------------------------------- the services


class Noting(FakeEmulatorAdapter):
    """The practice console, noting what it is told about save states."""

    name = "noting"
    told: list[tuple[Path, bool]]

    def keep_states_in(self, folder: Path) -> None:
        self.told = [*getattr(self, "told", []), (folder, self.is_connected)]
        super().keep_states_in(folder)


def service(states_dir: Path | None) -> ConnectionService:
    registry = default_registry()
    registry.register(Noting, listed=False)
    bus = EventBus(on_error=lambda exc, event, handler: (_ for _ in ()).throw(exc))
    if states_dir is None:
        return ConnectionService(registry, bus)
    return ConnectionService(registry, bus, states_dir=states_dir)


def test_the_connection_service_names_the_folder_before_it_connects(tmp_path: Path) -> None:
    folder = tmp_path / "states"
    connection = service(folder)
    connection.connect("noting")
    adapter = connection.adapter
    assert isinstance(adapter, Noting) and adapter.told == [(folder, False)]
    # The practice console has no files to keep: it takes the call and does nothing.
    assert not folder.exists()
    connection.call(lambda a: a.save_state("a"))
    connection.call(lambda a: a.load_state("a"))
    connection.disconnect()
    connection.connect("noting")  # every adapter it creates
    again = connection.adapter
    assert isinstance(again, Noting) and again is not adapter and again.told == [(folder, False)]
    connection.disconnect()


def test_the_connection_service_with_no_folder_names_none() -> None:
    connection = service(None)
    connection.connect("noting")
    adapter = connection.adapter
    assert isinstance(adapter, Noting) and not hasattr(adapter, "told")
    connection.disconnect()


def test_the_connection_service_passes_the_folder_on_to_bizhawk(tmp_path: Path) -> None:
    folder = tmp_path / "states"
    with BizHawkSimulator() as sim:
        connection = service(folder)
        connection.connect("bizhawk", port=sim.port)
        connection.call(lambda a: a.save_state("a"))
        connection.call(lambda a: a.load_state("a"))
        assert [r.get("dir") for r in asked(sim, "save")] == [str(folder)]
        assert [r.get("dir") for r in asked(sim, "load")] == [str(folder)]
        assert folder.is_dir()
        connection.disconnect()


def test_an_adapter_of_any_kind_takes_the_call(tmp_path: Path) -> None:
    """The default does nothing, so an adapter written without it still connects."""
    registry = AdapterRegistry()
    registry.register(FakeEmulatorAdapter)
    adapter = registry.create("fake")
    assert adapter.keep_states_in(tmp_path / "states") is None  # type: ignore[func-returns-value]
    assert not (tmp_path / "states").exists()
