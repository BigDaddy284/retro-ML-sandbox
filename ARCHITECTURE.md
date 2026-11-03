# Architecture

Living description of how Retro AI Lab is structured. Updated whenever a task changes module boundaries or interfaces.

## 1. Layers

```
┌──────────────────────────────────────────────────────────────┐
│ ui/            PySide6 views + view-models (no business logic) │
├──────────────────────────────────────────────────────────────┤
│ services/      Session orchestration: connect, observe, train │
├──────────┬──────────────┬─────────────┬──────────────────────┤
│ training/│ learning/    │ rewards/    │ understanding/       │
│ modes,   │ algorithms   │ rewards,    │ variable discovery,  │
│ runners, │ (Q-learning, │ objectives, │ events, relations,   │
│ history  │ DQN, ...)    │ personalities│ confidence          │
├──────────┴──────────────┴─────────────┴──────────────────────┤
│ observation/   frames, memory snapshots, diffs, input logs   │
├──────────────────────────────────────────────────────────────┤
│ emulator/      Adapter interface + adapters (BizHawk, VBA-M,  │
│                Fake)                                          │
├──────────────────────────────────────────────────────────────┤
│ storage/       SQLite + JSON: projects, game knowledge,       │
│                checkpoints, training history                  │
└──────────────────────────────────────────────────────────────┘
core/  shared domain types, events, errors, config (used by every layer)
```

Dependencies point downward only. `ui` depends on `services`; nothing below `services` imports Qt.

## 2. Package layout (planned)

```
src/retro_ai_lab/
  core/            types, errors, event bus, settings
  emulator/
    base.py        EmulatorAdapter protocol, Capabilities, Button, Frame, MemoryRegion
    fake/          synthetic game + FakeEmulatorAdapter
    bizhawk/       adapter + Lua bridge script
    vbam/          adapter (later)
  observation/
  understanding/
  rewards/
  learning/
  training/
  storage/
  services/
  ui/
  testing/         reusable fakes for tests and demo mode
tests/
  unit/ integration/ emulator/ real_emulator/ ui/ learning/
```

## 3. Emulator adapter contract

Every adapter subclasses `retro_ai_lab.emulator.EmulatorAdapter` (T-003). The public methods are `final` and perform the checks every backend must share; subclasses implement `_do_*` hooks only.

```python
class EmulatorAdapter(ABC):
    name: ClassVar[str]           # registry id, e.g. "bizhawk"
    display_name: ClassVar[str]   # child-facing name

    # lifecycle (idempotent)
    def connect(self, timeout_s: float = 5.0) -> None
    def disconnect(self) -> None
    is_connected: bool
    def mark_disconnected(self) -> None   # backend reports a dropped link

    # description (require connection)
    def capabilities(self) -> Capabilities
    def game_info(self) -> GameInfo
    def buttons(self) -> Sequence[Button]
    def memory_regions(self) -> Sequence[MemoryRegion]        # memory_read

    # operations (require connection + capability; arguments validated)
    def read_memory(self, region: str, offset: int, length: int) -> bytes   # memory_read
    def capture_frame(self) -> Frame                                        # screen_capture
    def set_input(self, pressed: Iterable[Button]) -> None                  # input
    def advance_frames(self, n: int = 1) -> None                            # frame_advance
    def set_speed(self, multiplier: float) -> None                          # speed_control
    def save_state(self, slot: str) -> None / load_state(...)               # save_states
```

Rules enforced by the base class:
- Any operation while disconnected raises `EmulatorDisconnectedError`.
- Any operation whose capability flag is off raises `UnsupportedCapabilityError(capability)` before the backend is touched.
- Bad arguments (unknown region or button, out-of-range read, non-positive speed or frame count, unsafe slot name) raise `ConfigurationError`.
- A backend that returns the wrong number of bytes from a read is reported as an error rather than passed through.
- Hooks for optional capabilities have defaults that raise `UnsupportedCapabilityError`, so an adapter only implements what it supports.

## 4. Decision log

Status: **Accepted** (in force), **Proposed** (needs user sign-off before work depends on it irreversibly).

| ID | Decision | Status | Reason |
|---|---|---|---|
| ADR-001 | Python 3.11+, PySide6, PyTorch, Gymnasium, SQLite/JSON | Accepted | Specified in DESIGN.md §14. |
| ADR-002 | `src/` layout, single package `retro_ai_lab`, layered as above | Accepted | Keeps components replaceable (DESIGN.md §14). |
| ADR-003 | Develop and CI on Linux + Windows; Windows is the release target | Accepted | Development runs in Linux containers; Windows CI guards the target platform. |
| ADR-004 | A deterministic Fake Emulator with a synthetic game is a first-class adapter | Accepted | No emulator or ROMs are available in CI; needed to test every layer and the first demo honestly. |
| ADR-005 | BizHawk / EmuHawk is the first real adapter, via a shipped Lua script that listens on a local TCP socket (LuaSocket) and speaks newline-delimited JSON; see §5 | Accepted (route and protocol fixed by T-110; behaviour on a real EmuHawk is verified in T-117 on the user's Windows PC) | The route is the one the Archipelago multiworld connector uses on current EmuHawk releases, so it is known to work; it needs no launch flags, and the whole bridge is one Lua file a child opens in the Lua Console. The built-in `comm.socketServer*` route needs `--socket_ip/--socket_port` at launch and changed its framing in 2.6.2. |
| ADR-006 | First real demo game: a simple racing or arcade title the user owns | Proposed | DESIGN.md §17 recommends a racing game first. The user chooses the specific game since no ROMs are supplied. |
| ADR-007 | Real-emulator tests are opt-in (env vars) and backed by manual verification on the user's Windows PC | Accepted | Only honest way to cover emulator behaviour that CI can't run. |
| ADR-008 | NumPy is a runtime dependency from Phase 2 (screen diffs) onwards | Accepted | Pixel work in pure Python is too slow at real console resolutions; PyTorch (Phase 5) depends on NumPy anyway, so this adds nothing new to the final package. |
| ADR-009 | Phase 7 neural learners (DQN) are implemented on NumPy behind a `Learner` interface; PyTorch is a later, optional backend | Accepted | The states here are a few dozen binned numbers, which a two-layer network in NumPy trains in real time. PyTorch adds ~200 MB to the Windows build (and gigabytes with CUDA on Linux), which is a lot for a children's app and for CI; the `Learner` contract (T-701) makes a PyTorch learner a drop-in later without touching the training service, the brain store or the panels. Overturn by adding `torch` as an optional extra and registering a learner. |

## 5. BizHawk bridge protocol (ADR-005, T-110)

The app never links to BizHawk. A Lua script shipped with the app, `retro_ai_lab/emulator/bizhawk/retro_ai_lab.lua`, runs inside EmuHawk (Lua Console → Open script, or `EmuHawk.exe --lua=<path>`), listens on `127.0.0.1:43110` with LuaSocket (bundled with EmuHawk, `require("socket")`), and the `bizhawk` adapter connects to it. The script is the server so that nothing has to be running before the child opens the emulator, and so a dropped app can reconnect.

**Sources checked (T-110):** the BizHawk Lua function reference (tasvideos.org/BizHawk/LuaFunctions) for every API named below, and Archipelago's `connector_bizhawk_generic.lua` as a worked example of LuaSocket + newline-delimited JSON on current EmuHawk. Nothing here was run on a real EmuHawk yet; T-117 does that on the user's Windows PC, and the "verify" column below says what it has to confirm.

### 5.1 Framing

One JSON object per line (`\n`), UTF-8, no newlines inside a message. The app sends requests, the script answers every request in order, one response per request:

```
→ {"id": 7, "cmd": "read", "reads": [{"domain": "RAM", "addr": 0, "len": 256}]}
← {"id": 7, "ok": true, "frame": 1234, "reads": ["AAEC…"]}
← {"id": 8, "ok": false, "error": "Unknown domain 'VRAM'."}
```

Binary data (memory bytes, screenshots) is base64. `frame` is `emu.framecount()` when the response was built. An unknown `cmd` or malformed line gets an `ok: false` response and the connection stays open; a closed socket ends the session and the script goes back to waiting for a connection.

### 5.2 Commands

| cmd | request fields | response fields | Lua API behind it | verify on real EmuHawk |
|---|---|---|---|---|
| `hello` | `script` (protocol version the app expects) | `script` (version), `bizhawk` (`client.getversion()`), `system` (`emu.getsystemid()`), `rom` (`gameinfo.getromname()`), `hash` (`gameinfo.getromhash()`), `domains` (`[{name,size}]` from `memory.getmemorydomainlist()` + `getmemorydomainsize`), `buttons` (keys of `joypad.get(1)`) | as listed | domain list format; button names per core |
| `read` | `reads: [{domain, addr, len}]` | `reads: [base64…]` | `memory.read_bytes_as_array(addr, len, domain)` | 1-indexed table order |
| `input` | `buttons: {name: bool}` | – | held set; `joypad.set(table, 1)` applied before every frame until changed | `joypad.set` must be repeated each frame |
| `step` | `frames` (≥1), optional `buttons`, `reads`, `screenshot` | `reads`, `image` (base64 PNG or null) | sets input, `emu.frameadvance()` × frames, then reads and screenshot | frame advance while paused (see 5.3) |
| `screenshot` | – | `image` (base64 of the file `client.screenshot(path)` wrote) | `client.screenshot(path)` into the temp folder; file read back and deleted; the adapter decodes it with QtGui's `QImage` into an RGB `Frame` | format written (PNG expected), cost |
| `speed` | `percent` | – | `client.speedmode(percent)` | accepted range |
| `save` / `load` | `slot` (safe name) | `load` returns `loaded: bool` | `savestate.save(path)` / `savestate.load(path)` under the script's temp dir | path handling |
| `mode` | `free_run: bool` | – | see 5.3 | which mode works |
| `ping` | – | – | – | – |
| `bye` | – | – | closes the connection, unpauses | – |

Names the emulator uses (domains such as `System Bus`, buttons such as `P1 A`) are kept on the Lua side; the adapter maps them to the app's lowercase identifiers (`system_bus`, `a`) and back, so the app's types stay unchanged.

### 5.3 Timing and the two run modes

The script's main loop serves whatever requests are waiting, then gives control back to the emulator, so each round trip costs at least one turn of the emulator loop (about 16 ms at 100% speed). That is why `step` bundles input, frames, reads and a screenshot into one request: a training step is one round trip.

* **Hold mode (default):** on `hello` the script pauses EmuHawk (`client.pause()`) and keeps itself responsive with `emu.yield()`. The game advances only when a `step` calls `emu.frameadvance()`. This matches the fake adapter exactly: nothing happens between the app's steps. It relies on Lua's `frameadvance` moving a paused emulator one frame; the Lua reference says it "signals to the emulator to resume emulation", and T-117 confirms it.
* **Free-run mode (fallback, `mode free_run=true`):** the emulator is not paused, the loop calls `emu.frameadvance()` on every turn, and `step` waits for `frames` more frames to pass with the held input. The game then runs at the emulator's speed between steps, and the adapter's `advance_frames(n)` means "let n more frames go by".

The adapter reports `frame_advance` only in hold mode.

### 5.4 Throughput expectations

Memory reads are small (the app reads a few regions of up to a few KB per step; base64 adds a third). A screenshot is a PNG written to disk, read back and base64'd in Lua: 10–100 KB and a few milliseconds per frame on a console-sized image, which is fine at the app's observation rate (the services capture frames at a few per second, not every step). If T-117 finds screenshots too slow, `comm.mmfScreenshot()` (a memory-mapped file the app can `mmap` by name on Windows) is the drop-in optimisation.

### 5.5 Testing without an emulator

The Lua script runs in CI inside `lupa` (Lua 5.4 in-process, matching EmuHawk's NLua), with the `emu`, `memory`, `joypad`, `client`, `savestate`, `gameinfo` and `socket` globals stubbed by a small Python harness that plays the Toy Racer. A pure-Python `BizHawkSimulator` speaks the same protocol over a real TCP socket for adapter and service tests. Tests under `tests/real_emulator/` are opt-in (`RETRO_AI_LAB_EMULATOR_PATH` and `RETRO_AI_LAB_TEST_ROM`, ADR-007) and run only on a machine with EmuHawk and a ROM; `docs/BIZHAWK.md` tells the user how. Between requests the app sends nothing, so the adapter's `check_link()` probes the socket (a readable socket with no bytes is a closed peer) for the UI's link watchdog.
