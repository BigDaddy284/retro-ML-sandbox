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
| ADR-010 | The second real adapter targets mGBA (Lua scripting with sockets, 0.10+) rather than VBA-M; Game Boy family games are reachable today through BizHawk's cores; VBA-M is revisited when its Lua engine ships in a release; see §6 | Proposed (T-808 spike, docs only; a real-machine check is part of the adapter task) | The released VBA-M (v2.2.3) has no scripting or remote-control interface: its only external route is a GDB stub (SDL port, GBA only) that reads memory but can't press buttons, step frames or take save states. VBA-M's unreleased Lua engine has no socket library, no screenshot and no range reads. mGBA's scripting API has everything the bridge protocol needs, one call each. |

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

## 6. Second emulator: VBA-M feasibility (ADR-010, T-808)

DESIGN.md §7 names "VisualBoyAdvance / VBA-M" beside BizHawk as a practical emulator to target. The spike checked what each candidate offers against the adapter contract (§3) and the bridge protocol (§5), from the projects' own sources and documentation. Nothing here was run; the first task on a chosen emulator does that, as T-117 did for BizHawk.

**Sources checked:** VBA-M's `README.md`, `CMakeLists.txt`, the `src/wx/lua/` sources on `master`, the `v2.2.3` tag, the SDL port's man page (`src/sdl/vbam.6`, the `--gdb` option); mGBA's scripting reference (mgba.io/docs/scripting.html) and the `mGBA-http` project as a worked example of a socket server in an mGBA script; the TASVideos VBA Lua function list (the old VBA-rr fork).

### 6.1 What each emulator offers

| need (adapter hook) | VBA-M v2.2.3 (released) | VBA-M `master` Lua engine (unreleased) | mGBA 0.10+ scripting |
|---|---|---|---|
| a way in | GDB remote stub, SDL port only, GBA only (`-G tcp[:PORT]`, port 55555) | Lua Console in the wx/Qt ports; `luaL_openlibs` only, **no LuaSocket** | Tools → Scripting; Lua with a `socket` API (`socket.bind`, `listen`, `accept`, `receive`, `send`, `received` callback) |
| game info | none | `emu.romname()`, `emu.getsystem()` | `getGameTitle()`, `getGameCode()`, `checksum(C.CHECKSUM.CRC32)`, `platform()` |
| memory_read | GDB `m addr,len` (any bus address) | `memory.readbyte/word/dword` only: **one call per byte**, no range read; bus addresses | `readRange(address, length)`; `emu.memory` domains with `base`, `size`, `name` (GBA: wram, iwram, vram, oam, io…; GB: wram, hram, vram…) |
| input | none | `joypad.set(port, table)` | `setKeys(mask)`, `addKey`, `clearKeys`; `keysRead` callback |
| frame_advance | none (instruction step only) | `emu.frameadvance()` (a coroutine yield; the host resumes the script once per frame), `emu.pause()`/`unpause()`, `registerbefore/after` | `runFrame()` ("run until the next frame"), `frame` callback; pausing from a script is **not documented** → verify |
| screen_capture | none | **none** (`gui.text/box/line/pixel` draw only) | `screenshot(filename)` |
| speed_control | none | `turbo` is a frontend global, not exposed | not documented → verify |
| save_states | none | `savestate.create/save/load` | `saveStateSlot/loadStateSlot`, `saveStateBuffer/loadStateBuffer` |
| systems | GBA (GB has no stub) | GB, GBC, GBA | GB, GBC, GBA |

### 6.2 Findings

1. **A VBA-M adapter is not feasible on the released VBA-M.** Every release up to v2.2.3 has no scripting and no remote-control interface. The GDB stub reads memory, but an adapter needs to press buttons, step frames and take save states, none of which the GDB protocol can do; it would give `memory_read` alone, which the training loop can't use.
2. **VBA-M's Lua engine on `master` gets most of the way, with three gaps.** It appeared after v2.2.3 and is not in any release or changelog yet. It can read memory, hold buttons and step frames (its `emu.frameadvance()` is exactly the hold-mode model of §5.3) and has save states. It cannot take a screenshot, reads memory one byte per call (a 256-byte region is 256 Lua calls per step, fine for the practice-sized reads but slow for scans), and has **no socket library**, so the bridge would need another transport: Lua's standard `io` library is open, so a file- or named-pipe-based exchange polled once per frame would work on Windows but at frame-rate latency per request and with the bundled `step` command doing all the work. Feasible, with `screen_capture` off, once a release ships it.
3. **mGBA matches the bridge protocol one call to one.** Its scripting API (0.10 and later) has sockets, a game title and checksum for the game id, range reads with named domains and sizes (the `hello` domain list), key setting, `runFrame()`, screenshots, and save states to slots or buffers. The `mGBA-http` project runs a socket server inside an mGBA script with these calls, so the route is known to work. The adapter would reuse the §5 protocol unchanged and the `BizHawkSimulator`/`lupa` test approach; the Lua script is a second file with the same commands. To verify on a real machine: whether `runFrame()` steps a paused emulator (hold mode) or whether the script must drive from the `frame` callback (free-run mode), and whether speed can be set from a script.
4. **Game Boy family games are already covered.** BizHawk runs GB, GBC and GBA through its Gambatte and mGBA cores, so the child can play these games with the existing adapter today. A second adapter adds a second emulator, not new games.

### 6.4 The mGBA bridge script (T-810)

`retro_ai_lab/emulator/mgba/retro_ai_lab.lua` speaks the §5 protocol unchanged, so the adapter, the simulator and the tests carry over. The differences come from mGBA's API, not the protocol:

| | BizHawk script | mGBA script |
|---|---|---|
| loop | the script owns the loop (`emu.frameadvance()` / `emu.yield()`) | event-driven: `callbacks:add("frame")`, `callbacks:add("keysRead")`, socket `received` events polled once a frame |
| run modes | hold (paused between requests) or free run | free run only: mGBA has no pause from a script, so `mode` is accepted and does nothing |
| `step` | runs the frames inside the handler and answers | arms a counter; the answer is sent from the `frame` callback once the frames have passed; another request meanwhile is refused ("A step is still running") |
| input | `joypad.set` of a table every frame | a key mask (`C.GBA_KEY`/`C.GB_KEY` bits) set with `emu:setKeys` on every `keysRead` |
| memory | named domains, `read_bytes_as_array` | `emu.memory` domains; `readRange(base + addr, len)` |
| game id | `gameinfo.getromname()`, `getromhash()` | `getGameTitle()` (or the game code), CRC32 from `checksum()` as 8 hex digits; `hello` also carries `code` and `emulator = "mGBA"` |
| save states | files in the temp folder | `saveStateBuffer()` / `loadStateBuffer()` kept in a table by slot name (lost when the script stops) |
| speed | `client.speedmode` | no `speed` command, so `speed_control` is off |

Still to verify on a real mGBA (the first T-811 run): that `readRange` takes bus addresses (`base + addr`) rather than offsets; the error value `receive` returns when the app closes the socket; that `saveStateBuffer` without flags is accepted; whether `platform()` names match `C.PLATFORM` keys; screenshot file format and cost; how many frames a round trip costs at full speed (the game runs on between requests, so a training step's reads describe the frame the answer was built on, up to a frame later than the last held input).

### 6.3 Decision (ADR-010, Proposed)

The second real adapter is mGBA, built as `emulator/mgba/` with a shipped `retro_ai_lab.lua` that speaks the §5 protocol, after the BizHawk adapter has been verified on the user's PC (T-117's real run). VBA-M stays on the list: the objective in DESIGN.md §7 is kept, and the adapter is revisited when a VBA-M release ships the Lua engine, with the limitation (no screen capture, a slower transport) recorded then. Overturn by a VBA-M release that adds sockets or a screenshot call, or by the user preferring VBA-M for a reason the spike didn't see.
