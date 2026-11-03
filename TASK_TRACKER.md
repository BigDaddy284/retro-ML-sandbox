# Task Tracker

Single source of progress. Follows the template in `DEVELOPMENT_RULES.md` §3.
Phases refer to DESIGN.md §16. Phases 1–2 are broken into small tasks now; later phases are listed as epics and will be split into tasks before work on them starts.

## Status

- **Current task:** T-203 (Ready)
- **Last approved checkpoint:** `main` after T-202
- **Open blockers:** none. Two decisions are *Proposed* in `ARCHITECTURE.md` (ADR-005 first emulator protocol, ADR-006 first demo game); they block only T-110 onward and T-200 onward respectively.

## Summary

| ID | Title | Phase | Status | Tests (pass/total) | Commit |
|---|---|---|---|---|---|
| T-000 | Project documentation baseline | 0 | Done | n/a (docs only) | 29249db (#1) |
| T-001 | Python project scaffold, tooling and CI | 1 | Done | 13/13 | see PR #2 |
| T-002 | Core domain types and errors | 1 | Done | 73/73 | see PR #3 |
| T-003 | Emulator adapter interface and capabilities model | 1 | Done | 72/72 | see PR #4 |
| T-004 | Synthetic game (deterministic toy racer) | 1 | Done | 39/39 | see PR #5 |
| T-005 | Fake emulator adapter over the synthetic game | 1 | Done | 52/52 | see PR #6 |
| T-006 | Adapter registry and connection service | 1 | Done | 39/39 | see PR #7 |
| T-007 | App shell: main window launches with empty panels | 1 | Done | 15/15 | see PR #8 |
| T-008 | Game window panel shows live frames | 1 | Done | 20/20 | see PR #9 |
| T-009 | Connect / disconnect UI with capability display | 1 | Done | 17/17 | see PR #10 |
| T-010 | Manual controller input from UI to adapter | 1 | Done | 22/22 | see PR #11 |
| T-011 | Graceful handling of emulator disconnection | 1 | Done | 16/16 | see PR #12 |
| T-012 | Windows packaging smoke test (launchable build) | 1 | Done | 5/5 + CI build | see PR #13 |
| T-110 | Spike: confirm BizHawk Lua ↔ socket protocol | 1 | Blocked on ADR-005 | – | – |
| T-111 | BizHawk adapter: connect and handshake | 1 | Ready after T-110 | – | – |
| T-112 | BizHawk adapter: capabilities and game info | 1 | – | – | – |
| T-113 | BizHawk adapter: frame capture | 1 | – | – | – |
| T-114 | BizHawk adapter: controller input | 1 | – | – | – |
| T-115 | BizHawk adapter: memory reads | 1 | – | – | – |
| T-116 | BizHawk adapter: frame advance, speed, save states | 1 | – | – | – |
| T-117 | BizHawk adapter: disconnect handling and real-emulator verification | 1 | – | – | – |
| T-201 | Memory snapshot capture and storage | 2 | Done | 46/46 | see PR #14 |
| T-202 | Memory change detection (frame-to-frame diff) | 2 | Done | 26/26 | see PR #15 |
| T-203 | Screen observation: frame diff and change regions | 2 | – | – | – |
| T-204 | Input recording (who pressed what, when) | 2 | – | – | – |
| T-205 | Save-state experiment harness (repeatable trials) | 2 | – | – | – |
| E-3 | Game Understanding (variables, events, relationships, confidence) | 3 | Epic | – | – |
| E-4 | Reward System (controls, categories, personalities, goals, objectives) | 4 | Epic | – | – |
| E-5 | Basic Learning (tabular Q-learning, save/load) | 5 | Epic | – | – |
| E-6 | Training Interface (live/background, modes, graphs, history) | 6 | Epic | – | – |
| E-7 | Advanced Learning (DQN, hierarchical RL, demonstrations) | 7 | Epic | – | – |
| E-8 | Expansion (VBA-M, comparison, sharing, competitions) | 8 | Epic | – | – |

The **first working demonstration** (DESIGN.md §17) is reached at the end of E-5 plus the minimum of E-6, first on the Fake Emulator and then on a real game via BizHawk.

---

## Task details

### T-000 – Project documentation baseline
- **Status:** Done
- **Phase:** 0
- **Objective:** Create the permanent development documents required by DESIGN.md §18.10 and CLAUDE.md §2.
- **Scope:** README, DEVELOPMENT_RULES, TASK_TRACKER, TESTING_STRATEGY, ARCHITECTURE, CHANGELOG, `.gitignore`. **Out of scope:** any code.
- **Acceptance criteria:**
  1. All five governing documents exist and are consistent with DESIGN.md and CLAUDE.md.
  2. Phase 1 and 2 work is split into small tasks with clear titles.
  3. Architecture decisions needing user sign-off are marked *Proposed*.
- **Test plan:** Manual review against DESIGN.md §16–18 and CLAUDE.md. No automated tests (no code).
- **Results:** Manual review complete. Excluded automated tests: not applicable, documentation only.
- **Completion report:** Documents written. Note: the committed `CLAUDE.md` appears truncated (ends mid-sentence in §5); flagged to the user.

### T-001 – Python project scaffold, tooling and CI
- **Status:** Done
- **Phase:** 1
- **Objective:** A pip-installable `retro_ai_lab` package with test, lint and type tooling running in CI on Windows and Linux.
- **Scope:** `pyproject.toml`, `src/retro_ai_lab/__init__.py` with version, `tests/` skeleton, pytest/ruff/mypy config, GitHub Actions workflow. **Out of scope:** any feature code, Qt.
- **Acceptance criteria:**
  1. `pip install -e .[dev]` succeeds on Python 3.11 and 3.12.
  2. `pytest`, `ruff check .` and `mypy src` all pass locally.
  3. CI runs those on `windows-latest` and `ubuntu-latest` and is green.
  4. `retro_ai_lab.__version__` is a valid PEP 440 string.
- **Expected changes:** `pyproject.toml`, `src/retro_ai_lab/`, `tests/`, `.github/workflows/ci.yml`.
- **Test plan:** version string valid PEP 440 and equal to pyproject; package importable; `__all__` controlled; `--version` prints the version and exits 0; `--help` and no-args exit 0; unknown flag, bad flag value and stray positional exit non-zero with usage on stderr; `python -m retro_ai_lab` works as a subprocess for both success and error paths; CI matrix (Windows + Linux × 3.11 + 3.12) green.
- **Results:** 13 / 13 passed / 0 failed / 0 excluded. `ruff check`, `ruff format --check`, `mypy --strict` clean.
- **Completion report:** `pyproject.toml` with `src/` layout and dev extras; `retro_ai_lab.cli` with `main(argv) -> int`; `conftest.py` auto-skips `real_emulator` tests unless configured; GitHub Actions matrix. Ruff format excludes Markdown so docs code blocks stay as written. No runtime dependencies yet; PySide6 is added in T-007.

### T-002 – Core domain types and errors
- **Status:** Done
- **Phase:** 1
- **Objective:** Shared immutable types (`Button`, `Frame`, `MemoryRegion`, `GameInfo`, `Capabilities`) and the error hierarchy (`RetroAILabError`, `EmulatorError`, `EmulatorDisconnected`, `UnsupportedCapability`, `ProtocolError`).
- **Scope:** `retro_ai_lab.core.types` and `retro_ai_lab.core.errors`. **Out of scope:** any behaviour that uses them.
- **Acceptance criteria:**
  1. Each type rejects invalid values with `ConfigurationError` and a readable message.
  2. Types are frozen, hashable and compare by value.
  3. `to_dict`/`from_dict` round-trip through JSON; unknown or missing keys are rejected.
  4. Errors form the documented hierarchy; `UnsupportedCapabilityError` carries the capability name.
- **Test plan:** construction and validation (negative sizes, empty names, bad identifiers, bools-as-ints, frame buffer length vs width×height×channels, bad channel counts), immutability, equality/hash, pixel access incl. out-of-range, `MemoryRegion.contains` boundaries, error hierarchy `isinstance` relationships, JSON round trips (property-based for `Capabilities` and `Frame`), bad serialised input.
- **Results:** 73 / 73 passed / 0 failed / 0 excluded (86 in the full suite). Lint, format and strict mypy clean.
- **Completion report:** Error classes use an `Error` suffix (`EmulatorDisconnectedError`, `UnsupportedCapabilityError`) per lint rule N818; ARCHITECTURE.md updated to match.

### T-003 – Emulator adapter interface and capabilities model
- **Status:** Done
- **Phase:** 1
- **Objective:** The `EmulatorAdapter` base class from ARCHITECTURE.md §3 that enforces connection, capability and argument checks identically for every backend.
- **Scope:** `retro_ai_lab.emulator.base`. **Out of scope:** any concrete adapter.
- **Acceptance criteria:**
  1. A subclass implementing only the abstract hooks works and reports no capabilities.
  2. Every operation while disconnected raises `EmulatorDisconnectedError` without touching the backend.
  3. Every optional operation raises `UnsupportedCapabilityError` naming the capability when its flag is off, without touching the backend.
  4. Invalid arguments raise `ConfigurationError` before the backend is called.
  5. `connect`/`disconnect` are idempotent; a failed connect leaves the adapter disconnected.
- **Test plan:** abstract class cannot be instantiated; lifecycle and idempotency; timeout validation; failed connect; `mark_disconnected`; parametrised "requires connection" over all 11 operations; parametrised "requires capability" over all 8 gated operations; default hooks raise even when an adapter lies about capabilities; `read_memory` boundaries, unknown region, type errors, short backend reads; `set_input` dedupe and unknown button; `advance_frames` and `set_speed` validation; save-state slot validation; return types are tuples.
- **Results:** 72 / 72 passed / 0 failed / 0 excluded (158 in the full suite). Lint, format and strict mypy clean.
- **Completion report:** Chose an ABC with `final` public methods and `_do_*` hooks over a bare `Protocol`, so the checks cannot be forgotten by a backend. ARCHITECTURE.md §3 rewritten to match.

### T-004 – Synthetic game (deterministic toy racer)
- **Status:** Done
- **Phase:** 1
- **Objective:** A tiny, pure-Python game with a screen buffer and a byte-addressable RAM holding position, speed, lap, checkpoint, lives and score at fixed but *undocumented-to-callers* addresses, so later discovery tasks have something real to find.
- **Scope:** `retro_ai_lab.emulator.fake.toy_racer`. **Out of scope:** the adapter wrapping it (T-005).
- **Acceptance criteria:**
  1. Same seed and inputs produce identical RAM and frames every time.
  2. Up/Down/B/Left/Right have their documented effects; A has none; Start restarts only after game over.
  3. A lap counts only after all three checkpoints; obstacles cost a life and stop the car; three hits end the game.
  4. RAM values stay in range; the header is constant; noise and timer bytes change each frame.
  5. `snapshot`/`restore` are exact; bad snapshots are rejected.
  6. Rendering is 64×48 greyscale and reflects lane, lives, obstacles and game over.
  7. ≥ 2,000 steps/s headless.
- **Test plan:** initial state; seed validation; button set; obstacle seeding and placement; property-based determinism; unknown button; acceleration/deceleration/brake/cancel; lane bounds; A no-op; input mirror byte; checkpoints→lap; lap requires checkpoints (via doctored snapshot); scoring; score and lap saturation; collision; game over freezes gameplay but not the frame counter; Start semantics; property-based RAM range invariants; noise/timer; 16-bit frame wrap; snapshot/restore exactness and divergence-free continuation; bad snapshots; reset; render size, player, lane movement, game-over bar, approaching obstacles; performance.
- **Results:** 39 / 39 passed / 0 failed / 0 excluded (197 in the full suite). Lint, format and strict mypy clean.
- **Completion report:** RAM layout is 32 bytes with state at fixed addresses, 4 noise bytes, a free-running timer and a constant header, so later discovery tasks must separate signal from noise. Movement is simulated one unit per sub-step so obstacles and checkpoints cannot be skipped at speed.

### T-005 – Fake emulator adapter over the synthetic game
- **Status:** Done
- **Phase:** 1
- **Objective:** `FakeEmulatorAdapter` implementing the full adapter contract, including save/load state and speed, with configurable capability flags and fault injection (timeouts, disconnects).
- **Scope:** `retro_ai_lab.emulator.fake.adapter` and the reusable `retro_ai_lab.testing.AdapterContract` suite. **Out of scope:** registry/service (T-006).
- **Acceptance criteria:**
  1. Passes the full adapter contract suite with all capabilities enabled.
  2. Inputs are held until changed; the game advances only on `advance_frames`.
  3. Memory and frames match the underlying game exactly.
  4. Save/load per slot restores RAM exactly; unknown slot is an error.
  5. Reduced capability sets are honoured by the base-class gating.
  6. Fault injection: refused connect, timed-out connect, link drop after N operations, `drop_link()`; all surface as the documented errors and leave the adapter disconnected.
- **Test plan:** contract suite (identity, stable description, lifecycle, idempotency, reconnect, 11 ops require connection, 8 gated ops raise when disabled, memory reads/repeatability/bounds, frame capture, input accept/reject, frame advance, speed, save/load exactness, unknown slot) plus fake-specific tests (game info, RAM mirror, held input, parity with direct play, frame numbers, speed, per-slot states, disconnect releases input, reduced caps, each fault type, operation counting, bad fault settings).
- **Results:** 52 / 52 passed / 0 failed / 0 excluded / 0 skipped (249 in the full suite). Lint, format and strict mypy clean.
- **Completion report:** `AdapterContract` lives in `src/retro_ai_lab/testing/` so BizHawk and VBA-M adapters reuse it unchanged; it skips a capability test only when the adapter genuinely lacks that capability, which is recorded in the pytest summary.

### T-006 – Adapter registry and connection service
- **Status:** Done
- **Phase:** 1
- **Objective:** Discover available adapters, connect by name, expose session state and events to upper layers without Qt.
- **Scope:** `core.events.EventBus`, `emulator.registry.AdapterRegistry` + `default_registry()`, `services.connection.ConnectionService` with its events. **Out of scope:** threading, UI.
- **Acceptance criteria:**
  1. Registry lists, validates and instantiates adapters by name; unknown names and bad options are `ConfigurationError`.
  2. Service state machine DISCONNECTED→CONNECTING→CONNECTED→DISCONNECTED, with failure returning to DISCONNECTED and a `ConnectionFailed` event.
  3. Double connect is rejected; disconnect when idle is a silent no-op.
  4. `call()` runs operations on the live adapter and reports a dropped link exactly once (state + `EmulatorDisconnected(expected=False)`).
  5. Events are delivered in order; a failing handler cannot break publishing.
- **Test plan:** event bus (order, subclass filtering, unsubscribe both ways, error isolation, default logging, self-unsubscribe during publish, counts, argument validation); registry (register/list/create with options, unknown, bad options, duplicates, invalid names, missing display name, non-adapter classes, default registry ships `fake`); service (initial state, happy path with exact event sequence, timeout passthrough, unknown adapter and bad options leave state untouched, failed connect, empty error message, double connect, disconnect, idle disconnect, reconnect, `call` happy path, `call` without connection, dropped link reported once, other errors pass through, custom registry).
- **Results:** 39 / 39 passed / 0 failed / 0 excluded (288 in the full suite). Lint, format and strict mypy clean.
- **Completion report:** Connection is synchronous in this task; the UI task that needs it off the main thread (T-009) will wrap it in a worker. `EmulatorDisconnected` the event is distinct from `EmulatorDisconnectedError` the exception.

### T-007 – App shell
- **Status:** Done
- **Phase:** 1
- **Objective:** `python -m retro_ai_lab` opens a main window with the five panels from DESIGN.md §10 (Game, AI Activity, Rewards, Progress, Learning) as placeholders, each expandable.
- **Scope:** `ui.app`, `ui.main_window`, `ui.panels`; CLI launches the UI when run with no arguments; PySide6 and pytest-qt dependencies; Linux CI installs Qt system libraries. **Out of scope:** any real panel content.
- **Acceptance criteria:**
  1. The real entry point launches headless in a fresh process and exits 0.
  2. Window title "Retro AI Lab"; five panels in the documented order with child-friendly titles and a plain-English placeholder each.
  3. Each panel collapses and expands by click and by API, emitting `toggled`; collapsing one leaves the others.
  4. `--version`/`--help` never load Qt; no arguments launches the app.
- **Test plan (pytest-qt, offscreen):** open/close; panel keys, titles, placeholders, visibility; spec uniqueness and no jargon; collapse/expand by click with arrow state and signal; toggle API idempotency; independence of panels; `set_body` replaces placeholder; status bar message and version label; `create_app` singleton + metadata; `run_app` with hook; `exit_after_ms`; real subprocess launch smoke test. CLI tests updated: no-args launches via injected launcher, `--version`/`--help` never launch, hidden `--exit-after-ms` validated, process argv used when `argv` is None.
- **Results:** 15 / 15 UI tests passed + 4 updated CLI tests (303 in the full suite) / 0 failed / 0 excluded. Lint, format and strict mypy clean (PySide6 stubs under `--strict`).
- **Completion report:** Found and fixed a real bug while writing the smoke test: `main()` dropped `sys.argv` when called from `__main__`. Hidden `--exit-after-ms` flag exists only for smoke tests and tooling. Linux CI now installs `libegl1` etc. for Qt offscreen.

### T-008 – Game window panel shows live frames
- **Status:** Done
- **Phase:** 1
- **Objective:** While an emulator that can capture its screen is connected, the Game panel shows its frames live, scaled like pixel art, and says in plain words what is going on otherwise.
- **Scope:** `ui.context.AppContext`, `ui.frame_view.FrameView` + `frame_to_image`, `ui.game_panel.GamePanel` + `FramePump`; `MainWindow` takes a context and hosts the panel. **Out of scope:** connect/disconnect controls (T-009), input (T-010).
- **Acceptance criteria:**
  1. Grey, RGB and RGBA frames convert to correct `QImage`s.
  2. The view keeps aspect ratio, uses integer scaling when it can, centres the image, and shows a placeholder when empty.
  3. On `EmulatorConnected` with screen capture the pump starts and the first frame appears immediately; frames follow the game; on `EmulatorDisconnected` the pump stops and the view clears with a plain-English status (different for an unexpected loss).
  4. An emulator without screen capture gets a friendly message, no pump.
  5. Pump survives transient emulator errors and stops quietly on a dropped link; teardown stops the timer and unsubscribes from the bus.
- **Test plan:** image conversion for each channel count, deep copy, type errors; view placeholder painting, centred integer scaling checked by pixel, downscaling keeps ratio, clear, type errors; panel initial state, connect → frame + pump, live updates via timer, disconnect, dropped link wording, no-screen emulator, pump without connection, pump error counting, bad fps, main window hosting, window teardown stops pump and unsubscribes.
- **Results:** 20 / 20 passed / 0 failed / 0 excluded (323 in the full suite). Lint, format and strict mypy clean.
- **Completion report:** Frame capture runs on the UI thread for now (fast with the fake adapter); real adapters will move it to a worker in T-113. Teardown captures the pump and unsubscribe callbacks instead of `self` because the Python wrapper is already gone when `destroyed` fires.

### T-009 – Connect / disconnect UI with capability display
- **Status:** Done
- **Phase:** 1
- **Objective:** A child can pick a console, press Connect, see in plain words what that emulator can do, and press Disconnect; connecting never freezes the window.
- **Scope:** `ui.bridge.EventBridge` (bus events delivered on the GUI thread), `ui.connect_bar.ConnectBar` (picker, buttons, status, capability line, background connect via `QThreadPool`), `AppContext.bridge`, toolbar in `MainWindow`; `GamePanel` now subscribes via the bridge. **Out of scope:** per-adapter connection options UI (needed when BizHawk arrives).
- **Acceptance criteria:**
  1. Events published from worker threads reach UI handlers on the GUI thread; same-thread events are immediate.
  2. Connect runs in the background; the event loop keeps turning during a slow connect; a second connect while busy is refused.
  3. Buttons and picker enable/disable correctly in every state; status wording for idle, connecting, connected, failed, expected and unexpected disconnect.
  4. Capability line lists what the emulator can and can't do in child-friendly words.
  5. Connect bar in the main window drives the game panel end to end.
- **Test plan:** bridge (same thread immediate, worker thread queued to GUI thread, unsubscribe/counts, detach + local publish); capability wording (all, none, mixed, labels plain); initial state; click connect/disconnect; failure wording and retry; responsiveness during a 0.4 s connect (timer ticks counted); refuse when already connected; no adapters; unexpected loss; main window integration.
- **Results:** 17 / 17 passed / 0 failed / 0 excluded (340 in the full suite). Lint, format and strict mypy clean.
- **Completion report:** `ConnectionService` is not thread-safe by design; the only cross-thread call is `connect()` inside the worker, during which the service is in CONNECTING and refuses other work. Disconnect stays on the GUI thread (fast for every adapter we plan).

### T-010 – Manual controller input from UI to adapter
- **Status:** Done
- **Phase:** 1
- **Objective:** A child can play the connected game with the keyboard or an on-screen pad, and pause/resume time (Manual Mode foundation, DESIGN.md §8).
- **Scope:** `ui.controller_pad.ControllerPad` (built from the emulator's buttons, keyboard map, `input_changed` signal), `ui.play_clock.PlayClock` (advances one frame per tick while playing; Play/Pause), both hosted in `GamePanel`; `MainWindow` forwards key events to the pad. **Out of scope:** input recording (T-204), rebinding keys.
- **Acceptance criteria:**
  1. The pad shows one button per emulator button and sends the held set on every change; releasing all sends an empty set.
  2. Keyboard mapping works, auto-repeat is ignored, unmapped keys are ignored, custom maps are supported.
  3. Emulators without `input` get a disabled pad with a message; disconnect releases everything.
  4. The clock runs on connect when `frame_advance` is available, can be paused and resumed, keeps the child's choice across reconnects, and stops on disconnect or a dropped link.
  5. Pressing Up in the main window makes the Toy Racer car accelerate and move; pausing freezes time.
- **Test plan:** pad empty state; built from emulator; press/release/release_all and signal sequence; unknown button; mouse on on-screen buttons; keyboard mapping incl. unmapped key; auto-repeat; custom keymap; no-input emulator; disconnect releases; dropped link while sending; transient send error counted. Clock: idle; runs and advances; pause/play/toggle; choice survives reconnect; `tick(n)`; no frame-advance emulator; dropped link; transient error; bad fps. End-to-end keyboard play in the main window.
- **Results:** 22 / 22 passed / 0 failed / 0 excluded (362 in the full suite). Lint, format and strict mypy clean.
- **Completion report:** Found a test-infrastructure bug: once `QApplication.exec()` has returned inside the pytest process, Qt refuses further event loops on that thread and every timer-driven test afterwards hangs. The two in-process `run_app` tests now run in a fresh interpreter. Time is modelled as "advance when we say", which matches how the BizHawk Lua bridge will work.

### T-011 – Graceful handling of emulator disconnection
- **Status:** Done
- **Phase:** 1
- **Objective:** Whatever way the emulator goes away, the app notices promptly, says so once in plain words, resets every part of the window, and offers a one-click Reconnect.
- **Scope:** `ConnectionService.check()` (silent-drop detection) and resilient `disconnect()`; `ui.link_watchdog.LinkWatchdog`; Reconnect in `ConnectBar`; status-bar messages in `MainWindow`; friendlier reason for generic drops. **Out of scope:** automatic retry loops.
- **Acceptance criteria:**
  1. A link that dies while nothing is being sent is detected by the watchdog within its interval and reported exactly once.
  2. A backend error while disconnecting still results in a clean DISCONNECTED state and a logged, readable reason; unexpected exceptions propagate but leave the state clean.
  3. After an unexpected loss every widget resets (pump, clock, pad, play button, capability line, game view) and the status bar and connect bar explain what happened.
  4. The Connect button turns into Reconnect and reuses the last console and options; an expected disconnect does not offer Reconnect.
  5. Nothing keeps hammering a dead connection.
- **Test plan:** service `check()` idle / healthy / silent drop reported once / reconnect after; `disconnect()` return value, backend error, unexpected exception. UI: watchdog timer and interval validation; watchdog notices silent drop; whole-window recovery after silent drop; Reconnect reuses options; reconnect without history falls back to the picker; expected disconnect wording; failed connect in status bar; backend error on Disconnect click; drop during manual play handled once (error counters stay flat).
- **Results:** 16 / 16 passed / 0 failed / 0 excluded (378 in the full suite). Lint, format and strict mypy clean.
- **Completion report:** Whichever component notices a drop first (pump, clock, pad or watchdog) the service reports it once with the same child-readable reason ("The emulator went away.") unless the adapter gave a more specific one.

### T-012 – Windows packaging smoke test
- **Status:** Done
- **Phase:** 1
- **Objective:** A one-folder Windows build (`RetroAILab.exe`) that launches and exits cleanly, produced and smoke-tested by CI on every push to `main`, downloadable as a workflow artifact.
- **Scope:** `packaging/retro_ai_lab.spec` (PyInstaller, windowed, unused Qt modules excluded), `packaging/smoke_test.py`, `.github/workflows/package.yml` (windows-latest build → smoke test → upload artifact). **Out of scope:** installer, code signing, icons.
- **Acceptance criteria:**
  1. The spec builds from the real entry point and the built app honours `--exit-after-ms` and `--version`.
  2. The package workflow builds on Windows, runs the smoke test and uploads `RetroAILab-windows`.
  3. The smoke script fails clearly on bad usage or a missing executable.
- **Test plan:** spec parses and references the entry point; entry point exists; workflow content; smoke script usage and missing path; smoke script passes against the unpackaged entry point (slow). Manual/CI: PyInstaller build on Linux locally (same spec) passed the smoke test; Windows build verified by the Package workflow on the PR.
- **Results:** 5 / 5 passed / 0 failed / 0 excluded (383 in the full suite). Local Linux PyInstaller build: window run exit 0, version run exit 0, 198 MB folder. Lint, format and strict mypy clean.
- **Completion report:** The Windows exe is windowed (no console), so `--version` output is not visible on Windows; the smoke test checks only the exit code there. Build size will be revisited when PyTorch is added (Phase 5), probably by shipping a CPU-only wheel.

### T-201 – Memory snapshot capture and storage
- **Status:** Done
- **Phase:** 2
- **Objective:** Capture the emulator's readable memory as immutable, comparable snapshots; keep the recent ones in memory and archive sessions durably in SQLite.
- **Scope:** `observation.memory` (`MemorySnapshot`, `MemoryCapture`, `SnapshotRing`), `storage.snapshots.SnapshotArchive`. **Out of scope:** diffing (T-202), wiring into the UI loop (done with T-202 so the ring is fed by the play clock).
- **Acceptance criteria:**
  1. Snapshots validate inputs, are immutable, compare and hash by value, and round-trip through JSON.
  2. Capture reads every region (or a chosen subset) exactly as the game holds it, tagged with the frame; unsupported memory or unknown regions are clear errors.
  3. Ring keeps the newest N in order with `latest`/`previous`.
  4. Archive stores sessions and snapshots in SQLite with exact binary integrity, survives reopen, continues sequence numbers, isolates sessions, rejects bad paths, and stores 1000 × 8 KiB snapshots in well under 10 s.
- **Test plan:** snapshot basics, rejection matrix, bad ranges, immutability/equality/hash, input copy, property-based JSON round trip, bad serialised shapes; capture exact read, subset/unknown, no memory capability, not connected; ring order/capacity/previous/clear, bad capacity, non-snapshot; archive sessions lifecycle, append/count/iterate/latest/history, isolation, unknown session, reopen, awkward bytes, directory path, unwritable path, performance.
- **Results:** 46 / 46 passed / 0 failed / 0 excluded (429 in the full suite). Lint, format and strict mypy clean.
- **Completion report:** One row per region per snapshot so a single region's history can be read cheaply by the discovery engine. WAL journal mode for concurrent reads later.

### T-202 – Memory change detection (frame-to-frame diff)
- **Status:** Done
- **Phase:** 2
- **Objective:** Tell exactly which bytes changed between frames, accumulate per-byte behaviour over a run (change rate, direction, range), and feed snapshots automatically while the game plays so the understanding engine has data without touching the emulator.
- **Scope:** `observation.diff` (`ByteChange`, `SnapshotDiff`, `diff_snapshots`, `ByteStats`, `ChangeTracker` with `classify()` into constant/noise/state/unknown), `services.observer.ObservationService` (+ `MemoryObserved` event) owned by `AppContext` and driven by `PlayClock.tick`. `EventBridge` now drops its bus subscription when its Qt object is destroyed. **Out of scope:** screen diffs (T-203), persisting diffs (the archive stores snapshots; diffs are recomputed).
- **Acceptance criteria:**
  1. Diff lists every changed byte with old/new values in region/offset order, is empty for equal snapshots, and rejects mismatched regions or sizes.
  2. Tracker stats are exact (samples, changes, ups/downs, min/max, distinct) and the classifier separates the toy racer's noise bytes and timer from its header constants and its real state bytes without any game knowledge.
  3. The observation service is idle until a memory-capable emulator connects, snapshots on every advance, publishes `MemoryObserved`, resets on reconnect, counts transient read errors, and goes quiet on link loss without counting it as an error.
  4. The play clock feeds the service on every tick.
- **Test plan:** diff empty/ordered/rejections, property-based diff vs naive comparison, immutability; tracker seed/transition/ups-downs/rates, unknown byte, bad input, reset, classify groups, unknown before enough data, bad noise rate, intersection helper, toy-racer discovery; service idle/connect/advance/reset/no-memory/dropped link/transient error/ring capacity/negative; UI clock feeds service; deleted bridge stops forwarding.
- **Results:** 26 / 26 passed / 0 failed / 0 excluded (455 in the full suite). Lint, format and strict mypy clean.
- **Completion report:** The noise/state split is the first step of discovery (Phase 3): on the toy racer it already isolates the four noise bytes, the timer and the low frame-counter byte as noise and the header as constant, leaving position/lane/speed/score/input as candidate state. A bridge deletion bug surfaced when the clock began publishing events after a test window was torn down; fixed by unsubscribing on `destroyed` and checking validity before emitting.

### Later Phase 2 tasks
Details will be written into this file when each becomes the next task, following the template.

### T-110 – Spike: confirm BizHawk Lua ↔ socket protocol
- **Status:** Blocked on ADR-005 sign-off and on access to a Windows machine with BizHawk for verification.
- **Objective:** Prove the communication route (BizHawk Lua `comm` socket API vs. alternatives), message framing, and achievable throughput, and record the result in ARCHITECTURE.md.

---

## Session log

### 2026-11-02 – Session 1
- Repository was empty. Committed `CLAUDE.md` and `DESIGN.md` to `main` as the initial commit.
- T-000 completed on branch `init/project-docs`; PR opened for review.
- T-001 completed and merged (PR #2). User asked for auto-merge once tests pass; continuing autonomously.
- T-002 completed (PR #3).
- T-003 completed (PR #4).
- T-004 completed (PR #5).
- T-005 completed (PR #6).
- T-006 completed (PR #7).
- T-007 completed (PR #8).
- T-008 completed (PR #9).
- T-009 completed (PR #10).
- T-010 completed (PR #11).
- T-011 completed (PR #12).
- T-012 completed (PR #13). Phase 1 is complete on the fake emulator; T-110+ (BizHawk) wait on ADR-005 sign-off.
- T-201 completed (PR #14).
- T-202 completed (PR #15).
- Next: T-203.
- Blockers: none for T-001 to T-012. ADR-005 / ADR-006 need user input before real-emulator work.
