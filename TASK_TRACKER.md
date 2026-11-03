# Task Tracker

Single source of progress. Follows the template in `DEVELOPMENT_RULES.md` §3.
Phases refer to DESIGN.md §16. Phases 1–2 are broken into small tasks now; later phases are listed as epics and will be split into tasks before work on them starts.

## Status

- **Current task:** T-408 – Goal service
- **Last approved checkpoint:** `main` after T-308 (PRs #23–#26 merging in order)
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
| T-203 | Screen observation: frame diff and change regions | 2 | Done | 20/20 | see PR #16 |
| T-204 | Input recording (who pressed what, when) | 2 | Done | 19/19 | see PR #17 |
| T-205 | Save-state experiment harness (repeatable trials) | 2 | Done | 12/12 | see PR #18 |
| E-3 | Game Understanding (variables, events, relationships, confidence) | 3 | Epic | – | – |
| T-301 | Knowledge model: variables, evidence, confidence, JSON store | 3 | Done | 40/40 | see PR #19 |
| T-302 | Variable candidate finder from change statistics (counters, bounded values, 16-bit pairs) | 3 | Done | 9/9 (+ diff) | see PR #20 |
| T-303 | Input linking via experiments: which inputs move which bytes | 3 | Done | 7/7 | see PR #21 |
| T-304 | Event detection: progression steps, resets, game over | 3 | Done | 8/8 | see PR #22 |
| T-305 | Relationship discovery between events and variables | 3 | Done | 7/7 | see PR #23 |
| T-306 | Discovery runner: automatic observe-experiment-propose loop with progress events | 3 | Done | 10/10 | see PR #24 |
| T-307 | "What the AI is doing" panel: discoveries with confidence badges, confirm / correct / reject | 3 | Done | 7/7 | see PR #25 |
| T-308 | Knowledge persistence wiring: auto-save per game, export / import from the UI | 3 | Done | 10/10 | see PR #26 |
| E-4 | Reward System (controls, categories, personalities, goals, objectives) | 4 | Epic | – | – |
| T-401 | Reward model: behaviours with category, trigger, +/- controls, suggested value, enable toggle, JSON round-trip | 4 | Done | 18/18 | see PR #27 |
| T-402 | Reward suggestions: derive behaviours and suggested values from the game knowledge | 4 | Done | 7/7 | see PR #28 |
| T-403 | AI personalities: Explorer, Collector, Completionist, Speedrunner, Battler, Survivor, Balanced presets | 4 | Done | 6/6 | see PR #29 |
| T-404 | Reward service: per-game profile persistence, live reward tally from events | 4 | Done | 15/15 | see PR #30 |
| T-405 | Reward exploitation detection with plain-words explanation | 4 | Done | 8/8 | see PR #31 |
| T-406 | Rewards panel: categories that expand, +/- buttons, value, toggle, suggested, personality picker | 4 | Done | 7/7 | see PR #32 |
| T-407 | Goals and objectives model: overall goal, objectives with detection, dependencies, completion | 4 | Done | 9/9 | see PR #33 |
| T-408 | Goal service: per-game goal persistence, completion tracking from events, suggested objectives | 4 | Planned | – | – |
| T-409 | Goals panel: edit the objective structure, see completion and confidence | 4 | Planned | – | – |
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
- **Completion report:** The noise/state split is the first step of discovery (Phase 3): on the toy racer it already isolates the four noise bytes, the timer and the low frame-counter byte as noise and the header as constant, leaving position/lane/speed/score/input as candidate state. A bridge deletion bug surfaced when the clock began publishing events after a test window was torn down; fixed by unsubscribing on `destroyed` and checking validity before emitting. CI on Windows/py3.11 then failed the T-201 archive performance gate (11.4 s for 1000 snapshots): each append did a full fsync. Switched the archive to `PRAGMA synchronous = NORMAL` under WAL (safe against corruption; 1000 snapshots now take ~0.1 s). Also made `observation.memory`'s import of `services` type-only, which removed a circular import that only import order had been hiding.

### T-203 – Screen observation: frame diff and change regions
- **Status:** Done
- **Phase:** 2
- **Objective:** Detect which parts of the picture change between frames (pixel count, bounding box, per-cell counts on a coarse grid) and accumulate per-cell activity over a run so static HUD areas can be told from the playfield.
- **Scope:** `observation.screen` (`FrameDiff`, `diff_frames` with threshold and grid, `ScreenTracker` with `activity`, `activity_map`, `classify` into static/active/busy, automatic reset on resolution change), `ObservationService.observe_frame` publishing `ScreenObserved`, `GamePanel` pump feeding it. NumPy added as a runtime dependency (ADR-008). **Out of scope:** object/sprite detection, OCR (Phase 3).
- **Acceptance criteria:**
  1. Diff counts changed pixels exactly, reports the inclusive bounding box, distributes counts over grid cells that tile the frame, honours a per-channel threshold, and rejects shape mismatches, bad thresholds and bad grids.
  2. Tracker activity is the exact fraction of transitions in which each cell changed; classification thresholds are validated.
  3. Service feeds the tracker from every captured frame while connected, publishes `ScreenObserved`, resets on reconnect; the game panel's pump calls it.
  4. 60 diffs of a 256x224 RGB frame complete in under 2 s.
- **Test plan:** array view, cell bounds; identical frames, counts/bbox/cells, threshold, multichannel, rejections, property-based vs naive count and bbox, immutability; tracker seed/accumulate, classify, resolution change/reset, bad input; toy racer moves; performance (slow); service observe_frame; panel pump feeds service.
- **Results:** 20 / 20 passed / 0 failed / 0 excluded (475 in the full suite). Lint, format and strict mypy clean.
- **Completion report:** Fixed a latent circular import (`observation.memory` imported `services` at runtime; now type-only). The panel ticks the pump once on connect so the first screen frame is observed immediately.

### T-204 – Input recording (who pressed what, when)
- **Status:** Done
- **Phase:** 2
- **Objective:** Record every change of the held button set with the frame it took effect on and its source (human, ai, replay), so inputs can be lined up against memory and screen changes and stored with a session.
- **Scope:** `observation.inputs` (`InputEvent`, `HeldSpan`, `InputLog` with `record`, `held_at`, `spans`, `frames_held`, `presses`, `by_source`, JSON round trip, bounded capacity), `ObservationService.record_input` publishing `InputRecorded` and tagging the current frame, `ControllerPad` recording every send as `human`, `SnapshotArchive.append_input`/`inputs` (new `inputs` table). **Out of scope:** replaying a log into the emulator (T-205 / Phase 4), AI inputs (Phase 5 plugs into the same `record_input`).
- **Acceptance criteria:**
  1. Events validate frame, names and source and round-trip through JSON.
  2. The log stores only changes, keeps frames monotonic, collapses several changes on one frame to the final state, answers `held_at` exactly for any frame, and forgets the oldest events when full (counting drops).
  3. Hold lengths and press counts are exact.
  4. Pad presses land in the log tagged `human` with the frame the observation service is on; the log clears on reconnect; inputs persist per session in the archive and are deleted with it.
- **Test plan:** event normalisation/round trip, rejection matrix, bad dicts; log changes-only, same-frame replacement, backwards frames, `held_at`/spans/by_source, hold lengths/presses, capacity, serialisation and order check, property-based `held_at` vs naive model; service tagging and publish, bad source, reset; archive per-session order, cascade delete, bad session, bad type; pad records human input.
- **Results:** 19 / 19 passed / 0 failed / 0 excluded (494 in the full suite). Lint, format and strict mypy clean.
- **Completion report:** Inputs are stored as names, not `Button` objects, so logs are portable across adapters and serialisable. Only the final held set of a frame is kept because the emulator only ever sees one input per frame.

### T-205 – Save-state experiment harness (repeatable trials)
- **Status:** Done
- **Phase:** 2
- **Objective:** Let the app try an input for a few frames from a saved checkpoint, see exactly what changed in memory and on screen, put the game back, and compare trials that differ only in their inputs to find the bytes those inputs influence (DESIGN.md §5 steps 2–5 and §10 "use save states to repeat experiments safely").
- **Scope:** `services.experiments` (`ExperimentHarness` with `available`, `checkpoint`, `restore`, `run`, `run_many`, `compare`; `Trial`, `Comparison`, `constant_plan`). Requires save states, memory read, input and frame advance; screen capture is optional. **Out of scope:** scheduling experiments automatically and interpreting the results (Phase 3 E-3), UI.
- **Acceptance criteria:**
  1. Unavailable (clear `UnsupportedCapabilityError`/`EmulatorDisconnectedError`) until a capable emulator is connected; refuses to run or restore without a checkpoint.
  2. A trial starts from the checkpoint, applies the per-frame plan, captures before/after memory (and screen when available), releases all buttons and restores the checkpoint by default; two identical trials give identical results.
  3. Empty plans and unknown buttons are rejected; link loss mid-trial propagates and forgets the checkpoint.
  4. `compare` reports every byte whose final value differs between trials and the bytes that changed in all of them; on the toy racer it isolates speed for "up" and lane for "left" without any game knowledge, and never flags header bytes.
- **Test plan:** constant plan validation; availability/capability gating; bad slot; checkpoint required; run + restore on the toy racer; repeatability; per-frame plan without restore then restore; empty plan/unknown buttons; no screen capture; link loss; compare discovery; compare rejections.
- **Results:** 12 / 12 passed / 0 failed / 0 excluded (506 in the full suite). Lint, format and strict mypy clean.
- **Completion report:** Phase 2 (observation) is complete on the fake emulator: memory snapshots, diffs, screen diffs, input log and the experiment harness give Phase 3 everything it needs to start discovering what bytes mean.

### T-301 – Knowledge model: variables, evidence, confidence, JSON store
- **Status:** Done
- **Phase:** 3
- **Objective:** Give the understanding engine a place to put what it believes, with every belief carrying its evidence and a confidence level that only a person can raise to Confirmed (DESIGN.md §3.5, §3.6).
- **Scope:** `understanding.knowledge` (`Confidence` ordered enum, `Encoding` with decoders, `VariableKind`, `Evidence`, `Variable` with derived confidence and `decode(snapshot)`, `Knowledge` with propose / add / update / support / confirm / correct / reject / remove, lookups by kind and confidence, `needs_help()`, JSON round trip with schema version), `storage.knowledge.KnowledgeStore` (one JSON file per game, atomic writes, `export`/`load` anywhere). **Out of scope:** producing beliefs (T-302+), events and relationships (T-304/305 extend the model), UI (T-307).
- **Acceptance criteria:**
  1. Confidence is ordered; automatic confidence comes from evidence weight thresholds and never reaches Confirmed; a rejected belief reads as Unknown.
  2. Decoders for u8, s8, u16 LE/BE, BCD and flag are exact and reject wrong widths or non-BCD bytes.
  3. Knowledge refuses duplicate ids and duplicate live locations, lets a rejected location be re-proposed, keeps user confirm/correct/reject as evidence, and round-trips through JSON (property-tested) including a newer-schema refusal.
  4. The store writes human-readable JSON atomically, maps awkward game ids to safe file names, ignores foreign files, and reports read/write failures as `ConfigurationError`.
- **Test plan:** confidence order/labels/thresholds/negative weight; decoder matrix and rejections; variable defaults, evidence accumulation, snapshot decode, rejection matrix, evidence and variable round trips; knowledge empty state, propose/add/lookup, update/support/queries, confirm/correct/reject/remove, round trip, bad dicts, property-based round trip; store folder/round trip, load_or_new/delete, awkward ids, foreign files, export/load errors, file-as-folder.
- **Results:** 40 / 40 passed / 0 failed / 0 excluded (546 in the full suite). Lint, format and strict mypy clean.
- **Completion report:** Phase 3 was broken into T-301..T-308 (table above). Confidence is derived, not stored, so editing evidence can never leave a stale level behind; `new_id()` reserves nothing until `add` succeeds.

### T-302 – Variable candidate finder from change statistics
- **Status:** Done
- **Phase:** 3
- **Objective:** Turn per-byte change statistics from ordinary play into first guesses about what each byte is (counter, score, lives, flag, progress, position, speed), each with plain-words evidence at Low or Medium confidence, with no knowledge of the game.
- **Scope:** `observation.diff.ByteStats` gains `wraps`, `carry_ins`, `unit_steps`, `max_step`, `resets`, `mostly_up`; `understanding.variables` (`Candidate`, `find_candidates(tracker)`, `propose_candidates(knowledge, candidates)`), 16-bit little-endian pairing by carry detection, noise and constant exclusion. **Out of scope:** input linking (T-303), events (T-304), big-endian / BCD detection (added when a real game needs it; documented limitation).
- **Acceptance criteria:**
  1. Nothing is proposed before `MIN_TRANSITIONS` transitions; constants and noise are never proposed.
  2. Synthetic bytes classify as counter / 16-bit counter / 16-bit score / lives / flag / progress / speed / position / unknown as documented.
  3. On a 900-frame toy racer drive with crashes and restarts, the finder labels the frame counter (16-bit counter), score (16-bit score), timer (counter), position, speed, lap and checkpoint (progress), lives, and game-over (flag), proposes none of the noise or header bytes, and every proposal lands at Low or Medium confidence with evidence.
  4. `propose_candidates` never touches a location the knowledge already has an opinion about (proposed, confirmed or rejected) and is idempotent.
- **Test plan:** min transitions and bad argument; counter/constant/noise; 16-bit pairs; lives/flag/progress/speed/unknown; position; pair precedence and ordering; propose rules; toy racer discovery; proposals' confidence and evidence.
- **Results:** 9 / 9 passed / 0 failed / 0 excluded (555 in the full suite). Lint, format and strict mypy clean.
- **Completion report:** The first version treated "monotonic" strictly and found nothing on a realistic drive, because restarts reset the score and position. Rules now tolerate resets (`mostly_up`, pair detection allows stray drops) and classification quality on the toy racer is as listed above. Lane and the input mask are labelled "progress" for now; T-303's input linking is expected to correct them.

### T-303 – Input linking via experiments
- **Status:** Done
- **Phase:** 3
- **Objective:** Run the active-discovery loop of DESIGN.md §3.4 for the controller: from a checkpoint, hold each button for a few frames and find which bytes it moves compared with doing nothing, then fold that into the knowledge as evidence, relabels and new proposals.
- **Scope:** `understanding.linking` (`InputLinks`, `find_input_links(harness, buttons, frames=8, repeats=2)` with unstable-byte exclusion from repeated idle trials, `apply_input_links(knowledge, links)`), new `VariableKind.INPUT` for the controller mirror. **Out of scope:** button combinations and held-duration effects (later discovery passes), scheduling (T-306).
- **Acceptance criteria:**
  1. On the toy racer, Up moves speed and position, Left/Right move the lane, B brakes, the input byte mirrors every button, and header/noise bytes are never linked; the game is restored afterwards.
  2. Bytes that differ between identical idle trials are reported as unstable and excluded from every link.
  3. Bad arguments (no buttons, non-positive frames/repeats, no checkpoint, unknown buttons) are `ConfigurationError`s.
  4. Applying links adds "changes when X is held" evidence once (idempotent), relabels unknown/progress values moved by a direction button as positions, labels a location moved by every tested button as the controller mirror (only with 3+ buttons tested), proposes linked bytes nobody knew about, and never changes a confirmed label or touches a rejected location.
  5. End to end: statistics (T-302) then linking relabel the lane as a position, find the controller byte, and raise speed to at least Medium.
- **Test plan:** toy racer links; unstable exclusion with a non-saved RNG; argument rejections; apply evidence/relabel/propose and idempotence; confirmed/rejected respected; too few buttons; end to end.
- **Results:** 7 / 7 passed / 0 failed / 0 excluded (562 in the full suite). Lint, format and strict mypy clean.
- **Completion report:** First attempt checkpointed at top speed, where Up changes nothing; the fixture now checkpoints at a middling speed, which is also what the discovery runner (T-306) should do: try buttons from several different game situations.

### T-304 – Event detection: progression steps, resets, game over
- **Status:** Done
- **Phase:** 3
- **Objective:** Read the snapshot stream as a story using the believed variables: progress steps, unusual score jumps, setbacks and recoveries (lives, health), flags switching, and resets (several values back to their starting values at once), so rewards and relationships have events to work with.
- **Scope:** `understanding.events` (`EventKind`, `GameEvent`, `EventDetector(knowledge, at_least=Low, gain_factor=3)`, `EventLog`). The detector only watches non-rejected progress / score / lives / health / flag variables at or above the confidence floor, re-arms when a variable is edited, and skips variables a snapshot does not cover. **Out of scope:** screen-based events (menus, transitions; a later pass once OCR exists), wiring into the live loop (T-306).
- **Acceptance criteria:**
  1. First sight of a variable starts watching without an event; unchanged values produce nothing.
  2. Progress up → PROGRESS; lives/health down → SETBACK, up → RECOVERY; flag on/off → FLAG_ON/OFF; progress going down alone is not an event.
  3. RESET only when at least two watched values return to their first-seen values in one frame and at least one dropped.
  4. SCORE_GAIN fires only for jumps at least `gain_factor` times the running typical step.
  5. Unknown-confidence, rejected and wrong-kind variables are ignored; the floor is adjustable; knowledge edits re-arm the watch.
  6. On a 900-frame toy racer drive: checkpoints register as progress, crashes as setbacks, bonuses as score gains, game over as a flag, and the restart as a reset after the flag.
- **Test plan:** first snapshot; rule matrix; reset rule (two values + drop, one value, no drop); score gain; filtering and floor; edits/reset/bad input/short snapshot; event log; toy racer story.
- **Results:** 8 / 8 passed / 0 failed / 0 excluded (570 in the full suite). Lint, format and strict mypy clean.
- **Completion report:** One test originally expected a plain recovery on a frame where three values returned home together; the detector's reset rule was right and the test was corrected rather than the rule.

### T-305 – Relationship discovery between events
- **Status:** Done
- **Phase:** 3
- **Objective:** Notice that one event tends to follow another (checkpoint → score bonus, last life lost → game over, game over → restart) and keep those links as editable, persisted discoveries (DESIGN.md §3.3, §3.6).
- **Scope:** `Relationship` record in the knowledge model (cause/effect event keys, support and side counts, follow/precede rates, derived confidence, user confirm/reject preserved across count refreshes, JSON schema 2 with backward-compatible load), `understanding.relationships.RelationshipFinder(knowledge, window=5)` counting pairs in the event stream (same-frame events link both ways; one effect confirms a cause once), `event_key` / `describe_key` helpers. **Out of scope:** multi-step chains and causal experiments (later), UI (T-307).
- **Acceptance criteria:**
  1. Rates and confidence derive from counts (Low ≥2 support at ≥60%, Medium ≥5 at ≥75%, High ≥10 at ≥90%); Confirmed only by a person; rejected reads Unknown; invalid counts rejected.
  2. Knowledge stores, refreshes, confirms, rejects, removes and round-trips relationships; schema-1 files still load.
  3. The finder pairs only inside the window, never pairs an event with itself, counts same-frame events both ways, never lets one effect confirm a cause twice, and refreshes rates when only side counts change.
  4. On a 2400-frame toy racer drive: checkpoint progress ↔ score gain (support ≥3, follow rate ≥60%), lives setback → game over (precede rate 100%), game over → reset (precede rate 100%), with a readable description.
- **Test plan:** relationship model and validation; knowledge relationship API and persistence; keys and descriptions; window rule; same-key and same-frame rules; rate refresh; toy racer relationships.
- **Results:** 7 / 7 passed / 0 failed / 0 excluded (577 in the full suite). Lint, format and strict mypy clean.
- **Completion report:** Events raised on the same frame (score bonus and checkpoint step) arrive in memory-layout order, so a one-way count missed the checkpoint → bonus link; same-frame pairs now count both ways.

### T-306 – Discovery runner
- **Status:** Done
- **Phase:** 3
- **Objective:** Run the understanding engine automatically while a game is connected: detect events and relationships on every snapshot, review the statistics into new proposals on a schedule, run button experiments on a schedule when save states exist, and announce every step so the UI can show what the AI is doing.
- **Scope:** `services.discovery.DiscoveryService(connection, observation, bus, review_every=120, experiment_every=600)` with `knowledge`, `events`, `stage`, `summary()`, `review()`, `experiment()`, `can_experiment()`; events `DiscoveryProgress(stage, detail)` and `DiscoveryUpdated(added, changed, events, relationships)`; `game_id_for`; owned by `AppContext`. **Out of scope:** persistence (T-308), UI (T-307), choosing *when* in a game experiments are most informative (later; today they run from wherever the game is).
- **Acceptance criteria:**
  1. Idle until a memory-capable game connects; a fresh `Knowledge` per connect keyed by ROM hash (or system:title); the previous knowledge stays readable after disconnect.
  2. Reviews run every `review_every` snapshots and propose variables; events and relationships are published as they happen.
  3. Experiments run every `experiment_every` snapshots when save states are available (and on demand), relabelling the lane as a position and finding the controller byte on the toy racer; without save states they are reported unavailable and the schedule passes quietly.
  4. Link loss during an experiment leaves the service idle and the connection service in charge of reporting it.
  5. Bad schedules are rejected; experiments can be switched off.
- **Test plan:** idle/connect/summary; game id; review schedule and progress stages; events and relationships while playing; experiments on schedule and on demand; no save states; disconnect/reconnect; link loss mid-experiment; bad schedules; app context wiring.
- **Results:** 10 / 10 passed / 0 failed / 0 excluded (587 in the full suite). Lint, format and strict mypy clean.
- **Completion report:** Experiments are synchronous and run inside the snapshot handler, so on the GUI thread they block for the length of the trials (about 70 frames on the fake emulator, instant). On a real emulator this will need either turbo during trials or a background worker; noted for T-11x.

### T-307 – "What the AI is doing" panel
- **Status:** Done
- **Phase:** 3
- **Objective:** Let a child see what the AI is doing and what it thinks it has found, in plain words with an honest confidence badge, and answer back: yes, no, or a name (DESIGN.md §3.5 "children should be able to inspect suggested discoveries and confirm or correct them").
- **Scope:** `ui.discovery_panel.DiscoveryPanel` in the "What the AI is doing" slot of the main window: stage line and one-line summary from `DiscoveryProgress`, list of discoveries ("Score (Score, ram[8]) — Probably"), evidence of the selected one, "Yes, that's right" / "No, that's wrong" / rename box (naming confirms), "Try the buttons" (on-demand experiment, enabled only when save states exist), last eight events, top relationships. Badges: Not sure / A guess / Probably / Pretty sure / Confirmed by you / You said no. **Out of scope:** editing kind or encoding from the UI (T-308 or later), persistence (T-308).
- **Acceptance criteria:**
  1. Idle text and disabled controls before a game connects; stage text follows discovery progress; disconnect returns to idle.
  2. Discoveries appear with the right badge and evidence; selection survives refreshes.
  3. Confirm, reject and rename update the knowledge and the list; blank names and empty selections are no-ops; the buttons and the rename box drive the same actions.
  4. Events and relationships are listed, newest events first, capped.
  5. The experiment button runs an experiment and is disabled when unavailable, with the reason shown.
  6. The main window hosts the panel in the activity slot.
- **Test plan:** idle; discoveries with badges and evidence; confirm/reject/rename; widgets; events and relationships; experiment button incl. unavailable; main window hosting.
- **Results:** 7 / 7 passed / 0 failed / 0 excluded (594 in the full suite). Lint, format and strict mypy clean.
- **Completion report:** Kept the vocabulary child-sized ("A guess", "Pretty sure") and kept the panel free of modal dialogs so it stays testable offscreen and unobtrusive while playing.

### T-308 – Knowledge persistence wiring
- **Status:** Done
- **Phase:** 3
- **Objective:** Make what the AI learns about a game survive restarts and be shareable (DESIGN.md §3.6 "discoveries should be reusable, editable and exportable").
- **Scope:** `core.paths.default_data_dir()` (env override `RETRO_AI_LAB_DATA_DIR`, else the platform's per-user app data folder); `AppContext.data_dir` and `knowledge_store`; `DiscoveryService(store=…)` loads the game's knowledge on connect (damaged file → start over), saves whenever a belief is added or changed, on disconnect and on `save()`, and offers `export_to(path)` / `import_from(path)` (import must be for the same game); the panel saves after confirm/reject/rename ("Remembered for next time.") and has "Share what it learned…" / "Load shared knowledge…" buttons (file dialogs) backed by testable `export_to` / `import_from` with outcomes shown in a note line. **Out of scope:** merging two knowledge files (import replaces), cloud sharing (Phase 8).
- **Acceptance criteria:**
  1. Data dir resolves per platform and honours the override.
  2. Knowledge is remembered between connections to the same game (confirmed verdicts included) and a different game starts empty; a damaged file does not crash the app.
  3. Export writes a loadable file; import replaces the knowledge for the same game, is refused for another game, and discovery carries on afterwards.
  4. Panel verdicts persist to the store; export/import outcomes and errors are shown in plain words; the file buttons are disabled when no game is connected.
- **Test plan:** paths override/default/platform matrix; remembered between connections; damaged file; export/import incl. wrong game; panel save/export/import/errors/disabled.
- **Results:** 10 / 10 passed / 0 failed / 0 excluded (604 in the full suite). Lint, format and strict mypy clean.
- **Completion report:** Phase 3 (Game Understanding) is complete on the fake emulator: the app watches a game, proposes what its bytes mean with evidence and honest confidence, experiments with the buttons, detects events and relationships, asks a person to confirm or correct, and remembers it all per game.

### T-401 – Reward model
- **Status:** Done
- **Phase:** 4
- **Objective:** Give children the simple controls DESIGN.md §5.1 asks for: every behaviour the AI can be rewarded for has a +, a -, a current value, an enable/disable toggle and a suggested value, organised into the eight categories of §5.2, with no mathematics on show.
- **Scope:** `rewards.model`: `Category` enum (Exploration … Completion, with labels), `Trigger` (event kind + optional variable, matches a `GameEvent`), `Behaviour` (id, name, category, trigger, suggested, value, enabled, custom flag, notes; validated and clamped to ±100), `RewardProfile(game_id)` with add/get/remove/by_category, `increase`/`decrease` by one step, `set_value`, `set_enabled`, `reset_to_suggested`, `suggest` (moves the value too unless a person has set it), `reward_for(event)` / `reward_for_events`, JSON `to_dict`/`from_dict` (schema 1). **Out of scope:** deriving behaviours from knowledge (T-402), personalities (T-403), persistence and live tally (T-404), UI (T-406).
- **Acceptance criteria:**
  1. Categories and behaviours validate (empty id/name, non-finite or out-of-range values, bad category rejected) and round-trip through JSON; newer schema refused.
  2. `+` and `-` move the value by one step, clamp at ±100, and mark the behaviour as set by a person; `reset_to_suggested` clears that mark.
  3. A suggestion updates the suggested value and moves the current value only while a person has not set it.
  4. Disabled behaviours contribute nothing; `reward_for` sums every enabled matching behaviour; a trigger with no variable matches any variable of that kind; events that match nothing give 0.
  5. `by_category` lists every category (empty ones too) in the §5.2 order.
- **Test plan:** category labels/order; trigger matching incl. wildcard; behaviour validation and clamping; profile add/duplicate/get/remove; +/-/clamp/custom; set_value/enable; suggest with and without custom; reward_for with enabled/disabled/multiple; JSON round-trip, bad data, newer schema; hypothesis property: +/- never leave the range.
- **Results:** 18 / 18 passed / 0 failed / 0 excluded (622 in the full suite). Lint, format and strict mypy clean.
- **Completion report:** Importing `retro_ai_lab.understanding` first went round in a circle (understanding → linking → services → discovery → understanding.linking); the linking module now imports the experiment harness lazily, and an import-order test guards every package. `reset_to_suggested` clears the person's mark so suggestions follow again.

### T-402 – Reward suggestions from the game knowledge
- **Status:** Done
- **Phase:** 4
- **Objective:** Fill the reward profile automatically from what the AI believes about the game (DESIGN.md §5.4): every believed variable that can raise an event becomes a behaviour with a sensible suggested value, bigger for things that lead to progress, smaller while the belief is shaky, and always editable.
- **Scope:** `rewards.suggestions`: `suggest_behaviours(knowledge, at_least=LOW)` maps (event kind, variable kind) pairs to a category, name and base value (progress +10, score jump +5, losing a life -10, gaining one +5, health drop -3 / recovery +2, flag on +2 / off 0, starting over -5); scales by belief confidence (Low 0.5, Medium 0.75, High and Confirmed 1.0); a cause of a believed relationship whose effect is progress gets 1.5x ("leads to progress"); a flag that leads to a reset reads as game over (Survival, -15). `apply_suggestions(profile, knowledge)` adds new behaviours, re-suggests changed values (a person's own value is kept), refreshes names and notes, and removes automatic behaviours whose variable is gone or rejected unless a person set them. **Out of scope:** personalities (T-403), persistence and wiring (T-404), UI (T-406).
- **Acceptance criteria:**
  1. Each tracked variable kind yields the expected behaviours, category and base value; unknown, rejected and too-uncertain variables yield nothing; "Starting over" is always offered.
  2. Confidence scales the value; a progress-causing relationship boosts it with a note; a flag that leads to a reset is suggested as a game-over penalty.
  3. Applying twice changes nothing; a renamed variable renames its behaviours; a raised confidence re-suggests the value but not a person's value; a rejected variable removes its automatic behaviours unless a person set them; a child's own behaviours are untouched.
  4. On the toy racer knowledge, the profile has checkpoint/lap progress, score jump, lives lost/gained, game over and starting over, and a live drive earns positive reward for checkpoints and negative for crashes.
- **Test plan:** per-kind mapping; filters; confidence scaling; relationship boost and game-over flag; apply idempotence, rename, re-suggest, custom kept, removal, foreign behaviours; toy racer end to end.
- **Results:** 7 / 7 passed / 0 failed / 0 excluded (629 in the full suite). Lint, format and strict mypy clean.
- **Completion report:** Added `RewardProfile.update` for descriptive fields (name, category, notes) so suggestions can refresh wording without going through the child's controls. Game over is spotted from data (a flag followed by a reset) rather than from the variable's name.

### T-403 – AI personalities
- **Status:** Done
- **Phase:** 4
- **Objective:** Offer the seven starting configurations of DESIGN.md §5.3 (Explorer, Collector, Completionist, Speedrunner, Battler, Survivor, Balanced Player) as reward configurations, not separate AIs, with every setting still changeable by the child.
- **Scope:** `RewardProfile.weights` (one multiplier per category, default 1.0, saved with the profile): the current value of a behaviour nobody has touched is its suggested value times its category weight, so suggestions and resets keep the personality in effect. `rewards.personalities`: `Personality(id, name, description, weights)`, the seven presets with child-friendly descriptions, `personality(id)`, `apply_personality(profile, personality)` (sets the name and weights, rescales untouched values, keeps values a person set and the enabled flags). **Out of scope:** picking a personality in the UI (T-406).
- **Acceptance criteria:**
  1. Seven presets with unique ids, names and descriptions; every weight positive; Balanced is all 1.0; unknown ids rejected.
  2. Applying a personality sets the profile's personality, rescales untouched values (clamped), keeps a person's values and every enabled flag; applying Balanced restores the suggested values.
  3. `suggest` and `reset_to_suggested` honour the weights; bad weights rejected; weights round-trip through JSON and default to 1.0 for older files.
  4. On the toy racer, a Survivor loses more for a crash than an Explorer, and a Speedrunner gains more for a checkpoint than a Collector.
- **Test plan:** preset table; lookup; apply rescale/keep/clamp; Balanced restore; suggest/reset with weights; validation; JSON; toy racer comparison.
- **Results:** 6 / 6 passed / 0 failed / 0 excluded (635 in the full suite). Lint, format and strict mypy clean.
- **Completion report:** The toy racer test caught that a newly suggested behaviour ignored the profile's weights; `apply_suggestions` now resets each new behaviour to its weighted value.

### T-404 – Reward service
- **Status:** Done
- **Phase:** 4
- **Objective:** Make rewards live: keep one profile per game on disk, fill it from the knowledge as discoveries arrive, turn every detected event into a reward the learner (Phase 5) can use, and keep a running tally the UI can show.
- **Scope:** `rewards.tally.RewardTally` (total, count, per-behaviour counts and sums, per-category sums, recent entries, `since(frame)`); `storage.rewards.RewardStore` (same shape as `KnowledgeStore`, sharing new `storage.jsonfiles` helpers); `services.rewards.RewardService(connection, discovery, bus, store)`: loads or creates the game's profile on connect (damaged file → new), applies suggestions on connect and on every `DiscoveryUpdated` that adds or changes beliefs, scores every event (`RewardEarned(frame, amount, behaviour_ids, event)` for non-zero amounts) into the tally, exposes the child's controls (`increase`, `decrease`, `set_value`, `set_enabled`, `reset_to_suggested`, `choose_personality`) which save and publish `RewardProfileChanged`, saves on disconnect; `AppContext.reward_store` and `AppContext.rewards`. **Out of scope:** exploitation detection (T-405), UI (T-406).
- **Acceptance criteria:**
  1. Tally sums, counts, groups and trims correctly; bad input rejected.
  2. Store creates its folder, round-trips, lists games, loads-or-creates and deletes; export and load anywhere; damaged files raise.
  3. Service: no profile until a memory-capable game connects; profile has the automatic behaviours for the current knowledge; new beliefs add behaviours and announce the change; events earn rewards into the tally and publish `RewardEarned`; disabled behaviours earn nothing.
  4. Controls change the profile, save it and announce; the profile (values, personality, enabled flags) is remembered on reconnect; a damaged file starts fresh; a different game starts fresh.
  5. The app context owns the service and the store under the data dir.
- **Test plan:** tally unit tests; store tests mirroring the knowledge store; service connect/filter/suggest/earn/controls/persist/damaged/other game; context wiring.
- **Results:** 15 / 15 passed / 0 failed / 0 excluded (650 in the full suite). Lint, format and strict mypy clean.
- **Completion report:** The knowledge store's file helpers were lifted into `storage.jsonfiles` so both stores share one atomic-write and safe-name path. Rewards are scored from `DiscoveryUpdated` events rather than a second detector, so the learner and the panel see exactly the events the engine saw.

### T-405 – Reward exploitation detection
- **Status:** Done
- **Phase:** 4
- **Objective:** Notice when the AI seems to be farming its rewards instead of getting anywhere (DESIGN.md §5.5), flag it, and explain in plain words why that might be happening and what the child can do.
- **Scope:** `rewards.exploitation`: `ExploitWarning(kind, behaviour_ids, share, window, explanation, suggestion)`; `ExploitDetector(profile, tally, window=3600, min_reward=20, dominant=0.7)` whose `check(now)` looks at the rewards of the last `window` frames and returns a warning when positive reward reached `min_reward` with nothing earned from a Progression or Completion behaviour: `farming` when one behaviour gave at least `dominant` of it, `no_progress` otherwise. No verdict while the profile has no Progression or Completion behaviour switched on (progress can't be judged yet). `RewardEntry` keeps per-behaviour shares. `RewardService` runs the check as events are scored, publishes `ExploitSuspected(warning)` at most once per window per cause, and keeps `warnings` / `current_warning` (cleared when the window passes without a repeat or on connect). **Out of scope:** showing warnings (T-406), automatic reward changes (never; the child decides).
- **Acceptance criteria:**
  1. One dominant positive behaviour with no progress in the window gives a `farming` warning naming it, with its share, an explanation and a suggestion; several behaviours give `no_progress`.
  2. Progress earned in the window, too little reward, entries outside the window, or no progress behaviour switched on give no warning; the dominant behaviour being progression itself gives no warning.
  3. Bad settings rejected.
  4. The service announces a warning once per window per cause, clears it when play moves on, and resets on connect.
- **Test plan:** detector scenarios (farming, no-progress, progress clears, too little, old entries, no progress behaviour, progression dominant); settings; service announce / no repeat / clear / reset.
- **Results:** 8 / 8 passed / 0 failed / 0 excluded (658 in the full suite). Lint, format and strict mypy clean.
- **Completion report:** `RewardEntry` now keeps each behaviour's share so the detector can tell progress reward apart inside a shared entry. The check runs on the frame of the newest scored event, so it costs nothing while no events arrive.

### T-406 – Rewards panel
- **Status:** Done
- **Phase:** 4
- **Objective:** Put the reward system in front of the child (DESIGN.md §5.1–§5.3, §5.5): behaviours grouped under categories that expand, each with its current reward, suggested reward and an on/off tick, + and - buttons, a "Use suggested" button, a personality picker, a running total, and the exploitation warning in plain words.
- **Scope:** `ui.rewards_panel.RewardsPanel(context)` in the main window's "Rewards" slot: personality combo (`personality_combo`), summary line (`rewards_summary`), a tree (`rewards_tree`) with one expandable row per category ("Survival (2)") and child rows "name | reward | suggested | on" where the tick box switches the behaviour on or off, `plus_button` / `minus_button` / `suggested_button` acting on the selected row, `rewards_warning` (the farming warning and its suggestion, empty when none), `rewards_note`. Driven by `RewardProfileChanged`, `RewardEarned`, `ExploitSuspected` and connect/disconnect through the GUI bridge; every control goes through `RewardService` so changes are saved. Testable API: `behaviour_ids()`, `row_text(id)`, `category_texts()`, `select(id)`, `selected_id`, `increase_selected()`, `decrease_selected()`, `use_suggested_selected()`, `set_selected_enabled()`, `choose_personality(id)`, `personality_id`, `summary_text`, `warning_text`, `note_text`, `refresh()`. **Out of scope:** goals and objectives (T-407/T-408), adding a behaviour by hand (later; suggestions cover discovered behaviours).
- **Acceptance criteria:**
  1. Idle: no rows, controls disabled, "No game to reward."; after disconnect the rows stay readable but the controls are off.
  2. Behaviours appear under their category in design order with reward, suggested value and tick; new discoveries add rows without losing the selection.
  3. + / - / Use suggested change the selected behaviour's reward (shown with a sign, and "you set this" once a person changed it); the tick switches it on or off; all through the service (saved, announced).
  4. The personality picker shows the current personality, applies a chosen one (values rescale) and shows "Custom" for a profile that has no preset.
  5. The total updates as rewards are earned; a farming warning shows its explanation and suggestion and clears when it lapses.
  6. The main window hosts the panel.
- **Test plan:** idle; rows and categories; selection kept across updates; plus/minus/suggested/tick; personality; total and warning; main window; disconnect.
- **Results:** 7 / 7 passed / 0 failed / 0 excluded (665 in the full suite). Lint, format and strict mypy clean.
- **Completion report:** After a disconnect the tree rows are disabled, so re-selecting the kept row no longer fires Qt's current-item signal; the panel now sets the control buttons' state itself at the end of every refresh. Windows CI then caught a use-after-free: ticking a row rebuilt the tree while Qt was still inside that row's signal. A refresh that arrives during an item signal is now deferred by one event-loop tick.

### T-407 – Goals and objectives model
- **Status:** Done
- **Phase:** 4
- **Objective:** Model DESIGN.md §4: an overall goal a child picks or writes, broken into objectives that each have a child-friendly description, a detection method, a reward, an optional penalty, a completion status, a confidence level, an on/off setting and dependencies on other objectives; plus the AI's first attempt at breaking a goal down from what it knows.
- **Scope:** `rewards.goals`: `Objective(id, description, trigger, reward, penalty, setback_trigger, completed, completed_frame, confidence, enabled, depends_on, notes)` with validation and JSON; `Goal(game_id, title)` with `add`/`get`/`remove`/`update`/`move`, `set_dependencies` (unknown ids and cycles rejected), `ready()` (enabled, not done, dependencies done), `next()`, `progress()`, `complete(id, frame)`, `reset_progress()`, `observe(event)` (completes ready objectives whose trigger fires, applies a penalty when a setback trigger fires before completion) returning `(reward, completed, penalised)`, `to_dict`/`from_dict` (schema 1); `GOAL_IDEAS` (the §4.1 examples); `suggest_goal(knowledge, title="")` builds objectives from believed progress variables ("Get further in Lap"), ordered by believed relationships (a cause comes before its effect) with confidence from the variable; `merge_suggestions(goal, knowledge)` adds missing suggested objectives without touching a person's edits. `storage.goals.GoalStore` on the shared JSON helpers. **Out of scope:** the service and panel (T-408, T-409).
- **Acceptance criteria:**
  1. Objective validation (blank description, bad reward, unknown confidence, self-dependency) and JSON round-trip.
  2. Goal add/duplicate/get/remove/update/move; dependencies reject unknown ids and cycles; removing an objective drops dependencies on it.
  3. `ready`/`next`/`progress` follow dependencies and the enabled flag; `observe` completes once, pays the reward, applies the penalty on a setback before completion only, ignores disabled or blocked objectives; `reset_progress` clears completion.
  4. Suggestions: one objective per believed progress variable with its confidence, ordered by relationships; merging twice changes nothing and keeps edits; the toy racer goal completes "Get further in Checkpoint" during a drive.
  5. Store round-trips, lists and deletes like the other stores.
- **Test plan:** objective validation/JSON; goal container and dependencies; readiness and observe; suggestions and merge; toy racer; store.
- **Results:** 9 / 9 passed / 0 failed / 0 excluded (674 in the full suite). Lint, format and strict mypy clean.
- **Completion report:** Dependencies may point at objectives later in the list, so loading adds every objective first and sets dependencies second. The penalty is defined as a setback that happens before the objective is done (DESIGN.md §4.2 "optional penalty"), which is the one reading that can't punish a finished step.

### Later Phase 4 tasks
T-402 to T-409 as listed in the summary table; each gets its own section when started.

### Later Phase 3 tasks
None outstanding; refinements will be filed as they come up on real games.

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
- T-203 completed (PR #16). NumPy is now a runtime dependency (ADR-008).
- T-204 completed (PR #17).
- T-205 completed (PR #18). Phase 2 complete on the fake emulator.
- Phase 3 broken into T-301..T-308. T-301 completed (PR #19).
- T-302 completed (PR #20).
- T-303 completed (PR #21).
- T-304 completed (PR #22).
- T-305 completed (PR #23).
- T-306 completed (PR #24).
- T-307 completed (PR #25).
- T-308 completed (PR #26). Phase 3 complete on the fake emulator.
- Phase 4 broken into T-401..T-408. T-401 completed (PR #27).
- T-402 completed (PR #28).
- T-403 completed (PR #29).
- T-404 completed (PR #30).
- T-405 completed (PR #31).
- T-406 completed (PR #32).
- T-407 completed (PR #33).
- Blockers: none for T-001 to T-012. ADR-005 / ADR-006 need user input before real-emulator work.
