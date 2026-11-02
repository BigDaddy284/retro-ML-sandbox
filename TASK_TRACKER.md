# Task Tracker

Single source of progress. Follows the template in `DEVELOPMENT_RULES.md` §3.
Phases refer to DESIGN.md §16. Phases 1–2 are broken into small tasks now; later phases are listed as epics and will be split into tasks before work on them starts.

## Status

- **Current task:** T-005 (Ready)
- **Last approved checkpoint:** `main` after T-004
- **Open blockers:** none. Two decisions are *Proposed* in `ARCHITECTURE.md` (ADR-005 first emulator protocol, ADR-006 first demo game); they block only T-110 onward and T-200 onward respectively.

## Summary

| ID | Title | Phase | Status | Tests (pass/total) | Commit |
|---|---|---|---|---|---|
| T-000 | Project documentation baseline | 0 | Done | n/a (docs only) | 29249db (#1) |
| T-001 | Python project scaffold, tooling and CI | 1 | Done | 13/13 | see PR #2 |
| T-002 | Core domain types and errors | 1 | Done | 73/73 | see PR #3 |
| T-003 | Emulator adapter interface and capabilities model | 1 | Done | 72/72 | see PR #4 |
| T-004 | Synthetic game (deterministic toy racer) | 1 | Done | 39/39 | see PR #5 |
| T-005 | Fake emulator adapter over the synthetic game | 1 | Ready | – | – |
| T-006 | Adapter registry and connection service | 1 | Ready | – | – |
| T-007 | App shell: main window launches with empty panels | 1 | Ready | – | – |
| T-008 | Game window panel shows live frames | 1 | Ready | – | – |
| T-009 | Connect / disconnect UI with capability display | 1 | Ready | – | – |
| T-010 | Manual controller input from UI to adapter | 1 | Ready | – | – |
| T-011 | Graceful handling of emulator disconnection | 1 | Ready | – | – |
| T-012 | Windows packaging smoke test (launchable build) | 1 | Ready | – | – |
| T-110 | Spike: confirm BizHawk Lua ↔ socket protocol | 1 | Blocked on ADR-005 | – | – |
| T-111 | BizHawk adapter: connect and handshake | 1 | Ready after T-110 | – | – |
| T-112 | BizHawk adapter: capabilities and game info | 1 | – | – | – |
| T-113 | BizHawk adapter: frame capture | 1 | – | – | – |
| T-114 | BizHawk adapter: controller input | 1 | – | – | – |
| T-115 | BizHawk adapter: memory reads | 1 | – | – | – |
| T-116 | BizHawk adapter: frame advance, speed, save states | 1 | – | – | – |
| T-117 | BizHawk adapter: disconnect handling and real-emulator verification | 1 | – | – | – |
| T-201 | Memory snapshot capture and storage | 2 | – | – | – |
| T-202 | Memory change detection (frame-to-frame diff) | 2 | – | – | – |
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
- **Objective:** `FakeEmulatorAdapter` implementing the full adapter contract, including save/load state and speed, with configurable capability flags and fault injection (timeouts, disconnects).
- **Test plan:** full contract test suite (reused later for every real adapter); save/load state restores RAM exactly; disabled capabilities raise; injected disconnect surfaces as `EmulatorDisconnected`.

### T-006 – Adapter registry and connection service
- **Objective:** Discover available adapters, connect by name, expose session state and events to upper layers without Qt.
- **Test plan:** registry lists fake adapter; unknown name error; connect/disconnect state machine including double-connect and disconnect-when-idle; events emitted in order.

### T-007 – App shell
- **Objective:** `python -m retro_ai_lab` opens a main window with the five panels from DESIGN.md §10 (Game, AI Activity, Rewards, Progress, Learning) as placeholders, each expandable.
- **Test plan (pytest-qt):** window opens and closes cleanly; all five panels present with child-friendly titles; expand/collapse works; app exits 0.

### T-008 … T-012
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
- Next: T-005.
- Blockers: none for T-001 to T-012. ADR-005 / ADR-006 need user input before real-emulator work.
