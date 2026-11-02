# Task Tracker

Single source of progress. Follows the template in `DEVELOPMENT_RULES.md` §3.
Phases refer to DESIGN.md §16. Phases 1–2 are broken into small tasks now; later phases are listed as epics and will be split into tasks before work on them starts.

## Status

- **Current task:** T-002 (Ready)
- **Last approved checkpoint:** `main` after T-001
- **Open blockers:** none. Two decisions are *Proposed* in `ARCHITECTURE.md` (ADR-005 first emulator protocol, ADR-006 first demo game); they block only T-110 onward and T-200 onward respectively.

## Summary

| ID | Title | Phase | Status | Tests (pass/total) | Commit |
|---|---|---|---|---|---|
| T-000 | Project documentation baseline | 0 | Done | n/a (docs only) | 29249db (#1) |
| T-001 | Python project scaffold, tooling and CI | 1 | Done | 13/13 | see PR #2 |
| T-002 | Core domain types and errors | 1 | Ready | – | – |
| T-003 | Emulator adapter interface and capabilities model | 1 | Ready | – | – |
| T-004 | Synthetic game (deterministic toy racer) | 1 | Ready | – | – |
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
- **Objective:** Shared immutable types (`Button`, `Frame`, `MemoryRegion`, `GameInfo`, `Capabilities`) and the error hierarchy (`RetroAILabError`, `EmulatorError`, `EmulatorDisconnected`, `UnsupportedCapability`, `ProtocolError`).
- **Test plan:** construction and validation (negative sizes, empty names, frame buffer length vs width×height×channels), immutability, equality/hash, error hierarchy `isinstance` relationships, serialisation round trip.

### T-003 – Emulator adapter interface and capabilities model
- **Objective:** The `EmulatorAdapter` protocol from ARCHITECTURE.md §3 plus a base class that enforces capability checks.
- **Test plan:** a minimal test adapter satisfies the protocol; calling an unsupported operation raises `UnsupportedCapability` with a clear message; operations while disconnected raise `EmulatorDisconnected`; capability flags round-trip to JSON.

### T-004 – Synthetic game (deterministic toy racer)
- **Objective:** A tiny, pure-Python game with a screen buffer and a byte-addressable RAM holding position, speed, lap, checkpoint, lives and score at fixed but *undocumented-to-callers* addresses, so later discovery tasks have something real to find.
- **Test plan:** determinism (same seed + inputs ⇒ identical RAM and frames); each button has its intended effect; lap and checkpoint progression; crash/life loss; game over and reset; RAM values stay in range; frame rendering size and content sanity; performance (≥ 2,000 steps/s headless).

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
- Next: T-002.
- Blockers: none for T-001 to T-012. ADR-005 / ADR-006 need user input before real-emulator work.
