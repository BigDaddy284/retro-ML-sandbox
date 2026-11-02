# Testing Strategy

How Retro AI Lab is tested, and the quality gates every task must pass.

## 1. Tooling

| Purpose | Tool |
|---|---|
| Test runner | `pytest` |
| Qt UI tests | `pytest-qt`, run with `QT_QPA_PLATFORM=offscreen` |
| Property / fuzz tests | `hypothesis` (parsers, protocol framing, reward maths) |
| Coverage | `pytest-cov` (reported per task; no global percentage gate replaces real criteria coverage) |
| Lint / format | `ruff` |
| Types | `mypy` |
| CI | GitHub Actions on `windows-latest` (target platform) and `ubuntu-latest` |

Single command for the full regression suite: `pytest` (plus `ruff check .` and `mypy src`).

## 2. Test layers

### 2.1 Unit tests (`tests/unit/`)
Pure logic with no I/O: reward calculation, objective graphs, confidence scoring, memory-diff analysis, Q-table updates, serialisation round trips. Fast (whole layer under ~30 s). Every public function covers normal input, boundaries, invalid input and failure paths.

### 2.2 Integration tests (`tests/integration/`)
Components wired together through their real interfaces: adapter ↔ observation pipeline, rewards ↔ learning loop, storage ↔ SQLite on a temp file. Use the **Fake Emulator** (below) instead of a real emulator.

### 2.3 Emulator protocol tests (`tests/emulator/`)
Exercise each adapter's wire protocol against a scripted fake peer (a local socket server that replays recorded or hand-written emulator messages). Cover: connect, handshake, capability report, frame capture, input send, memory read, save state, timeouts, malformed messages, mid-session disconnect, reconnect.

### 2.4 Real-emulator tests (`tests/real_emulator/`, marker `real_emulator`)
Run against an installed BizHawk / VBA-M on Windows with a user-supplied ROM. These cannot run in CI (no emulator, no ROMs). They are collected only when `RETRO_AI_LAB_EMULATOR_PATH` and `RETRO_AI_LAB_TEST_ROM` are set. When a task's acceptance criteria depend on real-emulator behaviour, the task's results must record either a passing real run or an explicit exclusion plus a manual verification checklist for the user.

### 2.5 UI tests (`tests/ui/`)
`pytest-qt` drives widgets offscreen: panels render, controls change view-model state, long operations don't block the UI thread, error states display a child-friendly message.

### 2.6 Learning tests (`tests/learning/`)
Deterministic (fixed seeds) checks that algorithms actually learn on synthetic environments: e.g. tabular Q-learning reaches the goal in a small grid within N episodes; changing the reward changes the learned policy (DESIGN.md §17 item 10). Marked `slow` where needed but still mandatory in the regression gate.

### 2.7 Manual verification
For anything not automatable (look and feel, real-emulator behaviour on the user's PC), the task records a numbered manual procedure with expected results. A task depending on it is `Done` only after the procedure has been carried out and its outcome recorded.

## 3. Test doubles

- **Fake Emulator adapter** – an in-process implementation of the emulator adapter interface backed by a small deterministic **synthetic game** (a toy racing / grid game with real "memory" bytes for position, speed, lap, lives, score). It lets every layer above the adapter be tested end to end, including the Game Understanding Engine discovering those variables without being told where they are.
- **Scripted protocol peer** – replays message sequences for adapter protocol tests, including faults.
- Fakes live in `src/retro_ai_lab/testing/` so they are importable by tests and by the demo mode, and have their own tests.

## 4. Coverage of each task's test plan

Each task's test plan must address, where relevant:
normal operation · expected behaviour · invalid input · unexpected input · edge cases · failure handling · data integrity · integration · regression · performance · UI behaviour.

"Not relevant" is allowed per item but must be stated, not omitted.

## 5. Quality gates

A task may be marked `Done` only when all of these hold:

1. Every test in the task's plan exists and passes (100%).
2. The full regression suite passes (100%), including lint and type checks.
3. CI is green on both Windows and Linux.
4. Any exclusions are recorded with justification and a covering manual procedure.
5. Every acceptance criterion maps to at least one test or manual check.
6. The application still launches (smoke test, from the task that creates the app shell onward).
7. Docs (`TASK_TRACKER.md`, `CHANGELOG.md`, and `ARCHITECTURE.md` if structure changed) are updated.

## 6. Prohibited practices

Skipping or disabling failing tests · `xfail` on a real failure · weakening assertions · swallowing exceptions to make a test pass · test-only branches in production code · marking unfinished functionality complete · moving forward with a known regression.

## 7. Flaky tests

A flaky test is a failing test. Fix the cause (usually timing, ordering or unseeded randomness). Use fixed seeds and injected clocks; never add retries to hide flakiness.
