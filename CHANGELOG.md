# Changelog

All notable changes are recorded here, one entry per completed task. Format loosely follows [Keep a Changelog](https://keepachangelog.com/).

## [Unreleased]

### Added
- T-404: Reward service: one saved reward profile per game, filled from the knowledge as discoveries arrive, every detected event scored into a running tally and announced as `RewardEarned`; child controls save and announce changes.
- T-403: Seven AI personalities (Explorer, Collector, Completionist, Speedrunner, Battler, Survivor, Balanced Player) as per-category reward weights on the profile; untouched values follow them.
- T-402: Suggested behaviours and reward values derived from the game knowledge (kind, confidence and what leads to progress), applied to a profile without overriding a person's choices.
- T-401: Reward model: behaviours filed under the eight categories with a trigger, +/- controls, current and suggested values, enable toggle and JSON round-trip; packages now import cleanly in any order.
- T-308: Knowledge is saved per game in the user's app data folder, restored on reconnect, and can be shared and loaded from the panel.
- T-307: "What the AI is doing" panel: discoveries with plain-words confidence badges and evidence, yes/no/rename answers, on-demand button experiments, recent events and relationships.
- T-306: Discovery service: automatic event/relationship tracking, scheduled reviews and button experiments while a game is connected, with progress and update events.
- T-305: Relationship discovery: cause/effect links between events with rates and confidence, stored in the knowledge (schema 2) and found live by `RelationshipFinder`.
- T-304: Event detection from believed variables: progress, score gains, setbacks, recoveries, flags and resets, with an event log.
- T-303: Input linking: experiments that find which bytes each button moves (excluding unstable bytes) and fold the result into the knowledge as evidence, relabels and proposals.
- T-302: Variable candidate finder: proposes counters, scores, lives, flags, progress, position and speed bytes (with 16-bit pairing) from change statistics, with evidence.
- T-301: Knowledge model (variables with evidence and derived confidence, user confirm/correct/reject) and a per-game JSON knowledge store with export/import.
- T-205: Save-state experiment harness: repeatable trials from a checkpoint with memory and screen diffs, and trial comparison to find input-dependent bytes.
- T-204: Input recording: `InputLog` of held-button changes per frame and source, `InputRecorded` events, pad inputs recorded as human, inputs stored per session in the archive.
- T-203: Pixel-level frame diffs with grid cell counts, a screen activity tracker, `ScreenObserved` events fed by the game panel; NumPy dependency.
- T-202: Byte-level memory diffs, per-byte change statistics with noise/state/constant classification, and an observation service fed by the play clock.
- T-201: Memory snapshots (`MemorySnapshot`, `MemoryCapture`, `SnapshotRing`) and a SQLite `SnapshotArchive`.
- T-012: PyInstaller spec, launch smoke-test script and a Windows package workflow that uploads a downloadable build.
- T-011: Disconnection handling: link watchdog, resilient disconnect, one-click Reconnect, status-bar messages.
- T-010: Manual play: on-screen controller pad built from the emulator's buttons, keyboard mapping, and a Play/Pause clock that advances frames.
- T-009: Connect/disconnect toolbar with background connect, plain-English capability line, and `EventBridge` delivering service events on the GUI thread.
- T-008: Game panel shows live emulator frames (`FrameView`, `FramePump`, `AppContext`).
- T-007: PySide6 application shell: main window with the five collapsible panels, status bar, `retro-ai-lab` launches the UI, hidden `--exit-after-ms` for smoke tests.
- T-006: `EventBus`, `AdapterRegistry` and `ConnectionService` (state machine with connected/disconnected/failed events, dropped-link detection).
- T-005: `FakeEmulatorAdapter` (full contract over Toy Racer, with fault injection) and the reusable `AdapterContract` test suite.
- T-004: Toy Racer, a deterministic synthetic racing game with RAM, screen rendering and save/restore, used to test every layer without an emulator.
- T-003: `EmulatorAdapter` base class enforcing connection, capability and argument checks for every backend.
- T-002: Core domain types (`Button`, `Capabilities`, `MemoryRegion`, `GameInfo`, `Frame`) with validation and JSON round-trip, and the error hierarchy.
- T-001: Python project scaffold (`src/retro_ai_lab`), CLI with `--version`, pytest/ruff/mypy configuration, GitHub Actions CI on Windows and Linux.
- T-000: Project documentation baseline: README, DEVELOPMENT_RULES, TASK_TRACKER, TESTING_STRATEGY, ARCHITECTURE, CHANGELOG, `.gitignore`.
