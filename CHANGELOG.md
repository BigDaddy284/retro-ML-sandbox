# Changelog

All notable changes are recorded here, one entry per completed task. Format loosely follows [Keep a Changelog](https://keepachangelog.com/).

## [Unreleased]

### Added
- T-008: Game panel shows live emulator frames (`FrameView`, `FramePump`, `AppContext`).
- T-007: PySide6 application shell: main window with the five collapsible panels, status bar, `retro-ai-lab` launches the UI, hidden `--exit-after-ms` for smoke tests.
- T-006: `EventBus`, `AdapterRegistry` and `ConnectionService` (state machine with connected/disconnected/failed events, dropped-link detection).
- T-005: `FakeEmulatorAdapter` (full contract over Toy Racer, with fault injection) and the reusable `AdapterContract` test suite.
- T-004: Toy Racer, a deterministic synthetic racing game with RAM, screen rendering and save/restore, used to test every layer without an emulator.
- T-003: `EmulatorAdapter` base class enforcing connection, capability and argument checks for every backend.
- T-002: Core domain types (`Button`, `Capabilities`, `MemoryRegion`, `GameInfo`, `Frame`) with validation and JSON round-trip, and the error hierarchy.
- T-001: Python project scaffold (`src/retro_ai_lab`), CLI with `--version`, pytest/ruff/mypy configuration, GitHub Actions CI on Windows and Linux.
- T-000: Project documentation baseline: README, DEVELOPMENT_RULES, TASK_TRACKER, TESTING_STRATEGY, ARCHITECTURE, CHANGELOG, `.gitignore`.
