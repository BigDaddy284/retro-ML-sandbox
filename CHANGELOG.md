# Changelog

All notable changes are recorded here, one entry per completed task. Format loosely follows [Keep a Changelog](https://keepachangelog.com/).

## [Unreleased]

### Added
- T-703: Feature vectors: a `FeatureMap` turns a situation into a fixed-length vector (present flag and scaled bucket per believed variable) for the coming network learners, and saves its slot order.
- T-702: Experience replay: a bounded, seeded memory of past steps (`Transition`, `ReplayBuffer`) that saves and reloads with the brain.
- T-701: Every learning algorithm sits behind one `Learner` contract with a registry by name; saved brains record their algorithm (older brains load as `q_table`); the training service takes `algorithm=`.
- T-607: "What's this?" on every panel opens a shared tips window explaining the panel and the ideas behind it (AI, machine learning, reinforcement learning, rewards, neural networks, training, decisions, exploring, reward exploitation, learning in steps) in simple or grown-up words.
- T-606: First working demonstration on the Fake Emulator (DESIGN.md §17) as an end-to-end test; counters left out of the AI's view of the game; stale suggested goal steps retired; a checkpoint restore no longer ends a go as a "reset"; PySide6 is kept below 6.12 on Python 3.11 (6.12.0 leaks a reference per signal and aborts after a few thousand events) with a guard test.
- T-605: The Learning panel says how the AI's favourite moves changed over recent goes and suggests what to try next.
- T-604: The activity panel says what the AI is pressing and why (trying something new or its best known move), its last moves, and the goal step it is working on; `TrainingStepped` events.
- T-603: The Progress panel counts training tries, how often a goal was reached, and the trend.
- T-602: Learning graph in the Learning panel: a dot per go, the rolling average line, the best go ringed, and a caption in plain words.
- T-601: Learning curve (points per go, rolling average, best and longest go, trend, success rate, described in plain words); goes record how many objectives they completed.
- T-507: Learning panel (who plays, speed, Start / Pause / Stop, Start from scratch, what the AI is pressing and knows, recent goes); the play clock and pad yield to the AI while it plays.
- T-506: The AI's brain (Q-table, settings, exploration, random state and the history of goes) is saved per game after every go and on disconnect, and restored on connect; `forget()` starts from scratch.
- T-505: Training service with independent, guided and manual (learn from the child) modes, start / pause / resume / stop, a history of goes, checkpoint restarts and fast or live stepping; wired into the app context.
- T-504: Episode runner: the agent acts, frames advance, rewards announced by the services are collected and learned from; episodes end on the game's reset, a step limit or link loss; checkpoint restarts; Start pressed when stuck.
- T-503: Tabular Q-learning agent: epsilon-greedy choice, textbook update, exploration decaying per episode, plain-words stats, JSON round-trip that refuses mismatched controls.
- T-502: Action space built from the emulator's buttons: do nothing, singles, direction + button combos; pause buttons left out; JSON round-trip.
- T-501: State encoder: believed variables decoded from a snapshot and binned into a small, stable state for the learner.
- T-409: Goals panel in the Progress slot: name the goal, see each step with its detection, confidence and status, add steps (detected or by hand), reorder, tick off, set what a step comes after, switch on or off, start again.
- T-408: Goal service: one saved goal per game with suggested objectives added as the AI learns, objectives ticked off from game events, setback penalties, and child edits that are saved and announced.
- T-407: Goals and objectives model: an overall goal broken into objectives with detection triggers, reward, optional setback penalty, completion, confidence, on/off and dependencies; suggested objectives from believed progress; a per-game goal store.
- T-406: Rewards panel: behaviours under expandable categories with reward, suggested value and on/off tick, + / - / Use suggested buttons, personality picker, running total and the farming warning.
- T-405: Reward exploitation detection: a plain-words warning (with a suggestion) when the AI keeps earning reward without any progress, announced once per window as `ExploitSuspected`.
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
