# Retro AI Lab

An emulator-based machine learning sandbox that lets children train an AI to play retro video games by choosing goals and adjusting rewards, with no code and no hexadecimal.

> Status: **pre-alpha, planning complete, implementation not started.** See [`TASK_TRACKER.md`](TASK_TRACKER.md).

## Documents

| Document | Purpose |
|---|---|
| [`DESIGN.md`](DESIGN.md) | Product vision and requirements (source of truth for *what* to build) |
| [`CLAUDE.md`](CLAUDE.md) | Rules for the autonomous developer (source of truth for *how* to build) |
| [`DEVELOPMENT_RULES.md`](DEVELOPMENT_RULES.md) | Day-to-day workflow, task template, Git workflow, stop conditions |
| [`TASK_TRACKER.md`](TASK_TRACKER.md) | Task list, current task, test results, commits, blockers |
| [`TESTING_STRATEGY.md`](TESTING_STRATEGY.md) | Test architecture and quality gates |
| [`ARCHITECTURE.md`](ARCHITECTURE.md) | Module layout, interfaces and recorded architecture decisions |
| [`CHANGELOG.md`](CHANGELOG.md) | Notable changes per task |

## Planned stack

Python 3.11+, PySide6 (Qt) desktop UI, PyTorch and Gymnasium for learning, SQLite and JSON for storage. Target platform is Windows; the code is kept cross-platform so it can be developed and tested on Linux CI.

## Legal

Retro AI Lab ships no ROMs and never modifies ROM files. Use games you legally own.
