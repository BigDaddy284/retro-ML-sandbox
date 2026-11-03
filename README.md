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

## Running it

```bash
python -m venv .venv && . .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
retro-ai-lab            # opens the desktop app
pytest                  # runs the test suite (headless Qt)
```

On Linux, Qt needs `libegl1 libopengl0 libxkbcommon-x11-0 libxcb-cursor0 libdbus-1-3 libfontconfig1`; see `.github/workflows/ci.yml`.

## Playing on BizHawk

The app connects to BizHawk's EmuHawk through a shipped Lua script. See
[docs/BIZHAWK.md](docs/BIZHAWK.md) for setup and for the opt-in real-emulator
test that verifies the bridge on your machine.

## Windows build

Every push to `main` runs the **Package** workflow, which builds `RetroAILab.exe` with PyInstaller, launches it as a smoke test and uploads the folder as the `RetroAILab-windows` artifact (Actions tab → latest Package run → Artifacts). To build locally: `pip install pyinstaller && pyinstaller packaging/retro_ai_lab.spec --noconfirm`.

## Planned stack

Python 3.11+, PySide6 (Qt) desktop UI, NumPy for screen and memory analysis, PyTorch and Gymnasium for learning, SQLite and JSON for storage. Target platform is Windows; the code is kept cross-platform so it can be developed and tested on Linux CI.

## Legal

Retro AI Lab ships no ROMs and never modifies ROM files. Use games you legally own.
