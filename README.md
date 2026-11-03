# Retro AI Lab

An emulator-based machine learning sandbox that lets children train an AI to play retro video games by choosing goals and adjusting rewards, with no code and no hexadecimal.

> Status: **alpha.** Phases 1–8 of [`DESIGN.md`](DESIGN.md) are implemented and tested on the built-in practice console (Toy Racer); the BizHawk and mGBA bridges are tested against simulators and wait for a run on a real emulator (see `docs/`). Task-by-task detail is in [`TASK_TRACKER.md`](TASK_TRACKER.md).

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

## Setting up on Windows

### Option A: the built app (no Python needed)

Every change to `main` builds a Windows app on GitHub. Signed in to GitHub,
open the repository's **Actions** tab, click the newest green **Package** run,
and download **RetroAILab-windows** under *Artifacts* at the bottom of the page.
Unzip it anywhere and run `RetroAILab.exe` (SmartScreen may ask once because
the app is unsigned: *More info > Run anyway*). The emulator bridge scripts are
inside that folder at `_internal\retro_ai_lab\emulator\bizhawk\retro_ai_lab.lua`
and `_internal\retro_ai_lab\emulator\mgba\retro_ai_lab.lua`. GitHub keeps each
build for 14 days, so fetch a fresh one when it is gone.

### Option B: from source with Python

If the PC cannot reach PyPI (pip times out), use option A instead. Once:

1. Install Python 3.11 or newer from <https://www.python.org/downloads/> and
   tick **Add python.exe to PATH** in the installer.
2. Get this repository onto the PC (GitHub's **Code > Download ZIP**, unzipped,
   or `git clone`).
3. Open a Command Prompt in that folder (type `cmd` in the folder's address bar
   in Explorer) and run these three lines one at a time, pressing Enter after
   each and waiting for the prompt to come back. The second takes a minute or
   two and prints a lot.

   ```
   python -m venv .venv
   .venv\Scripts\activate
   pip install -e .
   ```

Every time after that, in a Command Prompt in the same folder:

```
.venv\Scripts\activate
retro-ai-lab
```

The desktop window opens with the console box at the top. To play on a real
emulator, follow [docs/BIZHAWK.md](docs/BIZHAWK.md) or
[docs/MGBA.md](docs/MGBA.md).

If the app does not appear or closes on its own, every run writes a log at
`%LOCALAPPDATA%\RetroAILab\logs\retro-ai-lab.log` (paste `%LOCALAPPDATA%\RetroAILab\logs`
into Explorer's address bar); send that file with your report.

## Running it (developers)

```bash
python -m venv .venv && . .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
retro-ai-lab            # opens the desktop app
pytest                  # runs the test suite (headless Qt)
```

On Linux, Qt needs `libegl1 libopengl0 libxkbcommon-x11-0 libxcb-cursor0 libdbus-1-3 libfontconfig1`; see `.github/workflows/ci.yml`.

## Playing on BizHawk or mGBA

The app connects to BizHawk's EmuHawk, and to mGBA, through shipped Lua
scripts. See [docs/BIZHAWK.md](docs/BIZHAWK.md) and [docs/MGBA.md](docs/MGBA.md)
for setup and for the opt-in real-emulator tests that verify each bridge on
your machine.

## Windows build

Every push to `main` runs the **Package** workflow, which builds `RetroAILab.exe` with PyInstaller, launches it as a smoke test and uploads the folder as the `RetroAILab-windows` artifact (Actions tab → latest Package run → Artifacts). To build locally: `pip install pyinstaller && pyinstaller packaging/retro_ai_lab.spec --noconfirm`.

## Planned stack

Python 3.11+, PySide6 (Qt) desktop UI, NumPy for screen and memory analysis, PyTorch and Gymnasium for learning, SQLite and JSON for storage. Target platform is Windows; the code is kept cross-platform so it can be developed and tested on Linux CI.

## Legal

Retro AI Lab ships no ROMs and never modifies ROM files. Use games you legally own.
