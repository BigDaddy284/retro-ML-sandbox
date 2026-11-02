# Architecture

Living description of how Retro AI Lab is structured. Updated whenever a task changes module boundaries or interfaces.

## 1. Layers

```
┌──────────────────────────────────────────────────────────────┐
│ ui/            PySide6 views + view-models (no business logic) │
├──────────────────────────────────────────────────────────────┤
│ services/      Session orchestration: connect, observe, train │
├──────────┬──────────────┬─────────────┬──────────────────────┤
│ training/│ learning/    │ rewards/    │ understanding/       │
│ modes,   │ algorithms   │ rewards,    │ variable discovery,  │
│ runners, │ (Q-learning, │ objectives, │ events, relations,   │
│ history  │ DQN, ...)    │ personalities│ confidence          │
├──────────┴──────────────┴─────────────┴──────────────────────┤
│ observation/   frames, memory snapshots, diffs, input logs   │
├──────────────────────────────────────────────────────────────┤
│ emulator/      Adapter interface + adapters (BizHawk, VBA-M,  │
│                Fake)                                          │
├──────────────────────────────────────────────────────────────┤
│ storage/       SQLite + JSON: projects, game knowledge,       │
│                checkpoints, training history                  │
└──────────────────────────────────────────────────────────────┘
core/  shared domain types, events, errors, config (used by every layer)
```

Dependencies point downward only. `ui` depends on `services`; nothing below `services` imports Qt.

## 2. Package layout (planned)

```
src/retro_ai_lab/
  core/            types, errors, event bus, settings
  emulator/
    base.py        EmulatorAdapter protocol, Capabilities, Button, Frame, MemoryRegion
    fake/          synthetic game + FakeEmulatorAdapter
    bizhawk/       adapter + Lua bridge script
    vbam/          adapter (later)
  observation/
  understanding/
  rewards/
  learning/
  training/
  storage/
  services/
  ui/
  testing/         reusable fakes for tests and demo mode
tests/
  unit/ integration/ emulator/ real_emulator/ ui/ learning/
```

## 3. Emulator adapter contract (draft)

Every adapter implements one interface and reports what it can do, so callers never assume a feature exists (DESIGN.md §7).

```python
class EmulatorAdapter(Protocol):
    def connect(self, timeout_s: float) -> None: ...
    def disconnect(self) -> None: ...
    @property
    def is_connected(self) -> bool: ...
    def capabilities(self) -> Capabilities: ...          # flags: memory, screen, input, frame_advance, speed, save_states
    def game_info(self) -> GameInfo: ...                  # system, title/hash if exposed
    def buttons(self) -> Sequence[Button]: ...
    def memory_regions(self) -> Sequence[MemoryRegion]: ...
    def read_memory(self, region: str, offset: int, length: int) -> bytes: ...
    def capture_frame(self) -> Frame: ...
    def set_input(self, pressed: frozenset[Button]) -> None: ...
    def advance_frames(self, n: int = 1) -> None: ...
    def set_speed(self, multiplier: float) -> None: ...
    def save_state(self, slot: str) -> None: ...
    def load_state(self, slot: str) -> None: ...
```

Calling an unsupported operation raises `UnsupportedCapability`. Disconnects raise `EmulatorDisconnected`, which the service layer handles.

## 4. Decision log

Status: **Accepted** (in force), **Proposed** (needs user sign-off before work depends on it irreversibly).

| ID | Decision | Status | Reason |
|---|---|---|---|
| ADR-001 | Python 3.11+, PySide6, PyTorch, Gymnasium, SQLite/JSON | Accepted | Specified in DESIGN.md §14. |
| ADR-002 | `src/` layout, single package `retro_ai_lab`, layered as above | Accepted | Keeps components replaceable (DESIGN.md §14). |
| ADR-003 | Develop and CI on Linux + Windows; Windows is the release target | Accepted | Development runs in Linux containers; Windows CI guards the target platform. |
| ADR-004 | A deterministic Fake Emulator with a synthetic game is a first-class adapter | Accepted | No emulator or ROMs are available in CI; needed to test every layer and the first demo honestly. |
| ADR-005 | BizHawk / EmuHawk is the first real adapter, via a Lua script talking to the app over a local TCP socket | Proposed | BizHawk exposes memory, screenshots, joypad, frame advance and save states to Lua and supports socket comms. Exact protocol to be confirmed by a spike task. |
| ADR-006 | First real demo game: a simple racing or arcade title the user owns | Proposed | DESIGN.md §17 recommends a racing game first. The user chooses the specific game since no ROMs are supplied. |
| ADR-007 | Real-emulator tests are opt-in (env vars) and backed by manual verification on the user's Windows PC | Accepted | Only honest way to cover emulator behaviour that CI can't run. |
