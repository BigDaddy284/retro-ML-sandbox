# Playing on a real console: mGBA

Retro AI Lab talks to mGBA (Game Boy, Game Boy Color and Game Boy Advance)
through a small Lua script that ships with the app, the same way it talks to
BizHawk (see [BIZHAWK.md](BIZHAWK.md)). mGBA runs the game; the script listens
on a local port; the app connects, reads memory, takes screenshots, presses
buttons, waits for frames and saves states through it. Nothing is installed
into mGBA and nothing leaves the machine (the script only accepts connections
from 127.0.0.1).

## What you need

* **mGBA 0.10 or newer** (the first version with scripting). Download from
  <https://mgba.io/downloads.html>.
* A Game Boy or Game Boy Advance ROM you own.
* Retro AI Lab running on the same PC. There is no exe yet: follow **Setting
  up on Windows** in the [README](../README.md) once, then start it with
  `.venv\Scripts\activate` and `retro-ai-lab` from the checkout folder.

## Starting a session

1. Open mGBA and load the ROM (File > Load ROM).
2. Open the scripting window: **Tools > Scripting...**, then in that window
   **File > Load script**, browse to the checkout and pick
   `src\retro_ai_lab\emulator\mgba\retro_ai_lab.lua`. No Lua knowledge is
   needed. The scripting console prints
   `Retro AI Lab is waiting for the app on port 43110.` Leave the window open.
3. In Retro AI Lab pick **mGBA** in the console box and press **Connect**. The
   status line shows the game's name.
4. Disconnect in the app, or close the scripting window, to hand the game back.

If the app says *Couldn't find mGBA on port 43110*, mGBA is not running the
script yet (step 2), or something else uses that port. To use another port, set
the `RETRO_AI_LAB_PORT` environment variable before starting mGBA and tell the
app the same port.

## How mGBA differs from BizHawk

mGBA's scripts can't pause the emulator, so the game **keeps running between
the app's steps** ("free run" in ARCHITECTURE.md §5.3). A training step still
holds the buttons for exactly the frames asked and reads memory when they have
passed; what differs is that the game also moves while the AI is thinking. On
a fast PC that is a frame or two per step. mGBA has no speed control from a
script either, so the app's speed setting has no effect; use mGBA's own
fast-forward if you want training to run faster.

## Verifying the bridge (what the developer needs from you)

The bridge is tested in CI against a Lua runtime, but a few things can only be
checked on a real mGBA (ARCHITECTURE.md §6.4): that memory reads use bus
addresses, that save states work without flags, what the socket reports when
the app disconnects, and how fast a step is. The quickest way to check them is
the opt-in test module. In a checkout, with mGBA closed:

```
set RETRO_AI_LAB_MGBA_PATH=C:\mGBA\mGBA.exe
set RETRO_AI_LAB_TEST_ROM=C:\roms\game.gba
pytest tests/real_emulator/test_mgba.py -s -p no:cacheprovider
```

It launches mGBA with `--script` pointing at the shipped script, connects, and
prints what it measured. If your mGBA does not accept `--script`, start mGBA
yourself, load the script from Tools > Scripting, and run the same command with
`set RETRO_AI_LAB_MGBA_RUNNING=1` so the test connects to it instead. Please
paste the whole output, including any failure, into the project chat together
with the mGBA version. The scripting window usually shows the matching error
message; include that too.

## Known limits

* One app connection at a time; a second connection is closed until the
  first disconnects.
* Save states taken through the app live in the script's memory and are gone
  when the script is unloaded; mGBA's own save states are untouched.
* No speed control (see above). The game runs between steps.
