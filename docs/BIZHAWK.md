# Playing on a real console: BizHawk (EmuHawk)

Retro AI Lab talks to BizHawk's EmuHawk through a small Lua script that ships
with the app. EmuHawk runs the game; the script listens on a local port; the app
connects, reads memory, takes screenshots, presses buttons, steps frames and
saves states through it. Nothing is installed into BizHawk and nothing leaves
the machine (the script only accepts connections from 127.0.0.1).

## What you need

* Windows with **BizHawk 2.9 or newer** (tested design target: 2.10). Download
  the zip from <https://tasvideos.org/BizHawk> and unzip it anywhere (for
  example `C:\BizHawk`); run its prerequisites installer once (that only adds
  Windows runtimes, it does not install BizHawk itself).
* A game ROM you own, for a console BizHawk supports (NES, SNES, Game Boy, Mega
  Drive and many more).
* Retro AI Lab running on the same PC. There is no exe yet: follow **Setting
  up on Windows** in the [README](../README.md) once, then start it with
  `.venv\Scripts\activate` and `retro-ai-lab` from the checkout folder.

## Starting a session

No Lua knowledge is needed: the script is written, you only open it.

1. Run `EmuHawk.exe` from the BizHawk folder and load the ROM: **File > Open
   ROM**. The game starts playing.
2. In EmuHawk's menu bar click **Tools > Lua Console**. A small new window
   opens. In *that* window's menu click **Script > Open Script**, browse to the
   checkout and pick `src\retro_ai_lab\emulator\bizhawk\retro_ai_lab.lua`.
   The script appears in the list with a green tick and the console prints
   `Retro AI Lab is waiting for the app on port 43110.` Leave the window open.
   (A red X means the script hit an error; the text in the console says what.)
3. In Retro AI Lab pick **BizHawk (EmuHawk)** in the console box and press
   **Connect**. The game pauses (the script holds it so the AI can step one
   frame at a time) and the status line shows the game's name.
4. Disconnect in the app, or close the Lua script, to hand the game back.

If the app says *Couldn't find BizHawk on port 43110*, EmuHawk is not running
the script yet (step 2), or something else uses that port. To use another port,
set the `RETRO_AI_LAB_PORT` environment variable before starting EmuHawk and
tell the app the same port.

You can also start EmuHawk with the script from a command line:

```
EmuHawk.exe --lua="C:\path\to\retro_ai_lab.lua" "C:\path\to\game.nes"
```

## Verifying the bridge (what the developer needs from you)

The bridge is tested in CI against a Lua runtime and a protocol simulator, but
three things can only be checked on a real EmuHawk (ARCHITECTURE.md §5.3):

1. **Frame advance while held.** With the app connected the game should stand
   still, and each training step should move it by exactly the frames asked.
2. **Speed.** `client.speedmode` should accept the percentages the app sends.
3. **Save states.** Saving and loading should work from the temp folder.

The quickest way to check all three is the opt-in test module. In a checkout,
with EmuHawk closed:

```
set RETRO_AI_LAB_EMULATOR_PATH=C:\BizHawk\EmuHawk.exe
set RETRO_AI_LAB_TEST_ROM=C:\roms\game.nes
pytest tests/real_emulator -s -p no:cacheprovider
```

It launches EmuHawk with the script, connects, and prints what it measured
(frames per second when stepping, milliseconds per memory read and per
screenshot). Please paste the whole output, including any failure, into the
project chat together with the BizHawk version and the console the ROM is for.
If a test fails, EmuHawk's Lua console window usually has the matching error
message; include that too.

If you would rather check by hand: connect from the app, watch that the game
freezes, press **Step** (or start a short training go) and watch it move, then
disconnect and watch it run again.

## Known limits

* One app connection at a time; a second connection is refused until the
  first disconnects.
* Memory regions above 256 KiB (a whole system bus, for example) are not
  offered to the learning side, because the app reads a region whole every
  step. The handshake still lists them.
* Analog sticks and axes are not buttons and are ignored.
