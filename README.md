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
into Explorer's address bar); send that file with your report. The log names
each stage of the start before it begins, so its last line is the stage that
did not finish. If the window is not up after 10 seconds, or the app dies
inside one of its libraries, exactly where is written to a small file next to
the log (`start-1234.txt`), and the next time the app starts it copies that
into the log; so if it would not open, start it once more before sending the
log, or send the whole folder. While a game is
connected the log also records, every few seconds, how long the emulator took to
answer each kind of request, which is the first thing to look at if the game
runs slowly in the app.

## Running it (developers)

```bash
python -m venv .venv && . .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
retro-ai-lab            # opens the desktop app
pytest                  # runs the test suite (headless Qt)
```

The Console list offers only real emulators. To play on the built-in practice console (Toy Racer, what the tests use) set `RETRO_AI_LAB_PRACTICE_CONSOLE=1` before starting the app.

On Linux, Qt needs `libegl1 libopengl0 libxkbcommon-x11-0 libxcb-cursor0 libdbus-1-3 libfontconfig1`; see `.github/workflows/ci.yml`.

## Playing on BizHawk or mGBA

The app connects to BizHawk's EmuHawk, and to mGBA, through shipped Lua
scripts. See [docs/BIZHAWK.md](docs/BIZHAWK.md) and [docs/MGBA.md](docs/MGBA.md)
for setup and for the opt-in real-emulator tests that verify each bridge on
your machine.

While the game runs, your own controller or keyboard in the emulator's
window plays as usual: the app only takes the pad while a button is held on
its on-screen pad. In "You play" and "With your help" the AI learns from
whatever you press, on the real controller or on screen. While you play on
BizHawk the game runs at the emulator's own pace: the app lets it run when
you press Play, only watches it (memory, your pad, a picture) a frame at a
time, and holds it still again when you press Pause, so there are no
bursts and stalls between the app's requests. On an emulator that can only
be stepped, the game keeps real time: when the link is slower than a frame,
the app asks for several frames at once (a tenth of a second at most)
rather than letting the game crawl. When the AI starts playing it holds the
game between its steps again.

To see what playing costs on your PC, run the profiling tool while the
bridge script is loaded:

```bash
python -m retro_ai_lab.tools.profile_play --adapter bizhawk   # or: --adapter mgba
python -m retro_ai_lab.tools.profile_play                     # the built-in simulator, no emulator needed
```

It plays for five seconds (`--seconds` changes that) in the real window and
prints how many frames the clock managed against how many it wanted, what each
bridge command cost, what the app's own bookkeeping took per step and per
picture, and the longest stall of the window's thread.

To measure the AI itself without a game of your own, run the practice bench:

```bash
python -m retro_ai_lab.tools.practice_bench --goes 40
```

It runs the whole app on Toy Hopper, the built-in practice platform game
(gaps that need a held jump, walkers, coins), with nobody watching, and
prints how well the AI sees (how often it follows itself on the screen and
whether that really is the player), what killed it against what it thought
killed it, what it reads from memory, and how far it gets. The practice game
knows the truth, so each number has it beside it. `--seed` picks another
level, `--algorithm q_table` the simple table.

The app's own log does the same every five seconds while it runs: next to the
emulator link's line, an "App work" line counts the stalls of the window's
thread and names what that thread did, event by event (`on DiscoveryUpdated`),
panel refresh by panel refresh, and the picture's paint, leaving out anything
that never took a millisecond. When the window stutters, that line says which
part of the app was busy. One line says how big a game's knowledge is when it
is picked up (guesses, relationships, file size), and one per save adds how
long the save took. The AI's step is on the same line stage by stage:
`step: choose` (picking the buttons), `step: emulator` (the frames), `step:
observe` (reading the memory, with its parts as `observe: track` and the
`discovery: …` stages), `step: encode` and `step: learn`. When "Practise
fast" is not fast, those names say which stage of each step is slow.

Every go leaves two lines: "Go 12 ended by …" with the rewards that paid, and
"The brain after go 12: Simple table. Knows 840 situations … tries something
new 60% of the time. It sees 14 values: 2 positions (1 moving), 1 distance to
a moving thing, 1 life count, …". The second line is how to tell, from a log
alone, which brain the AI was using, how much it was still guessing, and what
it could see of the game. A go that lost lives says so on its line (", lost a
life 2 times, the first at step 120"), and every tenth go from the twentieth a
"Survival over the last 10 goes" line compares lives lost per go, and how many
steps the first loss took, with the ten goes before: the plain measure of
whether the AI is getting past what kills it. The same line gives the points
per go and names the reward that moved most between the two windows, so a
survival bought by hanging back ("x position goes down, -84.0 a go, was
-26.0") shows up as the trade it is. The counts are of the drops of the
value your own rewards penalise most, if you made one ("Lives goes down",
below zero; a life count or health value you penalise comes first, however
lightly, so "x position goes down" at -2 is a step back and not a death),
else of the AI's best life count and best health (yours first, if you
confirmed one), not of every guess of those kinds; the log names what is
counted whenever that changes. Beside it, when the ten goes had three deaths
or more, a line says what the AI saw a moment before them and seldom
otherwise ("Before the 12 deaths in the last 10 goes the AI saw Position at
ram[520] from x position=1 in 9 of them (15% of all steps)"), or that nothing
stood out and it may not see what kills it. Every go starts by putting the
game back to its checkpoint, and that jump is the AI's doing, not the game's:
the change statistics behind the "went up N times and down N" warnings skip
it, so a life count that only ever drops in play is not told it bounces.
If three goes running lose their first life at the same early step, the app
checks whether the spot the goes start from loses it by itself (it runs the
game from there holding nothing, then holding each of the AI's buttons in
turn); when the life goes whatever is held, the goes restart from just after
that loss, once the game listens again, and the log says so. A life a held
button keeps, or one the AI loses by what it presses, is left for it to learn
from, and the start is moved at most three times from one checkpoint.

## Letting the AI start the game

Press Start in the Learning panel wherever the game is, even on its title
screen. The AI first checks whether its buttons change anything in the game.
If the game ignores it (a demo is playing, or the game is over), it tries
every button on the controller, one at a time, until it finds the one that
gets the game going (pressing a button again, a few times at most, when a press
changed the screen but the game still ignores it, as Start does on a title
demo), remembers it for that game, and presses it whenever the game stops
listening again. If no button works after three tries, training stops and the
Learning panel says so: get the game going yourself, or pin a spot inside it,
then press Start again. It checks again after a go that ended stuck or
with the game starting over. The Discovery panel's summary says
what it found ("Start gets the game going."). To practise one part of a game,
play to it yourself and press "Pin this spot" first. Each time it checks
whether the game listens, it also notes which values change while nobody
presses anything and ignore the controller (timers, enemies, a demo) and, after
two such checks, drops any "score" or "progress" guess among them. The Discovery
panel still lists a dropped guess, and you can put it back.

Two boxes in the Learning panel set the go timers: "Each go lasts" (40
seconds unless you change it) and "Give up when nothing changes for" (about
two seconds), both in seconds and remembered for that game. Change them while
training is stopped.

"Speed" in the same panel is "Watch it play" (one step a game tick) or
"Practise fast": the emulator runs flat out, the picture drops to two a
second and the Discovery and Rewards panels refresh once every two seconds
instead of several times a second, since every refresh takes time from the
AI's steps and nobody is watching. They catch up the moment you pause, stop
or switch back to "Watch it play".

## Reading the rewards

The AI keeps every guess it makes about the game's memory, but the Rewards
panel, the Progress panel's steps and the step-detection list only show its
best guess of each role, in plain words (a suggested step or reward whose
guess the AI has changed its mind about, or no longer has at all, goes with
it; steps you wrote stay): "Your score jumps up", "Losing a
life", "Progress goes up" (and "Progress (2nd guess) goes up" when it offers
a second progress value, as a game can have a level and a checkpoint). To
call something by its real name, pick it in the "What the AI is doing"
panel, type the name and press Rename; every list follows. Picking a line
shows why the AI thinks so, and the "Yes, that's right", "No, that's wrong"
and Rename controls, just above the list. A game has one life count, one
health and one score, so confirming or naming one of those sets the AI's
own other guesses of that kind aside: they leave the rewards, the steps,
what the AI sees and the survival line together. That panel's
list can run to hundreds of guesses: the find box above it keeps only the
lines containing what you type (a name, an address such as `ram[12]`, or
"Confirmed by you"), and the box next to it sorts by address, by name or
"Most sure first", which puts what you confirmed at the top and what you
rejected at the bottom. When the AI cannot find a counter on its own (a life
count changes too rarely for it to be sure), tell it: the "Tell the AI:" row
has "I just lost a life", "I just got further" and "My score just went up".
Press one right after it happens; the AI keeps only the bytes that moved the
right way, and steadily, in the last five seconds (a life counter changes
once, down by one; a score never goes down), like a cheat finder, and after two
presses that agree it takes the one byte left as your word, names it, and
drops its own guesses of that kind that did not move. The note under the
buttons says what each press found. The AI presses "I just lost a life" for itself, too: every time the game throws the player back to a start point it looks for the byte that went down by one just before, and after a few restarts that agree it takes that as its lives guess (your own Yes or No still wins). The "What the AI is doing" panel, the log and
the Rewards table's last column (Address) still show the memory address for
grown-ups. Pick a reward and the line under
the table says what it means and what it is worth ("Losing a life: the AI
loses 7.5 points each time."); hover a group's name for what the group is.
The Points box next to + and − takes any number you type (0.65, -3); press
Enter or click elsewhere to apply it.
The AI offers first the values that moved a few times while it played, go
after go: after some goes in which you collected something, the counter
appears as a new "Progress" or "Your score" row, and the Memory table in the Research window shows its value now,
so the row reading the same number as the screen is the one to name.
"Progress goes up" and "Your score jumps up" only pay when the value beats
its best so far in the go: dying and covering the same ground again earns
nothing until the AI gets further. "Something switches on" and "switches
off" rows pay the first time in a go and not again until the next go: a door
that opens is worth its points once, and a value that flips every time the
AI presses a button does not pay for the pressing. Your own "goes up" rows
pay every time.
To reward something the AI did not think of, use "Add a reward" under the
table: pick a value ("Your position"), "goes up" or "goes down", and the
points (below zero takes points away), then press Add. "Going right is good,
going left is bad" is two rows: goes up +2 and goes down -2. "Remove" takes
any picked row out, and it stays out for that game: a suggested row such as
"Losing Lives" is not put back for the AI's next guess of that kind, and
"Starting over" and the button rows go the same way. Removing a row does not
tell the AI its guess is wrong; use "No, that's wrong" under What the AI is
doing for that. "Bring back suggestions" undoes every removal at once.
The Controls group lists every button of the connected game ("Pressing A")
and "Pressing the same button again"; all start at 0, so they change nothing
until you give one points (below zero to put the AI off hammering a button).
"Getting somewhere new" (under Progression) is the one reward that needs no
guess about the game's memory: once the AI has found itself on the screen, it
counts how far it has got by the picture alone, and every step of its own
size past the furthest it has been this go pays the row, 1 point to start
with. On a game whose screen scrolls, how far it has got is how far the
camera has travelled since the go began, so jumping on the spot and falling
into a pit are not somewhere new; on a screen that never scrolls it is where
the AI stands. The row appears the first time there is ground to pay for;
change its points, switch it off or remove it like any other. The same
ground never pays twice in a go: when the screen jumps to another place (a
life lost and put back, a door), the AI is taken to be back where the go
began. Up, down, left and right all count. Where the picture scrolls too fast
for the AI to follow (on a very small screen, or a very fast game), it says
so in the log ("It cannot follow the camera here") and counts no new ground
at all, rather than pay for the slow stretches it can measure.

The app uses the same look as Stills & Reels and StackSpace: dark by default,
with the blue-to-purple accent. The button at the right of the toolbar switches
between "Dark mode", "Light mode" and "Match Windows", and the app remembers
the choice.

Grown-ups can open the Research window (toolbar) for a Memory table: each
address the AI is watching, its value now, how often it changes, what the AI
thinks it is and how sure, and any name given to it. Filter by address
("ram[12]", "0x0c") or name, pick a row, type a name and press "Name it";
every panel then uses that name. The AI still learns everything itself; the
table only shows what it worked out and lets you name it. Its Learning
settings box has the learning numbers for researchers; "Exploring steps" is
how many learned steps it takes the AI to go from trying random moves to
mostly choosing (6,000, about ten 40-second goes). Exploring also follows
results: when the last ten goes score worse than the ten before, the AI goes
back to trying random moves "Exploring rebound" of the time (25%) and stays
there while the goes keep getting worse, so a bad habit is not locked in; the
log says when that happens. A random move is not a one-step twitch: the AI holds
it for a run of 1 to "Exploring run" steps (8, about half a second), so it
finds out what holding a button does, such as a long jump; runs start less
often, so the share of random steps is still what the other settings say.

To find a value yourself, like a cheat finder, use the Research window's
"Search values" tab. Type the number you see on the screen (12, a score of
300, or hex like 0x0C or $0C) and press New search, or leave the box empty to
start from every byte. A number bigger than one byte holds is looked for in
every way games store numbers (two to four bytes in either order, packed
decimal digits, one digit per byte), and each place found is one row that
reads as one number: a score kept one digit per byte over six bytes shows as
`ram[2013..2018]` with "6 digits, highest first" in the "Held as" column, and
the scans follow the whole number, so it still "went up" when it rolls from
800 to 1,000. Type the number as the game shows it, zeros and all (000600),
to say how many digits it has; otherwise the row shows the shortest reading
and counts the longer ones. Play a little, then press the scan that fits: "Is
exactly" (with the new number), "Went up", "Went down", "Changed" or "Didn't
change"; each compares with the scan before. A byte is shown in decimal and
hex, a longer number as it reads, and "Now" keeps up with the game. When a
few values are left, pick one, give it a name and points, and press "Reward
going up" or "Reward going down": the value is named for the AI in that
layout, the reward appears in the Rewards panel under Progression, and the
AI's own guesses on the other bytes of the value are set aside, since a byte
that is part of a value you named is not a value of its own. The app only
reads the game's memory; it never changes it.

After a long run on an older build, the AI may have dropped the real score or
lives for reasons a newer build no longer has. Press "Start from scratch" in
the Learning panel once so it forgets that game and can find the real values
again. It asks first, because it also deletes the brain the AI has trained for
that game. The brain itself needs no saving: it is written after every go and
picked up again when the same game connects, and the log says so ("Picked up
the saved brain for this game: 109 goes and 49,050 steps learned so far").

## What the AI sees

The AI's first eyes are the values it believes in (it also looks at the
picture; see the end of this section):
each go, it reads its best guesses (a position, the lives, the score, a
checkpoint, a few switches) and learns which button helps from how they
change. Its brain has room for 32 of them at a time, and they are handed
out kind by kind, the most trusted first: where things are, how it is
doing, then the switches, so a position the AI finds late still gets a
seat before the twentieth flag. If an enemy's position is in memory and
the AI has found it, the AI can learn to avoid the enemy; if it has not,
the enemy is invisible to it. A value that wanders both ways in small
steps is taken for the position of something that moves (an enemy, a
platform, a ball) and listed as "Something moving" until the AI finds a
button that moves it; a thing that is on screen only now and then counts
too, once it has moved through enough different values. A value that climbs
by the same amount every step is a clock, not a position, and gets no seat, even when the game throws it back to zero now and then. For each such thing the AI also sees how far it is
from its own position (the one the stick moves, or one you confirmed),
exactly when it is close and roughly when it is far, so a thing about to
touch it and one across the screen look different; those distances get
their own seats, so a moving thing is seen even when plainer positions
take all the position seats, and moving things are counted apart from
plain positions among the guesses the AI acts on. Copies of your own
position that the same buttons move count once, so the seats go to
different moving things. None of these labels is
final: a value gets its first label at the AI's first look, often after a
handful of changes, and at every review the AI rethinks any guess it made
itself whose behaviour now says otherwise, so an enemy slot that looked like
"a speed" on five values becomes "Something moving" once it has walked. It
rethinks a guess only once the value has changed twice as often as when it
last judged it, so a value on the edge of a rule does not flip back and
forth; those counts start again from zero each time a game connects, as
the statistics do. Guesses you confirmed, named or rejected are never rethought. The "What the AI is doing" list shows what it has
found, and the Research window's memory table lets you name a value for
it.

When the last ten goes' deaths mostly came with one sight (the AI's own
position in one of its wide buckets, say, and nothing moving near: a pit
or a fall trap), the AI looks closer there from then on, reading that
value in steps of 8 inside that bucket instead of one 128-wide lump, so
the trap and the ground before it become different situations and it can
learn the button for the spot. The log says where it now looks closer,
and those closer looks are saved with the brain, so a restart keeps them.

Every few minutes the log also prints a census of memory values, the
sixteen that change most, the sixteen smoothest and the sixteen rare ones
with the widest range, with the numbers behind the AI's word for each, so a
log can show why a value is or is not taken for something moving, or for a
clock. Each button experiment logs what the buttons moved, and the line
after every go names the position the distances are measured from. A
button that stops or replaces the whole game when held (a pause button, a
menu) is noted as such and linked to nothing.

It looks at the picture too. After Start it holds each button in turn from
the same spot and watches what moves: the thing that answers is itself
("Found itself on the screen" in the log). From the screens on which it saw
itself move it also learns what it looks like, the few colours it is drawn in
that nothing else on the screen shares, and from then on it picks itself out
of every screen by those colours alone ("It knows what it looks like"), so it
does not lose itself when it jumps, stands still or the screen scrolls. A
game saved before the AI could do this is looked at once more at the next
Start. Every other step of a go it then looks at the screen, follows the
other things on it, and sets what it touched against the lives it lost and
the points it gained, to learn which kinds of thing hurt and which are worth
picking up. At every look the brain is told what is around it, read off the
picture next to where it sees itself: how much floor there is to its right
and to its left before a drop, how far the way is clear at its own height on
each side, and whether it stands on something, has something below it or has
nothing under it at all; then how far the nearest moving thing and the
nearest thing that hurts are. So it can see a gap coming instead of having
to remember where in the level the gaps are. (It cannot yet tell a bush or a
far hill from a wall, and a game with a strip along the bottom of the screen
hides its gaps from this reading.) Twelve of the brain's 32 seats are kept
for what it reads off the picture. Between two looks it also measures how far
the camera moved, which is what tells a thing that walks from one the screen
merely slides past. It measures that by the scenery alone: the AI itself,
which the camera holds still on the screen, has no say, and neither have the
rows it has found to be painted on the glass (a strip of lives and score
stands still while everything else slides, so after two camera moves it is
left out); and one small thing changing on a still screen is not taken for
the camera moving. From the camera's travel (on a screen that never scrolls,
from where it stands) it counts the new ground of each go, which the reward "Getting
somewhere new" pays (see Reading the rewards). It hears of every life lost, also those lost
between two looks. A life lost after it went out of sight coming down, with
nothing under it but the backdrop, is a fall, and is not blamed on whatever
it touched or stood on before; coming down through the floor is not a fall
(many games throw a player that was hit off the bottom of the screen), and
then the thing it touched keeps the blame. Every ten goes the log says how
the looking is going ("On the screen so far"), with the lives lost after a
touch and after a fall.

## Arranging the panels

Every panel can be dragged by its title bar to another edge of the window,
dropped onto another panel to make tabs, pulled out as its own window, or
resized by the bars between panels; a panel made smaller than what it holds
scrolls, so the bars and the window itself go as small as you like. The
**Panels** button on the toolbar hides
or shows each one and has **Reset layout** to put everything back. The
arrangement is remembered between runs.

## Windows build

Every push to `main` runs the **Package** workflow, which builds `RetroAILab.exe` with PyInstaller, launches it as a smoke test and uploads the folder as the `RetroAILab-windows` artifact (Actions tab → latest Package run → Artifacts). To build locally: `pip install pyinstaller && pyinstaller packaging/retro_ai_lab.spec --noconfirm`.

## Planned stack

Python 3.11+, PySide6 (Qt) desktop UI, NumPy for screen and memory analysis, PyTorch and Gymnasium for learning, SQLite and JSON for storage. Target platform is Windows; the code is kept cross-platform so it can be developed and tested on Linux CI.

## Legal

Retro AI Lab ships no ROMs and never modifies ROM files. Use games you legally own.

## Credits

Retro AI Lab is built on other people's work. Everything taken from another source is listed here; keep this list current when something new is added.

**Shipped inside the app**

| What | Where it came from | Licence |
|---|---|---|
| LuaSocket DLLs (`emulator/bizhawk/luasocket/x64/socket-windows-5-4.dll`, `-5-1.dll`) | Prebuilt binaries from the [Archipelago](https://github.com/ArchipelagoMW/Archipelago) multiworld project (`data/lua/x64/`), unmodified copies of [LuaSocket](https://github.com/lunarmodules/luasocket) by Diego Nehab | MIT (both); full text in `emulator/bizhawk/luasocket/LICENSE` |
| `emulator/bizhawk/socket.lua` loader | Written for Retro AI Lab, modelled on Archipelago's `data/lua/socket.lua` (how the DLL is found and `bind`/`connect` are added) | MIT (Archipelago); our file is part of this project |
| BizHawk bridge design (Lua script as a TCP server, newline-delimited JSON) | Approach borrowed from Archipelago's `connector_bizhawk_generic.lua`; no code copied | n/a |

**Libraries used (installed with the app)**

| Library | Used for | Licence |
|---|---|---|
| [PySide6 / Qt](https://www.qt.io/qt-for-python) | Desktop UI | LGPL v3 |
| [NumPy](https://numpy.org) | Screen and memory analysis | BSD 3-Clause |
| [PyInstaller](https://pyinstaller.org) | Windows build | GPL v2 with bootloader exception (output is not covered) |

**Development tools (not shipped)**

pytest, pytest-qt, pytest-cov, Hypothesis, Ruff, mypy, and [lupa](https://github.com/scoder/lupa) (runs the Lua bridge scripts in tests). All MIT, Apache 2.0 or MPL 2.0.

**Emulators the app talks to (not shipped)**

[BizHawk](https://tasvideos.org/BizHawk) (MIT) and [mGBA](https://mgba.io) (MPL 2.0). The Lua API reference at tasvideos.org/BizHawk/LuaFunctions was the source for the EmuHawk calls in the bridge script.
