# Seeing the game: tagging enemies, ground and items on the screen

Status: T-970 (the finder) done; T-971 onwards proposed. ADR-011.

## 1. The question

Can the image-recognition tools of [Airtest](https://github.com/AirtestProject/Airtest) be joined to the learning AI so it recognises enemies on the screen and tags them, and more widely sees, understands and recognises its surroundings (ground, walls, platforms, hazards, items), not only what kills it?

Short answer: the idea is right and DESIGN.md asks for it (§3.2 "combine memory analysis, screen recognition…"; §14 "Image recognition"), but Airtest itself is the wrong thing to ship. The part of it that matters (template matching) is reimplemented here in NumPy for pixel-art games, and works.

## 2. Airtest, checked (1.4.3, November 2026)

| | Finding |
|---|---|
| Licence | Apache 2.0: compatible with the app's MIT licence (keep its NOTICE if code is copied). Not a blocker. |
| What it is | A UI test-automation framework for phone and desktop apps: connect to an Android/iOS/Windows device, find a button picture on its screen, tap it. The image part (`airtest.aircv`) is a thin layer over OpenCV: `matchTemplate` (normalised correlation), multi-scale template matching and SIFT/ORB keypoint matching. |
| Dependencies | `pip install airtest` into a clean Python 3.13 venv pulled **39 packages, 418 MB**: `opencv-contrib-python` pinned to ≤ 4.6.0.66 (2022), **`numpy<2.0`**, `pywinauto==0.6.3`, `facebook-wda`, `tidevice`, `mss==6.1.0`, `six<=1.16`, `ffmpeg-python`, Jinja2, requests, and 71 MB of Android helper binaries. The app runs on NumPy 2 (2.5.3 here), so taking Airtest on would pin the whole app back to NumPy 1.x, and roughly double the Windows build for a children's app (ADR-009 turned PyTorch down at 200 MB for the same reason). |
| Fit for retro games | Its matcher compares the whole rectangle, background included. Retro sprites are drawn over changing backgrounds and turn round. Measured with the same test sprite as `tests/unit/observation/test_sprites.py`: an enemy tagged against the sky and found in front of a bush scored 0.89 (passes, but with little margin); **the same enemy facing the other way was not found at all**. Keypoint matching (SIFT) finds few points on a 16×16 sprite. Scaling and rotation tolerance, its strengths on phones, are not needed: an emulator draws every sprite at exactly its size. |
| Speed | `find_all_template` on a 256×240 frame: about 1.1 ms per tag (OpenCV). Fine for the training loop. |
| Way in | It wants a device to screenshot; we already have frames from the emulator (`Frame`), so only `aircv` would be used. |

**Decision (ADR-011):** don't add Airtest or OpenCV. Write a masked template matcher for pixel art on NumPy (already a dependency, ADR-008). If a later feature needs real computer vision (OCR, learned detectors), add `opencv-python-headless` (current, NumPy 2 compatible, ~60 MB, Apache 2.0) as an optional extra then, not Airtest.

## 3. What T-970 built

`retro_ai_lab.observation.sprites`:

- `SpriteTemplate.from_frame(frame, x, y, w, h, name=…, tag=…)`: a person's box round a thing on the screen. Pixels of the background colour (the commonest colour on the box's edge, or one given) are see-through.
- `SpriteTag`: `player`, `enemy`, `hazard` (spikes, lava, pits), `item` (coins, power-ups), `solid` (ground, walls, platforms), `other`, each with child words.
- `SpriteFinder(templates).find(frame)`: every tagged thing on the screen as `SpriteMatch(name, tag, x, y, width, height, score, mirrored)`. Colour tolerance per channel, a minimum share of the sprite's pixels (so a partly covered enemy still counts), mirror images, one hit per thing (a shifted copy, the mirror or a repeating tile is not reported twice), adjacent ground tiles each found.
- Speed: about 1 ms for one tag and 2.5 ms for four on a 256×240 colour screen with five enemies, measured here; the same as Airtest's OpenCV path. One near-colour map per palette colour per frame serves every tag's anchor pixels, and only spots that pass the anchors are scored in full.
- Stored as JSON (`to_dict`/`from_dict`, pixels in base64), ready for a per-game store.

Nothing in the app uses it yet: the finder is the base the next tasks build on.

## 4. The plan from here (proposed tasks)

One task at a time, as CLAUDE.md asks. The aim is Boss's: the AI should see and recognise its surroundings, not only what kills it.

1. **T-971 Tags are kept per game.** `storage/sprites.py` beside the knowledge file; a service that holds the game's tags and a finder for them.
2. **T-972 Show the AI what things are.** In the Game panel: pause, draw a box round something, pick what it is (enemy, ground, platform, something that hurts, something to pick up, you) and give it a name. Everything the AI recognises is outlined on the live screen in its tag's colour, so a child sees what the AI sees.
3. **T-973 The AI looks at the screen while it plays.** When a game has tags, each training step asks the emulator for its frame too (`step(screenshot=True)`, about 3 ms on the user's PC per the T-960 log), and the finder runs on it.
4. **T-974 What is around me.** The AI's state gets a small map of the screen around the player (from the `player` tag, or the player's position the memory gives): for each cell of a coarse grid, what is there (enemy, hazard, solid, item, nothing), plus the nearest enemy's and hazard's distance across and up. These go into the state encoder and the feature map as parts of their own, beside the memory values, so the brain can learn "a gap ahead, jump", "an enemy two tiles left", "a platform above".
5. **T-975 Screen and memory agree.** A memory value that moves on its own (a mover, T-911) and a tagged enemy that moves the same way on the screen are the same thing: the guess is confirmed, named after the tag and trusted more. This is DESIGN §3.2's "combine memory analysis and screen recognition".
6. **T-976 Things that happen to you.** Events from the screen: touched an enemy, touched something that hurts, picked up an item, landed on solid ground; offered as rewards and used to explain deaths ("an enemy was touching you in 9 of 12 deaths"), which feeds the "Make the AI smarter at the game" work.
7. **Later: the AI asks.** It notices pictures that keep turning up and move on their own and asks the person "what is this?" (DESIGN §3.4: request help when not confident).
