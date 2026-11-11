# Seeing the game: tagging enemies, ground and items on the screen

Status: T-970 to T-974 done (find, follow, find itself, on the real game); T-975 onwards planned. ADR-011.

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

## 4. No drawing: the AI finds things and works out what they are (T-971)

Boss's direction (10 Nov): nobody should have to draw boxes or label anything; the AI recognises things itself, as the whole project is meant to be autonomous. So the person-tagging UI is dropped from the plan, and everything below is the AI's own work. A person can still look at what it decided, never has to.

`retro_ai_lab.observation.objects` (T-971) is the first step:

- `background_colour(frame)`: the commonest colour is the backdrop.
- `find_objects(frame)`: every patch of other colours is a thing; parts a pixel apart (eyes, feet) are joined; a patch over 64 pixels either way is *scenery* (ground, walls), the level itself.
- `ObjectCatalogue.observe(frame)`: sorts things screen after screen into kinds by exact look, a mirrored copy being the same kind, named "thing 1", "thing 2" in the order seen, each with a template the T-970 finder can look for.

Known limit, handled by T-972: on one still screen, a sprite touching the ground is part of the ground's patch. Motion between screens separates them.

## 5. The plan from here

One task at a time, as CLAUDE.md asks; each one the AI's own doing.

1. **T-972 (done) Follow things between screens.** Work out how far the screen scrolled, take it away, and what still moved is a thing moving in the level (which also cuts a walker out of the ground it stands on). Each thing gets a track: where it is, which way and how fast it moves.
2. **T-973 (done) Which one is me.** The AI already experiments with buttons from a save state (T-844); the kind that moves when it presses left or right, and not otherwise, is the player.
3. **T-974 (done) It looks for itself on the real game** at the first go after Start and remembers the answer.
4. **T-975 What each kind does to me.** Set what happens on the screen against what memory says happens (the life count, the score, already found): a kind the player was touching when a life went is something that hurts, and if it moves on its own an enemy; a kind that vanishes when touched as the score goes up is something to pick up; scenery the player stops against or stands on is solid. Each conclusion carries the same confidence levels as the memory guesses (low → confirmed) and grows with repeats.
5. **T-976 What is around me.** The AI's state gets a small map of what is near the player (enemy, hurts, solid, pick-up, nothing per cell) and the nearest enemy's distance, beside the memory values, so the brain can learn "gap ahead, jump", "enemy two tiles left".
6. **T-977 Screen and memory agree.** A memory value that moves with a tracked thing on the screen is that thing's position: the guess is confirmed and named ("enemy's x position"), which should cut the RAM guesses a person has to correct.
7. **T-978 Show what it sees.** Outlines on the Game panel in each kind's colour with the AI's name for it, so a child sees what the AI recognises. Read-only.
