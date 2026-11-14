"""The AI watches the screen while it plays and learns what things are (T-976).

Once the AI knows which thing on the screen is itself (T-974), every few steps
of a go it looks at the screen: the tracker follows the things (T-972), the
player's track is picked out (by its kind, or at the go's first look by where
the player was found), and the touches are set against the lives lost and the
score rises memory reported in the same steps (T-975). Between goes the
service asks :meth:`ScreenWatch.roles` and logs what changed.

The kinds ("thing 3") live as long as the app runs; keeping them with the
game across sessions is a later task.

Since T-987 the player is known by its looks when the finding has its colours
(:class:`~retro_ai_lab.observation.signature.ColourSignature`): each look picks
it out of the screen afresh, so a jump, a locked camera or standing on the
ground no longer lose it, and "me" is a box (:attr:`ScreenWatch.me`) rather
than a track. A finding without colours is followed as a track, as before.

Since T-989 every look in which it sees itself also reads what is around it
(:mod:`retro_ai_lab.observation.surroundings`): the floor either side, the way
at its own height and what is under it. The brain is given those first.

Since T-993 the watch also measures the new ground the go covers
(:mod:`retro_ai_lab.observation.ground`): the camera's travel on a game that
scrolls, where it stands on a screen that does not, in steps of its own
size. A cut to another place, or a life lost with a jump of the screen, puts
it back where the go began. The
runner asks :meth:`ScreenWatch.take_new_ground` after every look and the
reward service pays it.
"""

from __future__ import annotations

import logging
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass

from retro_ai_lab.core import ConfigurationError, Frame
from retro_ai_lab.observation.camera_memory import CameraMemory
from retro_ai_lab.observation.ground import Ground, Kept
from retro_ai_lab.observation.objects import ObjectCatalogue
from retro_ai_lab.observation.signature import Box
from retro_ai_lab.observation.sprites import SpriteTag
from retro_ai_lab.observation.surroundings import NOTHING_BELOW, Surroundings, read
from retro_ai_lab.observation.tracking import ObjectTracker, Track
from retro_ai_lab.understanding.player import PlayerFinding
from retro_ai_lab.understanding.roles import Happening, Role, RoleLearner, within

log = logging.getLogger(__name__)

DEFAULT_LOOK_EVERY = 2  # steps between two looks (a look costs a screenshot and ~15 ms)
SOON_LOOKS = 30  # looks after a touch in which a life lost is put down to it (~4 s)
FIRST_LOOK_REACH = 16  # pixels from where the player was found, at a go's first look
PART_REACH = 128  # pixels across within which a thing is in the brain's view (T-978)
NEAR = 48  # pixels across within which a thing is "close" rather than "coming"
LEVEL = 8  # pixels within which a thing is level with the player, or on top of it
HIDE_MARGIN = 1  # pixels round the player's own colours left out with it (its outline)
LOOK_SHIFT = 32  # pixels the camera may move between two looks, either way (T-992)
CUT = 0.25  # share of the screen no camera move explains: a cut to another place (T-993)
JOLT = 0.02  # ... and the share that, with a life lost about then, is a respawn
BACK_LOOKS = 4  # looks between the life lost and the jolt for the two to be one event
FOLLOWED_OVER = 200  # looks with the camera moving that its measure is judged over
LOST_LEAST = 20  # ... of which this many unmeasured, and more than the share below,
LOST_SHARE = 0.2  # mean the camera cannot be followed here: no ground is counted


def side(apart: int, *, near: int | None = NEAR) -> int:
    """A few coarse answers, not exact pixels, so the brain's table stays small:
    -2 far before, -1 close before, 0 level, 1 close after, 2 far after
    (without ``near``: only -1, 0 and 1)."""
    if abs(apart) <= LEVEL:
        return 0
    sign = 1 if apart > 0 else -1
    if near is None or abs(apart) <= near:
        return sign
    return 2 * sign


@dataclass(frozen=True, slots=True)
class Mark:
    """A moment of a go the watch can be put back to (T-1000): where the ground count
    stood and where the player was last seen; and the last screen the watch had
    looked at, with the box it hid in it (T-1023), or None before a go's first look."""

    ground: Kept
    last_me: tuple[int, int] | None
    screen: Frame | None = None
    hide: tuple[int, int, int, int] | None = None


class ScreenWatch:
    def __init__(
        self,
        player: PlayerFinding,
        *,
        catalogue: ObjectCatalogue | None = None,
        soon: int = SOON_LOOKS,
    ) -> None:
        self._player = player
        self._tracker = ObjectTracker(catalogue, max_shift=LOOK_SHIFT)
        self._learner = RoleLearner(soon=soon)
        self._sign = player.colours  # what the player looks like, when the finding says (T-987)
        self._me: int | None = None  # the player's track, when it has no colours to go by
        self._box: Box | None = None  # where the player was seen at the last look
        # Learned from the player's track when it has one. The finding's kind is a name
        # from the catalogue of whoever found it; it means something here only when
        # the caller handed over that same catalogue.
        self._my_kind = player.kind if catalogue is not None else None
        self._last_me: tuple[int, int] | None = None  # where the player was last seen
        self._looks = 0
        self._looks_this_go = 0
        self._told: dict[str, tuple[str, str]] = {}
        # What the summary every ten goes reports (T-977).
        self._seen_me = 0
        self._lives_lost = 0
        self._lives_after_touch = 0
        self._lives_after_fall = 0
        # Falling out of the bottom of the screen (T-988).
        self._soon = soon
        self._before: Box | None = None  # the sighting before the last, if a look apart
        self._last_box: Box | None = None  # the last sighting, kept while out of sight
        self._nothing_under = False  # at its last sighting, only backdrop below it
        self._fell_at: int | None = None  # the look at which it went out of sight, falling
        self._was_seen = False  # it was in sight at the look before
        self._around: Surroundings | None = None  # what was around it at the last look (T-989)
        # The new ground the go covers (T-993).
        self._ground = Ground()
        self._new_ground = 0  # steps found since the last take
        self._jolt_at: int | None = None  # the last jolt that found it somewhere else
        self._prior_box: Box | None = None  # its last sighting before this look
        self._lost_at: int | None = None  # the last look that heard of a life lost
        self._was_jolt = False  # the look before was a jolt too
        # Looks in which the camera moved: True where the move was measured, False
        # where it was not (a jolt). Kept from go to go: it is about the game.
        self._measured: deque[bool] = deque(maxlen=FOLLOWED_OVER)
        # The game's own count of where its camera is, found from the looks (T-1020):
        # which bytes of the memory handed over with each screen move as the screen
        # slides. Kept from go to go, like the above.
        self._camera = CameraMemory()
        self._layout: tuple[tuple[str, int], ...] = ()  # the regions handed over, and their sizes
        self._memory: bytes | None = None  # the memory at the last look, if the next can use it
        # The last screen looked at in this go and the box hidden in it (T-1023).
        self._shown: tuple[Frame | None, tuple[int, int, int, int] | None] = (None, None)
        self._said: set[tuple[int, int]] = set()  # the bytes found that the log has named

    @property
    def camera_bytes(self) -> tuple[tuple[str, int] | None, tuple[str, int] | None]:
        """Where the game keeps count of its camera, as far as the looks have shown
        (T-1020): the memory region and the byte in it that moves as the screen slides
        across, and the one for up and down; None for a way no byte has proved itself."""
        across, down = self._camera.found
        return (self._where(across), self._where(down))

    def _where(self, byte: int | None) -> tuple[str, int] | None:
        if byte is None:
            return None
        for name, size in self._layout:
            if byte < size:
                return (name, byte)
            byte -= size
        return None

    @property
    def player(self) -> PlayerFinding:
        return self._player

    @property
    def looks(self) -> int:
        return self._looks

    @property
    def seen_me(self) -> int:
        """Looks in which the thing it takes for itself was in sight."""
        return self._seen_me

    @property
    def lives_lost(self) -> int:
        return self._lives_lost

    @property
    def lives_after_touch(self) -> int:
        """Lives lost soon after touching something."""
        return self._lives_after_touch

    @property
    def lives_after_fall(self) -> int:
        """Lives lost after it went out of sight falling, with nothing under it."""
        return self._lives_after_fall

    @property
    def tracker(self) -> ObjectTracker:
        return self._tracker

    @property
    def player_track(self) -> int | None:
        """The track followed as the player (None when it is known by its colours)."""
        return self._me

    @property
    def me(self) -> Box | None:
        """Where the AI saw itself at the last look (x, y, width, height); None when
        it did not."""
        return self._box

    @property
    def spot(self) -> tuple[int, int]:
        """Where the AI is in the level, in pixels (T-1018): the camera's travel since
        the go began, plus where on the screen it last saw itself. On a screen that
        never scrolls that is where it stands on the screen; where it has not seen
        itself, the camera's travel alone. Only for telling whether it has moved."""
        x, y = self._ground.place if self._ground.scrolls else (0, 0)
        if self._last_me is not None:
            x, y = x + self._last_me[0], y + self._last_me[1]
        return (x, y)

    @property
    def around(self) -> Surroundings | None:
        """What was around it at the last look: the floor either side, the way at its
        own height, what is under it (T-989). None when it did not see itself."""
        return self._around

    @property
    def ground(self) -> Ground:
        """The new ground this go has covered so far (T-993)."""
        return self._ground

    @property
    def follows_camera(self) -> bool:
        """Whether the camera's moves can be measured well enough on this game to
        count new ground by (T-993). Not when, of the last looks in which it moved, a
        fifth or more (and 20 at least) could not be measured: a screen too small or a
        scroll too fast leaves too little of the picture to compare, and counting only
        the moves that can be measured would pay for going slowly."""
        unmeasured = len(self._measured) - sum(self._measured)
        return not (unmeasured >= LOST_LEAST and unmeasured > LOST_SHARE * len(self._measured))

    def take_new_ground(self) -> int:
        """The steps of new ground found since the last call (T-993); they are given
        once."""
        new, self._new_ground = self._new_ground, 0
        return new

    def begin_go(self) -> None:
        """A go starts from the checkpoint: new tracks, the player looked for afresh."""
        self._tracker.reset()
        self._shown = (None, None)
        self._memory = None  # the game was put to its start: no move to read across that
        self._ground.begin()
        self._new_ground = 0
        self._jolt_at = None
        self._prior_box = None
        self._lost_at = None
        self._was_jolt = False
        self._around = None
        self._me = None
        self._box = None
        self._last_me = None
        self._looks_this_go = 0
        self._before = None
        self._last_box = None
        self._fell_at = None
        self._nothing_under = False
        self._was_seen = False

    def mark(self) -> Mark:
        """This moment, to be put back to if the game is (T-1000)."""
        return Mark(self._ground.keep(), self._last_me, *self._shown)

    def back_to(self, mark: Mark) -> None:
        """The game was put back to the moment ``mark`` was taken (a save loaded).

        The tracker starts afresh, so the jump is not measured as a move of the
        camera or taken for a cut to another place. What was waiting from the last
        looks is dropped: a jolt, a life lost (it would be read as a respawn at the
        start of the go), a fall, touches not yet blamed. The ground is put back:
        what was reached stays reached. The counts for the summary stay.

        The tracker starts from the screen of the moment gone back to, where the mark
        has it (T-1023): the next look is then measured against that screen like any
        other. Started from nothing, that look read as no move at all, and up to a
        step of the camera's travel dropped out of the ground counted every time a
        go went back: most of what kept the count from matching the game's own.
        Showing it the screen is no look: nothing is counted or learnt from it. The
        game's memory is still not read across (T-1020): that look is the picture's
        alone.
        """
        if not isinstance(mark, Mark):
            raise ConfigurationError("Only a moment this watch marked can be gone back to.")
        if mark.screen is None:
            self._tracker.reset()
        else:
            self._tracker.start_from(mark.screen, hide=mark.hide)
        self._shown = (mark.screen, mark.hide)
        self._memory = None  # nor across this
        self._ground.put_back(mark.ground)
        self._learner.let_go()
        self._jolt_at = None
        self._prior_box = None
        self._lost_at = None
        self._was_jolt = False
        self._around = None
        self._me = None
        self._box = None
        self._last_me = mark.last_me
        self._before = None
        self._last_box = None
        self._fell_at = None
        self._nothing_under = False
        self._was_seen = False

    def trial(self, mark: Mark) -> ScreenWatch:
        """A watch for a try from the moment ``mark`` was taken: the same player, a
        copy of the ground put back to that moment, and whether the camera can be
        followed here. Its looks change nothing in this watch, and its
        :meth:`take_new_ground` is what the try would be paid."""
        if not isinstance(mark, Mark):
            raise ConfigurationError("Only a moment this watch marked can be tried from.")
        other = ScreenWatch(self._player, soon=self._soon)
        other._my_kind = None  # a kind is a name in this watch's own catalogue
        other._ground = self._ground.copy()
        other._ground.put_back(mark.ground)
        other._last_me = mark.last_me
        other._measured = deque(self._measured, maxlen=self._measured.maxlen)
        return other

    def look(
        self,
        frame: Frame,
        *,
        lost: int = 0,
        scored: int = 0,
        memory: Mapping[str, bytes] | None = None,
    ) -> None:
        """Look at a screen. ``memory`` is the game's memory as it was read with this
        screen, region by region (T-1020): from it and the last look's the watch
        learns which byte moves as the camera does, and once one has proved itself
        tells the tracker what the camera's move was, for it to take where the picture
        cannot say. Without ``memory`` the picture is all there is, as before. So it
        is at a look in which a life went: the game may have put the player back, and
        a count that jumped is no move for the tracker to try."""
        now = self._memory_now(memory)
        was, self._memory = self._memory, now
        expect = None
        if was is not None and now is not None and not lost:
            expect = self._camera.expect(was, now)
        self._prior_box = self._last_box
        if self._sign is not None:
            # By its colours: nearest where it was last seen, or at the start of a go
            # where the player was found (the go starts from that spot).
            near = self._last_me if self._last_me is not None else self._player.start
            self._box = self._sign.find(frame, near=near)
            # The tracker follows everything else: with the player left out of its
            # picture, nothing the player stands on or touches is one patch with it.
            hide = None
            if self._box is not None:
                x, y, w, h = self._box
                hide = (x - HIDE_MARGIN, y - HIDE_MARGIN, w + 2 * HIDE_MARGIN, h + 2 * HIDE_MARGIN)
            tracks = self._tracker.update(frame, hide=hide, expect=expect)
            self._shown = (frame, hide)
        else:
            tracks = self._tracker.update(frame, expect=expect)
            self._shown = (frame, None)
            self._me = self._find_me(tracks)
            me = next((t for t in tracks if t.id == self._me), None)
            if me is not None and me.missing == 0:
                self._box = (me.x, me.y, me.width, me.height)
                if me.kind is not None and not me.cut_out and not me.merged:
                    self._my_kind = me.kind
            else:
                self._box = None
        if was is not None and now is not None:
            self._learn_camera(was, now)
        if self._box is None and self._sign is not None and self._was_seen and self._falling():
            self._fell_at = self._looks
        happened: list[Happening] = []
        if lost and not self._a_fall(lost):
            happened.append(Happening.LIFE_LOST)
        self._around = None
        if self._box is not None:
            x, y, w, h = self._box
            self._seen_me += 1
            self._last_me = (x + w // 2, y + h // 2)
            self._before, self._last_box = self._last_box if self._was_seen else None, self._box
            self._around = read(frame, self._box, self._tracker.catalogue.background)
            self._nothing_under = (
                self._sign is not None
                and self._around is not None
                and self._around.under == NOTHING_BELOW
            )
            self._fell_at = None  # in sight again: it did not fall out of the screen
        self._was_seen = self._box is not None
        self._cover_ground(lost)
        if scored:
            happened.append(Happening.SCORE_UP)
        self._learner.observe(
            self._looks, tracks, self._me, happened, me=self._box if self._sign else None
        )
        self._looks += 1
        self._looks_this_go += 1

    def _memory_now(self, memory: Mapping[str, bytes] | None) -> bytes | None:
        """The memory handed over with a screen as one run of bytes, region by region
        in the order of their names. Other regions or sizes than before is another
        game, or this one read another way: what was learnt of the camera is void."""
        if memory is None:
            return None
        if (
            not isinstance(memory, Mapping)
            or not memory
            or any(
                not isinstance(k, str) or not isinstance(v, (bytes, bytearray)) or not v
                for k, v in memory.items()
            )
        ):
            raise ConfigurationError(
                "memory must be the game's memory by region: a name and its bytes each."
            )
        names = sorted(memory)
        layout = tuple((name, len(memory[name])) for name in names)
        if layout != self._layout:
            self._layout = layout
            self._camera = CameraMemory()
            self._said.clear()
            self._memory = None
        return b"".join(bytes(memory[name]) for name in names)

    def _learn_camera(self, was: bytes, now: bytes) -> None:
        """What this look says about where the game keeps its camera (T-1020): the move
        as the picture alone had it, against every byte's change. Not from a look the
        picture could not measure (a jolt), and never from what memory itself said:
        a look in which the tracker took the byte's word is one the picture could not
        tell, and says nothing for the byte or against it. (Counted against it, the
        byte would be dropped just where it is needed: beside a walker on bare ground
        the picture is wrong more often than right.)"""
        if self._tracker.scroll == (0, 0) and self._tracker.unexplained >= JOLT:
            return
        if self._tracker.scroll != self._tracker.sighted:
            return
        self._camera.note(was, now, self._tracker.sighted)
        for way, byte in enumerate(self._camera.found):
            if byte is None or (way, byte) in self._said:
                continue
            self._said.add((way, byte))
            where, agreed = self._where(byte), self._camera.agreed(way)
            assert where is not None and agreed is not None
            log.info(
                "The AI found where the game keeps count of its camera: %s[%d] moves as the "
                "screen slides %s (in %d of the %d looks it has compared). It goes by that "
                "where the screen cannot tell how far it slid.",
                where[0],
                where[1],
                "up and down" if way else "across",
                agreed[0],
                agreed[1],
            )

    def _cover_ground(self, lost: int) -> None:
        """Count the new ground of this look (T-993).

        A jolt is a look in which the screen changed and no camera move explains it.
        A quarter of the screen or more is a cut to another place. Being put back
        after dying can change far less of a plain level: that is a jolt that comes
        after a look that could be measured, finds the player somewhere else on the
        screen than it was last seen (further than its own size), and has a life lost
        within a few looks of it. Either way it is taken to be back where the go
        began, so ground it walks again pays nothing. A life lost without all that (a
        hit that costs health, a crash that leaves it where it was) changes nothing.

        Every other jolt is a move that could not be measured, and nothing is counted
        for it. Where those are common the count would pay for moving slowly enough to
        be measured, so nothing is counted at all while :attr:`follows_camera` is
        False.
        """
        moved, share = self._tracker.scroll, self._tracker.unexplained
        jolt = moved == (0, 0) and share >= JOLT
        if jolt and not self._was_jolt and self._elsewhere():
            self._jolt_at = self._looks
        if lost:
            self._lost_at = self._looks
        respawn = (
            self._jolt_at is not None
            and self._lost_at is not None
            and abs(self._jolt_at - self._lost_at) <= BACK_LOOKS
        )
        if respawn or (jolt and share >= CUT):
            self._ground.back()
            if respawn:
                self._jolt_at = self._lost_at = None
        elif jolt:
            self._measured.append(False)
        elif moved != (0, 0):
            self._measured.append(True)
        self._was_jolt = jolt
        if self._box is None:
            new = self._ground.look(moved)
        else:
            x, y, w, h = self._box
            new = self._ground.look(moved, (x + w // 2, y + h // 2), max(w, h))
        if self.follows_camera:
            self._new_ground += new  # otherwise counted and not paid, now or later

    def _elsewhere(self) -> bool:
        """It is seen further than its own size from where it was last seen."""
        if self._box is None or self._prior_box is None:
            return False
        (x, y, w, h), (px, py, _, _) = self._box, self._prior_box
        return max(abs(x - px), abs(y - py)) > max(w, h)

    def told(self, *, lost: int = 0) -> None:
        """Lives lost that no look will come for (the go ended): counted and put down to
        a fall or to the touch that was waiting, with no new screen to go by (T-988)."""
        if lost and not self._a_fall(lost):
            self._learner.life_lost()

    def _falling(self) -> bool:
        """At its last sighting it was coming down with nothing under it (and now it is
        gone). Coming down *through* the floor is no fall: many games throw a player
        that was hit off the bottom of the screen, and then the hit is the cause."""
        if self._before is None or self._last_box is None:
            return False
        return self._last_box[1] > self._before[1] and self._nothing_under

    def _a_fall(self, lost: int) -> bool:
        """Count ``lost`` lives, and say whether they went by a fall: within the last
        ``soon`` looks it went out of sight falling, and has not been seen since. A
        fall lets the touches still waiting go unblamed."""
        self._lives_lost += lost
        fell = self._fell_at is not None and self._looks - self._fell_at <= self._soon
        self._fell_at = None
        if fell:
            self._lives_after_fall += lost
            self._learner.let_go()
            return True
        if self._learner.waiting:
            self._lives_after_touch += lost
        return False

    def _find_me(self, tracks: tuple[Track, ...]) -> int | None:
        by_id = {t.id: t for t in tracks}
        if self._me is not None and self._me in by_id:
            return self._me
        seen = [t for t in tracks if t.missing == 0]
        if self._my_kind is not None:
            mine = [t for t in seen if t.kind == self._my_kind]
            if mine:
                return min(mine, key=lambda t: self._distance(t)).id
        if self._last_me is not None and seen:
            # Lost (cut out of the ground, half behind something): the nearest thing to
            # where it was last seen, if near enough, is it again (T-977).
            near = min(seen, key=lambda t: self._distance(t))
            if self._distance(near) <= 2 * FIRST_LOOK_REACH:
                return near.id
        if self._looks_this_go == 0 and seen:
            # The go starts where the player was found: the nearest thing there is it.
            px, py = self._player.start
            near = min(seen, key=lambda t: abs(t.centre[0] - px) + abs(t.centre[1] - py))
            if abs(near.centre[0] - px) + abs(near.centre[1] - py) <= FIRST_LOOK_REACH:
                return near.id
        return None

    def _distance(self, t: Track) -> int:
        if self._last_me is None:
            return 0
        return abs(t.centre[0] - self._last_me[0]) + abs(t.centre[1] - self._last_me[1])

    def describe(self) -> str:
        """What the AI's look at the screen has come to, for the log every ten goes (T-977)."""
        if self._looks == 0:
            return "The AI has not looked at the screen yet."
        share = round(100 * self._seen_me / self._looks)
        kinds = len(self._tracker.catalogue.kinds())
        roles = self.roles()
        hurts = sum(1 for r in roles if r.tag in (SpriteTag.ENEMY, SpriteTag.HAZARD))
        items = sum(1 for r in roles if r.tag is SpriteTag.ITEM)
        fell = (
            f" and {self._lives_after_fall} after falling off the bottom of the screen"
            if self._lives_after_fall
            else ""
        )
        lost = (
            ""
            if self.follows_camera
            else (
                f" It cannot follow the camera here ({len(self._measured) - sum(self._measured)} "
                f"of the last {len(self._measured)} moves could not be measured), so new "
                "ground is not counted."
            )
        )
        return (
            f"On the screen so far: {self._looks} looks, itself followed in {share}% of them; "
            f"{kinds} kinds of thing; {self._lives_lost} lives lost, "
            f"{self._lives_after_touch} of them soon after touching something{fell}; "
            f"{hurts} kinds found to hurt, {items} to pick up.{lost}"
        )

    def roles(self) -> tuple[Role, ...]:
        return self._learner.roles()

    def parts(self) -> tuple[tuple[str, int], ...]:
        """What the brain reads off the last look. First what is around the player
        (T-989): the floor either side, the way at its own height, what is under it.
        Then (T-978) how far the nearest moving thing and the nearest thing found to
        hurt are from it, across and up or down, in a few coarse answers (:func:`side`).
        Nothing when the player was not seen."""
        if self._box is None:
            return ()
        tracks = {t.id: t for t in self._tracker.tracks()}
        hurts = {r.kind for r in self.roles() if r.tag in (SpriteTag.ENEMY, SpriteTag.HAZARD)}
        x, y, w, h = self._box
        mx, my = x + w // 2, y + h // 2
        mine = (
            {self._me}
            if self._sign is None
            else {t.id for t in tracks.values() if within(t, self._box)}
        )
        out: list[tuple[str, int]] = list(self._around.parts()) if self._around else []

        def nearest(things: list[Track]) -> Track | None:
            return min(
                things,
                key=lambda t: abs(t.centre[0] - mx) + abs(t.centre[1] - my),
                default=None,
            )

        others = [t for t in tracks.values() if t.id not in mine and t.missing == 0]
        for name, pick in (
            ("mover", nearest([t for t in others if t.moves])),
            ("hurts", nearest([t for t in others if t.kind in hurts])),
        ):
            if pick is None or abs(pick.centre[0] - mx) > PART_REACH:
                continue
            out.append((f"screen:{name} dx", side(pick.centre[0] - mx)))
            out.append((f"screen:{name} dy", side(pick.centre[1] - my, near=None)))
        return tuple(out)

    def news(self) -> tuple[Role, ...]:
        """Roles that are new or changed since the last call (for the log)."""
        out: list[Role] = []
        for r in self.roles():
            now = (r.tag.value, r.confidence.name)
            if self._told.get(r.kind) != now:
                self._told[r.kind] = now
                out.append(r)
        return tuple(out)


__all__ = ["DEFAULT_LOOK_EVERY", "SOON_LOOKS", "Mark", "ScreenWatch", "side"]
