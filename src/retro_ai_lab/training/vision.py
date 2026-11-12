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
"""

from __future__ import annotations

from retro_ai_lab.core import Frame
from retro_ai_lab.observation.objects import ObjectCatalogue
from retro_ai_lab.observation.signature import Box
from retro_ai_lab.observation.sprites import SpriteTag
from retro_ai_lab.observation.tracking import ObjectTracker, Track
from retro_ai_lab.understanding.player import PlayerFinding
from retro_ai_lab.understanding.roles import Happening, Role, RoleLearner, within

DEFAULT_LOOK_EVERY = 2  # steps between two looks (a look costs a screenshot and ~15 ms)
SOON_LOOKS = 30  # looks after a touch in which a life lost is put down to it (~4 s)
FIRST_LOOK_REACH = 16  # pixels from where the player was found, at a go's first look
PART_REACH = 128  # pixels across within which a thing is in the brain's view (T-978)
NEAR = 48  # pixels across within which a thing is "close" rather than "coming"
LEVEL = 8  # pixels within which a thing is level with the player, or on top of it
HIDE_MARGIN = 1  # pixels round the player's own colours left out with it (its outline)


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


class ScreenWatch:
    def __init__(
        self,
        player: PlayerFinding,
        *,
        catalogue: ObjectCatalogue | None = None,
        soon: int = SOON_LOOKS,
    ) -> None:
        self._player = player
        self._tracker = ObjectTracker(catalogue)
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

    def begin_go(self) -> None:
        """A go starts from the checkpoint: new tracks, the player looked for afresh."""
        self._tracker.reset()
        self._me = None
        self._box = None
        self._last_me = None
        self._looks_this_go = 0

    def look(self, frame: Frame, *, lost: int = 0, scored: int = 0) -> None:
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
            tracks = self._tracker.update(frame, hide=hide)
        else:
            tracks = self._tracker.update(frame)
            self._me = self._find_me(tracks)
            me = next((t for t in tracks if t.id == self._me), None)
            if me is not None and me.missing == 0:
                self._box = (me.x, me.y, me.width, me.height)
                if me.kind is not None and not me.cut_out and not me.merged:
                    self._my_kind = me.kind
            else:
                self._box = None
        if self._box is not None:
            x, y, w, h = self._box
            self._seen_me += 1
            self._last_me = (x + w // 2, y + h // 2)
        happened: list[Happening] = []
        if lost:
            happened.append(Happening.LIFE_LOST)
            self._lives_lost += 1
            if self._learner.waiting:
                self._lives_after_touch += 1
        if scored:
            happened.append(Happening.SCORE_UP)
        self._learner.observe(
            self._looks, tracks, self._me, happened, me=self._box if self._sign else None
        )
        self._looks += 1
        self._looks_this_go += 1

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
        return (
            f"On the screen so far: {self._looks} looks, itself followed in {share}% of them; "
            f"{kinds} kinds of thing; {self._lives_lost} lives lost, "
            f"{self._lives_after_touch} of them soon after touching something; "
            f"{hurts} kinds found to hurt, {items} to pick up."
        )

    def roles(self) -> tuple[Role, ...]:
        return self._learner.roles()

    def parts(self) -> tuple[tuple[str, int], ...]:
        """What the brain reads off the last look (T-978): how far the nearest moving
        thing and the nearest thing found to hurt are from the player, across and up or
        down, in a few coarse answers (:func:`side`). Nothing when the player was not seen."""
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
        out: list[tuple[str, int]] = []

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


__all__ = ["DEFAULT_LOOK_EVERY", "SOON_LOOKS", "ScreenWatch", "side"]
