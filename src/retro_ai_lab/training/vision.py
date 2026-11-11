"""The AI watches the screen while it plays and learns what things are (T-976).

Once the AI knows which thing on the screen is itself (T-974), every few steps
of a go it looks at the screen: the tracker follows the things (T-972), the
player's track is picked out (by its kind, or at the go's first look by where
the player was found), and the touches are set against the lives lost and the
score rises memory reported in the same steps (T-975). Between goes the
service asks :meth:`ScreenWatch.roles` and logs what changed.

The kinds ("thing 3") live as long as the app runs; keeping them with the
game across sessions is a later task.
"""

from __future__ import annotations

from retro_ai_lab.core import Frame
from retro_ai_lab.observation.objects import ObjectCatalogue
from retro_ai_lab.observation.tracking import ObjectTracker, Track
from retro_ai_lab.understanding.player import PlayerFinding
from retro_ai_lab.understanding.roles import Happening, Role, RoleLearner

DEFAULT_LOOK_EVERY = 2  # steps between two looks (a look costs a screenshot and ~15 ms)
SOON_LOOKS = 30  # looks after a touch in which a life lost is put down to it (~4 s)
FIRST_LOOK_REACH = 16  # pixels from where the player was found, at a go's first look


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
        self._me: int | None = None
        self._looks = 0
        self._looks_this_go = 0
        self._told: dict[str, tuple[str, str]] = {}

    @property
    def player(self) -> PlayerFinding:
        return self._player

    @property
    def looks(self) -> int:
        return self._looks

    @property
    def tracker(self) -> ObjectTracker:
        return self._tracker

    @property
    def player_track(self) -> int | None:
        return self._me

    def begin_go(self) -> None:
        """A go starts from the checkpoint: new tracks, the player looked for afresh."""
        self._tracker.reset()
        self._me = None
        self._looks_this_go = 0

    def look(self, frame: Frame, *, lost: int = 0, scored: int = 0) -> None:
        tracks = self._tracker.update(frame)
        self._me = self._find_me(tracks)
        happened: list[Happening] = []
        if lost:
            happened.append(Happening.LIFE_LOST)
        if scored:
            happened.append(Happening.SCORE_UP)
        self._learner.observe(self._looks, tracks, self._me, happened)
        self._looks += 1
        self._looks_this_go += 1

    def _find_me(self, tracks: tuple[Track, ...]) -> int | None:
        by_id = {t.id: t for t in tracks}
        if self._me is not None and self._me in by_id:
            return self._me
        seen = [t for t in tracks if t.missing == 0]
        if self._player.kind is not None:
            mine = [t for t in seen if t.kind == self._player.kind]
            if mine:
                return min(mine, key=lambda t: t.id).id
        if self._looks_this_go == 0 and seen:
            # The go starts where the player was found: the nearest thing there is it.
            px, py = self._player.start
            near = min(seen, key=lambda t: abs(t.centre[0] - px) + abs(t.centre[1] - py))
            if abs(near.centre[0] - px) + abs(near.centre[1] - py) <= FIRST_LOOK_REACH:
                return near.id
        return None

    def roles(self) -> tuple[Role, ...]:
        return self._learner.roles()

    def news(self) -> tuple[Role, ...]:
        """Roles that are new or changed since the last call (for the log)."""
        out: list[Role] = []
        for r in self.roles():
            now = (r.tag.value, r.confidence.name)
            if self._told.get(r.kind) != now:
                self._told[r.kind] = now
                out.append(r)
        return tuple(out)


__all__ = ["DEFAULT_LOOK_EVERY", "SOON_LOOKS", "ScreenWatch"]
