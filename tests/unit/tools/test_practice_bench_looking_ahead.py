"""T-1052: the practice bench counts the plays ahead of its goes and the searches made
for a life seen going, read where a go and its result have them."""

from __future__ import annotations

from retro_ai_lab.tools.practice_bench import Report, _train
from tests.unit.tools.test_practice_bench_count import TO_THE_NEXT, Step, Story


class Looking(Story):
    """A story whose goes look ahead: ``looked[go]`` is the runner's count after each
    of the go's steps, ``ahead[go]`` what the go's result says of its searches."""

    def __init__(self, goes: list[list[Step]], looked: list[list[int]], ahead: list[int]) -> None:
        super().__init__(goes)
        self._looked = looked
        self._ahead = ahead
        self._step = 0

    def step(self) -> None:
        go = self.episodes
        super().step()
        if self.episodes == go:
            self.episode.looked_ahead = self._looked[go][self._step]
            self._step += 1
            return
        self.episode.looked_ahead = self._looked[go][self._step]
        self._step = 0
        result = self.history[-1].result
        result.ways_sought, result.ways_found, result.ways_ahead = 5, 3, self._ahead[go]


def test_the_bench_counts_the_looks_ahead_and_the_searches_before_a_life_went() -> None:
    goes = [list(TO_THE_NEXT), list(TO_THE_NEXT), list(TO_THE_NEXT)]
    # A go's count is read at every step and the most it showed is the go's: by the
    # step that ends a go the service may have begun the next, with the count at 0.
    story = Looking(
        goes, looked=[[0, 1, 1, 4, 0], [0, 0, 0, 0, 0], [2, 2, 7, 7, 0]], ahead=[2, 0, 1]
    )
    report = Report(seed=0, goes=3, max_steps=100, algorithm="q_table")
    _train(story, story.game, report)  # type: ignore[arg-type]
    assert (report.looked_ahead, report.ways_ahead) == (4 + 0 + 7, 3)
    assert (report.ways_sought, report.ways_found) == (15, 9)
    assert report.lines()[-1].startswith("Looking ahead: 11 times a thing")
    assert report.lines()[-1].endswith("that would have cost a life in 3.")
    # A go and a result that know nothing of looking ahead count as none.
    plain = Story(goes)
    report = Report(seed=0, goes=3, max_steps=100, algorithm="q_table")
    _train(plain, plain.game, report)  # type: ignore[arg-type]
    assert (report.looked_ahead, report.ways_ahead) == (0, 0)
    assert not any(line.startswith("Looking ahead") for line in report.lines())
