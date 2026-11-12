"""Hierarchical learning: a skill per goal step (DESIGN.md §6.2).

"Get to the checkpoint" and "finish the lap" can want different moves in
the same situation, and one table or network has to average them. The
skills learner keeps a separate inner learner per goal step and routes
every choice and every lesson to the skill for the step the AI is working
on; with no goal step it uses a general skill. Each skill is an ordinary
registered learner, so a skill can be a table or a network.
"""

from __future__ import annotations

from random import Random
from typing import Any, ClassVar

from retro_ai_lab.core import ConfigurationError
from retro_ai_lab.learning.actions import ActionSpace
from retro_ai_lab.learning.learner import (
    Q_TABLE,
    Decision,
    Learner,
    algorithms,
    learner_from_dict,
    make_learner,
    register_learner,
)
from retro_ai_lab.learning.qlearning import Settings, tries_words
from retro_ai_lab.learning.state import State

SCHEMA_VERSION = 1
SKILLS = "skills"
DEFAULT_SKILL = "*"  # the general skill, used when there is no goal step


@register_learner
class SkillsLearner:
    algorithm: ClassVar[str] = SKILLS

    def __init__(
        self,
        space: ActionSpace,
        settings: Settings | None = None,
        *,
        seed: int | None = None,
        inner: str = Q_TABLE,
    ) -> None:
        if inner == SKILLS or inner not in algorithms():
            raise ConfigurationError(
                f"A skill can't be learned with {inner!r} "
                f"(choices: {', '.join(n for n in algorithms() if n != SKILLS)})."
            )
        self._space = space
        self._settings = settings or Settings()
        self._inner = inner
        self._rng = Random(seed)
        self._skills: dict[str, Learner] = {}
        self._active = DEFAULT_SKILL
        self._episodes = 0
        self._skill_for(DEFAULT_SKILL)

    # ------------------------------------------------------------ skills

    @property
    def inner(self) -> str:
        """What kind of learner each skill is."""
        return self._inner

    @property
    def skill(self) -> str:
        return self._active

    @property
    def skills(self) -> tuple[str, ...]:
        return tuple(self._skills)

    def skill_learner(self, skill_id: str) -> Learner:
        try:
            return self._skills[skill_id]
        except KeyError:
            raise ConfigurationError(f"There is no skill {skill_id!r}.") from None

    def _skill_for(self, skill_id: str) -> Learner:
        learner = self._skills.get(skill_id)
        if learner is None:
            learner = make_learner(
                self._inner, self._space, self._settings, seed=self._rng.getrandbits(32)
            )
            self._skills[skill_id] = learner
        return learner

    def set_skill(self, skill_id: str | None) -> Learner:
        """Make the skill for a goal step (None: the general skill) the one that plays."""
        if skill_id is None:
            skill_id = DEFAULT_SKILL
        if not isinstance(skill_id, str) or not skill_id:
            raise ConfigurationError("A skill needs a name.")
        self._active = skill_id
        return self._skill_for(skill_id)

    def adopt(self, skill_id: str, learner: Learner) -> None:
        """Use an existing learner as a skill (how a plain brain becomes skills)."""
        if learner.space != self._space:
            raise ConfigurationError("The learner's controls don't match.")
        if learner.algorithm == SKILLS:
            raise ConfigurationError("A skill can't be made of skills.")
        self._skills[skill_id] = learner
        self._inner = learner.algorithm

    @property
    def _current(self) -> Learner:
        return self._skills[self._active]

    # ------------------------------------------------------------ Learner contract

    @property
    def space(self) -> ActionSpace:
        return self._space

    @property
    def settings(self) -> Settings:
        return self._settings

    @property
    def epsilon(self) -> float:
        return self._current.epsilon

    @property
    def updates(self) -> int:
        return sum(s.updates for s in self._skills.values())

    @property
    def episodes(self) -> int:
        return self._episodes

    @property
    def known(self) -> int:
        return sum(s.known for s in self._skills.values())

    @property
    def last_decision(self) -> Decision | None:
        return self._current.last_decision

    def stats(self) -> str:
        n = len(self._skills)
        names = ", ".join("general" if s == DEFAULT_SKILL else s for s in self._skills)
        active = "general" if self._active == DEFAULT_SKILL else self._active
        known, updates = self.known, self.updates
        return (
            f"{n} skill{'s' if n != 1 else ''} ({names}); using {active}. "
            f"Knows {known} situation{'s' if known != 1 else ''} in all, learned from "
            f"{updates} step{'s' if updates != 1 else ''} over {self._episodes} "
            f"go{'es' if self._episodes != 1 else ''}; "
            f"{tries_words(self.epsilon, self._settings.explore_run)}"
        )

    def choose(self, state: State) -> int:
        return self._current.choose(state)

    def values(self, state: State) -> tuple[float, ...]:
        return self._current.values(state)

    def configure(self, settings: Settings) -> None:
        if not isinstance(settings, Settings):
            raise ConfigurationError("Learning settings must be a Settings.")
        self._settings = settings
        for skill in self._skills.values():
            skill.configure(settings)

    def greedy(self, state: State) -> int:
        return self._current.greedy(state)

    def learn(
        self,
        state: State,
        action: int,
        reward: float,
        next_state: State,
        done: bool = False,
        *,
        steps: int = 1,
    ) -> float:
        return self._current.learn(state, action, reward, next_state, done, steps=steps)

    def end_episode(self, reward: float | None = None) -> float:
        self._episodes += 1
        for s in self._skills.values():
            s.end_episode(reward)
        return self._current.epsilon

    def resume_from(self, other: Learner) -> None:
        """Carry on another learner's goes (its exploration lives in the skill adopted)."""
        self._episodes = other.episodes

    def reset(self) -> None:
        self._skills.clear()
        self._active = DEFAULT_SKILL
        self._episodes = 0
        self._skill_for(DEFAULT_SKILL)

    def policy(self) -> dict[str, int]:
        """Every skill's policy, each situation prefixed with ``skill=<n>`` (creation order)."""
        merged: dict[str, int] = {}
        for n, learner in enumerate(self._skills.values()):
            for key, action in learner.policy().items():
                merged[f"skill={n};{key}" if key else f"skill={n}"] = action
        return merged

    # ------------------------------------------------------------ persistence

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA_VERSION,
            "algorithm": self.algorithm,
            "space": self._space.to_dict(),
            "settings": self._settings.to_dict(),
            "inner": self._inner,
            "active": self._active,
            "episodes": self._episodes,
            "rng": self._rng.getstate(),
            # A list, so the order (which numbers the skills in policy()) survives a save
            # that sorts keys.
            "skills": [{"id": sid, "learner": s.to_dict()} for sid, s in self._skills.items()],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any], *, space: ActionSpace | None = None) -> SkillsLearner:
        from retro_ai_lab.learning.qlearning import _rng_state

        try:
            schema = int(data.get("schema", SCHEMA_VERSION))
            if schema > SCHEMA_VERSION:
                raise ConfigurationError(f"This brain is from a newer version (schema {schema}).")
            if data.get("algorithm", cls.algorithm) != cls.algorithm:
                raise ConfigurationError("This brain was not learned as skills.")
            saved = ActionSpace.from_dict(data["space"])
            if space is not None and space != saved:
                raise ConfigurationError(
                    "This brain was trained with different controls than this game has."
                )
            agent = cls(
                saved, Settings(**data.get("settings", {})), inner=str(data.get("inner", Q_TABLE))
            )
            skills = data.get("skills", [])
            if isinstance(skills, dict):  # tolerated: {id: learner}
                skills = [{"id": sid, "learner": item} for sid, item in skills.items()]
            if not isinstance(skills, list):
                raise ConfigurationError("Bad brain data: skills must be a list.")
            agent._skills = {
                str(item["id"]): learner_from_dict(item["learner"], space=saved) for item in skills
            }
            if DEFAULT_SKILL not in agent._skills:
                agent._skill_for(DEFAULT_SKILL)
            agent._active = str(data.get("active", DEFAULT_SKILL))
            if agent._active not in agent._skills:
                agent._active = DEFAULT_SKILL
            agent._episodes = int(data.get("episodes", 0))
            if "rng" in data:
                agent._rng.setstate(_rng_state(data["rng"]))
        except (KeyError, TypeError, ValueError) as exc:
            raise ConfigurationError(f"Bad brain data: {exc}") from exc
        return agent


__all__ = ["DEFAULT_SKILL", "SCHEMA_VERSION", "SKILLS", "SkillsLearner"]
