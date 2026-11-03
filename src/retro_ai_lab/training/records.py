"""Plain records shared by the training service and the saved brain."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from retro_ai_lab.core import ConfigurationError
from retro_ai_lab.training.episode import EpisodeResult


class Mode(Enum):
    INDEPENDENT = "independent"
    GUIDED = "guided"
    MANUAL = "manual"

    @property
    def label(self) -> str:
        return {
            Mode.INDEPENDENT: "On its own",
            Mode.GUIDED: "With your help",
            Mode.MANUAL: "You play",
        }[self]

    @property
    def description(self) -> str:
        return {
            Mode.INDEPENDENT: "The AI plays and learns by itself.",
            Mode.GUIDED: "The AI plays while you change rewards and goals.",
            Mode.MANUAL: "You play; the AI watches and learns from you.",
        }[self]

    @property
    def ai_plays(self) -> bool:
        return self is not Mode.MANUAL

    @property
    def allows_guidance(self) -> bool:
        """May the child change rewards and goals while a go is running?"""
        return self is not Mode.INDEPENDENT


class Speed(Enum):
    LIVE = "live"  # one step per game tick: watch the AI play
    FAST = "fast"  # as many steps as fit: background practice

    @property
    def label(self) -> str:
        return "Watch it play" if self is Speed.LIVE else "Practise fast"


class Status(Enum):
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"


@dataclass(frozen=True, slots=True)
class EpisodeRecord:
    number: int  # 1-based, over the life of the brain
    mode: Mode
    result: EpisodeResult
    frame: int  # observation frame when the go ended

    def to_dict(self) -> dict[str, Any]:
        return {
            "number": self.number,
            "mode": self.mode.value,
            "result": self.result.to_dict(),
            "frame": self.frame,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EpisodeRecord:
        try:
            return cls(
                int(data["number"]),
                Mode(data["mode"]),
                EpisodeResult.from_dict(data["result"]),
                int(data["frame"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ConfigurationError(f"Bad go record: {exc}") from exc


__all__ = ["EpisodeRecord", "Mode", "Speed", "Status"]
