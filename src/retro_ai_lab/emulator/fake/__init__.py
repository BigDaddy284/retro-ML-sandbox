"""A deterministic fake emulator and the toy games it runs."""

from retro_ai_lab.emulator.fake.adapter import FakeEmulatorAdapter
from retro_ai_lab.emulator.fake.toy_hopper import HopperState, ToyHopper
from retro_ai_lab.emulator.fake.toy_racer import RacerState, ToyRacer

__all__ = ["FakeEmulatorAdapter", "HopperState", "RacerState", "ToyHopper", "ToyRacer"]
