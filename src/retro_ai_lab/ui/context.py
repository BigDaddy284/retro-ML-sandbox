"""Everything the UI needs from the layers below, bundled once."""

from __future__ import annotations

from dataclasses import dataclass, field

from retro_ai_lab.core.events import EventBus
from retro_ai_lab.emulator.registry import AdapterRegistry, default_registry
from retro_ai_lab.services import ConnectionService, DiscoveryService, ObservationService
from retro_ai_lab.ui.bridge import EventBridge


@dataclass
class AppContext:
    bus: EventBus = field(default_factory=EventBus)
    registry: AdapterRegistry = field(default_factory=default_registry)
    connection: ConnectionService = field(init=False)
    observation: ObservationService = field(init=False)
    discovery: DiscoveryService = field(init=False)
    bridge: EventBridge = field(init=False)

    def __post_init__(self) -> None:
        self.connection = ConnectionService(self.registry, self.bus)
        self.observation = ObservationService(self.connection, self.bus)
        self.discovery = DiscoveryService(self.connection, self.observation, self.bus)
        self.bridge = EventBridge(self.bus)


__all__ = ["AppContext"]
