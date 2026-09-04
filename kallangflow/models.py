"""Core data models for KallangFlow.

Everything here is a plain dataclass — no AI involved. This is the
"world" the agents perceive and act on. In this MVP the world is
simulated (see simulator.py); swapping in real transport/crowd APIs
later means changing simulator.py only, not these shapes.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class CongestionLevel(str, Enum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"


@dataclass
class TransportNode:
    id: str
    name: str
    kind: str  # "mrt" | "bus" | "taxi" | "walk"
    capacity: int
    load: int = 0

    def congestion(self) -> CongestionLevel:
        ratio = self.load / max(self.capacity, 1)
        if ratio < 0.4:
            return CongestionLevel.LOW
        if ratio < 0.8:
            return CongestionLevel.MODERATE
        return CongestionLevel.HIGH


@dataclass
class Gate:
    id: str
    name: str
    capacity: int
    load: int = 0

    def congestion(self) -> CongestionLevel:
        ratio = self.load / max(self.capacity, 1)
        if ratio < 0.4:
            return CongestionLevel.LOW
        if ratio < 0.8:
            return CongestionLevel.MODERATE
        return CongestionLevel.HIGH


@dataclass
class Facility:
    id: str
    kind: str  # "toilet" | "food" | "merch"
    near_gate: str
    queue_minutes: int = 0


@dataclass
class Attendee:
    id: str
    home_location: str
    transport_preference: str  # "mrt" | "bus" | "taxi" | "any"
    accessibility_needs: str | None = None   # e.g. "step-free access"
    travelling_with: str | None = None       # e.g. "elderly parent"
    event_start: str = "19:30"
    assigned_transport_node: str | None = None  # filled in once recommended


@dataclass
class WorldState:
    """The single source of truth the agents read from and (via the
    simulator) write to. Deliberately simple / in-memory — a real
    deployment would back this with a live feed."""
    current_time_minutes: int  # minutes since midnight, simulated clock
    event_start_minutes: int
    event_end_minutes: int
    weather: str = "clear"
    disruptions: list[str] = field(default_factory=list)
    transport_nodes: dict[str, TransportNode] = field(default_factory=dict)
    gates: dict[str, Gate] = field(default_factory=dict)
    facilities: list[Facility] = field(default_factory=list)
