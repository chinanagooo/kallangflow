"""Simulated environment for KallangFlow.

Standing in for real Singapore transport/crowd APIs, which are not
available for a hackathon MVP. Everything in this file is clearly
synthetic — labelled as such everywhere it surfaces in the demo/UI.
"""
from __future__ import annotations

import random

from .models import Facility, Gate, TransportNode, WorldState


def create_world_state(event_start_minutes: int = 19 * 60 + 30,
                        event_end_minutes: int = 22 * 60,
                        seed: int | None = 7) -> WorldState:
    """Build a fresh simulated world: Kallang precinct transport nodes,
    gates, and venue facilities, all starting at low load."""
    if seed is not None:
        random.seed(seed)

    transport_nodes = {
        "stadium_mrt": TransportNode("stadium_mrt", "Stadium MRT", "mrt", capacity=9000),
        "kallang_mrt": TransportNode("kallang_mrt", "Kallang MRT", "mrt", capacity=6000),
        "mountbatten_mrt": TransportNode("mountbatten_mrt", "Mountbatten MRT", "mrt", capacity=5000),
        "nicoll_highway_mrt": TransportNode("nicoll_highway_mrt", "Nicoll Highway MRT", "mrt", capacity=4000),
        "bus_hub": TransportNode("bus_hub", "Kallang Bus Interchange", "bus", capacity=6000),
        "taxi_ride_hail": TransportNode("taxi_ride_hail", "Taxi / Ride-hailing pickup", "taxi", capacity=3000),
    }
    gates = {
        "gate_3": Gate("gate_3", "Gate 3", capacity=6000),
        "gate_5": Gate("gate_5", "Gate 5", capacity=6000),
        "gate_7": Gate("gate_7", "Gate 7 (step-free)", capacity=3000),
    }
    facilities = [
        Facility("toilet_near_3", "toilet", "gate_3"),
        Facility("toilet_near_5", "toilet", "gate_5"),
        Facility("food_near_5", "food", "gate_5"),
        Facility("merch_near_7", "merch", "gate_7"),
    ]

    return WorldState(
        current_time_minutes=event_start_minutes - 105,  # 1h45m before doors
        event_start_minutes=event_start_minutes,
        event_end_minutes=event_end_minutes,
        transport_nodes=transport_nodes,
        gates=gates,
        facilities=facilities,
    )


def advance_time(world: WorldState, minutes: int) -> None:
    world.current_time_minutes += minutes


def _proximity_factor(world: WorldState, anchor_minutes: int, window: int = 120) -> float:
    """0..1 factor that rises as current time approaches `anchor_minutes`
    (used for both pre-event buildup and post-event exodus)."""
    delta = abs(world.current_time_minutes - anchor_minutes)
    if delta >= window:
        return 0.0
    return 1.0 - (delta / window)


def simulate_crowd_buildup(world: WorldState) -> None:
    """Advance the simulated congestion at gates and transport nodes,
    following simple logistic-ish curves around event start and end.
    This replaces a real crowd-sensing feed."""
    pre_event = _proximity_factor(world, world.event_start_minutes)
    post_event = _proximity_factor(world, world.event_end_minutes)

    # Gates fill up before the event starts.
    for gate in world.gates.values():
        target = int(gate.capacity * pre_event * random.uniform(0.7, 1.0))
        gate.load = max(gate.load, target)

    # Transport nodes fill up mainly after the event ends (the exodus),
    # with a small baseline from ordinary pre-event arrivals.
    for node in world.transport_nodes.values():
        baseline = int(node.capacity * pre_event * 0.15)
        exodus = int(node.capacity * post_event * random.uniform(0.75, 1.0))
        node.load = max(node.load, baseline, exodus)

    # Facility queues loosely track nearby gate congestion.
    for fac in world.facilities:
        gate = world.gates.get(fac.near_gate)
        if gate:
            ratio = gate.load / max(gate.capacity, 1)
            fac.queue_minutes = int(ratio * 15)


def inject_disruption(world: WorldState, node_id: str, note: str = "service disruption") -> None:
    """Manually trigger a disruption during the live demo, e.g. an MRT
    line going down. Spikes that node's load and records the event."""
    node = world.transport_nodes.get(node_id) or world.gates.get(node_id)
    if node is None:
        raise ValueError(f"Unknown node: {node_id}")
    node.load = int(node.capacity * 0.95)
    world.disruptions.append(f"{node_id}: {note}")


def project_congestion_in(world: WorldState, node_id: str, minutes_ahead: int = 8) -> str:
    """Cheap deterministic projection used by the predict tool: extrapolate
    current load trend forward `minutes_ahead` using the same proximity
    curve. Not a real forecasting model — good enough for a hackathon demo."""
    node = world.transport_nodes.get(node_id)
    if node is None:
        return "unknown"
    future_time = world.current_time_minutes + minutes_ahead
    future_post = _proximity_factor(
        WorldState(future_time, world.event_start_minutes, world.event_end_minutes),
        world.event_end_minutes,
    )
    projected_load = max(node.load, int(node.capacity * future_post * 0.9))
    ratio = projected_load / max(node.capacity, 1)
    if ratio < 0.4:
        return "low"
    if ratio < 0.8:
        return "moderate"
    return "high"
