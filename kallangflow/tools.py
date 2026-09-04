"""Agent tools.

Each tool is a thin wrapper around the simulator/world state. The
docstrings are what the model reads to decide when to call each tool —
kept short and specific, per the hackathon's own guidance (a vague
docstring is the #1 cause of an agent calling the wrong tool).

Tools are built via `make_tools(world)` so they close over a live
WorldState instance rather than a global — this keeps things testable
and lets a demo run multiple worlds side by side if needed.
"""
from __future__ import annotations

from langchain_core.tools import tool

from .models import WorldState
from .simulator import project_congestion_in

# Populated per-world by make_tools(); simple module-level registry
# keeps the @tool-decorated functions free of extra arguments (the
# model must only see the schema declared in the decorator).
_ACTIVE_WORLD: WorldState | None = None
GUARDIAN_LOG: list[str] = []


def make_tools(world: WorldState):
    """Bind all tools to `world` and return the tool list to hand to
    the model via .bind_tools([...])."""
    global _ACTIVE_WORLD
    _ACTIVE_WORLD = world
    return [
        get_transport_status,
        get_gate_congestion,
        get_weather,
        get_venue_facility_status,
        calculate_route,
        notify_guardian,
    ]


@tool
def get_transport_status(node_id: str) -> str:
    """Return the current congestion level and an 8-minute-ahead
    projection for a transport node. Valid node_id values: stadium_mrt,
    kallang_mrt, mountbatten_mrt, nicoll_highway_mrt, bus_hub,
    taxi_ride_hail."""
    node = _ACTIVE_WORLD.transport_nodes.get(node_id)
    if node is None:
        return f"No such transport node: {node_id}"
    projected = project_congestion_in(_ACTIVE_WORLD, node_id, minutes_ahead=8)
    return (f"{node.name}: current congestion = {node.congestion().value}, "
            f"projected in 8 min = {projected}")


@tool
def get_gate_congestion(gate_id: str) -> str:
    """Return the current congestion level for a venue entrance gate.
    Valid gate_id values: gate_3, gate_5, gate_7 (gate_7 is step-free)."""
    gate = _ACTIVE_WORLD.gates.get(gate_id)
    if gate is None:
        return f"No such gate: {gate_id}"
    return f"{gate.name}: congestion = {gate.congestion().value}"


@tool
def get_weather(location: str) -> str:
    """Return the current simulated weather for the venue precinct.
    location is informational only (e.g. 'Kallang')."""
    return f"Weather at {location}: {_ACTIVE_WORLD.weather}"


@tool
def get_venue_facility_status(facility_kind: str) -> str:
    """Return the nearest facility of a given kind and its queue time.
    facility_kind must be one of: toilet, food, merch."""
    matches = [f for f in _ACTIVE_WORLD.facilities if f.kind == facility_kind]
    if not matches:
        return f"No {facility_kind} facilities found."
    best = min(matches, key=lambda f: f.queue_minutes)
    return (f"Nearest {facility_kind} is near {best.near_gate}, "
            f"queue ~{best.queue_minutes} min.")


@tool
def calculate_route(origin: str, destination_node_id: str,
                     accessibility_needs: str = "") -> str:
    """Estimate travel time from origin (a home location name) to a
    transport node, adjusted for accessibility needs if given (e.g.
    'step-free access' routes avoid stairs and may take longer)."""
    node = _ACTIVE_WORLD.transport_nodes.get(destination_node_id)
    if node is None:
        return f"No such destination: {destination_node_id}"
    base_minutes = 35  # flat heuristic; a real system would call a routing API
    if accessibility_needs:
        base_minutes += 10
    congestion_penalty = {"low": 0, "moderate": 5, "high": 15}[node.congestion().value]
    total = base_minutes + congestion_penalty
    return f"Estimated {total} min from {origin} to {node.name} (accessibility: {accessibility_needs or 'none'})."


@tool
def notify_guardian(attendee_id: str, message: str) -> str:
    """Send a short status update to the attendee's designated Journey
    Guardian (a family member/loved one). In this simulation the
    message is logged, not actually sent."""
    entry = f"[GUARDIAN][{attendee_id}] {message}"
    GUARDIAN_LOG.append(entry)
    return "Guardian notified (simulated)."
