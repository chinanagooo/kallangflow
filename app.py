"""Minimal Flask web app wrapping the KallangFlow agent.

Sandbox/demo server.

The server is the source of truth for:
- attendee profile
- simulated clock
- event start/end
- transport crowd state
- the milestone passed to the LangGraph agent

The browser should never decide the event phase itself.
"""

from __future__ import annotations

import time

from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request

load_dotenv()

from kallangflow.agent import run_agent_for_attendee
from kallangflow.db import get_attendee, init_db, save_attendee
from kallangflow.models import Attendee
from kallangflow.simulator import (
    advance_time,
    create_world_state,
    simulate_crowd_buildup,
)

app = Flask(__name__)

init_db()

# --------------------------------------------------------------------
# Event configuration
# --------------------------------------------------------------------
# Keep these values explicit so the app does not silently inherit a
# different default from simulator.py.
EVENT_START_MINUTES = 19 * 60       # 19:00
EVENT_END_MINUTES = 22 * 60         # 22:00

# --------------------------------------------------------------------
# Shared world state
# --------------------------------------------------------------------
world = create_world_state(
    event_start_minutes=EVENT_START_MINUTES,
    event_end_minutes=EVENT_END_MINUTES,
)
simulate_crowd_buildup(world)

DEMO_ATTENDEE_ID = "A-DEMO"


def _default_attendee() -> Attendee:
    """Fallback used only if no profile has ever been saved."""
    return Attendee(
        id=DEMO_ATTENDEE_ID,
        home_location="Tampines",
        transport_preference="mrt",
    )


# --------------------------------------------------------------------
# Simulated clock
# --------------------------------------------------------------------
CLOCK = {
    "real_start": time.time(),
    "sim_start_minutes": world.current_time_minutes,
    "scale": 1.0,
    "manual_offset_minutes": 0,
}


def sync_world_clock() -> None:
    """Synchronise the in-memory world clock and crowd state."""
    elapsed_real_seconds = time.time() - CLOCK["real_start"]
    ticked_minutes = elapsed_real_seconds * CLOCK["scale"]

    world.current_time_minutes = int(
        CLOCK["sim_start_minutes"]
        + ticked_minutes
        + CLOCK["manual_offset_minutes"]
    )

    simulate_crowd_buildup(world)


def reset_world_clock() -> None:
    """Return the simulated world to its original state.

    Rebuilding the world is intentional. The crowd simulator can update
    transport-node prediction/load values over time, so simply moving the
    clock backwards is not enough to restore those values. A fresh world
    guarantees that the projected crowd state is reset as well.
    """
    global world

    world = create_world_state(
        event_start_minutes=EVENT_START_MINUTES,
        event_end_minutes=EVENT_END_MINUTES,
    )

    CLOCK["real_start"] = time.time()
    CLOCK["sim_start_minutes"] = world.current_time_minutes
    CLOCK["manual_offset_minutes"] = 0

    simulate_crowd_buildup(world)


def format_clock(minutes: int) -> str:
    minutes = int(minutes) % (24 * 60)
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


def get_event_phase(minutes: int) -> str:
    """Return the authoritative event phase for the current time."""
    if minutes < world.event_start_minutes:
        return "before"
    if minutes < world.event_end_minutes:
        return "during"
    return "after"


# --------------------------------------------------------------------
# Pages
# --------------------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html")


# --------------------------------------------------------------------
# World / simulated time API
# --------------------------------------------------------------------
@app.route("/api/world")
def get_world():
    """Return the current transport state and simulated clock."""
    sync_world_clock()

    nodes = {
        node_id: {
            "name": node.name,
            "load": node.load,
            "capacity": node.capacity,
            "congestion": node.congestion().value,
        }
        for node_id, node in world.transport_nodes.items()
    }

    return jsonify({
        "nodes": nodes,
        "current_time_minutes": world.current_time_minutes,
        "current_time_display": format_clock(world.current_time_minutes),
        "event_start_minutes": world.event_start_minutes,
        "event_end_minutes": world.event_end_minutes,
        "event_phase": get_event_phase(world.current_time_minutes),
    })


@app.route("/api/fast-forward", methods=["POST"])
def fast_forward():
    """Jump the simulated clock forward."""
    data = request.get_json(silent=True) or {}

    try:
        minutes = int(data.get("minutes", 30))
    except (TypeError, ValueError):
        return jsonify({"error": "minutes must be a whole number."}), 400

    if minutes <= 0:
        return jsonify({"error": "minutes must be positive."}), 400

    CLOCK["manual_offset_minutes"] += minutes

    # Keep compatibility with the simulator helper.
    advance_time(world, minutes=minutes)

    # Re-sync so the automatic clock and manual offset remain consistent.
    sync_world_clock()

    return jsonify({
        "current_time_minutes": world.current_time_minutes,
        "current_time_display": format_clock(world.current_time_minutes),
        "event_phase": get_event_phase(world.current_time_minutes),
    })


@app.route("/api/reset-time", methods=["POST"])
def reset_time():
    """Reset the simulated clock AND all simulated crowd/prediction state."""
    reset_world_clock()

    nodes = {
        node_id: {
            "name": node.name,
            "load": node.load,
            "capacity": node.capacity,
            "congestion": node.congestion().value,
        }
        for node_id, node in world.transport_nodes.items()
    }

    return jsonify({
        "nodes": nodes,
        "current_time_minutes": world.current_time_minutes,
        "current_time_display": format_clock(world.current_time_minutes),
        "event_start_minutes": world.event_start_minutes,
        "event_end_minutes": world.event_end_minutes,
        "event_phase": get_event_phase(world.current_time_minutes),
    })


# --------------------------------------------------------------------
# Attendee profile API
# --------------------------------------------------------------------
@app.route("/api/attendee", methods=["GET"])
def get_attendee_route():
    """Return the saved profile, if any."""
    attendee = get_attendee(DEMO_ATTENDEE_ID)

    if attendee is None:
        return jsonify(None)

    return jsonify({
        "home_location": attendee.home_location,
        "transport_preference": attendee.transport_preference,
        "accessibility_needs": attendee.accessibility_needs or "",
        "travelling_with": attendee.travelling_with or "",
    })


@app.route("/api/attendee", methods=["POST"])
def save_attendee_route():
    """Save/overwrite the attendee profile."""
    data = request.get_json(silent=True) or {}

    home_location = (data.get("home_location") or "").strip()

    # Frontend supports multiple transport selections.
    transport_preference = data.get("transport_preference") or ["mrt"]

    if isinstance(transport_preference, str):
        transport_preference = [transport_preference]

    # Flatten comma-separated values.
    transport_options = []

    for option in transport_preference:
        for value in str(option).split(","):
            value = value.strip().lower()
            if value:
                transport_options.append(value)

    # Remove duplicates while preserving selection order.
    transport_preference = list(dict.fromkeys(transport_options))

    accessibility_needs = (
        data.get("accessibility_needs") or ""
    ).strip() or None

    travelling_with = (
        data.get("travelling_with") or ""
    ).strip() or None

    if not home_location:
        return jsonify({"error": "Home location is required."}), 400

    allowed_transport = {
        "mrt",
        "bus",
        "walk",
        "taxi",
        "cycling",
    }

    if not transport_preference:
        return jsonify({"error": "Select at least one transport option."}), 400

    invalid_transport = [
        option
        for option in transport_preference
        if option not in allowed_transport
    ]

    if invalid_transport:
        return jsonify({
            "error": f"Invalid transport option: {', '.join(invalid_transport)}"
        }), 400

    transport_preference_string = ",".join(transport_preference)

    attendee = Attendee(
        id=DEMO_ATTENDEE_ID,
        home_location=home_location,
        transport_preference=transport_preference_string,
        accessibility_needs=accessibility_needs,
        travelling_with=travelling_with,
    )

    save_attendee(attendee)

    return jsonify({"status": "saved"})


# --------------------------------------------------------------------
# LangGraph recommendation API
# --------------------------------------------------------------------
@app.route("/api/plan", methods=["POST"])
def plan_journey():
    """Run the LangGraph agent using the authoritative current phase."""
    sync_world_clock()

    attendee = get_attendee(DEMO_ATTENDEE_ID) or _default_attendee()

    try:
        # IMPORTANT:
        # Calculate the phase on the server immediately before calling
        # LangGraph. The frontend cannot accidentally become out of sync.
        milestone = get_event_phase(world.current_time_minutes)

        result = run_agent_for_attendee(
            attendee,
            world,
            milestone=milestone,
        )

        response = {
            "recommendation": result["recommendation"],
            "current_time_minutes": world.current_time_minutes,
            "current_time_display": format_clock(world.current_time_minutes),
            "event_start_minutes": world.event_start_minutes,
            "event_end_minutes": world.event_end_minutes,
            "event_phase": milestone,
        }

        # Preserve route information if the agent/app layer provides it.
        # This is intentionally optional so the map still works even if
        # the current agent result only contains a recommendation.
        for key in ("route_info", "node_id", "gate", "geometry", "profile"):
            if key in result:
                response[key] = result[key]

        return jsonify(response)

    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)
