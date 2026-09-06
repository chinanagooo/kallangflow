"""Minimal Flask web app wrapping the KallangFlow agent.

Run with:  python app.py
Then open: http://localhost:5000

This is a starting point, not a production deployment. See the notes
inline below for what would need to change before real multi-user use.

Attendee profiles are saved to the same SQLite database (attendees.db,
via kallangflow/db.py) that demo.py uses — a profile set up through this
web form is what demo.py will offer to reuse, and vice versa.

Simulated clock: world.current_time_minutes now advances on its own in
real time (scaled up, so you don't have to wait real hours to see the
event progress), and a "fast-forward" button lets you jump ahead
instantly for demo purposes. Both paths update the same world object
that agent.py's monitor_node reads when building its prompt, so the
agent's next call genuinely reflects the new time — no agent.py changes
needed.
"""
from __future__ import annotations

import time

from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request

load_dotenv()

from kallangflow.agent import run_agent_for_attendee
from kallangflow.db import get_attendee, init_db, save_attendee
from kallangflow.models import Attendee
from kallangflow.simulator import advance_time, create_world_state, simulate_crowd_buildup

app = Flask(__name__)

init_db()

# --------------------------------------------------------------------
# Event milestones. Change these two lines to move the whole simulation
# — the LLM agent already reads world.event_start_minutes /
# world.event_end_minutes when building its prompt (see agent.py's
# monitor_node), so no agent.py changes are needed to shift these.
# --------------------------------------------------------------------
EVENT_START_MINUTES = 19 * 60       # 19:00
EVENT_END_MINUTES = 22 * 60         # 22:00

# --------------------------------------------------------------------
# Shared world state, held in memory for this single Flask process.
#
# NOTE: this only works because Flask's dev server runs one process /
# one worker. If you deploy this behind gunicorn with multiple workers,
# or scale to multiple machines, each worker gets its OWN copy of
# `world` and they will disagree with each other. Before doing that,
# move this into something genuinely shared — Redis, a small database,
# or a single dedicated process the workers talk to over an internal API.
# --------------------------------------------------------------------
world = create_world_state(
    event_start_minutes=EVENT_START_MINUTES,
    event_end_minutes=EVENT_END_MINUTES,
)
simulate_crowd_buildup(world)

# Still one fixed attendee id for this demo — a real app would look
# attendees up by session/login instead. The PROFILE ITSELF, though,
# now comes from the database rather than being hardcoded here.
DEMO_ATTENDEE_ID = "A-DEMO"


def _default_attendee() -> Attendee:
    """Fallback used only if no profile has ever been saved yet."""
    return Attendee(id=DEMO_ATTENDEE_ID, home_location="Tampines", transport_preference="mrt")


# --------------------------------------------------------------------
# Simulated clock.
#
# CLOCK["scale"] = simulated minutes that pass per real second. At 1.0,
# one real minute = 60 simulated minutes, so a ~5-hour event (300 sim
# minutes) plays out over ~5 real minutes of the app just sitting open
# — fast enough to actually watch happen, without needing to babysit
# it. Tune this if you want it faster/slower.
#
# CLOCK["manual_offset_minutes"] accumulates extra simulated minutes
# added by the fast-forward button, on top of whatever the automatic
# ticking has already added.
# --------------------------------------------------------------------
CLOCK = {
    "real_start": time.time(),
    "sim_start_minutes": world.current_time_minutes,
    "scale": 1.0,
    "manual_offset_minutes": 0,
}


def sync_world_clock() -> None:
    """Recompute world.current_time_minutes from real elapsed time plus
    any manual fast-forward, then recompute crowd levels for that time.
    Call this before anything reads world state (dashboard refresh,
    agent calls) so both always see the current simulated moment."""
    elapsed_real_seconds = time.time() - CLOCK["real_start"]
    ticked_minutes = elapsed_real_seconds * CLOCK["scale"]
    world.current_time_minutes = int(
        CLOCK["sim_start_minutes"] + ticked_minutes + CLOCK["manual_offset_minutes"]
    )
    simulate_crowd_buildup(world)


def format_clock(minutes: int) -> str:
    minutes = int(minutes) % (24 * 60)
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/world")
def get_world():
    """Return current transport node loads and the simulated clock, for
    the dashboard cards and clock display."""
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
    })


@app.route("/api/fast-forward", methods=["POST"])
def fast_forward():
    """Jump the simulated clock ahead instantly, for demo purposes —
    e.g. skip straight to the exodus surge instead of waiting for the
    automatic ticking to get there. Uses the same advance_time() helper
    demo.py uses, so behaviour matches the CLI demo exactly."""
    data = request.get_json(silent=True) or {}
    try:
        minutes = int(data.get("minutes", 30))
    except (TypeError, ValueError):
        return jsonify({"error": "minutes must be a whole number."}), 400
    if minutes <= 0:
        return jsonify({"error": "minutes must be positive."}), 400

    # Fold the jump into the manual offset so future automatic ticking
    # keeps counting up from this new point, rather than the next
    # sync_world_clock() call overwriting the jump.
    CLOCK["manual_offset_minutes"] += minutes
    advance_time(world, minutes=minutes)
    sync_world_clock()

    return jsonify({
        "current_time_minutes": world.current_time_minutes,
        "current_time_display": format_clock(world.current_time_minutes),
    })


@app.route("/api/reset-time", methods=["POST"])
def reset_time():
    """Fully reset the simulation: rebuilds `world` from scratch (fresh
    transport nodes, gates, no disruptions) AND resets the clock back to
    its original starting point.

    This has to rebuild the whole world, not just the clock, because
    simulate_crowd_buildup() ratchets node/gate load upward with max() —
    load never decreases on its own, by design, so it doesn't fall back
    down just because the clock display goes back to an earlier time.
    Resetting only CLOCK left load values stuck wherever they'd already
    climbed to from prior polling/fast-forwarding — this is the actual
    fix for that."""
    global world
    world = create_world_state(
        event_start_minutes=EVENT_START_MINUTES,
        event_end_minutes=EVENT_END_MINUTES,
    )
    simulate_crowd_buildup(world)

    CLOCK["real_start"] = time.time()
    CLOCK["sim_start_minutes"] = world.current_time_minutes
    CLOCK["manual_offset_minutes"] = 0

    return jsonify({
        "current_time_minutes": world.current_time_minutes,
        "current_time_display": format_clock(world.current_time_minutes),
    })


@app.route("/api/attendee", methods=["GET"])
def get_attendee_route():
    """Return the saved profile, if any, so the form can prefill itself."""
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
    """Save (or overwrite) the attendee's profile from the form.

    transport_preference now accepts a comma-separated list (e.g.
    "bus,walk") to match the multi-select transport chips in the UI —
    people can genuinely be fine with more than one mode. Still stored
    as a single string field on Attendee/in the database; only the
    validation and the set of allowed tokens changed here."""
    data = request.get_json(silent=True) or {}
    home_location = (data.get("home_location") or "").strip()
    transport_preference_raw = (data.get("transport_preference") or "mrt").strip().lower()
    accessibility_needs = (data.get("accessibility_needs") or "").strip() or None
    travelling_with = (data.get("travelling_with") or "").strip() or None

    if not home_location:
        return jsonify({"error": "Home location is required."}), 400

    allowed_modes = {"mrt", "bus", "walk", "taxi", "cycling"}
    modes = [m.strip() for m in transport_preference_raw.split(",") if m.strip()]
    if not modes or any(m not in allowed_modes for m in modes):
        return jsonify({
            "error": "Preferred transport must be one or more of: mrt, bus, walk, taxi, cycling."
        }), 400
    transport_preference = ",".join(modes)

    attendee = Attendee(
        id=DEMO_ATTENDEE_ID,
        home_location=home_location,
        transport_preference=transport_preference,
        accessibility_needs=accessibility_needs,
        travelling_with=travelling_with,
    )
    save_attendee(attendee)
    return jsonify({"status": "saved"})


@app.route("/api/plan", methods=["POST"])
def plan_journey():
    """Run the real LangGraph agent (Bedrock call included) and return
    its recommendation, using whichever profile is currently saved (or
    the hardcoded default if none has been saved yet). Syncs the clock
    first so the agent's prompt reflects the current simulated time.
    This is the one genuinely slow endpoint — expect a few seconds,
    since it makes multiple sequential Bedrock calls
    (monitor -> predict -> recommend)."""
    sync_world_clock()
    attendee = get_attendee(DEMO_ATTENDEE_ID) or _default_attendee()
    try:
        result = run_agent_for_attendee(attendee, world, milestone="before")
        return jsonify({"recommendation": result["recommendation"]})
    except Exception as exc:
        # Surface the real error message during development. Before any
        # real deployment, log the full exception server-side and return
        # a generic message here instead — don't leak internals to clients.
        return jsonify({"error": str(exc)}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)
