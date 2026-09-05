"""Minimal Flask web app wrapping the KallangFlow agent.

Run with:  python app.py
Then open: http://localhost:5000

This is a starting point, not a production deployment. See the notes
inline below for what would need to change before real multi-user use.

Attendee profiles are now saved to the same SQLite database (attendees.db,
via kallangflow/db.py) that demo.py uses — a profile set up through this
web form is what demo.py will offer to reuse, and vice versa.
"""
from __future__ import annotations

from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request

load_dotenv()

from kallangflow.agent import run_agent_for_attendee
from kallangflow.db import get_attendee, init_db, save_attendee
from kallangflow.models import Attendee
from kallangflow.simulator import create_world_state, simulate_crowd_buildup

app = Flask(__name__)

init_db()

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
world = create_world_state()
simulate_crowd_buildup(world)

# Still one fixed attendee id for this demo — a real app would look
# attendees up by session/login instead. The PROFILE ITSELF, though,
# now comes from the database rather than being hardcoded here.
DEMO_ATTENDEE_ID = "A-DEMO"


def _default_attendee() -> Attendee:
    """Fallback used only if no profile has ever been saved yet."""
    return Attendee(id=DEMO_ATTENDEE_ID, home_location="Tampines", transport_preference="mrt")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/world")
def get_world():
    """Return current transport node loads, for the dashboard cards."""
    nodes = {
        node_id: {
            "name": node.name,
            "load": node.load,
            "capacity": node.capacity,
            "congestion": node.congestion().value,
        }
        for node_id, node in world.transport_nodes.items()
    }
    return jsonify(nodes)


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
    """Save (or overwrite) the attendee's profile from the form."""
    data = request.get_json(silent=True) or {}
    home_location = (data.get("home_location") or "").strip()
    transport_preference = (data.get("transport_preference") or "mrt").strip().lower()
    accessibility_needs = (data.get("accessibility_needs") or "").strip() or None
    travelling_with = (data.get("travelling_with") or "").strip() or None

    if not home_location:
        return jsonify({"error": "Home location is required."}), 400
    if transport_preference not in ("mrt", "bus", "taxi", "any"):
        return jsonify({"error": "Transport preference must be mrt, bus, taxi, or any."}), 400

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
    the hardcoded default if none has been saved yet). This is the one
    genuinely slow endpoint — expect a few seconds, since it makes
    multiple sequential Bedrock calls (monitor -> predict -> recommend)."""
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
