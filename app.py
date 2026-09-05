"""Minimal Flask web app wrapping the KallangFlow agent.

Run with:  python app.py
Then open: http://localhost:5000

This is a starting point, not a production deployment. See the notes
inline below for what would need to change before real multi-user use.
"""
from __future__ import annotations

from dotenv import load_dotenv
from flask import Flask, jsonify, render_template

load_dotenv()

from kallangflow.agent import run_agent_for_attendee
from kallangflow.models import Attendee
from kallangflow.simulator import create_world_state, simulate_crowd_buildup

app = Flask(__name__)

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

# Also hardcoded for this demo — a real app would look attendees up by
# session/login instead of using one fixed demo attendee.
DEMO_ATTENDEE = Attendee(
    id="A-DEMO",
    home_location="Tampines",
    transport_preference="mrt",
)


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


@app.route("/api/plan", methods=["POST"])
def plan_journey():
    """Run the real LangGraph agent (Bedrock call included) and return
    its recommendation. This is the one genuinely slow endpoint — expect
    a few seconds, since it makes multiple sequential Bedrock calls
    (monitor -> predict -> recommend)."""
    try:
        result = run_agent_for_attendee(DEMO_ATTENDEE, world, milestone="before")
        return jsonify({"recommendation": result["recommendation"]})
    except Exception as exc:
        # Surface the real error message during development. Before any
        # real deployment, log the full exception server-side and return
        # a generic message here instead — don't leak internals to clients.
        return jsonify({"error": str(exc)}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)
