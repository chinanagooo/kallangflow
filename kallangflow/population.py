"""Population generation and crowd-scale aggregation.

We do NOT run 50,000 real LLM agent calls — that's neither necessary
nor affordable for a hackathon demo. Instead:

  * a small batch of REAL attendees (default 30) go through the full
    LangGraph agent loop (agent.py) and get genuine personalised
    recommendations.
  * the remaining synthetic population is redistributed statistically,
    using the SAME congestion signal the real agents saw, to produce
    an honest "at 50,000-attendee scale" projection.

This file is intentionally free of any LLM calls — pure Python — so
it's cheap to run repeatedly and easy to unit test.
"""
from __future__ import annotations

import random
from collections import Counter
from dataclasses import replace

from .models import Attendee, WorldState

HOME_LOCATIONS = [
    "Tampines", "Woodlands", "Jurong West", "Punggol", "Bishan",
    "Toa Payoh", "Ang Mo Kio", "Bedok", "Clementi", "Yishun",
]

TRANSPORT_PREFS = ["mrt", "mrt", "mrt", "bus", "taxi", "any"]  # mrt weighted


def generate_population(n: int, seed: int | None = 11) -> list[Attendee]:
    """Create n synthetic attendees with plausible, varied preferences."""
    rng = random.Random(seed)
    attendees: list[Attendee] = []
    for i in range(n):
        accessibility = None
        travelling_with = None
        roll = rng.random()
        if roll < 0.06:
            accessibility = "step-free access"
        elif roll < 0.10:
            travelling_with = "elderly parent"

        attendees.append(Attendee(
            id=f"A{i:05d}",
            home_location=rng.choice(HOME_LOCATIONS),
            transport_preference=rng.choice(TRANSPORT_PREFS),
            accessibility_needs=accessibility,
            travelling_with=travelling_with,
        ))
    return attendees


def aggregate_signals(attendees: list[Attendee]) -> Counter:
    """Count how many attendees are currently assigned to each transport
    node. This is the 'anonymised aggregate movement signal' from the
    pitch — plain counting, no model involved."""
    counts: Counter = Counter()
    for a in attendees:
        if a.assigned_transport_node:
            counts[a.assigned_transport_node] += 1
    return counts


def redistribute_simulated(n: int, world: WorldState, avoid_node: str | None = None,
                            seed: int | None = 13) -> Counter:
    """Statistically distribute `n` *simulated* (non-agent) attendees
    across transport options, weighted AWAY from congested nodes and
    toward the ones the real agents are already being routed to.

    This mimics — without literally running — what 50,000 individual
    agents would collectively do if each saw the same congestion signal
    and made an individually-optimal choice.
    """
    rng = random.Random(seed)
    nodes = list(world.transport_nodes.values())

    weights = []
    for node in nodes:
        congestion = node.congestion().value
        base_weight = {"low": 1.0, "moderate": 0.5, "high": 0.15}[congestion]
        if avoid_node and node.id == avoid_node:
            base_weight *= 0.1
        weights.append(base_weight)

    # A "wait" option is not a transport node but a valid choice.
    choices = [n.id for n in nodes] + ["wait"]
    node_weights = weights + [0.25]  # flat weight for "choosing to wait"

    result: Counter = Counter()
    for _ in range(n):
        pick = rng.choices(choices, weights=node_weights, k=1)[0]
        result[pick] += 1
    return result


def without_intervention_baseline(n: int, world: WorldState, default_node: str) -> Counter:
    """What happens with NO agent system: everyone independently makes
    the same 'obvious' choice (e.g. everyone defaults to Stadium MRT
    because it's the nearest exit). This is the 'before' bar in the
    before/after chart."""
    return Counter({default_node: n})
