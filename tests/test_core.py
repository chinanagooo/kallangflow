"""Sanity tests that need NO API key — they exercise the simulator,
population/aggregation logic, and raw tool functions (via .invoke),
which is everything except the actual LLM calls in agent.py.

Run with: python -m tests.test_core   (from the project root)
"""
from kallangflow.models import Attendee
from kallangflow.population import (
    aggregate_signals,
    generate_population,
    redistribute_simulated,
    without_intervention_baseline,
)
from kallangflow.simulator import (
    advance_time,
    create_world_state,
    inject_disruption,
    project_congestion_in,
    simulate_crowd_buildup,
)
from kallangflow.tools import get_gate_congestion, get_transport_status, get_weather, make_tools


def test_world_and_buildup():
    world = create_world_state()
    assert world.current_time_minutes < world.event_start_minutes
    simulate_crowd_buildup(world)
    # gates should have picked up SOME load pre-event
    assert any(g.load > 0 for g in world.gates.values())
    print("[ok] world creation + crowd buildup")


def test_disruption_and_projection():
    world = create_world_state()
    simulate_crowd_buildup(world)
    inject_disruption(world, "stadium_mrt", note="test disruption")
    assert world.transport_nodes["stadium_mrt"].congestion().value == "high"
    proj = project_congestion_in(world, "stadium_mrt", minutes_ahead=8)
    assert proj in ("low", "moderate", "high")
    print("[ok] disruption injection + congestion projection")


def test_tools_direct_invoke():
    world = create_world_state()
    simulate_crowd_buildup(world)
    make_tools(world)  # binds the module-level active world
    status = get_transport_status.invoke({"node_id": "stadium_mrt"})
    assert "Stadium MRT" in status
    gate_status = get_gate_congestion.invoke({"gate_id": "gate_5"})
    assert "Gate 5" in gate_status
    weather = get_weather.invoke({"location": "Kallang"})
    assert "Weather" in weather
    print("[ok] tools callable directly (no LLM needed):")
    print("     ", status)
    print("     ", gate_status)


def test_advance_time_and_exodus():
    world = create_world_state()
    advance_time(world, 260)  # land close to event end -> exodus surge
    simulate_crowd_buildup(world)
    stadium = world.transport_nodes["stadium_mrt"]
    assert stadium.load > 0
    assert stadium.congestion().value in ("moderate", "high")
    print(f"[ok] post-event exodus: Stadium MRT load={stadium.load}/{stadium.capacity} "
          f"({stadium.congestion().value})")


def test_population_and_redistribution():
    world = create_world_state()
    advance_time(world, 150)
    simulate_crowd_buildup(world)

    pop = generate_population(200)
    assert len(pop) == 200
    assert any(a.accessibility_needs for a in pop), "expected some accessibility cases to be seeded"

    baseline = without_intervention_baseline(50_000, world, default_node="stadium_mrt")
    assert baseline["stadium_mrt"] == 50_000

    redistributed = redistribute_simulated(50_000, world, avoid_node="stadium_mrt")
    assert sum(redistributed.values()) == 50_000
    # the avoided node should get meaningfully less than the naive baseline
    assert redistributed.get("stadium_mrt", 0) < baseline["stadium_mrt"] * 0.5
    print("[ok] population generated + redistribution reduces peak node load")
    print("     baseline stadium_mrt:", baseline["stadium_mrt"])
    print("     redistributed stadium_mrt:", redistributed.get("stadium_mrt", 0))
    print("     redistributed full split:", dict(redistributed))


def test_aggregate_signals():
    a1 = Attendee(id="A1", home_location="Bishan", transport_preference="mrt",
                  assigned_transport_node="mountbatten_mrt")
    a2 = Attendee(id="A2", home_location="Bedok", transport_preference="mrt",
                  assigned_transport_node="mountbatten_mrt")
    counts = aggregate_signals([a1, a2])
    assert counts["mountbatten_mrt"] == 2
    print("[ok] aggregate_signals counts correctly")


if __name__ == "__main__":
    test_world_and_buildup()
    test_disruption_and_projection()
    test_tools_direct_invoke()
    test_advance_time_and_exodus()
    test_population_and_redistribution()
    test_aggregate_signals()
    print("\nALL NON-LLM TESTS PASSED")
