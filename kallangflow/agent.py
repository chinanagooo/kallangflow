"""The personal Journey Agent — one LangGraph loop per attendee.

Implements the Learn -> Monitor -> Predict -> Recommend -> Act -> Adapt
cycle from the pitch. "Learn" is folded into the initial prompt (the
attendee's known preferences); the remaining five stages are explicit
graph nodes so the loop is visible and inspectable, not a black box.
"""
from __future__ import annotations

import os
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_aws import ChatBedrockConverse
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from .models import Attendee, WorldState
from .tools import make_tools, notify_guardian

load_dotenv()

def chat_model(temperature: float = 0) -> ChatBedrockConverse:
    return ChatBedrockConverse(
        model=os.environ.get("BEDROCK_MODEL", "us.anthropic.claude-haiku-4-5-20251001-v1:0"),
        region_name=os.environ.get("AWS_REGION", "ap-southeast-1"),
        temperature=temperature,
    )

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    attendee: Attendee
    world: WorldState
    recommendation: str
    journey_log: list[str]
    loop_count: int
    max_loops: int
    milestone: str  # "before" | "gate" | "after" | "home"


# ---------------------------------------------------------------- MONITOR
SYSTEM_MONITOR = """You are the monitoring stage of a personal event-journey
agent. Use the available tools to check transport status, gate congestion,
and weather relevant to THIS attendee's journey. Call at most 2-3 tools,
then reply with a short plain-text summary of what you found. Do not make
a recommendation yet — that happens later."""


def monitor_node(state: AgentState) -> dict:
    attendee = state["attendee"]
    world = state["world"]
    tools = make_tools(world)
    model = chat_model().bind_tools(tools)
    by_name = {t.name: t for t in tools}

    prefs = f"prefers {attendee.transport_preference} transport"
    if attendee.accessibility_needs:
        prefs += f", requires {attendee.accessibility_needs}"
    if attendee.travelling_with:
        prefs += f", travelling with {attendee.travelling_with}"

    prompt = (
        f"Attendee {attendee.id} lives in {attendee.home_location}, {prefs}. "
        f"Event starts at simulated minute {world.event_start_minutes}; current "
        f"simulated time is {world.current_time_minutes}. Check whatever transport "
        f"node(s) and gate(s) are relevant to this attendee right now."
    )

    # This scratch list holds the raw tool-calling loop (toolUse/toolResult blocks).
    # It stays LOCAL to this node — it is never merged into the graph's shared
    # state["messages"], because passing raw tool-call blocks to a model invoked
    # later without a bound toolConfig causes Bedrock's Converse API to silently
    # return empty content (see README "Architecture notes").
    scratch = [SystemMessage(SYSTEM_MONITOR), HumanMessage(prompt)]
    summary_text = ""
    for _ in range(3):  # bounded ReAct loop — see Session 2 "the cap is the safety net"
        reply = model.invoke(scratch)
        scratch.append(reply)
        if not reply.tool_calls:
            summary_text = reply.content
            break
        for call in reply.tool_calls:
            fn = by_name.get(call["name"])
            if fn is None:
                result = f"No tool named {call['name']}. Available: {', '.join(by_name)}."
            else:
                try:
                    result = fn.invoke(call["args"])
                except Exception as exc:  # tool misuse must not crash the run
                    result = f"{call['name']} failed: {exc}"
            scratch.append(ToolMessage(result, tool_call_id=call["id"]))
    else:
        # loop exhausted without a clean final reply — fall back to whatever text we have
        summary_text = getattr(scratch[-1], "content", "") or "No findings captured."

    # Only a clean, tool-call-free summary is passed downstream.
    clean_summary = HumanMessage(f"Monitoring findings: {summary_text}")
    return {"messages": [clean_summary]}


# ---------------------------------------------------------------- PREDICT
SYSTEM_PREDICT = """You are the prediction stage. Given the monitoring
findings above, state in one or two sentences what is likely to change
in the next 10-15 minutes for THIS attendee (e.g. a gate or station
becoming more congested, or nothing notable)."""


def predict_node(state: AgentState) -> dict:
    reply = chat_model().invoke(state["messages"] + [SystemMessage(SYSTEM_PREDICT)])
    return {"messages": [reply]}


# -------------------------------------------------------------- RECOMMEND
SYSTEM_RECOMMEND = """You are the recommendation stage of a personal
event-journey agent. Based on everything above, give ONE specific,
actionable recommendation: when to leave or wait, and which gate or
transport option to use, and why. If the attendee has accessibility
needs or is travelling with someone needing extra care, prioritise the
most APPROPRIATE option over the fastest one, and say so explicitly.
2-3 sentences, written directly to the attendee, no preamble."""


def recommend_node(state: AgentState) -> dict:
    reply = chat_model().invoke(state["messages"] + [SystemMessage(SYSTEM_RECOMMEND)])
    return {"messages": [reply], "recommendation": reply.content}


# -------------------------------------------------------------------- ACT
SYSTEM_GUARDIAN = """Write ONE short, reassuring message (under 25 words)
from the attendee to their Journey Guardian (a family member), matching
the milestone described. Plain text, first person. Example style:
"I'm heading home via Mountbatten MRT, ETA 11:15 PM." """

MILESTONE_PROMPTS = {
    "before": "Milestone: about to leave home for the event.",
    "gate": "Milestone: entering the venue now.",
    "after": "Milestone: the event has ended, leaving the venue now.",
    "home": "Milestone: arrived home safely.",
}


def act_node(state: AgentState) -> dict:
    attendee = state["attendee"]
    milestone = state.get("milestone", "before")
    prompt = MILESTONE_PROMPTS.get(milestone, "Milestone: journey update.")
    reply = chat_model().invoke([
        SystemMessage(SYSTEM_GUARDIAN),
        HumanMessage(f"{prompt}\n\nContext / recommendation: {state.get('recommendation', '')}"),
    ])
    notify_guardian.invoke({"attendee_id": attendee.id, "message": reply.content})
    journey_log = state.get("journey_log", []) + [reply.content]
    return {"journey_log": journey_log}


# ------------------------------------------------------------------ ADAPT
def adapt_node(state: AgentState) -> dict:
    return {"loop_count": state.get("loop_count", 0) + 1}


def should_continue(state: AgentState) -> str:
    if state.get("loop_count", 0) >= state.get("max_loops", 1):
        return END
    return "monitor"


def build_graph():
    builder = StateGraph(AgentState)
    builder.add_node("monitor", monitor_node)
    builder.add_node("predict", predict_node)
    builder.add_node("recommend", recommend_node)
    builder.add_node("act", act_node)
    builder.add_node("adapt", adapt_node)

    builder.add_edge(START, "monitor")
    builder.add_edge("monitor", "predict")
    builder.add_edge("predict", "recommend")
    builder.add_edge("recommend", "act")
    builder.add_edge("act", "adapt")
    builder.add_conditional_edges("adapt", should_continue, ["monitor", END])
    return builder.compile()


def run_agent_for_attendee(attendee: Attendee, world: WorldState,
                            milestone: str = "before", max_loops: int = 1) -> AgentState:
    """Run one attendee through one or more Monitor->Adapt cycles.
    max_loops=1 means: monitor once, predict, recommend, act, done. Raise
    it to demonstrate re-adapting to a changed world without changing the
    graph (see demo.py step 2/3, which mutates `world` between calls)."""
    graph = build_graph()
    initial: AgentState = {
        "messages": [],
        "attendee": attendee,
        "world": world,
        "recommendation": "",
        "journey_log": [],
        "loop_count": 0,
        "max_loops": max_loops,
        "milestone": milestone,
    }
    return graph.invoke(initial)
