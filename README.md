# KallangFlow — working MVP

A software-only, simulation-based agentic AI system for coordinating crowd
movement at Singapore's major events (Kallang precinct). Built on
LangGraph + Groq (swap to AWS Bedrock by changing `chat_model()` in
`kallangflow/agent.py` — see the hackathon's `section_3_bedrock` labs).

**Everything transport/crowd-related here is SIMULATED** (`kallangflow/simulator.py`).
There is no live Singapore transport feed in this MVP — that's flagged
in the demo output and should be flagged on your slides too.

## What's implemented

| Module | What it does |
|---|---|
| `models.py` | Attendee, TransportNode, Gate, Facility, WorldState — plain dataclasses |
| `simulator.py` | Simulated crowd buildup, disruptions, 8-min congestion projection |
| `population.py` | Synthetic 50,000-attendee population + the redistribution logic behind the "50,000 agents" story |
| `tools.py` | The 6 tools the agent can call (transport status, gate congestion, weather, facility queues, route estimate, guardian notify) |
| `agent.py` | The LangGraph loop: **Monitor → Predict → Recommend → Act → Adapt**, bounded and tool-call-safe |
| `demo.py` | Scripted 6-step demo matching the MVP scenario in the pitch |

## Setup

Instructions below apply to both Windows and macOS; anywhere a command
differs, both are given.

### 1. Install Python and dependencies

**macOS:**

If you don't already have Python 3 (check with `python3 --version`),
install it via [Homebrew](https://brew.sh):

```
brew install python
```

It's recommended to use a virtual environment rather than installing
packages system-wide:

```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

You'll need to run `source .venv/bin/activate` again in any new terminal
tab/window before running the project (your prompt will show `(.venv)`
when it's active).

**Windows:**

```
pip install -r requirements.txt
cp .env.example .env
# edit .env and paste your Groq key from console.groq.com
```

## Run the non-LLM sanity tests (no API key needed)

```bash
python -m tests.test_core
```
This exercises the simulator, population/redistribution math, and all
tool functions directly — everything except the actual model calls.

## Run the full scripted demo (needs GROQ_API_KEY)

```bash
python -m kallangflow.demo
```

Without a key set, the demo still runs and prints the simulated
crowd/transport data and the before/after redistribution chart — it
just skips the live LLM recommendation/guardian-message calls and
tells you so inline, so you can sanity-check everything else first.

## What to say if a judge asks "is this real data?"

No — Singapore doesn't expose live MRT/crowd data publicly, so the
transport and crowd figures are a deterministic simulator
(`simulator.py`) designed to produce realistic pre-event buildup and
post-event exodus curves. The **agent architecture and reasoning are
real**: a live LangGraph loop actually calls tools, actually reads
changing congestion state, and actually changes its recommendation
when you inject a disruption live (Step 2/3 of the demo). Swapping the
simulator for a real feed later is a `simulator.py`-only change — the
agent and tool interfaces don't need to change.

## Next steps (see the earlier build-sequence doc, Phases 5–9)

- Wire `act_node`'s guardian messages into a small Streamlit dashboard
  showing attendee view + family view side by side (Phase 7)
- Optional: refactor into a DeepAgents supervisor with journey/venue/guardian
  sub-agents (Phase 6) — requires Bedrock/Claude, not Groq
- Optional: deploy via Bedrock AgentCore for a live HTTPS endpoint (Phase 9, Step 35)
