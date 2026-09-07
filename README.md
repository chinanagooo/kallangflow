# KallangFlow

KallangFlow is an agentic crowd-flow simulation and journey recommendation system for events around the Kallang area in Singapore.

The project combines a simulated transport network, crowd/congestion modelling, an LLM-powered journey agent, and a web interface. The agent observes the current simulated conditions and attendee preferences before producing a personalised recommendation.

## Overview

Large events can create rapidly changing crowd conditions around transport nodes and venue entrances. A route that is suitable before an event may no longer be suitable while the event is in progress or after it ends.

KallangFlow addresses this by combining:

- **Crowd-flow simulation** — models changing loads and congestion at transport nodes.
- **Agentic recommendations** — uses an LLM agent to interpret the current world state and attendee needs.
- **Event-aware reasoning** — distinguishes between the periods before, during, and after the event.
- **Personalisation** — considers home location, preferred transport modes, accessibility needs, and whether the attendee is travelling with someone.
- **Interactive simulation controls** — allows the simulated clock to be advanced and reset.
- **Interactive map visualisation** — displays relevant transport nodes and the recommended journey route.

## Key Features

### 1. Dynamic crowd simulation

The simulator maintains a world state containing transport nodes, capacities, loads, congestion levels, and event timing.

As simulated time advances, crowd conditions change. This allows recommendations to be tested against different points in the event journey.

### 2. Personalised journey planning

Attendees can provide:

- Home location
- Preferred transport mode(s)
- Accessibility requirements
- Travelling-with information

Multiple transport modes can be selected, allowing the agent to choose between alternatives instead of being restricted to one mode.

Supported transport options include:

- MRT / LRT
- Bus
- Walking
- Taxi / ride-hailing
- Cycling

### 3. Event-aware agent

The recommendation agent uses the event start and end times to determine the current journey phase.

| Phase | Agent focus |
|---|---|
| Before event | When to leave home and which transport/gate to use |
| Event in progress | Respond to the attendee's current situation and transport conditions |
| After event | Leave the venue and choose an appropriate journey home |

The agent should not recommend travelling toward the venue after the event has already ended.

### 4. Accessibility-aware recommendations

Accessibility and care requirements are treated as decision factors rather than simply additional profile information.

For example, when an attendee has mobility requirements or is travelling with someone who needs additional care, the agent can prioritise an appropriate and accessible route over the theoretically fastest option.

### 5. Interactive map

The frontend uses Leaflet with OpenStreetMap tiles to visualise the transport network and recommended routes.

The interface can display:

- Transport nodes
- Stadium area
- Recommended transport node
- Recommended gate
- Route geometry when available
- Different route presentation for accessibility-oriented journeys

The route decision is made by the backend/agent layer; the browser is responsible for displaying the returned route.

### 6. Simulated clock

The application provides controls for moving through the event timeline.

The simulation can be:

- Started before the event
- Fast-forwarded through the event
- Viewed after the event
- Reset to its initial state

Resetting the simulation rebuilds the simulated world so that crowd and prediction values return to their initial state rather than retaining values accumulated during a previous simulation run.

---

## Architecture

```text
                         ┌───────────────────────┐
                         │      Web Browser      │
                         │                       │
                         │  Attendee Preferences │
                         │  Simulation Controls  │
                         │  Map / Recommendation │
                         └──────────┬────────────┘
                                    │
                              HTTP / JSON
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │      Flask App       │
                         │       app.py         │
                         │                      │
                         │ /api/world           │
                         │ /api/attendee        │
                         │ /api/plan            │
                         │ /api/fast-forward    │
                         │ /api/reset-time      │
                         └───────┬───────┬──────┘
                                 │       │
                    world state  │       │ recommendation
                                 │       ▼
                                 │  ┌───────────────┐
                                 │  │  Agent Layer  │
                                 │  │    agent.py   │
                                 │  │               │
                                 │  │ LangGraph     │
                                 │  │ + Bedrock     │
                                 │  └───────┬───────┘
                                 │          │
                                 │          ▼
                         ┌───────▼──────────────────┐
                         │       Simulator          │
                         │       simulator.py       │
                         │                          │
                         │ Transport nodes          │
                         │ Crowd loads              │
                         │ Congestion               │
                         │ Event timeline           │
                         └──────────────────────────┘

                         Supporting models
                         ┌───────────────────────┐
                         │      models.py        │
                         │ Attendee / WorldState │
                         │ Transport nodes       │
                         └───────────────────────┘
```

## Project Structure

```text
kallangflow/
│
├── app.py                    # Flask web application and API endpoints
│
├── kallangflow/
│   ├── agent.py              # LangGraph agent and recommendation logic
│   ├── simulator.py          # Crowd-flow simulation and world state
│   ├── models.py             # Attendee and simulation data models
│   └── ...
│
├── test/
│   ├── app.py
│   └── kallangflow/
│       ├── agent.py
│       ├── simulator.py
│       └── models.py
│
├── index.html                # Web interface
├── requirements.txt          # Python dependencies
├── README.md
└── ...
```

> The exact contents may vary between branches. The `test/` implementation contains the sandbox version used during development.

## Technology Stack

| Component | Technology |
|---|---|
| Backend | Python + Flask |
| Agent orchestration | LangGraph |
| Language model | Amazon Bedrock |
| Simulation | Python |
| Frontend | HTML, CSS, JavaScript |
| Map | Leaflet |
| Map tiles | OpenStreetMap |
| API communication | REST / JSON |

## Requirements

Before running KallangFlow, install:

- Python 3.10+ recommended
- An AWS account with access to Amazon Bedrock
- A Bedrock model enabled for the AWS region being used
- Git

Install the Python dependencies from the repository:

```bash
pip install -r requirements.txt
```

For a cleaner setup, a virtual environment is recommended:

```bash
python -m venv .venv
```

### Windows

```bash
.venv\Scripts\activate
pip install -r requirements.txt
```

### macOS / Linux

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

## AWS / Amazon Bedrock Setup

KallangFlow uses Amazon Bedrock for the LLM-powered recommendation stage.

Configure AWS credentials using one of the standard AWS authentication methods, for example the AWS CLI:

```bash
aws configure
```

Then verify that the credentials are available:

```bash
aws sts get-caller-identity
```

Make sure the configured AWS identity has permission to invoke the Bedrock model used by the project.

If the application expects environment variables for the model or AWS configuration, use the names already defined by the repository's configuration/code rather than committing credentials to source control.

If using a sandbox account, load API keys, secret access key and session ID directly into .env and save before running the application.

```bash
AWS_ACCESS_KEY_ID="your_api_key"
AWS_SECRET_ACCESS_KEY="your_access_key"
AWS_SESSION_TOKEN="your_session_token"
AWS_REGION=us-east-1
BEDROCK_MODEL=us.anthropic.claude-haiku-4-5-20251001-v1:0
```

## Running the Application

From the repository root:

```bash
python app.py
```

The Flask development server should start locally.

Open the local address shown by Flask in a browser. A common development address is:

```text
http://127.0.0.1:5000
```

If the repository's current `app.py` specifies another port, use that port instead.

## Using the Application

### 1. Enter attendee information

Provide the attendee's:

- Home location
- Transport preference(s)
- Accessibility needs, if applicable
- Travelling-with information, if applicable

### 2. View the simulated world

The interface displays the current simulated time and transport-node conditions.

Transport nodes may show changing load/congestion as the simulation progresses.

### 3. Get a recommendation

Select **Get my recommendation**.

The backend:

1. Reads the current simulation time.
2. Determines whether the event is before, during, or after the event.
3. Retrieves the attendee profile.
4. Runs the recommendation agent.
5. Considers relevant transport conditions.
6. Produces a specific journey recommendation.
7. Returns route information to the frontend.
8. Displays the recommendation and route on the map.

### 4. Fast-forward the simulation

Use the fast-forward control to move the simulation through time.

This is useful for testing how recommendations change:

```text
Before event
     ↓
Event approaching
     ↓
Event in progress
     ↓
Event ending
     ↓
After event
```

### 5. Reset the simulation

Use the reset control to return to the initial simulation state.

A reset rebuilds the simulated world instead of simply moving the clock backwards. This ensures that accumulated crowd/load/prediction state from the previous run does not remain in the new simulation.

---

## API Endpoints

The Flask application exposes endpoints for the frontend.

### `GET /api/world`

Returns the current simulation state, including:

- Current simulated time
- Event start/end times
- Current event phase
- Transport node information
- Node loads
- Node capacities
- Congestion state

Example:

```http
GET /api/world
```

### `POST /api/attendee`

Saves the current attendee profile.

Example request:

```json
{
  "home_location": "Tampines",
  "transport_preference": ["mrt", "bus"],
  "accessibility_needs": "",
  "travelling_with": ""
}
```

Supported transport values:

```text
mrt
bus
walk
taxi
cycling
```

### `POST /api/plan`

Generates a personalised journey recommendation using the current simulation state and attendee information.

The server determines the event milestone from the simulation clock rather than relying on the browser to specify the journey phase.

### `POST /api/fast-forward`

Advances the simulation clock.

### `POST /api/reset-time`

Resets:

- Simulated time
- Crowd state
- Transport-node loads
- Prediction state
- Current journey state

The endpoint returns the freshly rebuilt world state so the frontend can render the reset values.

---

## Event Timeline

The default event configuration used by the application is:

```text
Event start: 19:00
Event end:   22:00
```

The application determines the current phase from these values.

```text
                 EVENT TIMELINE

       BEFORE             DURING              AFTER
          │                 │                   │
          ▼                 ▼                   ▼
      ────────────────┬───────────────────┬──────────────
                      │                   │
                    19:00               22:00
                  Event starts         Event ends
```

The recommendation objective changes with the phase.

### Before 19:00

The agent focuses on getting the attendee to the event.

### 19:00–22:00

The event is in progress. Recommendations should reflect the current situation instead of treating the journey as a pre-event trip.

### At or after 22:00

The event is considered finished. The agent focuses on leaving the venue and travelling home.

---

## Simulation Model

The simulator represents the event environment as a collection of transport nodes.

Examples of nodes used by the application include:

- Stadium MRT
- Kallang MRT
- Mountbatten MRT
- Nicoll Highway MRT
- Bus hub
- Taxi / ride-hailing

Each node can have a capacity and changing load.

A simplified congestion relationship is:

```text
congestion = current load / node capacity
```

The simulator then exposes the resulting conditions to the agent so that recommendations can account for changing crowd pressure.

---

## Agent Flow

The recommendation process can be viewed as:

```text
Attendee Profile
       │
       ▼
Current Simulation State
       │
       ▼
Determine Event Phase
       │
       ▼
Monitor Relevant Transport Nodes
       │
       ▼
LangGraph Agent
       │
       ▼
Recommendation
       │
       ├──────────────► Journey / route information
       │
       ▼
Frontend
       │
       ├── Recommendation text
       └── Map route
```

The important design principle is that the recommendation should be based on the **current simulated world**, not only on static transport preferences.

---

## Reset Behaviour

The simulator contains mutable state that can change as time advances.

Simply changing:

```text
22:00 → 17:15
```

does not necessarily restore all previously accumulated simulation values.

Therefore, the reset operation reconstructs the world from its initial configuration.

This gives:

```text
Initial World
     │
     ▼
Fast-forward
     │
     ▼
Changed crowd state
     │
     ▼
Reset
     │
     ▼
Fresh Initial World
```

This is important when repeatedly testing different scenarios.

---

## Development Notes

### Changing the event time

The event start and end times are defined in the Flask application and passed into the simulator.

If the event schedule changes, keep the following consistent:

- Flask configuration
- Simulator configuration
- Agent event-phase logic
- Frontend clock display
- Any attendee/event defaults

### Adding transport modes

When adding a new transport mode, update both sides of the application:

1. Frontend transport options.
2. Backend validation.

For example, adding a new mode requires the backend to accept the corresponding value instead of rejecting it as an invalid transport option.

### Route rendering

The backend should decide the recommended route/node. The frontend should primarily visualise the route returned by the server.

This keeps route selection separate from presentation.

---

## Troubleshooting

### `Invalid transport option`

Check that the frontend is sending one or more supported values:

```text
mrt
bus
walk
taxi
cycling
```

If the backend still reports an old validation message, make sure the Flask process has been restarted after changing `app.py`.

### Recommendations ignore the event phase

Check that:

- `/api/plan` derives the milestone from the server-side simulation clock.
- The agent prompt contains the event start/end times.
- The recommendation system prompt explicitly handles before/during/after event behaviour.

### Reset changes the clock but not crowd values

The world must be rebuilt during reset.

Do not only move the simulated clock backwards. Recreate the world state so accumulated load/prediction values are discarded.

### Map does not update

Check that:

- `/api/plan` returns route information.
- The frontend calls the route drawing function after receiving the response.
- Leaflet has been initialised.
- The map container has a valid height.

---

## Security

Do not commit:

```text
.env
AWS credentials
AWS secret keys
API keys
Bedrock credentials
```

Use AWS's normal credential mechanisms or environment variables and keep secrets outside version control.

For production deployment, replace the Flask development server with an appropriate production WSGI/ASGI deployment architecture and apply normal authentication, HTTPS, logging, and secret-management practices.

---

## Project Purpose

KallangFlow is intended as a prototype for exploring how an agentic AI system can make personalised mobility recommendations in a dynamic crowd environment.

Rather than treating journey planning as a static shortest-path problem, the project experiments with:

- Dynamic crowd conditions
- Time-aware decision making
- Personal preferences
- Accessibility considerations
- Event lifecycle awareness
- Agent-based recommendations
- Visual route presentation

The result is a simulation environment where different attendee profiles and event times can be tested against changing transport conditions.

---

## Future Improvements

Potential extensions include:

- More realistic crowd-flow models
- Real-time transport data integration
- More detailed pedestrian routes
- More venue gates and entrances
- Real MRT/bus travel times
- Multi-attendee simulations
- Historical crowd calibration
- Recommendation evaluation metrics
- Automated scenario testing
- Persistent simulation scenarios
- Production deployment
- More robust route optimisation
- Explicit explanation of why a route was selected

---

## License

Add the project's intended license here if one has been selected.

If no license has been added to the repository yet, the project should be treated according to the repository owner's rights rather than assuming an open-source license.

---

## Repository

GitHub: https://github.com/chinanagooo/kallangflow
