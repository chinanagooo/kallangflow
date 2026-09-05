# KallangFlow

Agentic crowd-flow simulation for event journeys, built on LangGraph. Each
attendee gets a personal LangGraph agent that runs a
Monitor → Predict → Recommend → Act → Adapt loop, using live LLM calls
(**Amazon Bedrock**, Claude models via `langchain-aws`) to reason about
transport disruptions, gate congestion, and personalised recommendations.

## What this is

- `kallangflow/simulator.py` — synthetic world state (transport nodes, gates,
  disruptions). All data is SYNTHETIC, not a live feed.
- `kallangflow/agent.py` — the LangGraph agent definition, one graph per
  attendee. Builds its Bedrock chat client via `chat_model()`.
- `kallangflow/population.py` — statistical modelling for simulating the
  full 50,000-attendee scale from a small real-agent sample.
- `kallangflow/demo.py` — scripted 6-step end-to-end demo. Now prompts
  interactively for the attendee's profile (see "Personalisation" below).
- `kallangflow/db.py` — SQLite persistence for attendee profiles, shared
  between `demo.py` and `app.py`.
- `app.py` — minimal Flask web app: a dashboard of transport node
  congestion, a profile form, and a "Plan my journey" button that calls
  the real LangGraph agent (same Bedrock calls as the CLI demo).
- `templates/index.html` — the single page `app.py` serves.
- `tests/test_core.py` — non-LLM tests (simulator, population, tools). No
  AWS credentials needed to run these.

Run the CLI demo with:

```
python -m kallangflow.demo
```

Or run the web app:

```
python app.py
```
then open `http://localhost:5000`.

### Personalisation

Attendee profiles (home location, transport preference, accessibility
needs, who they're travelling with) are no longer hardcoded. Both
`demo.py` and `app.py` read/write the same SQLite database,
**`attendees.db`**, created automatically at the project root the first
time either one runs:

- **`demo.py`**: on first run, prompts for each field one at a time and
  saves the result. On later runs, detects the saved profile and asks
  `Reuse this profile? [Y/n]` instead of asking again.
- **`app.py`**: a "Your journey profile" form on the page — `GET
  /api/attendee` loads the saved profile to prefill it, `POST
  /api/attendee` saves whatever's submitted. `/api/plan` then uses
  whichever profile is currently saved.

Both currently share one fixed attendee id (`"A-DEMO"`) — this is "one
shared profile for whoever's using the app right now", not per-user
accounts. Delete `attendees.db` any time to wipe all saved profiles and
start fresh.

---

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
```

If you hit `ModuleNotFoundError` for something already listed in
`requirements.txt` (e.g. `dotenv`, `langgraph`, `langchain-aws`), it usually
means packages were installed piecemeal rather than via the full
`pip install -r` command above — re-run the full install to be sure
everything's actually present (on macOS, also double check your virtual
environment is activated — a common cause of "it's installed but Python
can't find it").

If you want to run the web app (`app.py`), also install Flask:

```
pip install flask
```

(and add it to `requirements.txt` if it isn't already there). No extra
package is needed for the SQLite persistence — `sqlite3` is part of the
Python standard library.

### 2. Configure AWS Bedrock access

This project uses **Amazon Bedrock** (Anthropic Claude models) as its LLM
backend via `langchain-aws`'s `ChatBedrockConverse`. You need:

- An AWS account/sandbox with Bedrock model access enabled for at least one
  Claude model, in a region you can confirm has that access.
- Credentials the AWS SDK (`boto3`) can find — either a normal AWS profile,
  or (common for hackathon/training sandbox accounts) a set of **temporary**
  credentials issued via a sandbox portal.

If you don't have the AWS CLI installed yet:

- **macOS:** `brew install awscli`
- **Windows:** download the MSI installer from AWS's official site, or
  `winget install Amazon.AWSCLI`

#### If you have a normal AWS account

```
aws configure
```

or, for SSO-based accounts:

```
aws configure sso
aws sso login --profile <your-profile>
```

#### If you have a sandbox/training account with a web portal

(e.g. AWS Skill Builder / Innovation Sandbox style portals)

These typically issue **temporary** credentials (Access Key ID + Secret
Access Key + **Session Token**) directly from a web page, rather than
supporting `aws configure sso` or `aws sso login`. **Do not run
`aws configure` or `aws login`** for this kind of account — there's nothing
to log into. Instead, look for a "Command line or programmatic access" link
on the portal and copy all **three** values into your `.env` file directly
(see step 3). A session token is required for temporary credentials —
omitting it is a common cause of silent auth failures.

These credentials **expire** (often within a few hours). If Bedrock calls
that were working suddenly start failing with `ExpiredTokenException`, this
is the first thing to check — see "Troubleshooting" below. Fetch a fresh
set of all three values from the portal and overwrite all three lines in
`.env` (don't mix old and new values).

### 3. Set environment variables

Create a `.env` file in the project root. Use `.env.example` as a starting
point:

- **macOS:** `cp .env.example .env` in Terminal, then edit it with any text
  editor (`nano .env`, `open -e .env`, or `code .env` if using VS Code).
  Files starting with `.` are hidden in Finder by default — if you want to
  see it there too, press `Cmd+Shift+.` in Finder to toggle hidden files.
- **Windows:** copy `.env.example` to `.env`, then edit — but **not** with
  a plain double-click into Notepad's Save dialog, since Notepad defaults
  to appending `.txt` unless you explicitly choose "Save as type: All
  Files". Using `copy .env.example .env` in PowerShell first, then editing
  the existing `.env` file, avoids this trap entirely.

```
AWS_ACCESS_KEY_ID=ASIA...
AWS_SECRET_ACCESS_KEY=...
AWS_SESSION_TOKEN=...
AWS_REGION=us-east-1
BEDROCK_MODEL=us.anthropic.claude-haiku-4-5-20251001-v1:0
```

Notes:

- `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` / `AWS_SESSION_TOKEN` are
  only needed if you're not using a locally-configured AWS profile (e.g.
  sandbox-issued temporary credentials). Omit them if `aws configure` /
  `aws sso login` already set up credentials your machine can find globally.
- **`AWS_REGION` must match the region prefix of `BEDROCK_MODEL`.**
  Cross-region inference profile IDs are prefixed `us.`, `apac.`, `eu.`,
  etc. A `us.`-prefixed model ID will fail with
  `ValidationException: The provided model identifier is invalid` if
  `AWS_REGION` is set to a non-US region (e.g. `ap-southeast-1`), even
  though the model genuinely exists and is enabled on your account. This
  was the single most time-consuming issue during initial setup — check
  this first if you see that specific error. If your sandbox is based
  outside the US, the simplest fix is usually to keep `AWS_REGION=us-east-1`
  and a `us.`-prefixed model — Bedrock inference profiles don't require you
  to be physically located where the region is.
- To see what Claude models/profiles are actually available on your
  account in a given region (this works with sandbox env-var credentials
  too — the AWS CLI does **not** read `.env` automatically, so set the
  three `AWS_*` values as shell env vars first if you haven't run
  `aws configure`):

  **macOS/Linux (bash/zsh):**
  ```
  export AWS_ACCESS_KEY_ID="<your key>"
  export AWS_SECRET_ACCESS_KEY="<your secret>"
  export AWS_SESSION_TOKEN="<your token>"
  ```

  **Windows (PowerShell):**
  ```
  $env:AWS_ACCESS_KEY_ID="<your key>"
  $env:AWS_SECRET_ACCESS_KEY="<your secret>"
  $env:AWS_SESSION_TOKEN="<your token>"
  ```

  Then, on either platform:

  ```
  aws bedrock list-foundation-models --region us-east-1 \
    --query "modelSummaries[?contains(modelId, 'claude')].modelId" --output table
  aws bedrock list-inference-profiles --region us-east-1 \
    --query "inferenceProfileSummaries[?contains(inferenceProfileId, 'claude')].inferenceProfileId" --output table
  ```

  Prefer a dated, pinned model ID/profile (e.g.
  `us.anthropic.claude-haiku-4-5-20251001-v1:0`) over a bare alias where
  possible — models do get retired
  (`ResourceNotFoundException: This model version has reached the end of
  its life`), and a pinned ID makes that failure mode explicit rather than
  silently drifting.

### 4. Verify before running the full demo

Cheapest possible sanity check — one LLM call, no LangGraph, no tools:

```
python -c "from dotenv import load_dotenv; load_dotenv(); from kallangflow.agent import chat_model; print(chat_model().invoke('Say hello in one sentence.').content)"
```

This should print a real sentence. If it errors, resolve that before
running `python -m kallangflow.demo` — debugging inside the full 6-step,
multi-agent demo is much slower than isolating the problem here first.

### 5. Run

```
python -m kallangflow.demo
```

Phases 1–5's non-LLM parts (simulator, population modelling, redistribution
math) run and print correctly even without valid AWS credentials — only the
agent's own reasoning/recommendation calls are skipped, with a clear
`[AWS Bedrock credentials not set — ...]` message in their place. The
`HAVE_KEY` gate in `demo.py` checks for `AWS_ACCESS_KEY_ID` and
`BEDROCK_MODEL` being set, not any Groq-related variable.

---

## Architecture notes

- **Model client caching**: `chat_model()` in `agent.py` caches the
  `ChatBedrockConverse` client per temperature setting, rather than
  reconstructing it (and its underlying boto3 client) on every node call.
- **Tool-call message hygiene**: only `monitor_node` binds tools and
  produces `toolUse`/`toolResult` message blocks. That raw tool-calling
  loop is kept in a **local scratch list**, not merged into the graph's
  shared `state["messages"]` — only a clean, plain-text summary is passed
  downstream to `predict_node` / `recommend_node`. Passing raw tool-call
  history to a model invoked without a bound `toolConfig` causes Bedrock's
  Converse API to emit `RuntimeWarning: Tool messages were passed without
  toolConfig` and can silently return an empty response
  (`reply.content == []`) instead of raising an error.
- **Step 5 concurrency**: the 10 real per-attendee agent runs (each a full
  Monitor→Adapt LangGraph invocation) execute concurrently via
  `ThreadPoolExecutor`, since Bedrock calls are I/O-bound. Note this means
  the shared `WorldState` object is read (and, depending on tool
  implementation, possibly written) from multiple threads concurrently
  during that step — acceptable for demo purposes, but not thread-safe by
  design.
- **Attendee persistence**: `db.py` is one flat SQLite table, no ORM, no
  migrations — intentionally simple for a single-attendee-at-a-time demo.
  SQLite handles concurrent writers poorly; a real multi-user version would
  want a proper database (e.g. Postgres) plus per-user accounts, not one
  shared `"A-DEMO"` row.
- **`app.py`'s in-memory `world`** has the same single-process caveat as
  before: fine for `python app.py`'s dev server, but multiple `gunicorn`
  workers (or multiple machines) would each get their own disconnected copy
  — see the "shared world state" note from the architecture discussion.

---

## Troubleshooting

| Symptom | Likely cause |
| --- | --- |
| `ModuleNotFoundError: No module named 'X'` | Packages installed piecemeal instead of via `pip install -r requirements.txt`. Re-run the full install. |
| `botocore.errorfactory.ValidationException: The provided model identifier is invalid` | `AWS_REGION` doesn't match the region prefix (`us.`/`apac.`/etc.) of `BEDROCK_MODEL`. |
| `botocore.exceptions.ClientError: ... ExpiredTokenException: The security token included in the request is expired` | Sandbox/temporary credentials have expired (often within a few hours). Fetch fresh Access Key ID + Secret Access Key + Session Token from your sandbox portal and overwrite all three lines in `.env`. |
| `botocore.errorfactory.ResourceNotFoundException: This model version has reached the end of its life` | The pinned model ID has been retired by AWS. Re-check `aws bedrock list-foundation-models` / `list-inference-profiles` for a current one. |
| `aws sts get-caller-identity` → `Unable to locate credentials` | The AWS CLI does not read `.env` files automatically (unlike the Python app, via `python-dotenv`). Either run `aws configure`, or set the three `AWS_*` values in your shell session before running CLI commands — `export AWS_ACCESS_KEY_ID=...` (macOS/Linux) or `$env:AWS_ACCESS_KEY_ID="..."` (Windows PowerShell). |
| `python -c "...boto3.Session().get_credentials()..."` prints `None` | `.env` is missing, misnamed (check for a stray `.env.txt`), not in the current working directory, or missing one of the three `AWS_*` credential lines. |
| `RuntimeWarning: Tool messages were passed without toolConfig` + empty `recommendation` (`reply.content == []`) | A node is invoking the model on message history containing raw tool-call blocks without also binding tools. Fixed by keeping `monitor_node`'s tool-calling loop in a local scratch list and returning only a clean text summary to `state["messages"]` — see Architecture notes above. |
| `IndentationError` after editing `demo.py` | Check that every `if HAVE_KEY: / else:` pair is aligned to the same indentation level, and that `else:` isn't accidentally attached to a nested `for` loop instead of the intended `if`. |

## About

Agentic crowd-flow simulator for event journeys, made possible through LangGraph agents on Amazon Bedrock. Journeys are customiseable via user input. Created by the Techtitans2 team for the SimplifyNext Agentic AI Hackathon 2026.
