# ADK A2UI Orchestrator Demo

Local Python-first ADK 2.0+ A2UI orchestrator demo for business banking
relationship managers.

## Project Baseline

This repository is managed with `uv`. Use `uv` for dependency management,
lockfile updates, virtual environment synchronization, quality gates, and app
execution.

```bash
uv sync --locked
uv run pytest
uv run ruff check .
uv run mypy orchestrator_demo
```

Disallowed project-manager paths for this demo include package-manager install
commands outside `uv`, alternate Python lockfiles, committed virtual
environments, and dependency-manager files such as `requirements.txt`,
`poetry.lock`, `Pipfile.lock`, and `environment.yml`.

## Runtime Configuration

Runtime secrets must come from environment variables or a local `.env` file
based on `.env.example`. Real secret values must not be committed, logged, added
to fixtures, pasted into A2UI payloads, or written into this README.

Required variables:

| Variable | Purpose |
| --- | --- |
| `OPENROUTER_API_KEY` | Secret OpenRouter-compatible LiteLLM credential. |
| `LLM_MODEL` | LiteLLM model name, for example an `openrouter/...` model path. |

Optional variables:

| Variable | Default | Purpose |
| --- | --- | --- |
| `OPENROUTER_API_BASE` | `https://openrouter.ai/api/v1` | OpenRouter-compatible API base URL. |
| `OR_APP_NAME` | unset | Optional OpenRouter app name metadata. |
| `OR_SITE_URL` | unset | Optional OpenRouter site URL metadata. |
| `ORCHESTRATOR_APP_HOST` | `127.0.0.1` | Local HTTP bind host. |
| `ORCHESTRATOR_APP_PORT` | `8000` | Local HTTP bind port. |

The app fails fast when `OPENROUTER_API_KEY` or `LLM_MODEL` is missing. Error
messages name missing variables but do not reveal secret values.

## Local Startup

Synchronize the locked environment, provide the required runtime configuration
through your shell, secret manager, or local `.env`, then start the app.

```bash
uv sync --locked
uv run python -m orchestrator_demo.app
```

The startup line prints the local base URL, normally
`http://127.0.0.1:8000`. The basic renderer is available at `GET /`, and the
machine-readable endpoint contract is available at `GET /api`.

## Google ADK Dev UI

The repo also exposes a Google ADK-compatible `root_agent` for ADK Web. Run this
from the repository root after providing the required runtime configuration:

```bash
uv sync --locked
uv run adk web orchestrator_demo --host 0.0.0.0 --port 8001
```

Open the forwarded `8001` URL and select the `orchestrator` app. ADK Web exposes
the orchestrator through request, approve, and reject tools with JSON outputs.
It does not render A2UI surfaces; use `uv run python -m orchestrator_demo.app`
for the local A2UI renderer.

## Local Transport Contract

The local app uses JSON HTTP endpoints to exercise the A2A/A2UI loop:

| Interface | Endpoint | Purpose |
| --- | --- | --- |
| Request | `POST /api/request` | Submit a natural-language RM request. |
| Action | `POST /api/user-action` | Submit a structured A2UI `userAction`. |
| Status list | `GET /api/status` | Fetch accumulated status events. |
| Status stream | `GET /api/status/stream` | Fetch status events as `text/event-stream`. |
| Artifacts | `GET /api/artifacts` | Fetch the latest final artifacts and A2UI parts. |
| Renderer | `GET /` | Open the trusted local renderer. |

`POST /api/request` accepts JSON with `input` and returns `taskId`,
`contextId`, route `path`, routing `decision`, optional `approvalPlan`,
`a2uiParts`, `statusEvents`, and `artifacts`. Simple high-confidence requests
route directly to one specialist. Complex requests return `path:
plan_required`, an editable approval plan, and A2UI parts with
`application/json+a2ui` metadata before any specialist execution.

`POST /api/user-action` accepts the original A2UI event envelope:

```json
{
  "userAction": {
    "type": "approve_plan",
    "surfaceId": "surface_plan_example",
    "payload": {
      "planId": "plan_example",
      "approvedStepIds": ["step_1", "step_2"],
      "editedPlanVersion": 1
    }
  }
}
```

The `surfaceId` is required. Plan events are handled by structured approval
state, and downstream specialist events are routed through the surface ownership
registry instead of an LLM decision.

## Approval Flow

Complex requests always require A2UI plan approval:

1. `POST /api/request` classifies the request, generates an `approvalPlan`, and
   returns an editable A2UI workflow canvas using the plan
   `approval_surface_id`.
2. The renderer may emit `add_instruction`, `reject_plan`, or `approve_plan`
   events through `POST /api/user-action`.
3. Draft edit events update the unapproved plan and refresh the A2UI surface.
4. `reject_plan` records structured rejection state and does not execute the
   graph.
5. `approve_plan` freezes the approved plan, creates the ADK graph, executes
   the graph, emits progress status events, and stores final artifacts.

Approved plans are immutable for this MVP. Conversation text is not an approval
mechanism.

## Renderer Behavior

The renderer at `GET /` is a minimal trusted DOM mapper for the Basic Catalog
A2UI payloads used in the demo. It renders approval surfaces, downstream
specialist surfaces, status updates, and artifacts. It preserves A2UI
`surfaceId` values, emits structured `userAction` events back to
`POST /api/user-action`, and does not execute arbitrary code from generated UI
payloads.

Downstream specialist A2UI is validated and passed through without orchestrator
layout rewrites. If a downstream surface creates actions, the orchestrator
routes those actions deterministically by `surfaceId`.

## ADK Graph Acceptance

MVP acceptance requires approved complex plans to build and execute through the
ADK graph workflow API. The graph builder must produce an ADK-backed workflow
for sequential and fan-out/fan-in plans. If the installed ADK graph API is
unavailable or incompatible, the app and tests must fail with a clear
developer-facing error. There is no silent local fallback for MVP acceptance.

## Test And Harness Handoff

The harness should run the locked environment and quality gates through `uv`:

```bash
uv sync --locked
uv run pytest
uv run ruff check .
uv run mypy orchestrator_demo
```

Useful focused checks include `uv run pytest tests/test_app_transport.py` for
request/action/status/artifact behavior,
`uv run pytest tests/test_renderer_contract.py` for renderer behavior, and
`uv run pytest tests/test_adk_graph_acceptance.py` for ADK graph acceptance.

## Known Limitations

- synthetic data only.
- mocked SLM intent classification.
- mocked search for deterministic web-search behavior.
- local-only demo with no production deployment target.
- deterministic tests rather than live model-quality evaluation.
- minimal renderer intended to prove the A2UI loop, not production UX.
- no regulated decisions, binding credit decisions, pricing decisions, or
  production compliance determinations.
