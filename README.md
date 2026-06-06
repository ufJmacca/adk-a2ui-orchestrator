# ADK A2UI Orchestrator Demo

Local Python-first Google ADK A2A and A2UI orchestrator demo for business
banking relationship managers.

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
to fixtures, pasted into A2UI payloads, written to ADK session state, stored as
artifacts, or written into this README.

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

ADK loads the root agent lazily. `adk api_server` can start, serve the Dev UI,
and publish the A2A card before these variables are validated. Missing
`OPENROUTER_API_KEY` or `LLM_MODEL` is reported on the first agent request, and
error messages name missing variables but do not reveal secret values.

## Primary Local Runtime

The primary local runtime is the Google ADK API server with A2A and the bundled
Dev UI enabled. Synchronize the locked environment, provide the required runtime
configuration through your shell, secret manager, or local `.env`, then start
the runtime from the repository root.

```bash
uv sync --locked
uv run adk api_server --a2a --with_ui orchestrator_demo --host 0.0.0.0 --port 8000 --session_service_uri sqlite:///.adk/orchestrator_sessions.sqlite --artifact_service_uri file:./.adk/artifacts
```

Open the forwarded `8000` URL and select the `orchestrator` agent in the ADK Dev
UI. External A2A clients can discover the orchestrator at
`http://127.0.0.1:8000/a2a/orchestrator`. The command stores local ADK session
and artifact state under `.adk/`, which is ignored by git.

## Dev UI Debugging

Use ADK Web only for Dev UI debugging when the A2A server surface is not needed.
The command still loads the same `orchestrator` root agent, but it is not the
primary local runtime for this migration.

```bash
uv sync --locked
uv run adk web orchestrator_demo --host 0.0.0.0 --port 8000
```

## ADK Tool Contract

The root agent exposes stable tools for the full request, draft-edit, approval,
and rejection loop:

| Tool | Purpose |
| --- | --- |
| `submit_orchestrator_request` | Submit a natural-language RM request. |
| `add_plan_instruction` | Add a user instruction to one draft plan step. |
| `remove_plan_step` | Remove one draft plan step when allowed by plan invariants. |
| `replace_plan_agent` | Replace the agent assigned to one draft plan step. |
| `reorder_plan_steps` | Reorder the current draft plan steps. |
| `approve_orchestrator_plan` | Approve a current draft plan and execute its graph. |
| `reject_orchestrator_plan` | Reject a current draft plan without execution. |

Tool responses are JSON-serializable dictionaries with stable camelCase fields
for protocol consumers. Plan-required and draft-updated responses include both
structured plan JSON and A2UI data parts so clients can either render the plan
surface or inspect the JSON directly.

## Approval Flow

Complex requests always require structured plan approval:

1. `submit_orchestrator_request` classifies the request, generates a draft plan,
   persists pending approval state in ADK session state, and returns A2UI plan
   review payloads.
2. Draft edit tools mutate the current draft and return a refreshed A2UI plan
   surface with an incremented plan version.
3. `reject_orchestrator_plan` records structured rejection state and does not
   execute the graph.
4. `approve_orchestrator_plan` validates the current plan version and approved
   step IDs, freezes the approved plan, executes the ADK graph, emits status
   events, and stores final artifacts through ADK artifact storage.

Approved and rejected plans are immutable for this MVP. Conversation text is not
an approval mechanism.

## A2UI Presentation

A2UI remains the presentation format for draft plan review, draft refreshes,
approval and rejection controls, downstream specialist surfaces, and useful
artifact summaries. These payloads now travel through ADK Dev UI rendering and
A2A data parts with `application/json+a2ui` metadata. Structured JSON fields
remain available so clients that do not render A2UI can still perform explicit
edit, approve, or reject tool calls.

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

Useful focused checks include `uv run pytest tests/test_repo_hygiene.py` for
repository and documentation hygiene, `uv run pytest tests/test_adk_tool_flows.py`
for ADK tool behavior, and `uv run pytest tests/test_adk_a2a_plugin.py` for A2A
exposure.

## Known Limitations

- synthetic data only.
- mocked SLM intent classification.
- mocked search for deterministic web-search behavior.
- local-only demo with no production deployment target.
- deterministic tests rather than live model-quality evaluation.
- no regulated decisions, binding credit decisions, pricing decisions, or
  production compliance determinations.
