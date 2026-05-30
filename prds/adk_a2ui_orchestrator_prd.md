# PRD — ADK 2.0+ A2UI Multi-Agent Orchestrator for Business Banking Relationship Managers

**Version:** 0.3  
**Date:** 2026-05-30  
**Status:** Draft for agentic harness implementation  
**Primary audience:** Internal engineering  
**Primary implementation language:** Python  
**Primary user:** Business Banking Relationship Manager  
**Primary interaction surface:** A2A with A2UI payloads  
**Model access:** LiteLLM-compatible models, initially via OpenRouter runtime environment variables  
**Package/project manager:** `uv` only for project creation, dependency management, lockfile generation, virtual environment management, and command execution  

---

## 1. Product summary

This project will deliver a **Python-first orchestrator agent** for business banking relationship managers. The orchestrator will receive a user request, call a mocked external SLM intent API, enhance the SLM suggestion with LLM-based reasoning when needed, decide whether the request is simple or complex, and then either:

1. Route directly to one specialist agent for simple single-agent requests, or
2. Generate a structured execution plan for complex requests, present that plan through an A2UI approval surface, capture approval as a UI event, and execute the approved workflow through a dynamically created ADK 2.0+ graph.

The orchestrator is primarily a **multi-agent task router**. ADK 2.0+ graph orchestration and A2UI generative UI are enabling features that support explicit routing, plan approval, dynamic graph construction, and UI-based human-in-the-loop control.

The first demo is **local-only**, uses **synthetic business banking data**, and assumes **LiteLLM-compatible inference** using environment-provided OpenRouter credentials. The implementation should be structured so local specialist agents can later be replaced by remote A2A-compatible agents without redesigning the orchestrator. The repository must be managed with `uv` as the sole Python package and project manager, using `pyproject.toml`, `.python-version`, and `uv.lock` for reproducible local and harness execution.

---

## 2. Background and technical basis

ADK graph workflows provide a code-defined way to represent workflows as execution nodes and edges. Nodes can include agents, tools, human input, code functions, or nested workflows. This aligns with the need for deterministic orchestration and dynamic plan execution in a multi-agent system.

A2UI lets agents produce structured UI JSON that a trusted client renderer converts into interactive UI components. In the ADK integration, A2UI payloads can be parsed, validated, and wrapped into A2A `DataPart` messages with the MIME type `application/json+a2ui`.

The existing Google A2UI ADK orchestrator sample demonstrates relevant patterns:

- Routing requests to downstream expert subagents.
- Passing A2UI extension metadata to downstream A2A agents.
- Using `RemoteA2aAgent` for subagent communication.
- Tracking A2UI surface ownership.
- Routing later A2UI `userAction` events back to the owning subagent.

This PRD adapts those patterns but changes the orchestration behavior: simple requests may be routed directly, while complex requests require explicit A2UI approval before specialist execution.

---

## 3. Goals

The system must:

1. Provide a Python-based orchestrator agent using ADK 2.0+.
2. Use A2UI for plan approval and, where useful, complex long-form output.
3. Call a mocked external SLM intent endpoint for every user request.
4. Treat the SLM response as a suggestion rather than a binding decision.
5. Use LLM inference to enhance intent recognition when the SLM result is incomplete, ambiguous, or low confidence.
6. Distinguish simple single-agent requests from complex multi-agent or multi-step requests.
7. Route simple high-confidence requests directly to one specialist agent.
8. Require A2UI-based user approval for every complex request.
9. Capture plan approval through a structured A2UI `userAction`, not through natural-language chat.
10. Build an ADK graph workflow dynamically from the approved plan.
11. Support sequential, fan-out/fan-in, and mixed graph execution patterns.
12. Support a dynamically reloadable Python config file for specialist agent registration.
13. Use local `LlmAgent` specialists for the initial demo.
14. Wrap one or two local specialists behind the same interface expected from future remote A2A agents.
15. Preserve downstream specialist A2UI payloads without alteration.
16. Route downstream A2UI interactions deterministically to the agent that owns the UI surface.
17. Include a basic renderer in scope to prove the approval and UI event loop.
18. Provide clear runtime-secret handling instructions for an agentic harness.
19. Require `uv` as the only supported Python package/project manager for initialization, dependency changes, lockfile updates, virtual environment synchronization, command execution, and harness runs.

---

## 4. Non-goals

The first demo will not:

1. Provide a production deployment target such as Cloud Run, GKE, or enterprise infrastructure.
2. Implement real production remote agents.
3. Implement a live external SLM service; the SLM endpoint must be mocked.
4. Build a custom financial-services A2UI catalog; the demo will use the Basic Catalog first.
5. Make binding credit decisions, pricing decisions, loan approvals, risk-rating changes, or regulated recommendations.
6. Use conversation text as the approval mechanism for complex plans.
7. Modify, normalize, or rewrite A2UI payloads produced by downstream specialist agents.
8. Store real customer data.
9. Require real web search unless explicitly enabled later; mocked search is acceptable for deterministic demos.
10. Assume ADK graph Live Streaming support inside graph workflows. Progress updates should be implemented as status events/messages rather than relying on Live Streaming within the graph.
11. Support `pip`, Poetry, Pipenv, Conda, `requirements.txt`, or ad-hoc virtual environment workflows as project dependency management paths for this repository. These tools must not be used by the harness except where `uv` internally invokes compatible behavior.

---

## 5. Target users and use cases

### 5.1 Primary user

The primary user is a **Business Banking Relationship Manager** who needs help preparing for customer meetings, researching prospects, reviewing relationship context, and identifying next-best engagement opportunities.

### 5.2 Primary domain

The initial domain is **generic business banking**. The demo should not assume highly specialized commercial banking, private banking, or institutional banking workflows unless added later.

### 5.3 Hero journey 1 — Meeting preparation

Example request:

> Prepare me for tomorrow's meeting with ABC Manufacturing.

Expected behavior:

1. Orchestrator calls mocked SLM intent endpoint.
2. Orchestrator enhances the SLM result with LLM reasoning.
3. Request is classified as complex.
4. Orchestrator creates a plan before calling any specialist.
5. Orchestrator renders an editable A2UI workflow canvas.
6. RM approves or edits the plan through UI events.
7. Orchestrator freezes the approved plan.
8. Orchestrator builds a dynamic ADK graph.
9. Specialist agents execute.
10. Synthesis agent produces a meeting-prep briefing.

### 5.4 Hero journey 2 — Prospect research

Example request:

> Research this prospect and tell me whether they are worth pursuing.

Expected behavior:

1. Request is classified as complex because it likely requires web search, industry research, risk signals, opportunity analysis, and synthesis.
2. Orchestrator presents a multi-agent research plan.
3. User approves through A2UI.
4. Graph runs with parallel research where useful.
5. Final output provides a prospect briefing, caveats, opportunities, risks, and recommended follow-up questions.

### 5.5 Simple direct-route journey

Example request:

> Summarize the internal notes for ABC Manufacturing.

Expected behavior:

1. Request is classified as simple if confidence is at least `0.85`.
2. Orchestrator routes directly to the Internal Knowledge Agent.
3. Orchestrator returns the specialist response directly.
4. No plan approval UI is shown.

---

## 6. Functional requirements

### 6.1 Request classification

Every user request must pass through the following classification pipeline.

#### Stage 1 — Mocked SLM intent endpoint

The orchestrator must call a mocked SLM intent endpoint for **every** user request.

The mock must return:

```json
{
  "intent": "industry_research",
  "confidence": 0.87
}
```

Allowed initial intents:

- `industry_research`
- `web_search`
- `internal_knowledge`
- `meeting_prep`
- `prospect_research`
- `credit_risk`
- `relationship_summary`
- `product_opportunity`
- `compliance_policy`
- `data_quality`
- `unknown`

The mocked endpoint must be implemented behind an abstraction that can later be replaced by a real external API call.

Recommended initial implementation:

- An ADK tool-like Python function or class.
- Deterministic rule-based behavior by default.
- Optional test mode to simulate low-confidence or wrong SLM outputs.

#### Stage 2 — LLM-enhanced intent recognition

The orchestrator must evaluate whether the SLM result is sufficient. The LLM-based assessment should consider:

| Signal | Examples |
|---|---|
| Intent count | "Research this prospect and prepare talking points" implies multiple intents. |
| Agent count | A direct internal lookup may need one agent; prospect research may need several. |
| Step count | One inference versus research, analysis, compliance review, and synthesis. |
| Data source count | Internal-only, web-only, or combined internal and external context. |
| Output shape | Simple answer, briefing pack, table, plan, approval UI, or dashboard. |
| Sensitivity | Credit, compliance, customer-sensitive information, or advisory output. |
| Ambiguity | Vague entity names, unclear customer/prospect target, or missing objective. |

The LLM assessment should output:

```json
{
  "intents": ["meeting_prep", "relationship_summary", "industry_research"],
  "confidence": 0.91,
  "complexity": "complex",
  "required_agents": ["relationship_summary", "industry_research", "internal_knowledge", "synthesis"],
  "rationale": "The user requested meeting preparation, which requires multiple data sources and synthesis."
}
```

#### Stage 3 — Confidence-weighted merge

When the SLM and LLM assessment disagree, the orchestrator must perform a confidence-weighted merge.

Default formula:

```text
final_confidence = (0.4 * slm_confidence) + (0.6 * llm_confidence)
```

Default direct-route threshold:

```text
SIMPLE_DIRECT_ROUTE_THRESHOLD = 0.85
```

The LLM assessment carries more weight than the SLM because the SLM is a lightweight suggestion service and may be incomplete.

Clarification questions are allowed only when the orchestrator cannot safely route or plan the request.

---

## 7. Complexity classification

### 7.1 Simple request definition

A request is simple only when all of the following are true:

1. Exactly one intent is selected.
2. Exactly one specialist agent is required.
3. The request can be answered with a single specialist call or single model inference.
4. The merged route confidence is at least `0.85`.
5. The request does not trigger a sensitive or regulated-output path.
6. The request does not require plan approval for any other reason.

Examples:

- "Summarize the latest internal notes for ABC Manufacturing."
- "Give me a quick overview of the manufacturing industry."
- "What product opportunities should I consider for a café business?"

### 7.2 Complex request definition

A request is complex if it is anything more than a single-intent, single-agent, single-inference task.

A request is complex when any of the following are true:

1. Multiple intents are detected.
2. Multiple specialist agents are required.
3. Multiple execution steps are required.
4. Fan-out/fan-in execution would improve quality.
5. Synthesis is required.
6. The request asks for a meeting pack, prospect report, research memo, comparison, or workflow.
7. The request requires combining internal and external sources.
8. The request involves credit, risk, compliance, or advisory output.
9. Confidence is below the direct-route threshold.
10. The orchestrator cannot safely determine a single owner agent.

Complex requests must always require plan approval before specialist execution.

---

## 8. Orchestration behavior

### 8.1 Simple direct path

For a simple high-confidence request:

1. Receive the user request.
2. Call the mocked SLM intent endpoint.
3. Run LLM-enhanced intent assessment.
4. Merge SLM and LLM confidence.
5. Select one specialist agent.
6. Route the request directly to the selected specialist.
7. Return the specialist response directly.
8. Preserve any A2UI payload from the specialist.
9. If an A2UI surface is created, store surface ownership for future UI event routing.

No approval UI is shown in this path.

### 8.2 Complex plan path

For complex requests:

1. Receive the user request.
2. Call the mocked SLM intent endpoint.
3. Run LLM-enhanced intent assessment.
4. Merge SLM and LLM confidence.
5. Classify as complex.
6. Generate an execution plan **before any specialist agent is called**.
7. Render the plan as an editable A2UI workflow canvas.
8. Allow the user to approve, reject, edit, remove steps, reorder steps, choose agents, or add instructions.
9. Capture approval through an A2UI `userAction`.
10. Freeze the approved plan as immutable.
11. Build a dynamic ADK graph workflow from the approved plan.
12. Execute the graph.
13. Emit progress/status updates during execution.
14. Collect specialist outputs.
15. Run synthesis where required.
16. Return the final response and any downstream A2UI surfaces.

### 8.3 A2UI event-routing path

When the client sends an A2UI `userAction`:

1. Extract `surfaceId` from the A2UI payload.
2. Look up the owning specialist agent using the surface ownership registry.
3. Route the event directly to the owning agent.
4. Do not ask the orchestrator LLM to decide the event owner.
5. Preserve the original event payload.
6. Return the owning agent's response.
7. Update ownership if the owning agent creates a new surface.

This is a deterministic route-management requirement.

---

## 9. Plan approval requirements

The orchestrator must generate a plan for every complex request and must not call specialist agents before plan approval.

### 9.1 Plan content

The plan must include:

1. User objective.
2. Detected intent or intents.
3. Selected agents.
4. Ordered steps.
5. Parallel groups, where applicable.
6. Dependencies between steps.
7. Expected output per step.
8. Data source categories.
9. Risk and caveat notes.
10. Final synthesis step, where applicable.
11. Approval controls.
12. Edit controls.
13. Rejection controls.

### 9.2 Approval UI

The approval UI must be an **editable A2UI workflow canvas**.

Minimum capabilities:

- Display objective.
- Display selected agents.
- Display sequence and dependencies.
- Display parallel groups.
- Approve whole plan.
- Reject whole plan.
- Edit plan.
- Remove steps.
- Reorder steps.
- Choose or replace agents.
- Add instructions.
- Emit structured A2UI `userAction` events.

### 9.3 Approval state

Approval state must be stored as structured workflow state outside natural-language conversation history.

Once approved, the plan is immutable for this demo.

### 9.4 Example approval event

```json
{
  "userAction": {
    "type": "approve_plan",
    "surfaceId": "surface_plan_123",
    "payload": {
      "planId": "plan_123",
      "approvedStepIds": ["step_1", "step_2", "step_3"],
      "editedPlanVersion": 2
    }
  }
}
```

### 9.5 Example rejection event

```json
{
  "userAction": {
    "type": "reject_plan",
    "surfaceId": "surface_plan_123",
    "payload": {
      "planId": "plan_123",
      "reason": "Too broad; focus on credit risk only."
    }
  }
}
```

---

## 10. A2UI requirements

### 10.1 Component support

The demo should allow the following A2UI component classes from the Basic Catalog or equivalent supported renderer mappings:

| Component class | Intended use |
|---|---|
| Cards | Summary blocks and key findings. |
| Tables | Customer data, agent outputs, comparisons. |
| Forms | Clarifying questions and editable plan metadata. |
| Buttons | Approve, reject, rerun, and route actions. |
| Accordions | Expandable details and source sections. |
| Charts | Optional trend or opportunity visualisation. |
| Timelines | Relationship history and execution progress. |
| Citations | Source/provenance display. |
| Approval panels | Plan approval and rejection. |
| Status/progress indicators | Multi-step execution status. |

The first version should use the Basic Catalog rather than a custom banking catalog.

### 10.2 A2UI validation

All A2UI JSON generated by the orchestrator or specialist agents must be schema-validated before being sent to the client.

Validation flow:

1. Parse agent output.
2. Detect A2UI parts.
3. Validate against the selected catalog schema.
4. If invalid, provide structured validation feedback to the generating agent.
5. Retry repair once for demo scope.
6. If still invalid, fall back to text output plus a developer diagnostic.
7. Never emit invalid A2UI to the renderer.

### 10.3 Downstream A2UI pass-through

Specialist agents own their own A2UI. The orchestrator must not edit, normalize, rewrite, or reinterpret downstream A2UI payloads.

Allowed orchestrator responsibilities:

- Validate A2UI payloads.
- Preserve A2UI payloads.
- Track surface ownership.
- Route UI events to the owning agent.

Disallowed orchestrator behavior:

- Rewriting specialist UI layout.
- Changing labels, component structure, or actions.
- Using LLM inference to guess the target agent for a `userAction`.
- Combining downstream A2UI surfaces into a new UI unless the synthesis agent explicitly owns the generated surface.

---

## 11. Basic renderer requirements

A basic renderer is in scope for this PRD. It does not need to be production-grade, but it must prove the A2UI interaction loop.

Minimum renderer capabilities:

1. Render the A2UI Basic Catalog components used by the orchestrator.
2. Render the approval workflow canvas.
3. Emit A2UI `userAction` events.
4. Include `surfaceId` in UI events.
5. Send events back over A2A.
6. Display progress/status updates.
7. Display downstream A2UI surfaces from specialist agents.
8. Preserve ownership metadata needed for deterministic event routing.
9. Avoid executing arbitrary code from agent-provided payloads.

The renderer should map declarative A2UI JSON to trusted local components.

---

## 12. Dynamic agent graph requirements

The orchestrator must support:

1. Runtime-loaded agent registry.
2. ADK graph workflow nodes.
3. Conditional routing.
4. Dynamic graph creation based on the user request and approved plan.
5. Sequential flows.
6. Fan-out/fan-in flows.
7. Mixed sequential and parallel execution.
8. Dynamically reloadable Python config file for available agents.
9. Local `LlmAgent` specialists for the initial demo.
10. One or two local agents wrapped behind the same interface that future remote A2A agents will use.
11. Common specialist interface using:
    - `AgentDescriptor`
    - `capabilities`
    - `input_schema`
    - `output_schema`
    - `a2ui_catalogs`
    - `routing_examples`

### 12.1 Required graph patterns

| Pattern | Example |
|---|---|
| Direct single node | One specialist answers one simple question. |
| Sequential | Internal knowledge → credit risk → synthesis. |
| Fan-out/fan-in | Industry research + web search + internal knowledge in parallel → synthesis. |
| Conditional route | If missing internal data → data quality agent; otherwise synthesis. |
| Approval pause | Plan approval before graph creation/execution. |

### 12.2 Graph construction rule

For complex approved plans, graph-based execution is required. The graph must be created dynamically from the approved `ExecutionPlan`.

---

## 13. Specialist agents

The demo should include the following specialists.

| Agent | Purpose | Initial implementation |
|---|---|---|
| Industry Research Agent | Provides sector overview, market drivers, risks, and opportunities. | Local `LlmAgent` |
| Web Search Agent | Retrieves and summarizes public information about companies, industries, and market events. | Local `LlmAgent` with mocked or pluggable search |
| Internal Knowledge Agent | Answers from synthetic CRM notes, policy snippets, relationship records, and internal knowledge. | Local `LlmAgent` |
| Credit Risk Agent | Flags risk themes, credit concerns, covenants, repayment indicators, and missing data. | Local `LlmAgent` |
| Relationship Summary Agent | Produces relationship history, key contacts, prior meetings, products held, and open follow-ups. | Local `LlmAgent` |
| Product Opportunity Agent | Identifies banking product opportunities such as deposits, lending, merchant services, treasury, cards, or FX. | Local `LlmAgent` |
| Compliance / Policy Agent | Checks whether outputs include regulated, risky, or unsupported claims. | Local `LlmAgent` |
| Data Quality Agent | Highlights missing information, stale context, weak evidence, or low-confidence outputs. | Local `LlmAgent` |
| Meeting Prep Agent | Generates final meeting-prep brief structure. | Local `LlmAgent` |
| Synthesis Agent | Combines specialist outputs into final RM-ready answer. | Local `LlmAgent` |

For the initial demo, all specialist outputs may use synthetic data. In production, local specialists should be replaceable by remote A2A-compatible agents.

---

## 14. Runtime secret and LLM inference configuration

The agentic harness must provide LLM credentials through runtime secrets or environment variables only.

Secrets must not be placed in:

- The PRD.
- Prompts.
- Source code.
- Test fixtures.
- Checked-in config.
- A2UI payloads.
- Logs.
- Conversation history.

### 14.1 Required runtime environment variables

```bash
OPENROUTER_API_KEY=...
LLM_MODEL=openrouter/<provider>/<model>
```

### 14.2 Optional runtime environment variables

```bash
OPENROUTER_API_BASE=https://openrouter.ai/api/v1
OR_APP_NAME=adk-a2ui-orchestrator-demo
OR_SITE_URL=http://localhost:3000
```

### 14.3 `.env.example`

The repository should include a checked-in `.env.example`:

```env
# .env.example
OPENROUTER_API_KEY=
LLM_MODEL=openrouter/<provider>/<model>
OPENROUTER_API_BASE=https://openrouter.ai/api/v1
OR_APP_NAME=adk-a2ui-orchestrator-demo
OR_SITE_URL=http://localhost:3000
```

The real `.env` must be ignored:

```gitignore
.env
.env.*
!.env.example
```

### 14.4 Local operator flow

```bash
cp .env.example .env
# edit .env locally and paste the key
uv sync --locked
uv run python -m orchestrator_demo.app
```

Alternative direct environment injection:

```bash
export OPENROUTER_API_KEY="sk-or-..."
export LLM_MODEL="openrouter/<provider>/<model>"
uv sync --locked
uv run python -m orchestrator_demo.app
```

PowerShell:

```powershell
$env:OPENROUTER_API_KEY="sk-or-..."
$env:LLM_MODEL="openrouter/<provider>/<model>"
uv sync --locked
uv run python -m orchestrator_demo.app
```

### 14.5 Harness contract

The agentic harness must inject:

- `OPENROUTER_API_KEY` as a secret.
- `LLM_MODEL` as runtime config.

The implementation must fail fast if either required value is missing. The error message must not reveal the key or any secret-derived value. The harness must install dependencies with `uv sync --locked` and must execute application and test commands through `uv run`.

### 14.6 Example settings model

```python
from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    openrouter_api_key: SecretStr = Field(alias="OPENROUTER_API_KEY")
    llm_model: str = Field(alias="LLM_MODEL")
    openrouter_api_base: str = Field(
        default="https://openrouter.ai/api/v1",
        alias="OPENROUTER_API_BASE",
    )
    or_app_name: str | None = Field(default=None, alias="OR_APP_NAME")
    or_site_url: str | None = Field(default=None, alias="OR_SITE_URL")


settings = Settings()
```

### 14.7 Example ADK LiteLLM model builder

```python
import os

from google.adk.models.lite_llm import LiteLlm

from app.settings import settings


def configure_litellm_env() -> None:
    os.environ["OPENROUTER_API_KEY"] = settings.openrouter_api_key.get_secret_value()
    os.environ["OPENROUTER_API_BASE"] = settings.openrouter_api_base

    if settings.or_app_name:
        os.environ["OR_APP_NAME"] = settings.or_app_name

    if settings.or_site_url:
        os.environ["OR_SITE_URL"] = settings.or_site_url


def build_litellm_model() -> LiteLlm:
    configure_litellm_env()
    return LiteLlm(model=settings.llm_model)
```

### 14.8 Security defaults for the demo key

For the initial harness run:

1. Use a dedicated OpenRouter key only for this demo.
2. Apply the smallest practical credit or spend limit.
3. Rotate or delete the key after the run.
4. Do not paste the key into natural-language prompts.
5. Do not expose the key to downstream specialist agents.
6. Redact any environment dump or error message that might include it.

---

## 15. Package and project management requirements

`uv` is mandatory for this repository. The implementation harness and all local developer instructions must use `uv` as the single package and project manager for Python dependency management, virtual environment synchronization, lockfile generation, and command execution.

### 15.1 Required project files

The repository must include and maintain:

```text
pyproject.toml
uv.lock
.python-version
.gitignore
README.md
```

Requirements:

1. `pyproject.toml` is the source of truth for project metadata, Python version requirements, runtime dependencies, development dependencies, scripts, and tool configuration.
2. `uv.lock` must be committed and used by the harness for reproducible dependency resolution.
3. `.python-version` must be committed so the Python runtime is explicit.
4. `.venv/` must be excluded from version control.
5. `.env` and `.env.*` must be excluded from version control except `.env.example`.
6. `requirements.txt`, `poetry.lock`, `Pipfile`, `Pipfile.lock`, `conda.yaml`, `environment.yml`, and committed virtual environments are prohibited for this demo repository.

### 15.2 Required uv commands

The implementation must use the following command patterns:

```bash
# Initialize project, if starting from an empty repository
uv init --package adk-a2ui-orchestrator-demo

# Add runtime dependencies
uv add google-adk a2ui-agent-sdk litellm pydantic pydantic-settings python-dotenv

# Add development dependencies
uv add --dev pytest pytest-asyncio ruff mypy

# Resolve and lock dependencies
uv lock

# Synchronize exactly from the lockfile in the harness
uv sync --locked

# Run the local application
uv run python -m orchestrator_demo.app

# Run tests and quality checks
uv run pytest
uv run ruff check .
uv run mypy orchestrator_demo
```

The harness must not use `pip install -r requirements.txt`, `poetry install`, `pipenv install`, `conda env create`, or direct global Python package installation for this project.

### 15.3 Minimum pyproject expectations

The generated `pyproject.toml` must include at minimum:

```toml
[project]
name = "adk-a2ui-orchestrator-demo"
version = "0.1.0"
description = "Local ADK 2.0+ A2UI orchestrator demo for business banking relationship managers."
requires-python = ">=3.11"
dependencies = [
  "google-adk",
  "a2ui-agent-sdk",
  "litellm",
  "pydantic",
  "pydantic-settings",
  "python-dotenv",
]

[dependency-groups]
dev = [
  "pytest",
  "pytest-asyncio",
  "ruff",
  "mypy",
]
```

The exact dependency versions may be selected by the harness, but the generated `uv.lock` must capture the final resolved versions. When feasible, the harness should prefer stable versions compatible with ADK 2.0+ and the selected A2UI package.

### 15.4 Dependency change policy

Any dependency change must be made through `uv add`, `uv remove`, or direct `pyproject.toml` edits followed by `uv lock`. The implementation report must list any added dependencies and explain why they were needed.

### 15.5 CI and harness reproducibility

Before declaring success, the harness must prove that a clean checkout can run with:

```bash
uv sync --locked
uv run pytest
```

A missing, stale, or inconsistent `uv.lock` is a build failure.

---

## 16. Python implementation requirements

### 16.1 Suggested package structure

```text
adk-a2ui-orchestrator-demo/
  pyproject.toml
  uv.lock
  .python-version
  .env.example
  .gitignore
  README.md
  orchestrator_demo/
    app/
      __main__.py
      settings.py
      bootstrap_llm.py
    orchestrator/
      agent.py
      planner.py
      router.py
      graph_builder.py
      execution.py
      approval_state.py
      surface_routes.py
    intent/
      slm_mock_client.py
      classifier.py
      merge.py
    registry/
      agent_config.py
      agent_registry.py
      descriptors.py
    agents/
      industry_research.py
      web_search.py
      internal_knowledge.py
      credit_risk.py
      relationship_summary.py
      product_opportunity.py
      compliance_policy.py
      data_quality.py
      meeting_prep.py
      synthesis.py
    a2ui_support/
      schema_manager.py
      approval_canvas.py
      validation.py
      event_parser.py
      renderer_contract.py
    a2a_support/
      remote_agent_adapter.py
      local_remote_wrapper.py
      part_converters.py
  tests/
    test_intent_merge.py
    test_simple_routing.py
    test_complex_planning.py
    test_approval_events.py
    test_surface_routing.py
    test_graph_builder.py
```

### 16.2 Core Pydantic contracts

```python
from typing import Literal

from pydantic import BaseModel, Field


class IntentSuggestion(BaseModel):
    intent: str
    confidence: float = Field(ge=0.0, le=1.0)


class LlmIntentAssessment(BaseModel):
    intents: list[str]
    confidence: float = Field(ge=0.0, le=1.0)
    complexity: Literal["simple", "complex"]
    rationale: str
    required_agents: list[str]


class RoutingDecision(BaseModel):
    path: Literal["direct", "plan_required", "clarification_required"]
    selected_agent: str | None
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str


class PlanStep(BaseModel):
    step_id: str
    agent_id: str
    instruction: str
    depends_on: list[str] = []
    expected_output: str


class ExecutionPlan(BaseModel):
    plan_id: str
    objective: str
    steps: list[PlanStep]
    immutable_after_approval: bool = True


class AgentDescriptor(BaseModel):
    agent_id: str
    display_name: str
    capabilities: list[str]
    input_schema: dict
    output_schema: dict
    a2ui_catalogs: list[str]
    routing_examples: list[str]
    execution_mode: Literal["local_llm", "local_a2a_compatible", "remote_a2a"]
```

### 16.3 Agent registry

The first version should use a Python config file to register agents and support dynamic reload without orchestrator restart.

Example:

```python
AVAILABLE_AGENTS = [
    AgentDescriptor(
        agent_id="industry_research",
        display_name="Industry Research Agent",
        capabilities=["industry trends", "market risks", "sector outlook"],
        input_schema={"type": "object"},
        output_schema={"type": "object"},
        a2ui_catalogs=["basic"],
        routing_examples=[
            "Give me a quick overview of the manufacturing industry.",
            "What are key risks in retail trade this quarter?"
        ],
        execution_mode="local_llm",
    ),
]
```

Requirements:

1. Removed agents must not be selected for new plans.
2. Existing immutable approved plans should continue using the frozen agent IDs where possible.
3. If an approved plan references an unavailable agent, execution must fail gracefully with a clear developer-facing error.
4. Agent registry reloads must be logged.
5. Agent descriptors must not include secrets.

### 16.4 Mock SLM implementation

The SLM mock should be replaceable by an external HTTP client later.

Example interface:

```python
class SlmIntentClient:
    async def classify(self, user_input: str) -> IntentSuggestion:
        ...
```

Example mock behavior:

```python
class MockSlmIntentClient(SlmIntentClient):
    async def classify(self, user_input: str) -> IntentSuggestion:
        text = user_input.lower()

        if "meeting" in text or "prepare" in text:
            return IntentSuggestion(intent="meeting_prep", confidence=0.82)

        if "prospect" in text or "research" in text:
            return IntentSuggestion(intent="prospect_research", confidence=0.78)

        if "internal notes" in text or "crm" in text:
            return IntentSuggestion(intent="internal_knowledge", confidence=0.9)

        if "industry" in text or "sector" in text:
            return IntentSuggestion(intent="industry_research", confidence=0.88)

        return IntentSuggestion(intent="unknown", confidence=0.35)
```

### 16.5 Routing algorithm

Pseudo-flow:

```python
async def route_request(user_input: str) -> RoutingDecision:
    slm = await slm_client.classify(user_input)
    llm = await llm_classifier.assess(user_input, slm, registry.available_agents())

    final_confidence = 0.4 * slm.confidence + 0.6 * llm.confidence

    if (
        llm.complexity == "simple"
        and len(llm.intents) == 1
        and len(llm.required_agents) == 1
        and final_confidence >= 0.85
    ):
        return RoutingDecision(
            path="direct",
            selected_agent=llm.required_agents[0],
            confidence=final_confidence,
            reason="Single intent, single agent, high confidence.",
        )

    return RoutingDecision(
        path="plan_required",
        selected_agent=None,
        confidence=final_confidence,
        reason="Complex, multi-step, multi-agent, or below direct-route threshold.",
    )
```

---

## 17. Progress update requirements

For approved complex workflows, the orchestrator must emit status updates such as:

1. Plan approved.
2. Graph created.
3. Step started.
4. Step completed.
5. Step failed.
6. Parallel branch started.
7. Parallel branch completed.
8. Synthesis started.
9. Final response ready.

These should be implemented as graph execution events, A2A task status updates, app-level progress messages, or A2UI status/progress components.

---

## 18. Security, validation, and trust requirements

### 18.1 A2UI and remote-agent trust

Remote agents and generated UI definitions must be treated as untrusted input.

Required controls:

| Area | Requirement |
|---|---|
| A2UI validation | Validate all A2UI payloads before rendering. |
| UI event routing | Use deterministic `surfaceId` ownership lookup. |
| Remote-agent readiness | Treat future AgentCards, messages, artifacts, statuses, and UI definitions as untrusted. |
| Prompt injection | Do not blindly insert remote agent descriptions or UI payloads into orchestrator prompts. |
| Approval integrity | Store approval as structured state. |
| Plan immutability | Freeze plan after approval. |
| Secrets | Keep OpenRouter/LiteLLM keys in runtime env vars only. |
| Auditability | Log SLM suggestion, LLM assessment, merged decision, plan approval, graph execution, and UI events. |
| Renderer safety | Do not execute arbitrary code from A2UI payloads. |
| Credential handling | Redact secrets in all logs and exceptions. |

### 18.2 Dependency security

Because the implementation uses LiteLLM and ADK, the harness must install pinned dependency versions through `uv` and avoid known compromised versions. Dependency installation must be reproducible through the committed `uv.lock` file.

Required controls:

1. Use `uv.lock` as the only dependency lockfile.
2. Run `uv sync --locked` in the harness before running the app or tests.
3. Add, remove, and update dependencies only through `uv add`, `uv remove`, or `pyproject.toml` edits followed by `uv lock`.
4. Pin or lock `google-adk`, `a2ui-agent-sdk`, `litellm`, and related packages through `uv.lock`.
5. Run dependency scanning in the harness if available.
6. Rotate secrets if dependency compromise is suspected.
7. Avoid logging model provider headers.
8. Treat any `requirements.txt`, `poetry.lock`, `Pipfile.lock`, or `environment.yml` in the generated repo as a build defect unless added only as a generated export artifact outside the normal harness path.

---

## 19. Demo scenarios

### 19.1 Simple direct route scenarios

| User request | Expected route |
|---|---|
| "Summarize the internal notes for ABC Manufacturing." | Internal Knowledge Agent |
| "Give me a quick overview of the manufacturing industry." | Industry Research Agent |
| "What product opportunities should I consider for a café business?" | Product Opportunity Agent |

Acceptance condition:

- No approval UI is shown.
- SLM was still called.
- Merged confidence is at least `0.85`.
- Exactly one specialist was called.

### 19.2 Complex plan approval scenarios

| User request | Expected behavior |
|---|---|
| "Prepare me for tomorrow's meeting with ABC Manufacturing." | Generate editable A2UI plan before specialist calls. |
| "Research this prospect and give me risks, opportunities, and talking points." | Generate multi-agent research plan. |
| "Compare this customer's relationship history with current industry risks and suggest meeting priorities." | Generate mixed sequential and fan-out/fan-in plan. |

Acceptance condition:

- No specialist agent is called until approval is captured through A2UI.
- Approved plan is immutable.
- Execution graph is created from the approved plan.

### 19.3 A2UI event routing scenario

1. Product Opportunity Agent renders a product recommendation card with buttons.
2. User clicks "Show more detail."
3. Client emits A2UI `userAction` with the original `surfaceId`.
4. Orchestrator maps the `surfaceId` to Product Opportunity Agent.
5. Event is routed directly to Product Opportunity Agent.
6. Orchestrator does not ask an LLM where to route the event.

Acceptance condition:

- Routing is deterministic and traceable.

### 19.4 SLM disagreement scenario

Example request:

> Pull together what I need before seeing ABC Manufacturing tomorrow.

Potential SLM output:

```json
{
  "intent": "internal_knowledge",
  "confidence": 0.62
}
```

Expected enhanced classification:

```json
{
  "intents": ["meeting_prep", "relationship_summary", "internal_knowledge", "industry_research"],
  "complexity": "complex",
  "required_agents": ["relationship_summary", "internal_knowledge", "industry_research", "synthesis"]
}
```

Expected behavior:

- Confidence-weighted merge does not direct-route.
- Plan approval is required.

---

## 20. Success metrics

| Metric | Target for demo |
|---|---|
| Simple-route accuracy | 90%+ on demo golden set |
| Complex-plan approval enforcement | 100% of complex tasks require A2UI approval |
| Specialist pre-approval calls | 0 for complex tasks |
| A2UI validation | 100% of emitted A2UI payloads validated |
| Surface event routing | 100% deterministic by `surfaceId` |
| Dynamic registry reload | Add/remove agent without orchestrator restart |
| Graph creation | Approved plans generate valid ADK graph workflow |
| Fan-out/fan-in | At least one demo journey uses parallel specialist execution |
| Renderer loop | Approval, rejection, edit, and event routing work through UI events |
| Secret handling | No secrets in logs, code, test fixtures, or A2UI payloads |

---

## 21. MVP acceptance criteria

The MVP is complete when all of the following are true:

1. The repository is initialized and managed with `uv`.
2. `pyproject.toml`, `.python-version`, and `uv.lock` are present and committed.
3. The harness can run `uv sync --locked` successfully.
4. All application and test commands are executed with `uv run`.
5. No `requirements.txt`, `poetry.lock`, `Pipfile.lock`, `environment.yml`, or committed `.venv` is present.
6. A local Python orchestrator starts successfully with LiteLLM-compatible model configuration.
7. The app fails fast if `OPENROUTER_API_KEY` or `LLM_MODEL` is missing.
8. Secret values are never logged.
9. Every user request calls the mocked SLM intent tool.
10. The mocked SLM returns intent plus confidence.
11. The orchestrator performs LLM-enhanced intent evaluation.
12. SLM and LLM results are merged using a confidence-weighted strategy.
13. Single-agent simple requests above `0.85` confidence route directly.
14. Complex requests generate a plan before any specialist call.
15. The plan is rendered as an editable A2UI workflow canvas.
16. Approval is captured through A2UI `userAction`, not text.
17. The approved plan becomes immutable.
18. The orchestrator builds an ADK graph workflow from the approved plan.
19. The graph can execute sequential and fan-out/fan-in flows.
20. Specialist agents can generate their own A2UI.
21. The orchestrator validates but does not alter downstream A2UI.
22. A2UI surface ownership is stored.
23. A2UI events are routed directly to owning agents.
24. Agent registry is loaded from Python config.
25. Agent registry can reload dynamically.
26. At least one local specialist is wrapped behind a remote-A2A-compatible interface.
27. The basic renderer can display approval UI and emit A2UI events.
28. Logs capture routing, approval, graph execution, and event routing.
29. Unit tests cover intent merge, routing, planning, approval events, surface routing, and graph building.

---
## 22. Recommended MVP build order

1. Initialize the repository with `uv init --package adk-a2ui-orchestrator-demo`.
2. Add `.python-version`, `pyproject.toml`, `uv.lock`, `.env.example`, and `.gitignore`.
3. Add runtime and development dependencies with `uv add` and `uv add --dev`.
4. Run `uv lock` and verify `uv sync --locked`.
5. Define Pydantic contracts for intent, routing, plan, agent descriptors, approval events, graph steps, and A2UI events.
6. Implement settings and runtime-secret loading.
7. Build the mocked SLM intent tool.
8. Build static local specialist agents.
9. Build the Python config-based agent registry.
10. Implement registry reload.
11. Implement the LLM-enhanced classifier.
12. Implement the confidence-weighted merge.
13. Implement direct routing.
14. Implement complex-request planning.
15. Implement A2UI approval canvas generation.
16. Implement A2UI validation and one repair retry.
17. Implement approval event parser.
18. Implement immutable approval state.
19. Implement ADK graph builder.
20. Implement sequential graph execution.
21. Add fan-out/fan-in graph execution.
22. Add downstream A2UI pass-through.
23. Add surface ownership registry.
24. Add deterministic A2UI event routing.
25. Build basic renderer.
26. Add golden tests.
27. Add harness run instructions.

---
## 23. Testing requirements

### 23.1 Unit tests

Required tests:

- `test_slm_called_for_every_request`
- `test_simple_route_above_threshold`
- `test_low_confidence_simple_intent_routes_to_plan`
- `test_complex_request_requires_approval`
- `test_no_specialist_called_before_approval`
- `test_approved_plan_is_immutable`
- `test_plan_edit_event_updates_unapproved_plan`
- `test_a2ui_approval_event_parsing`
- `test_a2ui_validation_success`
- `test_a2ui_validation_failure_repair`
- `test_surface_owner_set`
- `test_user_action_routes_to_surface_owner`
- `test_registry_reload_add_agent`
- `test_registry_reload_remove_agent`
- `test_graph_builder_sequential`
- `test_graph_builder_fan_out_fan_in`
- `test_missing_openrouter_key_fails_fast`
- `test_secret_not_logged`
- `test_uv_lockfile_present`
- `test_pyproject_present`
- `test_no_disallowed_dependency_manager_files`

### 23.2 Golden scenario tests

Create a small golden dataset:

```json
[
  {
    "input": "Summarize the internal notes for ABC Manufacturing.",
    "expected_path": "direct",
    "expected_agent": "internal_knowledge"
  },
  {
    "input": "Prepare me for tomorrow's meeting with ABC Manufacturing.",
    "expected_path": "plan_required",
    "expected_agents": ["relationship_summary", "internal_knowledge", "industry_research", "synthesis"]
  },
  {
    "input": "Research this prospect and give me risks, opportunities, and talking points.",
    "expected_path": "plan_required",
    "expected_agents": ["web_search", "industry_research", "product_opportunity", "credit_risk", "synthesis"]
  }
]
```

---

## 24. Agentic harness handoff instructions

The implementation harness should be given this PRD plus the following operator instructions.

### 24.1 Required harness environment

The harness must install dependencies and run commands exclusively through `uv`. It must run `uv sync --locked` before tests or application startup.

The harness must run the implementation with:

```bash
OPENROUTER_API_KEY=<secret>
LLM_MODEL=openrouter/<provider>/<model>
```

Optional:

```bash
OPENROUTER_API_BASE=https://openrouter.ai/api/v1
OR_APP_NAME=adk-a2ui-orchestrator-demo
OR_SITE_URL=http://localhost:3000
```

### 24.2 Harness constraints

The harness must:

1. Treat `OPENROUTER_API_KEY` as a secret.
2. Never echo it into logs or generated files.
3. Ensure `.env` is ignored.
4. Prefer runtime secret injection over writing `.env`.
5. Use a dedicated demo key with a small spend limit.
6. Install pinned dependencies with `uv sync --locked`.
7. Run application and test commands through `uv run`.
8. Never use `pip install -r requirements.txt`, Poetry, Pipenv, Conda, or global Python package installation for this repository.
9. Fail the build if `uv.lock` is missing or stale.
10. Run tests before declaring completion.
11. Produce an implementation report summarizing:
   - What was built.
   - How to run it.
   - What tests pass.
   - Known limitations.
   - Any assumptions made.

### 24.3 Harness success condition

The harness can declare success when:

1. `uv sync --locked` succeeds from a clean checkout.
2. `uv run pytest` succeeds.
3. The local orchestrator starts.
4. A simple request routes directly.
5. A complex request renders an A2UI approval plan.
6. Approval is captured through a UI event.
7. The approved graph executes.
8. Final output is produced.
9. No secret is present in logs, code, or generated A2UI.
10. The test suite passes.

---
## 25. Open assumptions

This PRD assumes:

1. Demo data is synthetic.
2. The web search agent can use mocked search first for repeatability.
3. Specialist agents return structured outputs suitable for synthesis.
4. A synthesis agent is required for multi-agent outputs.
5. Compliance/policy review is advisory only and does not make regulated decisions.
6. The first renderer can be minimal as long as it proves A2UI approval and event handling.
7. Production remote agents will eventually use A2A.
8. Only one or two local remote-compatible wrappers are needed for the initial demo.
9. The implementation team will choose exact model names through `LLM_MODEL`.
10. The orchestrator is allowed to ask clarification questions only when route or plan quality would otherwise be unsafe or too ambiguous.
11. The harness has access to a `uv` binary or can install `uv` before repository setup begins.

---

## 26. References consulted

- ADK graph workflows: https://adk.dev/graphs/
- ADK graph routes: https://adk.dev/graphs/routes/
- ADK human input for graph workflows: https://adk.dev/graphs/human-input/
- ADK dynamic workflows: https://adk.dev/graphs/dynamic/
- ADK LiteLLM connector: https://adk.dev/agents/models/litellm/
- A2UI ADK integration: https://adk.dev/integrations/a2ui/
- A2UI GitHub repository: https://github.com/google/A2UI
- A2UI ADK orchestrator sample: https://github.com/google/A2UI/tree/main/samples/agent/adk/orchestrator
- LiteLLM OpenRouter provider docs: https://docs.litellm.ai/docs/providers/openrouter
- uv documentation: https://docs.astral.sh/uv/
- uv working on projects: https://docs.astral.sh/uv/guides/projects/
- uv project configuration: https://docs.astral.sh/uv/concepts/projects/config/
- OpenRouter: https://openrouter.ai/

---

## 27. Appendix — concise implementation checklist

```text
[ ] Initialize repository with uv
[ ] Commit pyproject.toml
[ ] Commit uv.lock
[ ] Commit .python-version
[ ] Ensure .venv is ignored
[ ] Ensure requirements.txt, poetry.lock, Pipfile.lock, and environment.yml are absent
[ ] Verify uv sync --locked succeeds
[ ] Run all app/test commands with uv run
[ ] Load runtime settings from env only
[ ] Fail fast when OPENROUTER_API_KEY or LLM_MODEL is missing
[ ] Implement MockSlmIntentClient
[ ] Implement LLM-enhanced classifier
[ ] Implement confidence-weighted merge
[ ] Implement direct route path
[ ] Implement complex plan path
[ ] Generate A2UI approval canvas
[ ] Parse approve/reject/edit A2UI userAction events
[ ] Freeze approved plan
[ ] Build ADK graph from approved plan
[ ] Execute sequential graph
[ ] Execute fan-out/fan-in graph
[ ] Create local LlmAgent specialists
[ ] Add remote-A2A-compatible wrapper for 1–2 local agents
[ ] Add dynamic Python config agent registry
[ ] Add registry reload
[ ] Validate A2UI payloads
[ ] Preserve downstream A2UI payloads
[ ] Track A2UI surface ownership
[ ] Route userAction events by surfaceId
[ ] Add basic renderer
[ ] Add golden tests
[ ] Add harness README
[ ] Ensure no secrets are logged or committed
```
