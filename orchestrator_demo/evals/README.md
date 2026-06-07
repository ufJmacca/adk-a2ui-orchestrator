# ADK Evaluation Compatibility

This directory is reserved for ADK evalsets and User Simulation artifacts. The
first compatibility slice keeps production behavior unchanged and records the
locked evaluator surface before adding checked-in eval cases.

## Locked Version

The repository remains pinned to `google-adk==2.1.0` in `pyproject.toml` and
`uv.lock`. No additional eval extras or dependencies are required for the
current compatibility checks.

Confirmed import command:

```bash
ORCHESTRATOR_DEMO_DETERMINISTIC_MODEL=1 \
  uv run --locked python -c "from google.adk.evaluation.agent_evaluator import AgentEvaluator; from google.adk.evaluation.eval_set import EvalSet; from google.adk.evaluation.eval_config import EvalConfig"
```

## Programmatic API

Under `google-adk==2.1.0`, these evaluator symbols import successfully:

- `AgentEvaluator` from `google.adk.evaluation.agent_evaluator`
- `EvalSet` from `google.adk.evaluation.eval_set`
- `EvalConfig` from `google.adk.evaluation.eval_config`

The confirmed programmatic entry points are:

```python
AgentEvaluator.evaluate(
    agent_module,
    eval_dataset_file_path_or_dir,
    num_runs=2,
    agent_name=None,
    initial_session_file=None,
    print_detailed_results=True,
)

AgentEvaluator.evaluate_eval_set(
    agent_module,
    eval_set,
    criteria=None,
    eval_config=None,
    num_runs=2,
    agent_name=None,
    print_detailed_results=True,
)
```

`EvalSet` accepts `eval_set_id`, optional `name`, optional `description`,
`eval_cases`, and `creation_timestamp`. `EvalConfig` accepts `criteria`,
`customMetrics`, and `userSimulatorConfig`.

The focused compatibility test is:

```bash
ORCHESTRATOR_DEMO_DETERMINISTIC_MODEL=1 \
  uv run --locked pytest tests/test_adk_evalsets.py -q
```

If a future locked ADK build removes or reshapes these imports, the test skips
with a locked-version reason that names the missing module or symbol.

## CLI Compatibility

CLI compatibility finding for this slice:

`uv run --locked adk eval --help` confirms that ADK 2.1.0 exposes this command
shape. Once the checked-in evalset and config exist, use the loader-compatible
orchestrator directory with the deterministic model enabled:

```bash
ORCHESTRATOR_DEMO_DETERMINISTIC_MODEL=1 \
  uv run --locked adk eval \
  orchestrator_demo/orchestrator \
  orchestrator_demo/evals/basic_evalset.evalset.json \
  --config_file_path orchestrator_demo/evals/basic_eval_config.json \
  --print_detailed_results
```

The CLI help describes `EVAL_SET_FILE_PATH_OR_ID` as one or more eval set file
paths or eval set IDs, with optional `:eval_case_id` suffix filtering. The
checked-in `basic_evalset.evalset.json` and config are added in a later slice,
so this slice confirms the CLI accepts an eval set file path argument but does
not yet run a repository evalset fixture.

## ADK Web Capture

Maintainers can use ADK Web to capture deterministic sessions, review and edit
cases, then promote cleaned synthetic cases into the fixed evalset:

```bash
uv run adk web orchestrator_demo --host 0.0.0.0 --port 8000
```

## User Simulation

User Simulation is an opt-in dynamic evaluation lane. The checked-in files in
this directory define scenario inputs and configuration, but generated User
Simulation evalsets are not treated as checked-in source fixtures by default.
Keep generated cases under `orchestrator_demo/evals/generated/` or use the
local ADK eval set store, then promote only cleaned, synthetic, deterministic
fixtures when they are intended to become source-controlled regression cases.

The repository lock intentionally keeps `google-adk==2.1.0` without the ADK eval
extra. User Simulation generation and dynamic `adk eval` execution require an
out-of-band environment with the same ADK version plus the eval extra installed;
do not run these dynamic commands as `uv run --locked adk ...` unless
`google-adk[eval]==2.1.0` has first been added to and locked in the project.

Example out-of-band setup:

```bash
uv venv .venv-adk-eval
. .venv-adk-eval/bin/activate
uv pip install -e . "google-adk[eval]==2.1.0"
```

Create a scenario-backed evalset:

```bash
adk eval_set create \
  orchestrator_demo/orchestrator \
  orchestrator_user_sim
```

Add eval cases from the checked-in scenario pack:

```bash
adk eval_set add_eval_case \
  orchestrator_demo/orchestrator \
  orchestrator_user_sim \
  --scenarios_file orchestrator_demo/evals/conversation_scenarios.json \
  --session_input_file orchestrator_demo/evals/session_input.json
```

Run the generated dynamic evalset:

```bash
ORCHESTRATOR_DEMO_DETERMINISTIC_MODEL=1 \
  adk eval \
  orchestrator_demo/orchestrator \
  orchestrator_user_sim \
  --config_file_path orchestrator_demo/evals/user_sim_eval_config.json \
  --print_detailed_results
```

Under the locked ADK 2.1 CLI, the loader-compatible local app path is
`orchestrator_demo/orchestrator`. Local `eval_set` commands store the generated
evalset at `orchestrator_demo/orchestrator/orchestrator_user_sim.evalset.json`;
that generated path is git-ignored by default. Promote only reviewed, cleaned,
synthetic, deterministic fixtures into `orchestrator_demo/evals/` when they are
intended to become checked-in regression cases.

These commands may require Google Cloud and Vertex credentials depending on the
ADK command path being used. Before enabling scheduled or blocking runs, define
Google Cloud/Vertex project ownership, API enablement, quota, budgets, possible
costs, and cost review ownership. Do not commit credentials, service-account
files, tokens, `.env` values, generated customer data, or generated eval results
that have not been reviewed for synthetic-only content.

`user_sim_eval_config.json` intentionally uses User Simulation metrics instead
of fixed expected-response criteria:

- `hallucinations_v1`
- `safety_v1`
- `multi_turn_task_success_v1`
- `multi_turn_trajectory_quality_v1`
- `multi_turn_tool_use_quality_v1`
- `per_turn_user_simulator_quality_v1`

The repository is locked to `google-adk==2.1.0`. Unsupported or unavailable
User Simulation metrics in ADK 2.1 are non-blocking until validated against the
locked ADK runtime and the chosen Google Cloud/Vertex project.
