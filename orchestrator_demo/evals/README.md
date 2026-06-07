# ADK Evaluation Compatibility

This directory contains ADK evalsets and User Simulation artifacts. The fixed
evalset keeps production behavior unchanged while exercising deterministic
tool routing, plan approval, rejection, and safe final-response behavior.

## Locked Version

The repository remains pinned to ADK 2.1.0 with `google-adk[eval]==2.1.0` in
`pyproject.toml` and `uv.lock`, so the locked environment includes the optional
eval dependencies required by the ADK CLI.

Confirmed import command:

```bash
ORCHESTRATOR_DEMO_DETERMINISTIC_MODEL=1 \
ORCHESTRATOR_DEMO_ADK_EVAL_MODE=1 \
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

## Local Fixed Eval Workflow

Synchronize the locked environment before running fixed evals:

```bash
uv sync --locked
```

The focused pytest wrapper for the checked-in fixed evalset is:

```bash
ORCHESTRATOR_DEMO_DETERMINISTIC_MODEL=1 \
ORCHESTRATOR_DEMO_ADK_EVAL_MODE=1 \
  uv run --locked pytest tests/test_adk_evalsets.py -m adk_eval_runner
```

If a future locked ADK build removes or reshapes these imports, the test skips
with a locked-version reason that names the missing module or symbol.

Fixed evals do not require `OPENROUTER_API_KEY` in deterministic mode. Eval
commands, pytest failures, ADK CLI output, and captured fixtures must not log
secrets, `.env` values, real customer data, credentials, or production banking
decisions.

## CI Eval Lane

GitHub Actions keeps the `quality` job focused on deterministic software gates:
lock check, locked sync, Ruff, Mypy, and the full pytest suite.

The separate `eval-basic` job runs only the fixed eval pytest wrapper with
`ORCHESTRATOR_DEMO_DETERMINISTIC_MODEL=1` and
`ORCHESTRATOR_DEMO_ADK_EVAL_MODE=1`. The default pytest configuration
deselects the `adk_eval_runner` marker so these metric-bearing runner tests do
not execute in the required `quality` job:

```bash
uv run --locked pytest tests/test_adk_evalsets.py -m adk_eval_runner -ra
```

The `-ra` flag keeps explicit skip reasons visible when the locked ADK eval API
is incompatible, and the pytest wrapper emits case-level diagnostics for eval
failures. CI also uploads the `eval-basic` log as a GitHub Actions artifact
named `eval-basic-results`.

The `eval-basic` lane is intentionally separate and non-required until
maintainers accept the baseline flake rate.

## CLI Compatibility

CLI compatibility finding for this slice:

`uv run --locked adk eval --help` confirms that ADK 2.1.0 exposes this command
shape. Use the loader-compatible orchestrator directory with both deterministic
eval flags enabled. The supported narrow CLI target is
`orchestrator_demo/orchestrator`; package-root `orchestrator_demo` eval loading
is not the documented path for this repository.

```bash
ORCHESTRATOR_DEMO_DETERMINISTIC_MODEL=1 \
ORCHESTRATOR_DEMO_ADK_EVAL_MODE=1 \
  uv run --locked adk eval \
  orchestrator_demo/orchestrator \
  orchestrator_demo/evals/basic_evalset.evalset.json \
  --config_file_path orchestrator_demo/evals/basic_eval_config.json \
  --print_detailed_results
```

The CLI help describes `EVAL_SET_FILE_PATH_OR_ID` as one or more eval set file
paths or eval set IDs, with optional `:eval_case_id` suffix filtering. The
checked-in `basic_evalset.evalset.json` and config run under this command in
the locked environment.

## ADK Web Capture

Maintainers can use ADK Web to capture deterministic sessions, review and edit
cases, then promote cleaned synthetic cases into the fixed evalset:

```bash
uv run adk web orchestrator_demo --host 0.0.0.0 --port 8000
```

Captured sessions must be cleaned, synthetic, and manually promoted into
`basic_evalset.evalset.json`. Before promotion, remove exploratory turns,
transient IDs that are not part of the assertion, real names, secrets, `.env`
values, credentials, and any regulated decision language.

## User Simulation

User Simulation is an opt-in dynamic evaluation lane. The checked-in files in
this directory define scenario inputs and configuration, but generated User
Simulation evalsets are not treated as checked-in source fixtures by default.
Keep generated cases under `orchestrator_demo/evals/generated/` or use the
local ADK eval set store, then promote only cleaned, synthetic, deterministic
fixtures when they are intended to become source-controlled regression cases.

The repository is locked to `google-adk[eval]==2.1.0`, with underlying
`google-adk==2.1.0` ADK APIs. Dynamic User Simulation can use the locked eval
extra, but live simulator or judge paths may still require Google Cloud and
Vertex credentials.

Create a scenario-backed evalset:

```bash
uv run --locked adk eval_set create \
  orchestrator_demo/orchestrator \
  orchestrator_user_sim
```

Add eval cases from the checked-in scenario pack:

```bash
uv run --locked adk eval_set add_eval_case \
  orchestrator_demo/orchestrator \
  orchestrator_user_sim \
  --scenarios_file orchestrator_demo/evals/conversation_scenarios.json \
  --session_input_file orchestrator_demo/evals/session_input.json
```

Run the generated dynamic evalset:

```bash
ORCHESTRATOR_DEMO_DETERMINISTIC_MODEL=1 \
ORCHESTRATOR_DEMO_ADK_EVAL_MODE=1 \
  uv run --locked adk eval \
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

## Manual User Simulation Workflow

The opt-in workflow for dynamic User Simulation is
`.github/workflows/eval-user-sim.yml`. It is triggered with
`workflow_dispatch` and does not run on pull requests, push events, or every PR
in the first release. This keeps `eval-user-sim` separate from the required
`quality` job and the deterministic `eval-basic` lane.

Before dispatching the workflow, configure these GitHub repository secrets and
variables without committing their values:

- `GOOGLE_CLOUD_PROJECT`: Google Cloud project that owns the eval run.
- `GOOGLE_APPLICATION_CREDENTIALS_JSON`: service-account JSON for the runner.
- `GOOGLE_CLOUD_LOCATION`: optional repository variable for the Vertex region;
  the workflow defaults to `us-central1` when it is not set.

The workflow writes `GOOGLE_APPLICATION_CREDENTIALS_JSON` to a temporary runner
file and exports only `GOOGLE_APPLICATION_CREDENTIALS`; it does not print the
credential value. The app still runs with
`ORCHESTRATOR_DEMO_DETERMINISTIC_MODEL=1` and
`ORCHESTRATOR_DEMO_ADK_EVAL_MODE=1`, while the User Simulation model and
`max_allowed_invocations` are controlled by `user_sim_eval_config.json`.

`max_allowed_invocations` is capped at 12 in the checked-in config and validated
by the workflow before the dynamic eval runs. The workflow uploads an
`eval-user-sim-results` artifact containing the eval log and, when ADK writes
one, the generated scenario-backed evalset. Treat those artifacts as review
outputs: do not promote generated content into source control until it has been
checked for synthetic-only content, secrets, and regulated decision language.

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
