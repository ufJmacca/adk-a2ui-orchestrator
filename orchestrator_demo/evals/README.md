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
  uv run --locked pytest tests/test_adk_evalsets.py
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
`ORCHESTRATOR_DEMO_ADK_EVAL_MODE=1`:

```bash
uv run --locked pytest tests/test_adk_evalsets.py -ra
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
