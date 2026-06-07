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
