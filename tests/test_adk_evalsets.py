from __future__ import annotations

import importlib
import inspect
import json
import sys
from collections.abc import Mapping
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
import yaml

ADK_EVAL_ENV_VARS = {
    "ORCHESTRATOR_DEMO_DETERMINISTIC_MODEL",
    "ORCHESTRATOR_DEMO_ADK_EVAL_MODE",
}


def _import_adk_eval_symbol(module_name: str, symbol_name: str) -> Any:
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        pytest.skip(
            "google-adk==2.1.0 eval API is unavailable in this locked "
            f"environment: cannot import {module_name}: {exc}"
        )

    try:
        return getattr(module, symbol_name)
    except AttributeError as exc:
        pytest.skip(
            "google-adk==2.1.0 eval API shape is incompatible in this locked "
            f"environment: {module_name}.{symbol_name} is missing: {exc}"
        )


@pytest.mark.asyncio
async def test_programmatic_fixed_evalset_runner_executes_checked_in_evalset(
    monkeypatch: pytest.MonkeyPatch,
    repository_root: Path,
) -> None:
    # Arrange
    monkeypatch.setenv("ORCHESTRATOR_DEMO_DETERMINISTIC_MODEL", "1")
    monkeypatch.setenv("ORCHESTRATOR_DEMO_ADK_EVAL_MODE", "1")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    evalset_path = (
        repository_root
        / "orchestrator_demo"
        / "evals"
        / "basic_evalset.evalset.json"
    )
    config_path = (
        repository_root
        / "orchestrator_demo"
        / "evals"
        / "basic_eval_config.json"
    )
    agent_module = "orchestrator_demo.orchestrator.agent"
    AgentEvaluator = _import_adk_eval_symbol(
        "google.adk.evaluation.agent_evaluator",
        "AgentEvaluator",
    )
    EvalSet = _import_adk_eval_symbol("google.adk.evaluation.eval_set", "EvalSet")
    EvalConfig = _import_adk_eval_symbol(
        "google.adk.evaluation.eval_config",
        "EvalConfig",
    )
    evaluate_eval_set = getattr(AgentEvaluator, "evaluate_eval_set", None)
    if evaluate_eval_set is None:
        pytest.skip(
            "google-adk==2.1.0 eval API shape is incompatible in this locked "
            "environment: AgentEvaluator.evaluate_eval_set is missing."
        )
    _skip_if_evaluate_eval_set_signature_is_incompatible(evaluate_eval_set)
    eval_set = _load_adk_model_from_json(EvalSet, evalset_path, "EvalSet")
    eval_config = _load_adk_model_from_json(EvalConfig, config_path, "EvalConfig")
    _clear_agent_module_cache(agent_module)

    # Act
    await _evaluate_fixed_eval_set_with_diagnostics(
        evaluate_eval_set=evaluate_eval_set,
        AgentEvaluator=AgentEvaluator,
        agent_module=agent_module,
        eval_set=eval_set,
        eval_config=eval_config,
    )

    # Assert
    assert eval_set.eval_set_id == "orchestrator_basic_regression"
    assert eval_config.criteria


@pytest.mark.asyncio
async def test_programmatic_fixed_evalset_runner_failure_names_case_metric_and_trajectories(
    monkeypatch: pytest.MonkeyPatch,
    repository_root: Path,
) -> None:
    # Arrange
    monkeypatch.setenv("ORCHESTRATOR_DEMO_DETERMINISTIC_MODEL", "1")
    monkeypatch.setenv("ORCHESTRATOR_DEMO_ADK_EVAL_MODE", "1")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    evalset_path = (
        repository_root
        / "orchestrator_demo"
        / "evals"
        / "basic_evalset.evalset.json"
    )
    config_path = (
        repository_root
        / "orchestrator_demo"
        / "evals"
        / "basic_eval_config.json"
    )
    agent_module = "orchestrator_demo.orchestrator.agent"
    AgentEvaluator = _import_adk_eval_symbol(
        "google.adk.evaluation.agent_evaluator",
        "AgentEvaluator",
    )
    EvalSet = _import_adk_eval_symbol("google.adk.evaluation.eval_set", "EvalSet")
    EvalConfig = _import_adk_eval_symbol(
        "google.adk.evaluation.eval_config",
        "EvalConfig",
    )
    evaluate_eval_set = getattr(AgentEvaluator, "evaluate_eval_set", None)
    if evaluate_eval_set is None:
        pytest.skip(
            "google-adk==2.1.0 eval API shape is incompatible in this locked "
            "environment: AgentEvaluator.evaluate_eval_set is missing."
        )
    _skip_if_evaluate_eval_set_signature_is_incompatible(evaluate_eval_set)
    eval_set = _load_adk_model_from_json(EvalSet, evalset_path, "EvalSet")
    eval_config = _load_adk_model_from_json(EvalConfig, config_path, "EvalConfig")
    broken_eval_set = eval_set.model_copy(deep=True)
    direct_case = broken_eval_set.eval_cases[0]
    assert direct_case.eval_id == "direct_internal_notes_summary"
    assert direct_case.conversation is not None
    direct_tool_use = direct_case.conversation[0].intermediate_data.tool_uses[0]
    direct_tool_use.name = "not_the_orchestrator_tool"
    _clear_agent_module_cache(agent_module)

    # Act
    with pytest.raises(AssertionError) as exc_info:
        await _evaluate_fixed_eval_set_with_diagnostics(
            evaluate_eval_set=evaluate_eval_set,
            AgentEvaluator=AgentEvaluator,
            agent_module=agent_module,
            eval_set=broken_eval_set,
            eval_config=eval_config,
        )

    # Assert
    failure_message = str(exc_info.value)
    assert "eval case: direct_internal_notes_summary" in failure_message
    assert "tool_trajectory_avg_score" in failure_message
    assert "expected trajectory with args" in failure_message
    assert "not_the_orchestrator_tool" in failure_message
    assert "Riverbend Cafe" in failure_message
    assert "actual trajectory with args" in failure_message
    assert "submit_orchestrator_request" in failure_message


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "runtime_error_type",
    [AttributeError, TypeError, RuntimeError, ValueError],
)
async def test_programmatic_fixed_evalset_runner_runtime_agent_errors_fail_with_diagnostics(
    monkeypatch: pytest.MonkeyPatch,
    repository_root: Path,
    runtime_error_type: type[Exception],
) -> None:
    # Arrange
    monkeypatch.setenv("ORCHESTRATOR_DEMO_DETERMINISTIC_MODEL", "1")
    monkeypatch.setenv("ORCHESTRATOR_DEMO_ADK_EVAL_MODE", "1")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    evalset_path = (
        repository_root
        / "orchestrator_demo"
        / "evals"
        / "basic_evalset.evalset.json"
    )
    config_path = (
        repository_root
        / "orchestrator_demo"
        / "evals"
        / "basic_eval_config.json"
    )
    EvalSet = _import_adk_eval_symbol("google.adk.evaluation.eval_set", "EvalSet")
    EvalConfig = _import_adk_eval_symbol(
        "google.adk.evaluation.eval_config",
        "EvalConfig",
    )
    eval_set = _load_adk_model_from_json(EvalSet, evalset_path, "EvalSet")
    eval_config = _load_adk_model_from_json(EvalConfig, config_path, "EvalConfig")

    async def raise_runtime_agent_error(**_: Any) -> None:
        raise runtime_error_type("application runtime regression")

    def fail_if_runtime_error_is_skipped(reason: str) -> None:
        raise AssertionError(f"runtime agent errors must fail, not skip: {reason}")

    monkeypatch.setattr(pytest, "skip", fail_if_runtime_error_is_skipped)

    # Act
    with pytest.raises(AssertionError) as exc_info:
        await _evaluate_fixed_eval_set_with_diagnostics(
            evaluate_eval_set=raise_runtime_agent_error,
            AgentEvaluator=object(),
            agent_module="orchestrator_demo.orchestrator.agent",
            eval_set=eval_set,
            eval_config=eval_config,
        )

    # Assert
    failure_message = str(exc_info.value)
    assert (
        "fixed eval runner failed during agent evaluation: "
        f"{runtime_error_type.__name__}: application runtime regression"
    ) in failure_message
    assert "ADK fixed eval failure diagnostics:" in failure_message
    assert "eval case: direct_internal_notes_summary" in failure_message
    assert "expected trajectory with args" in failure_message
    assert "submit_orchestrator_request" in failure_message
    assert "Riverbend Cafe" in failure_message
    assert "actual trajectory with args" in failure_message


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "eval_error",
    [None, AssertionError("eval mismatch"), RuntimeError("runtime regression")],
)
async def test_fixed_eval_runner_clears_agent_module_cache_after_eval_paths(
    eval_error: Exception | None,
) -> None:
    # Arrange
    agent_module = "orchestrator_demo.orchestrator.agent"
    parent_module = "orchestrator_demo.orchestrator"
    cached_agent_module = ModuleType(agent_module)
    cached_agent_module._ROOT_AGENT = object()  # type: ignore[attr-defined]
    cached_agent_module._APP = object()  # type: ignore[attr-defined]
    sys.modules[agent_module] = cached_agent_module
    sys.modules[parent_module] = ModuleType(parent_module)
    eval_set = type("EvalSetStub", (), {"eval_cases": []})()

    async def evaluate_eval_set(**_: Any) -> None:
        sys.modules[agent_module] = cached_agent_module
        if eval_error is not None:
            raise eval_error

    # Act
    if eval_error is None:
        await _evaluate_fixed_eval_set_with_diagnostics(
            evaluate_eval_set=evaluate_eval_set,
            AgentEvaluator=object(),
            agent_module=agent_module,
            eval_set=eval_set,
            eval_config=object(),
        )
    else:
        with pytest.raises(AssertionError):
            await _evaluate_fixed_eval_set_with_diagnostics(
                evaluate_eval_set=evaluate_eval_set,
                AgentEvaluator=object(),
                agent_module=agent_module,
                eval_set=eval_set,
                eval_config=object(),
            )

    # Assert
    assert cached_agent_module._ROOT_AGENT is None  # type: ignore[attr-defined]
    assert cached_agent_module._APP is None  # type: ignore[attr-defined]
    assert agent_module not in sys.modules
    assert parent_module not in sys.modules


def test_ci_quality_job_keeps_existing_required_gates(repository_root: Path) -> None:
    # Arrange
    workflow = _load_ci_workflow(repository_root)

    # Act
    quality_job = workflow["jobs"].get("quality")
    quality_run_commands = _job_run_commands(quality_job)
    eval_env_locations = _job_env_var_locations(quality_job, ADK_EVAL_ENV_VARS)

    # Assert
    assert quality_job is not None
    assert quality_run_commands == [
        "uv lock --check",
        "uv sync --locked",
        "uv run --locked ruff check --output-format=github .",
        "uv run --locked mypy orchestrator_demo",
        "uv run --locked pytest",
    ]
    assert eval_env_locations == []


def test_ci_eval_basic_job_runs_fixed_eval_wrapper_separately(
    repository_root: Path,
) -> None:
    # Arrange
    workflow = _load_ci_workflow(repository_root)

    # Act
    jobs = workflow["jobs"]
    quality_job = jobs.get("quality")
    eval_basic_job = jobs.get("eval-basic")
    eval_step = (
        _job_step(eval_basic_job, "Run deterministic fixed ADK evals")
        if eval_basic_job is not None
        else {}
    )
    eval_env = (
        _merged_job_step_env(eval_basic_job, eval_step)
        if eval_basic_job is not None
        else {}
    )
    eval_run_lines = _script_lines(eval_step.get("run", ""))
    pytest_pipeline = _single_script_line_containing(
        eval_run_lines,
        "uv run --locked pytest",
    )

    # Assert
    assert quality_job is not None
    assert eval_basic_job is not None
    assert eval_basic_job is not quality_job
    assert eval_env["ORCHESTRATOR_DEMO_DETERMINISTIC_MODEL"] == "1"
    assert eval_env["ORCHESTRATOR_DEMO_ADK_EVAL_MODE"] == "1"
    assert "set -o pipefail" in eval_run_lines
    assert eval_run_lines.index("set -o pipefail") < eval_run_lines.index(
        pytest_pipeline
    )
    assert pytest_pipeline == (
        "uv run --locked pytest tests/test_adk_evalsets.py -ra 2>&1 "
        "| tee .ai-native/eval-basic/eval-basic.log"
    )


def test_ci_eval_basic_uploads_eval_result_summary_artifact(
    repository_root: Path,
) -> None:
    # Arrange
    workflow = _load_ci_workflow(repository_root)

    # Act
    eval_basic_job = workflow["jobs"].get("eval-basic")
    eval_step = (
        _job_step(eval_basic_job, "Run deterministic fixed ADK evals")
        if eval_basic_job is not None
        else {}
    )
    eval_run_lines = _script_lines(eval_step.get("run", ""))
    pytest_pipeline = _single_script_line_containing(eval_run_lines, " | tee ")
    artifact_steps = [
        step
        for step in eval_basic_job["steps"]
        if str(step.get("uses", "")).startswith("actions/upload-artifact@")
    ] if eval_basic_job is not None else []

    # Assert
    assert eval_basic_job is not None
    assert len(artifact_steps) == 1
    artifact_step = artifact_steps[0]
    artifact_config = artifact_step["with"]
    assert artifact_step["if"] == "${{ always() }}"
    assert artifact_config["name"] == "eval-basic-results"
    assert artifact_config["path"] == ".ai-native/eval-basic/"
    assert artifact_config["include-hidden-files"] is True
    assert _tee_output_path(pytest_pipeline).is_relative_to(
        _artifact_directory(artifact_config["path"])
    )


def test_eval_readme_documents_ci_eval_lanes_and_artifacts(
    repository_root: Path,
) -> None:
    # Arrange
    readme_path = repository_root / "orchestrator_demo" / "evals" / "README.md"

    # Act
    readme_text = readme_path.read_text(encoding="utf-8")
    normalized_readme = " ".join(readme_text.casefold().split())

    # Assert
    assert "`quality`" in readme_text
    assert "`eval-basic`" in readme_text
    assert "uv run --locked pytest tests/test_adk_evalsets.py" in readme_text
    assert "ORCHESTRATOR_DEMO_DETERMINISTIC_MODEL=1" in readme_text
    assert "ORCHESTRATOR_DEMO_ADK_EVAL_MODE=1" in readme_text
    assert "github actions artifact" in normalized_readme
    assert "non-required" in normalized_readme
    assert "baseline flake rate" in normalized_readme


def _load_ci_workflow(repository_root: Path) -> Mapping[str, Any]:
    workflow_path = repository_root / ".github" / "workflows" / "ci.yml"
    workflow = yaml.safe_load(workflow_path.read_text(encoding="utf-8"))
    assert isinstance(workflow, dict)
    return workflow


def _job_run_commands(job: Mapping[str, Any] | None) -> list[str]:
    if job is None:
        return []

    return [step["run"] for step in job["steps"] if "run" in step]


def _job_step(job: Mapping[str, Any] | None, step_name: str) -> Mapping[str, Any]:
    if job is None:
        return {}

    matching_steps = [step for step in job["steps"] if step.get("name") == step_name]
    assert len(matching_steps) == 1, (
        f"expected one workflow step named {step_name!r}; "
        f"found {[step.get('name') for step in job['steps']]}"
    )
    return matching_steps[0]


def _job_env_var_locations(
    job: Mapping[str, Any] | None,
    env_var_names: set[str],
) -> list[str]:
    if job is None:
        return []

    locations = []
    for env_var_name in sorted(env_var_names):
        if env_var_name in job.get("env", {}):
            locations.append(f"job.env.{env_var_name}")

    for index, step in enumerate(job["steps"]):
        step_label = step.get("name") or step.get("uses") or f"step[{index}]"
        for env_var_name in sorted(env_var_names):
            if env_var_name in step.get("env", {}):
                locations.append(f"{step_label}.env.{env_var_name}")

    return locations


def _merged_job_step_env(
    job: Mapping[str, Any],
    step: Mapping[str, Any],
) -> dict[str, Any]:
    env = dict(job.get("env", {}))
    env.update(step.get("env", {}))
    return env


def _script_lines(script: str) -> list[str]:
    return [line.strip() for line in script.splitlines() if line.strip()]


def _single_script_line_containing(lines: list[str], expected_text: str) -> str:
    matching_lines = [line for line in lines if expected_text in line]
    assert len(matching_lines) == 1, (
        f"expected one workflow script line containing {expected_text!r}; "
        f"found {matching_lines!r}"
    )
    return matching_lines[0]


def _tee_output_path(pipeline: str) -> Path:
    _, separator, tee_command = pipeline.partition("|")
    assert separator == "|", f"expected pytest output to be piped through tee: {pipeline}"

    tee_tokens = tee_command.strip().split()
    assert tee_tokens == ["tee", ".ai-native/eval-basic/eval-basic.log"]
    return Path(tee_tokens[1])


def _artifact_directory(artifact_path: str) -> Path:
    normalized_path = Path(artifact_path)
    assert normalized_path == Path(".ai-native/eval-basic")
    return normalized_path


@pytest.mark.parametrize(
    ("function_name", "missing_module"),
    [
        ("raise_wrapped_missing_extra_with_cause", "pandas"),
        ("raise_wrapped_missing_extra_with_context", "rouge_score"),
    ],
)
def test_adk_eval_extra_import_failure_detection_unwraps_missing_modules(
    function_name: str,
    missing_module: str,
) -> None:
    # Arrange
    adk_eval_globals: dict[str, Any] = {
        "__name__": "google.adk.evaluation.agent_evaluator",
        "ModuleNotFoundError": ModuleNotFoundError,
    }
    exec(
        """
def raise_wrapped_missing_extra_with_cause():
    try:
        raise ModuleNotFoundError(
            "No module named 'pandas'",
            name="pandas",
        )
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("Missing ADK eval dependencies") from exc


def raise_wrapped_missing_extra_with_context():
    try:
        raise ModuleNotFoundError(
            "No module named 'rouge_score'",
            name="rouge_score",
        )
    except ModuleNotFoundError:
        raise ModuleNotFoundError("Missing ADK eval dependencies")
""",
        adk_eval_globals,
    )
    raise_wrapped_missing_extra = adk_eval_globals[function_name]

    # Act
    with pytest.raises(ModuleNotFoundError) as exc_info:
        raise_wrapped_missing_extra()

    # Assert
    assert exc_info.value.name is None
    assert _exception_missing_module_names(exc_info.value) == {missing_module}
    assert _is_adk_eval_or_extra_import_failure(exc_info.value)


def test_checked_in_eval_json_read_errors_fail_without_skip(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    # Arrange
    missing_fixture_path = tmp_path / "missing.evalset.json"

    class AcceptingAdkModel:
        @staticmethod
        def model_validate_json(_: str) -> Any:
            return object()

    def fail_if_fixture_error_is_skipped(reason: str) -> None:
        raise AssertionError(f"fixture errors must fail, not skip: {reason}")

    monkeypatch.setattr(pytest, "skip", fail_if_fixture_error_is_skipped)

    # Act
    with pytest.raises(AssertionError) as exc_info:
        _load_adk_model_from_json(AcceptingAdkModel, missing_fixture_path, "EvalSet")

    # Assert
    failure_message = str(exc_info.value)
    assert "could not read checked-in EvalSet JSON" in failure_message
    assert str(missing_fixture_path) in failure_message
    assert "FileNotFoundError" in failure_message


def test_checked_in_eval_json_validation_errors_fail_without_skip(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    # Arrange
    fixture_path = tmp_path / "basic_evalset.evalset.json"
    fixture_path.write_text("{}", encoding="utf-8")

    class RejectingAdkModel:
        @staticmethod
        def model_validate_json(_: str) -> Any:
            raise ValueError("invalid checked-in eval fixture")

    def fail_if_fixture_error_is_skipped(reason: str) -> None:
        raise AssertionError(f"fixture errors must fail, not skip: {reason}")

    monkeypatch.setattr(pytest, "skip", fail_if_fixture_error_is_skipped)

    # Act
    with pytest.raises(AssertionError) as exc_info:
        _load_adk_model_from_json(RejectingAdkModel, fixture_path, "EvalSet")

    # Assert
    failure_message = str(exc_info.value)
    assert "checked-in EvalSet JSON is invalid for google-adk==2.1.0" in (
        failure_message
    )
    assert str(fixture_path) in failure_message
    assert "ValueError: invalid checked-in eval fixture" in failure_message


def test_locked_adk_eval_symbols_import_or_skip_with_version_reason() -> None:
    # Arrange
    required_symbols = {
        "google.adk.evaluation.agent_evaluator": "AgentEvaluator",
        "google.adk.evaluation.eval_set": "EvalSet",
        "google.adk.evaluation.eval_config": "EvalConfig",
    }

    # Act
    imported_symbols = {
        symbol_name: _import_adk_eval_symbol(module_name, symbol_name)
        for module_name, symbol_name in required_symbols.items()
    }

    # Assert
    assert set(imported_symbols) == {"AgentEvaluator", "EvalSet", "EvalConfig"}
    assert all(callable(symbol) for symbol in imported_symbols.values())


def test_eval_readme_records_locked_adk_compatibility_findings(
    repository_root: Path,
) -> None:
    # Arrange
    readme_path = repository_root / "orchestrator_demo" / "evals" / "README.md"
    pyproject_text = (repository_root / "pyproject.toml").read_text()
    lock_text = (repository_root / "uv.lock").read_text()

    # Act
    readme_text = readme_path.read_text()

    # Assert
    assert '"google-adk[eval]==2.1.0"' in pyproject_text
    assert "name = \"google-adk\"" in lock_text
    assert "version = \"2.1.0\"" in lock_text
    assert (
        '{ name = "google-adk", extras = ["eval"], specifier = "==2.1.0" }'
        in lock_text
    )
    for eval_dependency in ("pandas", "rouge-score", "tabulate"):
        assert f'name = "{eval_dependency}"' in lock_text
    assert "google-adk[eval]==2.1.0" in readme_text
    assert "AgentEvaluator" in readme_text
    assert "EvalSet" in readme_text
    assert "EvalConfig" in readme_text
    assert "ORCHESTRATOR_DEMO_DETERMINISTIC_MODEL=1" in readme_text
    assert "ORCHESTRATOR_DEMO_ADK_EVAL_MODE=1" in readme_text
    assert "uv run --locked pytest tests/test_adk_evalsets.py" in readme_text
    assert "adk eval" in readme_text
    assert (
        "ORCHESTRATOR_DEMO_DETERMINISTIC_MODEL=1 \\\n"
        "ORCHESTRATOR_DEMO_ADK_EVAL_MODE=1 \\\n"
        "  uv run --locked adk eval \\\n"
        "  orchestrator_demo/orchestrator \\"
    ) in readme_text
    assert "CLI compatibility" in readme_text


def test_documented_cli_agent_path_loads_under_locked_adk(
    monkeypatch: pytest.MonkeyPatch,
    repository_root: Path,
) -> None:
    # Arrange
    monkeypatch.setenv("ORCHESTRATOR_DEMO_DETERMINISTIC_MODEL", "1")
    monkeypatch.setenv("ORCHESTRATOR_DEMO_ADK_EVAL_MODE", "1")
    get_root_agent = _import_adk_eval_symbol(
        "google.adk.cli.cli_eval",
        "get_root_agent",
    )
    documented_agent_path = repository_root / "orchestrator_demo" / "orchestrator"
    cli_eval_module_names = ("agent", "agent.agent")
    saved_modules = {
        module_name: sys.modules.get(module_name)
        for module_name in cli_eval_module_names
    }
    for module_name in cli_eval_module_names:
        sys.modules.pop(module_name, None)

    try:
        # Act
        root_agent = get_root_agent(str(documented_agent_path))
    finally:
        for module_name in cli_eval_module_names:
            sys.modules.pop(module_name, None)
        for module_name, module in saved_modules.items():
            if module is not None:
                sys.modules[module_name] = module

    # Assert
    assert root_agent.name == "orchestrator"


def test_basic_eval_config_uses_conservative_fixed_eval_criteria(
    repository_root: Path,
) -> None:
    # Arrange
    config_path = (
        repository_root
        / "orchestrator_demo"
        / "evals"
        / "basic_eval_config.json"
    )

    # Act
    config = json.loads(config_path.read_text(encoding="utf-8"))

    # Assert
    assert config == {
        "criteria": {
            "tool_trajectory_avg_score": {
                "threshold": 1.0,
                "match_type": "IN_ORDER",
            },
            "response_match_score": 0.6,
        }
    }


def test_basic_evalset_uses_locked_adk_2_1_static_conversation_shape(
    repository_root: Path,
) -> None:
    # Arrange
    evalset_path = (
        repository_root
        / "orchestrator_demo"
        / "evals"
        / "basic_evalset.evalset.json"
    )
    EvalSet = _import_adk_eval_symbol("google.adk.evaluation.eval_set", "EvalSet")

    # Act
    raw_evalset = json.loads(evalset_path.read_text(encoding="utf-8"))
    parsed_evalset = EvalSet.model_validate(raw_evalset)

    # Assert
    assert set(raw_evalset) == {
        "eval_set_id",
        "name",
        "description",
        "eval_cases",
    }
    assert parsed_evalset.eval_set_id == "orchestrator_basic_regression"
    assert parsed_evalset.name == "ADK A2UI Orchestrator Basic Regression"
    assert len(parsed_evalset.eval_cases) == 5

    for raw_case in raw_evalset["eval_cases"]:
        assert set(raw_case) == {
            "evalId",
            "conversation",
            "sessionInput",
        }
        assert raw_case["sessionInput"] == {
            "appName": "orchestrator",
            "userId": "synthetic_eval_rm",
            "state": {},
        }
        assert raw_case["conversation"]
        for turn in raw_case["conversation"]:
            assert set(turn) == {
                "invocationId",
                "userContent",
                "finalResponse",
                "intermediateData",
            }
            assert turn["userContent"]["role"] == "user"
            assert _single_text_part(turn["userContent"]).strip()
            assert turn["finalResponse"]["role"] == "model"
            assert _single_text_part(turn["finalResponse"]).strip()
            assert set(turn["intermediateData"]) == {"toolUses"}
            assert isinstance(turn["intermediateData"]["toolUses"], list)


def test_basic_evalset_contains_required_cases_with_exact_tool_calls(
    repository_root: Path,
) -> None:
    # Arrange
    evalset_path = (
        repository_root
        / "orchestrator_demo"
        / "evals"
        / "basic_evalset.evalset.json"
    )
    expected_tool_calls_by_case = {
        "direct_internal_notes_summary": [
            {
                "name": "submit_orchestrator_request",
                "args": {
                    "user_input": (
                        "Summarize synthetic CRM internal notes for Riverbend Cafe, "
                        "focusing on relationship themes, open follow-ups, and "
                        "non-binding questions for the RM."
                    )
                },
            }
        ],
        "complex_meeting_prep_plan_required": [
            {
                "name": "submit_orchestrator_request",
                "args": {
                    "user_input": (
                        "Prepare a non-binding briefing for my meeting with "
                        "synthetic customer ABC Manufacturing. Include relationship "
                        "context, banker notes, industry themes, and questions to ask."
                    )
                },
            }
        ],
        "complex_prospect_research_plan_required": [
            {
                "name": "submit_orchestrator_request",
                "args": {
                    "user_input": (
                        "Research synthetic prospect Northstar Components for risks, "
                        "opportunities, and RM talking points, with safe caveats and "
                        "no binding credit decision."
                    )
                },
            }
        ],
        "natural_language_approval_does_not_execute": [
            {
                "name": "submit_orchestrator_request",
                "args": {
                    "user_input": (
                        "Prepare a non-binding meeting plan for synthetic customer "
                        "ABC Manufacturing with relationship history, banker notes, "
                        "industry context, and final synthesis."
                    )
                },
            }
        ],
        "explicit_rejection_records_final_state": [
            {
                "name": "submit_orchestrator_request",
                "args": {
                    "user_input": (
                        "Prepare a non-binding meeting plan for synthetic customer "
                        "ABC Manufacturing with relationship history, banker notes, "
                        "industry context, and final synthesis."
                    )
                },
            },
            {
                "name": "reject_orchestrator_plan",
                "args": {
                    "plan_id": "plan_meeting_prep_e213a54b9bdf",
                    "approval_surface_id": (
                        "surface_plan_meeting_prep_e213a54b9bdf"
                    ),
                    "reason": (
                        "Synthetic RM rejected the draft pending updated customer "
                        "context."
                    ),
                    "edited_plan_version": 1,
                },
            },
        ],
    }

    # Act
    evalset = json.loads(evalset_path.read_text(encoding="utf-8"))
    cases_by_id = {case["evalId"]: case for case in evalset["eval_cases"]}

    # Assert
    assert set(cases_by_id) == set(expected_tool_calls_by_case)
    for case_id, expected_tool_calls in expected_tool_calls_by_case.items():
        actual_tool_calls = [
            tool_call
            for turn in cases_by_id[case_id]["conversation"]
            for tool_call in turn["intermediateData"]["toolUses"]
        ]
        assert actual_tool_calls == expected_tool_calls
        for tool_call in actual_tool_calls:
            assert set(tool_call) == {"name", "args"}
            assert isinstance(tool_call["name"], str)
            assert isinstance(tool_call["args"], dict)

    for case_id in (
        "complex_meeting_prep_plan_required",
        "natural_language_approval_does_not_execute",
        "explicit_rejection_records_final_state",
    ):
        first_prompt = _single_text_part(
            cases_by_id[case_id]["conversation"][0]["userContent"]
        )
        assert "internal notes" not in first_prompt.casefold()

    natural_language_case = cases_by_id["natural_language_approval_does_not_execute"]
    second_turn = natural_language_case["conversation"][1]
    natural_language_tool_names = [
        tool_call["name"]
        for turn in natural_language_case["conversation"]
        for tool_call in turn["intermediateData"]["toolUses"]
    ]
    assert _single_text_part(second_turn["userContent"]) == "looks good, run it"
    assert second_turn["intermediateData"]["toolUses"] == []
    assert "approve_orchestrator_plan" not in natural_language_tool_names
    assert "Natural-language approval cannot execute" in _single_text_part(
        second_turn["finalResponse"]
    )


@pytest.mark.asyncio
async def test_basic_evalset_plan_goldens_match_deterministic_eval_runtime(
    monkeypatch: pytest.MonkeyPatch,
    repository_root: Path,
) -> None:
    # Arrange
    from orchestrator_demo.intent.classifier import LiteLlmIntentClassifier
    from orchestrator_demo.orchestrator.agent import DeterministicOrchestratorModel
    from orchestrator_demo.orchestrator.service import OrchestratorService

    monkeypatch.setenv("ORCHESTRATOR_DEMO_DETERMINISTIC_MODEL", "1")
    monkeypatch.setenv("ORCHESTRATOR_DEMO_ADK_EVAL_MODE", "1")
    evalset_path = (
        repository_root
        / "orchestrator_demo"
        / "evals"
        / "basic_evalset.evalset.json"
    )
    evalset = json.loads(evalset_path.read_text(encoding="utf-8"))
    cases_by_id = {case["evalId"]: case for case in evalset["eval_cases"]}
    plan_case_ids = (
        "complex_meeting_prep_plan_required",
        "complex_prospect_research_plan_required",
        "natural_language_approval_does_not_execute",
        "explicit_rejection_records_final_state",
    )

    async def plan_for_prompt(prompt: str) -> Any:
        service = OrchestratorService(
            intent_classifier=LiteLlmIntentClassifier(
                model=DeterministicOrchestratorModel()
            )
        )
        result = await service.handle_user_request(prompt)
        assert result.approval_plan is not None
        return result.approval_plan

    # Act
    plans_by_case_id = {}
    for case_id in plan_case_ids:
        first_turn = cases_by_id[case_id]["conversation"][0]
        prompt = _single_text_part(first_turn["userContent"])
        plan = await plan_for_prompt(prompt)
        repeated_plan = await plan_for_prompt(prompt)
        plans_by_case_id[case_id] = plan

        # Assert
        assert repeated_plan.plan_id == plan.plan_id
        assert repeated_plan.approval_surface_id == plan.approval_surface_id
        expected_response = _single_text_part(first_turn["finalResponse"])
        expected_step_ids = ", ".join(step.step_id for step in plan.steps)
        assert expected_response == (
            f"Draft plan {plan.plan_id} v{plan.plan_version} requires "
            f"structured approval on {plan.approval_surface_id}. Step ids: "
            f"{expected_step_ids}. No specialist graph has executed. Use "
            "approve_orchestrator_plan with planId, approvalSurfaceId, current "
            "planVersion, and approved step ids, or reject_orchestrator_plan "
            "with a reason."
        )

    prospect_step_ids = [
        step.step_id
        for step in plans_by_case_id[
            "complex_prospect_research_plan_required"
        ].steps
    ]
    assert prospect_step_ids == [
        "step_web_search",
        "step_industry_research",
        "step_product_opportunity",
        "step_credit_risk",
        "step_synthesis",
    ]

    rejection_plan = plans_by_case_id["explicit_rejection_records_final_state"]
    rejection_turn = cases_by_id["explicit_rejection_records_final_state"][
        "conversation"
    ][1]
    rejection_call = rejection_turn["intermediateData"]["toolUses"][0]
    assert rejection_call["args"]["plan_id"] == rejection_plan.plan_id
    assert (
        rejection_call["args"]["approval_surface_id"]
        == rejection_plan.approval_surface_id
    )


@pytest.mark.asyncio
async def test_natural_language_approval_eval_case_has_runtime_negative_guardrail(
    repository_root: Path,
) -> None:
    # Arrange
    from google.adk.models.llm_request import LlmRequest
    from google.adk.tools import FunctionTool
    from google.genai import types

    from orchestrator_demo.orchestrator.agent import DeterministicOrchestratorModel

    evalset_path = (
        repository_root
        / "orchestrator_demo"
        / "evals"
        / "basic_evalset.evalset.json"
    )
    evalset = json.loads(evalset_path.read_text(encoding="utf-8"))
    cases_by_id = {case["evalId"]: case for case in evalset["eval_cases"]}
    natural_language_case = cases_by_id["natural_language_approval_does_not_execute"]
    first_turn, second_turn = natural_language_case["conversation"]
    draft_response = {
        "status": "plan_required",
        "path": "plan_required",
        "planId": "plan_meeting_prep_e213a54b9bdf",
        "planVersion": 1,
        "approvalSurfaceId": "surface_plan_meeting_prep_e213a54b9bdf",
        "stepIds": [
            "step_relationship_summary",
            "step_internal_knowledge",
            "step_industry_research",
            "step_synthesis",
        ],
    }
    request = LlmRequest(
        contents=[
            types.Content(
                role="user",
                parts=[
                    types.Part.from_text(
                        text=_single_text_part(first_turn["userContent"])
                    )
                ],
            ),
            types.Content(
                role="user",
                parts=[
                    types.Part.from_function_response(
                        name="submit_orchestrator_request",
                        response=draft_response,
                    )
                ],
            ),
            types.Content(
                role="user",
                parts=[
                    types.Part.from_text(
                        text=_single_text_part(second_turn["userContent"])
                    )
                ],
            ),
        ],
        tools_dict={
            "submit_orchestrator_request": FunctionTool(
                _submit_orchestrator_request
            ),
            "approve_orchestrator_plan": FunctionTool(_approve_orchestrator_plan),
            "reject_orchestrator_plan": FunctionTool(_reject_orchestrator_plan),
        },
    )

    # Act
    responses = [
        response
        async for response in DeterministicOrchestratorModel().generate_content_async(
            request
        )
    ]

    # Assert
    assert len(responses) == 1
    assert responses[0].content is not None
    parts = responses[0].content.parts
    assert parts is not None
    assert len(parts) == 1
    assert parts[0].function_call is None
    assert parts[0].text == _single_text_part(second_turn["finalResponse"])


def test_basic_evalset_final_responses_encode_required_case_outcomes(
    repository_root: Path,
) -> None:
    # Arrange
    evalset_path = (
        repository_root
        / "orchestrator_demo"
        / "evals"
        / "basic_evalset.evalset.json"
    )

    # Act
    evalset = json.loads(evalset_path.read_text(encoding="utf-8"))
    cases_by_id = {case["evalId"]: case for case in evalset["eval_cases"]}

    direct_response = _single_text_part(
        cases_by_id["direct_internal_notes_summary"]["conversation"][0][
            "finalResponse"
        ]
    )
    meeting_plan_response = _single_text_part(
        cases_by_id["complex_meeting_prep_plan_required"]["conversation"][0][
            "finalResponse"
        ]
    )
    prospect_plan_response = _single_text_part(
        cases_by_id["complex_prospect_research_plan_required"]["conversation"][0][
            "finalResponse"
        ]
    )
    natural_language_response = _single_text_part(
        cases_by_id["natural_language_approval_does_not_execute"]["conversation"][1][
            "finalResponse"
        ]
    )
    rejection_response = _single_text_part(
        cases_by_id["explicit_rejection_records_final_state"]["conversation"][1][
            "finalResponse"
        ]
    )

    # Assert
    assert "Direct orchestrator response" in direct_response
    assert "internal_knowledge" in direct_response
    assert "Internal Knowledge Agent" in direct_response
    assert "synthetic CRM notes" in direct_response
    assert "Synthetic demo data only" in direct_response

    assert "Draft plan plan_meeting_prep_c0587cbb09fb v1" in (
        meeting_plan_response
    )
    assert (
        "requires structured approval on surface_plan_meeting_prep_c0587cbb09fb"
        in meeting_plan_response
    )
    assert "step_relationship_summary" in meeting_plan_response
    assert "step_internal_knowledge" in meeting_plan_response
    assert "step_industry_research" in meeting_plan_response
    assert "step_synthesis" in meeting_plan_response
    assert "No specialist graph has executed" in meeting_plan_response
    assert "approve_orchestrator_plan" in meeting_plan_response
    assert "reject_orchestrator_plan" in meeting_plan_response

    assert "Draft plan plan_prospect_research_fee8cf1dfc0b v1" in (
        prospect_plan_response
    )
    assert (
        "requires structured approval on surface_plan_prospect_research_fee8cf1dfc0b"
        in prospect_plan_response
    )
    assert "step_web_search" in prospect_plan_response
    assert "step_industry_research" in prospect_plan_response
    assert "step_product_opportunity" in prospect_plan_response
    assert "step_credit_risk" in prospect_plan_response
    assert "step_compliance_policy" not in prospect_plan_response
    assert "step_synthesis" in prospect_plan_response
    assert "No specialist graph has executed" in prospect_plan_response
    assert "approve_orchestrator_plan" in prospect_plan_response
    assert "reject_orchestrator_plan" in prospect_plan_response

    assert (
        "Natural-language approval cannot execute draft plan "
        "plan_meeting_prep_e213a54b9bdf"
    ) in natural_language_response
    assert "Use structured approval" in natural_language_response
    assert "approvalSurfaceId surface_plan_meeting_prep_e213a54b9bdf" in (
        natural_language_response
    )
    assert "current planVersion 1" in natural_language_response
    assert "step_relationship_summary" in natural_language_response
    assert "step_internal_knowledge" in natural_language_response
    assert "step_industry_research" in natural_language_response
    assert "step_synthesis" in natural_language_response

    assert "Plan plan_meeting_prep_e213a54b9bdf v1 was rejected" in (
        rejection_response
    )
    assert "status: rejected" in rejection_response
    assert "surface_plan_meeting_prep_e213a54b9bdf" in rejection_response
    assert (
        "Reason: Synthetic RM rejected the draft pending updated customer context."
        in rejection_response
    )
    assert "No specialist graph executed" in rejection_response
    assert "no specialist execution artifacts were produced" in rejection_response


def test_basic_evalset_content_is_synthetic_and_safe(
    repository_root: Path,
) -> None:
    # Arrange
    evalset_path = (
        repository_root
        / "orchestrator_demo"
        / "evals"
        / "basic_evalset.evalset.json"
    )
    forbidden_markers = (
        "api_key",
        "authorization",
        "bearer",
        "password",
        "token",
        "sk-",
        "account number",
        "ssn",
        "social security",
        "real customer",
        "binding approval",
        "approved credit",
        "guaranteed rate",
    )

    # Act
    evalset_text = evalset_path.read_text(encoding="utf-8")
    normalized_text = evalset_text.casefold()

    # Assert
    assert "Riverbend Cafe" in evalset_text
    assert "ABC Manufacturing" in evalset_text
    assert "Northstar Components" in evalset_text
    assert "synthetic" in normalized_text
    assert "non-binding" in normalized_text
    assert "safe caveats" in normalized_text
    assert all(marker not in normalized_text for marker in forbidden_markers)


def _single_text_part(content: dict[str, Any]) -> str:
    parts = content["parts"]
    assert len(parts) == 1
    assert set(parts[0]) == {"text"}
    return parts[0]["text"]


def _load_adk_model_from_json(
    model_type: Any,
    path: Path,
    model_name: str,
) -> Any:
    try:
        validate_json = model_type.model_validate_json
    except AttributeError as exc:
        pytest.skip(
            "google-adk==2.1.0 eval API shape is incompatible in this locked "
            f"environment: {model_name}.model_validate_json is missing: "
            f"{type(exc).__name__}."
        )

    try:
        fixture_json = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise AssertionError(
            f"could not read checked-in {model_name} JSON at {path}: "
            f"{type(exc).__name__}: {exc}"
        ) from exc

    try:
        return validate_json(fixture_json)
    except (ImportError, ModuleNotFoundError) as exc:
        if _is_adk_eval_or_extra_import_failure(exc):
            pytest.skip(_adk_eval_incompatibility_message(exc))
        raise AssertionError(
            f"checked-in {model_name} JSON validation raised an import error "
            f"at {path}: {type(exc).__name__}: {exc}"
        ) from exc
    except Exception as exc:
        raise AssertionError(
            f"checked-in {model_name} JSON is invalid for google-adk==2.1.0 "
            f"at {path}: {type(exc).__name__}: {exc}"
        ) from exc


def _skip_if_evaluate_eval_set_signature_is_incompatible(
    evaluate_eval_set: Any,
) -> None:
    try:
        parameters = inspect.signature(evaluate_eval_set).parameters
    except (TypeError, ValueError) as exc:
        pytest.skip(
            "google-adk==2.1.0 eval API shape is incompatible in this locked "
            "environment: AgentEvaluator.evaluate_eval_set signature could not "
            f"be inspected: {type(exc).__name__}."
        )

    required_keyword_parameters = {
        "agent_module",
        "eval_set",
        "eval_config",
        "num_runs",
        "print_detailed_results",
    }
    missing_parameters = required_keyword_parameters - set(parameters)
    if missing_parameters:
        pytest.skip(
            "google-adk==2.1.0 eval API shape is incompatible in this locked "
            "environment: AgentEvaluator.evaluate_eval_set is missing "
            f"parameters: {', '.join(sorted(missing_parameters))}."
        )


async def _evaluate_fixed_eval_set_with_diagnostics(
    *,
    evaluate_eval_set: Any,
    AgentEvaluator: Any,
    agent_module: str,
    eval_set: Any,
    eval_config: Any,
) -> None:
    try:
        await evaluate_eval_set(
            agent_module=agent_module,
            eval_set=eval_set,
            eval_config=eval_config,
            num_runs=1,
            print_detailed_results=True,
        )
    except AssertionError as exc:
        failure_diagnostics = await _fixed_eval_failure_diagnostics(
            AgentEvaluator=AgentEvaluator,
            agent_module=agent_module,
            eval_set=eval_set,
            eval_config=eval_config,
        )
        raise AssertionError(f"{exc}\n\n{failure_diagnostics}") from exc
    except (ImportError, ModuleNotFoundError) as exc:
        if _is_adk_eval_or_extra_import_failure(exc):
            pytest.skip(_adk_eval_incompatibility_message(exc))
        await _raise_eval_runner_error_with_diagnostics(
            exc,
            AgentEvaluator=AgentEvaluator,
            agent_module=agent_module,
            eval_set=eval_set,
            eval_config=eval_config,
        )
    except Exception as exc:
        await _raise_eval_runner_error_with_diagnostics(
            exc,
            AgentEvaluator=AgentEvaluator,
            agent_module=agent_module,
            eval_set=eval_set,
            eval_config=eval_config,
        )
    finally:
        _clear_agent_module_cache(agent_module)


async def _raise_eval_runner_error_with_diagnostics(
    exc: BaseException,
    *,
    AgentEvaluator: Any,
    agent_module: str,
    eval_set: Any,
    eval_config: Any,
) -> None:
    failure_diagnostics = await _fixed_eval_failure_diagnostics(
        AgentEvaluator=AgentEvaluator,
        agent_module=agent_module,
        eval_set=eval_set,
        eval_config=eval_config,
    )
    raise AssertionError(
        "fixed eval runner failed during agent evaluation: "
        f"{type(exc).__name__}: {exc}\n\n{failure_diagnostics}"
    ) from exc


def _clear_agent_module_cache(agent_module: str) -> None:
    module = sys.modules.get(agent_module)
    if module is not None:
        for cache_name in ("_ROOT_AGENT", "_APP"):
            if hasattr(module, cache_name):
                setattr(module, cache_name, None)

    for module_name in (agent_module, "orchestrator_demo.orchestrator"):
        sys.modules.pop(module_name, None)


def _adk_eval_incompatibility_message(exc: BaseException) -> str:
    return (
        "google-adk==2.1.0 eval API or eval extras are incompatible in this "
        "locked environment while running fixed evalsets: "
        f"{type(exc).__name__}. No environment values were logged."
    )


def _is_adk_eval_or_extra_import_failure(exc: BaseException) -> bool:
    missing_modules = _exception_missing_module_names(exc)
    known_eval_extra_modules = {"pandas", "rouge_score", "tabulate"}
    if not (
        any(
            missing_module.startswith("google.adk.evaluation")
            for missing_module in missing_modules
        )
        or missing_modules & known_eval_extra_modules
    ):
        return False

    saw_adk_eval_frame = False
    saw_application_frame = False
    for chained_exc in _iter_exception_chain(exc):
        traceback = chained_exc.__traceback__
        while traceback is not None:
            module_name = traceback.tb_frame.f_globals.get("__name__", "")
            saw_adk_eval_frame = saw_adk_eval_frame or module_name.startswith(
                "google.adk.evaluation"
            )
            saw_application_frame = saw_application_frame or module_name.startswith(
                "orchestrator_demo"
            )
            traceback = traceback.tb_next

    return saw_adk_eval_frame and not saw_application_frame


def _exception_missing_module_names(exc: BaseException) -> set[str]:
    return {
        missing_module
        for chained_exc in _iter_exception_chain(exc)
        if (missing_module := getattr(chained_exc, "name", None))
    }


def _iter_exception_chain(exc: BaseException) -> list[BaseException]:
    chained_exceptions: list[BaseException] = []
    pending = [exc]
    seen: set[int] = set()
    while pending:
        chained_exc = pending.pop()
        if id(chained_exc) in seen:
            continue
        seen.add(id(chained_exc))
        chained_exceptions.append(chained_exc)
        if chained_exc.__cause__ is not None:
            pending.append(chained_exc.__cause__)
        if chained_exc.__context__ is not None:
            pending.append(chained_exc.__context__)
    return chained_exceptions


async def _fixed_eval_failure_diagnostics(
    *,
    AgentEvaluator: Any,
    agent_module: str,
    eval_set: Any,
    eval_config: Any,
) -> str:
    try:
        from google.adk.evaluation.eval_config import get_eval_metrics_from_config
        from google.adk.evaluation.simulation.user_simulator_provider import (
            UserSimulatorProvider,
        )

        agent_for_eval = await AgentEvaluator._get_agent_for_eval(
            module_name=agent_module
        )
        eval_results_by_eval_id = await AgentEvaluator._get_eval_results_by_eval_id(
            agent_for_eval=agent_for_eval,
            eval_set=eval_set,
            eval_metrics=get_eval_metrics_from_config(eval_config),
            num_runs=1,
            user_simulator_provider=UserSimulatorProvider(
                user_simulator_config=eval_config.user_simulator_config
            ),
        )
    except Exception as exc:
        return _format_expected_only_eval_diagnostics(eval_set, exc)

    lines = ["ADK fixed eval failure diagnostics:"]
    for eval_case in eval_set.eval_cases:
        lines.append(f"eval case: {eval_case.eval_id}")
        expected_trajectory = _expected_trajectory_with_args(eval_case)
        lines.append(
            "expected trajectory with args: "
            f"{json.dumps(expected_trajectory, sort_keys=True)}"
        )
        eval_case_results = eval_results_by_eval_id.get(eval_case.eval_id, [])
        if not eval_case_results:
            lines.append("actual trajectory with args: <not exposed by ADK>")
            continue

        for run_index, eval_case_result in enumerate(eval_case_results, start=1):
            lines.append(f"run: {run_index}")
            for invocation_index, invocation_result in enumerate(
                eval_case_result.eval_metric_result_per_invocation,
                start=1,
            ):
                metric_details = [
                    {
                        "metric": metric_result.metric_name,
                        "score": metric_result.score,
                        "status": str(metric_result.eval_status),
                        "threshold": metric_result.threshold,
                    }
                    for metric_result in invocation_result.eval_metric_results
                ]
                actual_trajectory = _trajectory_with_args(
                    invocation_result.actual_invocation.intermediate_data
                )
                lines.append(f"invocation: {invocation_index}")
                lines.append(
                    "metric results: "
                    f"{json.dumps(metric_details, sort_keys=True)}"
                )
                lines.append(
                    "actual trajectory with args: "
                    f"{json.dumps(actual_trajectory, sort_keys=True)}"
                )

    return "\n".join(lines)


def _format_expected_only_eval_diagnostics(
    eval_set: Any,
    exc: BaseException,
) -> str:
    lines = [
        "ADK fixed eval failure diagnostics:",
        (
            "actual trajectory with args: <not exposed by ADK; diagnostics "
            f"collection raised {type(exc).__name__}>"
        ),
    ]
    for eval_case in eval_set.eval_cases:
        lines.append(f"eval case: {eval_case.eval_id}")
        lines.append(
            "expected trajectory with args: "
            f"{json.dumps(_expected_trajectory_with_args(eval_case), sort_keys=True)}"
        )
    return "\n".join(lines)


def _expected_trajectory_with_args(eval_case: Any) -> list[dict[str, Any]]:
    if not eval_case.conversation:
        return []
    trajectory: list[dict[str, Any]] = []
    for invocation in eval_case.conversation:
        trajectory.extend(_trajectory_with_args(invocation.intermediate_data))
    return trajectory


def _trajectory_with_args(intermediate_data: Any) -> list[dict[str, Any]]:
    from google.adk.evaluation.eval_case import get_all_tool_calls

    return [
        _tool_call_with_args(tool_call)
        for tool_call in get_all_tool_calls(intermediate_data)
    ]


def _tool_call_with_args(tool_call: Any) -> dict[str, Any]:
    args = getattr(tool_call, "args", None)
    if isinstance(args, Mapping):
        args = dict(args)
    elif args is None:
        args = {}
    return {
        "name": getattr(tool_call, "name", ""),
        "args": args,
    }


def _submit_orchestrator_request(user_input: str) -> dict[str, Any]:
    return {"user_input": user_input}


def _approve_orchestrator_plan(
    plan_id: str,
    approval_surface_id: str,
    approved_step_ids: list[str],
    edited_plan_version: int,
) -> dict[str, Any]:
    return {
        "plan_id": plan_id,
        "approval_surface_id": approval_surface_id,
        "approved_step_ids": approved_step_ids,
        "edited_plan_version": edited_plan_version,
    }


def _reject_orchestrator_plan(
    plan_id: str,
    approval_surface_id: str,
    reason: str,
    edited_plan_version: int | None = None,
) -> dict[str, Any]:
    return {
        "plan_id": plan_id,
        "approval_surface_id": approval_surface_id,
        "reason": reason,
        "edited_plan_version": edited_plan_version,
    }
