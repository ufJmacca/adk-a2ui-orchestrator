import json
from pathlib import Path
import re
import shlex
import subprocess
import tomllib
from typing import Any, Mapping
from urllib.parse import urlparse

import yaml


ROOT = Path(__file__).resolve().parents[1]


REQUIRED_FILES = {
    "pyproject.toml",
    "uv.lock",
    ".python-version",
    ".env.example",
    ".gitignore",
    ".dockerignore",
    "README.md",
}

RUNTIME_DEPENDENCIES = {
    "google-adk",
    "a2ui-agent-sdk",
    "a2a-sdk",
    "litellm",
    "pydantic",
    "pydantic-settings",
    "python-dotenv",
}

DEV_DEPENDENCIES = {
    "pytest",
    "pytest-asyncio",
    "ruff",
    "mypy",
}

REQUIRED_UV_GATE_COMMANDS = (
    "uv sync --locked",
    "uv run pytest",
    "uv run ruff check .",
    "uv run mypy orchestrator_demo",
)

REQUIRED_PRIMARY_RUNTIME_COMMAND = (
    "uv run adk api_server --a2a --with_ui orchestrator_demo "
    "--host 0.0.0.0 --port 8000 "
    "--session_service_uri sqlite:///.adk/orchestrator_sessions.sqlite "
    "--artifact_service_uri file:./.adk/artifacts"
)

REQUIRED_DEV_UI_DEBUG_COMMAND = (
    "uv run adk web orchestrator_demo --host 0.0.0.0 --port 8000"
)

PROHIBITED_README_RUNTIME_MARKERS = (
    "python -m orchestrator_demo.app",
    "/api/request",
    "/api/user-action",
    "/api/status/stream",
    "/api/status",
    "/api/artifacts",
    "GET /",
)

PROHIBITED_DEPENDENCY_MANAGER_FILES = {
    "requirements.txt",
    "poetry.lock",
    "Pipfile",
    "Pipfile.lock",
    "conda.yaml",
    "environment.yml",
}

MALFORMED_LOCAL_STORAGE_URI_MARKERS = (
    "sqlite://.adk",
    "file://.adk",
)

UNSUPPORTED_CUSTOM_SURFACE_PATHS = {
    "orchestrator_demo/app/__main__.py",
    "orchestrator_demo/app/server.py",
    "orchestrator_demo/app/static/index.html",
    "orchestrator_demo/app/static/renderer.js",
    "orchestrator_demo/app/static/styles.css",
}

OBSOLETE_CUSTOM_SURFACE_TESTS = {
    "tests/test_app_entrypoint.py",
    "tests/test_app_transport.py",
    "tests/test_renderer_contract.py",
}


def _dependency_name(requirement: str) -> str:
    return re.split(r"[\[<>=!~; ]", requirement, maxsplit=1)[0].lower()


def _version_tuple(version: str) -> tuple[int, int, int]:
    major, minor, patch = version.split(".")[:3]
    return int(major), int(minor), int(patch)


def _readme_command_lines(readme: str) -> list[str]:
    command_lines: list[str] = []
    in_command_block = False

    for raw_line in readme.splitlines():
        line = raw_line.strip()
        if line in {"```bash", "```sh", "```shell"}:
            in_command_block = True
            continue
        if in_command_block and line == "```":
            in_command_block = False
            continue
        if in_command_block and line:
            command_lines.append(line)

    return command_lines


def _readme_h2_sections(readme: str) -> dict[str, str]:
    sections: dict[str, str] = {}
    current_heading: str | None = None
    current_lines: list[str] = []

    for line in readme.splitlines():
        if line.startswith("## "):
            if current_heading is not None:
                sections[current_heading] = "\n".join(current_lines).strip()
            current_heading = line.removeprefix("## ").strip()
            current_lines = []
            continue
        if current_heading is not None:
            current_lines.append(line)

    if current_heading is not None:
        sections[current_heading] = "\n".join(current_lines).strip()

    return sections


def _squash_whitespace(value: str) -> str:
    return " ".join(value.split())


def _load_github_actions_workflow(path: Path) -> Mapping[str, Any]:
    workflow = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(workflow, dict)

    if True in workflow and "on" not in workflow:
        workflow["on"] = workflow.pop(True)

    return workflow


def _workflow_step(job: Mapping[str, Any], step_name: str) -> Mapping[str, Any]:
    steps = job.get("steps", [])
    assert isinstance(steps, list)
    matching_steps = [
        step
        for step in steps
        if isinstance(step, dict) and step.get("name") == step_name
    ]
    assert len(matching_steps) == 1, (
        f"expected one workflow step named {step_name!r}; "
        f"found {[step.get('name') for step in steps if isinstance(step, dict)]}"
    )
    return matching_steps[0]


def _command_option_value(command: str, option: str) -> str:
    arguments = shlex.split(command)
    return arguments[arguments.index(option) + 1]


def test_required_bootstrap_files_exist() -> None:
    # Arrange
    expected_files = REQUIRED_FILES

    # Act
    missing_files = sorted(name for name in expected_files if not (ROOT / name).is_file())

    # Assert
    assert missing_files == []


def test_pyproject_declares_runtime_and_dev_dependencies() -> None:
    # Arrange
    pyproject_path = ROOT / "pyproject.toml"

    # Act
    pyproject = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    project = pyproject["project"]
    runtime_dependencies = {
        _dependency_name(requirement) for requirement in project["dependencies"]
    }
    dev_dependencies = {
        _dependency_name(requirement)
        for requirement in pyproject["dependency-groups"]["dev"]
    }

    # Assert
    assert project["name"] == "adk-a2ui-orchestrator-demo"
    assert project["requires-python"] == ">=3.11"
    assert RUNTIME_DEPENDENCIES <= runtime_dependencies
    assert DEV_DEPENDENCIES <= dev_dependencies


def test_pyproject_limits_package_discovery_to_application_package() -> None:
    # Arrange
    pyproject_path = ROOT / "pyproject.toml"

    # Act
    pyproject = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    package_discovery = pyproject["tool"]["setuptools"]["packages"]["find"]

    # Assert
    assert package_discovery["include"] == ["orchestrator_demo*"]
    assert "prds*" in package_discovery["exclude"]


def test_pyproject_includes_orchestrator_agent_card_package_data() -> None:
    # Arrange
    pyproject_path = ROOT / "pyproject.toml"

    # Act
    pyproject = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    package_data = pyproject["tool"]["setuptools"]["package-data"]

    # Assert
    assert package_data["orchestrator_demo.orchestrator"] == ["agent.json"]


def test_pyproject_does_not_package_static_renderer_assets() -> None:
    # Arrange
    pyproject_path = ROOT / "pyproject.toml"

    # Act
    pyproject = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    package_data = pyproject["tool"]["setuptools"]["package-data"]
    static_package_data = {
        package: patterns
        for package, patterns in package_data.items()
        if package == "orchestrator_demo.app"
        or any("static" in pattern for pattern in patterns)
    }

    # Assert
    assert static_package_data == {}


def test_custom_http_runtime_and_static_renderer_files_are_removed() -> None:
    # Arrange
    unsupported_paths = UNSUPPORTED_CUSTOM_SURFACE_PATHS

    # Act
    present_paths = sorted(
        path for path in unsupported_paths if (ROOT / path).exists()
    )

    # Assert
    assert present_paths == []


def test_obsolete_custom_transport_and_renderer_tests_are_removed() -> None:
    # Arrange
    obsolete_tests = OBSOLETE_CUSTOM_SURFACE_TESTS

    # Act
    present_tests = sorted(path for path in obsolete_tests if (ROOT / path).exists())

    # Assert
    assert present_tests == []


def test_pyproject_configures_pytest_ruff_and_mypy_quality_gates() -> None:
    # Arrange
    pyproject_path = ROOT / "pyproject.toml"

    # Act
    pyproject = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))

    # Assert
    assert pyproject["tool"]["pytest"]["ini_options"]["testpaths"] == ["tests"]
    assert pyproject["tool"]["ruff"]["target-version"] == "py311"
    assert pyproject["tool"]["mypy"] == {
        "python_version": "3.11",
        "files": ["orchestrator_demo"],
        "ignore_missing_imports": True,
        "show_error_codes": True,
        "warn_unused_ignores": True,
    }


def test_readme_documents_required_uv_quality_gate_commands() -> None:
    # Arrange
    readme_path = ROOT / "README.md"

    # Act
    readme = readme_path.read_text(encoding="utf-8")
    missing_commands = [
        command for command in REQUIRED_UV_GATE_COMMANDS if command not in readme
    ]

    # Assert
    assert missing_commands == []


def test_readme_documents_adk_lazy_missing_configuration_behavior() -> None:
    # Arrange
    readme_path = ROOT / "README.md"

    # Act
    readme = readme_path.read_text(encoding="utf-8")
    runtime_configuration_section = _readme_h2_sections(readme).get(
        "Runtime Configuration"
    )

    # Assert
    assert runtime_configuration_section is not None
    runtime_configuration_text = _squash_whitespace(
        runtime_configuration_section.lower()
    )

    assert "loads the root agent lazily" in runtime_configuration_text
    assert "can start, serve the dev ui, and publish the a2a card" in (
        runtime_configuration_text
    )
    assert "reported on the first agent request" in runtime_configuration_text
    assert "fails fast" not in runtime_configuration_text
    assert "fail-fast" not in runtime_configuration_text


def test_readme_primary_runtime_section_documents_required_adk_a2a_command() -> None:
    # Arrange
    readme_path = ROOT / "README.md"
    expected_command = REQUIRED_PRIMARY_RUNTIME_COMMAND

    # Act
    readme = readme_path.read_text(encoding="utf-8")
    readme_command_lines = _readme_command_lines(readme)
    primary_runtime_section = _readme_h2_sections(readme).get(
        "Primary Local Runtime"
    )
    primary_runtime_command_lines = _readme_command_lines(
        primary_runtime_section or ""
    )

    # Assert
    assert primary_runtime_section is not None
    primary_runtime_text = _squash_whitespace(primary_runtime_section.lower())

    assert "primary local runtime" in primary_runtime_text
    assert primary_runtime_command_lines.count(expected_command) == 1
    assert REQUIRED_DEV_UI_DEBUG_COMMAND not in primary_runtime_command_lines
    assert readme_command_lines.count(expected_command) == 1


def test_readme_primary_runtime_command_uses_adk_accepted_local_storage_uris() -> None:
    # Arrange
    readme_path = ROOT / "README.md"
    expected_session_uri = "sqlite:///.adk/orchestrator_sessions.sqlite"
    expected_artifact_uri = "file:./.adk/artifacts"

    # Act
    readme = readme_path.read_text(encoding="utf-8")
    session_uri = _command_option_value(
        REQUIRED_PRIMARY_RUNTIME_COMMAND,
        "--session_service_uri",
    )
    artifact_uri = _command_option_value(
        REQUIRED_PRIMARY_RUNTIME_COMMAND,
        "--artifact_service_uri",
    )

    # Assert
    assert REQUIRED_PRIMARY_RUNTIME_COMMAND in readme
    assert session_uri == expected_session_uri
    assert artifact_uri == expected_artifact_uri
    assert all(marker not in readme for marker in MALFORMED_LOCAL_STORAGE_URI_MARKERS)
    assert all(
        marker not in REQUIRED_PRIMARY_RUNTIME_COMMAND
        for marker in MALFORMED_LOCAL_STORAGE_URI_MARKERS
    )

    session_parts = urlparse(session_uri)
    artifact_parts = urlparse(artifact_uri)
    adk_sqlite_path = Path(session_parts.path.removeprefix("/"))
    artifact_path = Path(artifact_parts.path)

    assert session_parts.scheme == "sqlite"
    assert session_parts.netloc == ""
    assert adk_sqlite_path == Path(".adk/orchestrator_sessions.sqlite")
    assert artifact_parts.scheme == "file"
    assert artifact_parts.netloc == ""
    assert artifact_parts.path == "./.adk/artifacts"
    assert artifact_path == Path(".adk/artifacts")
    assert not artifact_path.is_absolute()


def test_primary_runtime_storage_uris_construct_adk_services() -> None:
    # Arrange
    from google.adk.cli.service_registry import get_service_registry

    session_uri = _command_option_value(
        REQUIRED_PRIMARY_RUNTIME_COMMAND,
        "--session_service_uri",
    )
    artifact_uri = _command_option_value(
        REQUIRED_PRIMARY_RUNTIME_COMMAND,
        "--artifact_service_uri",
    )
    registry = get_service_registry()

    # Act
    session_service = registry.create_session_service(session_uri, agents_dir=str(ROOT))
    artifact_service = registry.create_artifact_service(
        artifact_uri,
        agents_dir=str(ROOT),
    )

    # Assert
    assert session_service is not None
    assert getattr(session_service, "_db_path") == ".adk/orchestrator_sessions.sqlite"
    assert artifact_service is not None
    assert getattr(artifact_service, "root_dir") == (ROOT / ".adk/artifacts").resolve()


def test_readme_dev_ui_debugging_section_keeps_adk_web_debugging_only() -> None:
    # Arrange
    readme_path = ROOT / "README.md"
    expected_command = REQUIRED_DEV_UI_DEBUG_COMMAND

    # Act
    readme = readme_path.read_text(encoding="utf-8")
    readme_command_lines = _readme_command_lines(readme)
    dev_ui_debugging_section = _readme_h2_sections(readme).get("Dev UI Debugging")
    dev_ui_debugging_command_lines = _readme_command_lines(
        dev_ui_debugging_section or ""
    )

    # Assert
    assert dev_ui_debugging_section is not None
    dev_ui_debugging_text = _squash_whitespace(dev_ui_debugging_section.lower())

    assert "only for dev ui debugging" in dev_ui_debugging_text
    assert "not the primary local runtime" in dev_ui_debugging_text
    assert dev_ui_debugging_command_lines.count(expected_command) == 1
    assert REQUIRED_PRIMARY_RUNTIME_COMMAND not in dev_ui_debugging_command_lines
    assert readme_command_lines.count(expected_command) == 1


def test_eval_readme_documents_local_fixed_eval_workflow_and_capture_safety() -> None:
    # Arrange
    readme_path = ROOT / "orchestrator_demo" / "evals" / "README.md"
    expected_pytest_command = "uv run --locked pytest tests/test_adk_evalsets.py"
    expected_cli_command = (
        "uv run --locked adk eval \\\n"
        "  orchestrator_demo/orchestrator \\\n"
        "  orchestrator_demo/evals/basic_evalset.evalset.json \\\n"
        "  --config_file_path orchestrator_demo/evals/basic_eval_config.json \\\n"
        "  --print_detailed_results"
    )
    unsupported_package_root_cli_command = (
        "uv run --locked adk eval \\\n"
        "  orchestrator_demo \\\n"
        "  orchestrator_demo/evals/basic_evalset.evalset.json"
    )

    # Act
    readme = readme_path.read_text(encoding="utf-8")
    normalized_readme = _squash_whitespace(readme.casefold())

    # Assert
    assert "uv sync --locked" in readme
    assert expected_pytest_command in readme
    assert expected_cli_command in readme
    assert unsupported_package_root_cli_command not in readme
    assert REQUIRED_DEV_UI_DEBUG_COMMAND in readme
    assert "captured sessions" in normalized_readme
    assert "cleaned" in normalized_readme
    assert "synthetic" in normalized_readme
    assert "manually promoted" in normalized_readme
    assert "fixed evals do not require `openrouter_api_key`" in normalized_readme
    assert "deterministic mode" in normalized_readme
    assert "must not log secrets" in normalized_readme


def test_user_sim_workflow_is_manual_and_documents_cost_controls() -> None:
    # Arrange
    workflow_path = ROOT / ".github" / "workflows" / "eval-user-sim.yml"
    readme_path = ROOT / "orchestrator_demo" / "evals" / "README.md"
    config_path = ROOT / "orchestrator_demo" / "evals" / "user_sim_eval_config.json"

    # Act
    workflow = _load_github_actions_workflow(workflow_path)
    readme = readme_path.read_text(encoding="utf-8")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    normalized_readme = _squash_whitespace(readme.casefold())
    trigger = workflow["on"]
    jobs = workflow["jobs"]
    eval_job = jobs["eval-user-sim"]
    eval_job_env = eval_job.get("env", {})
    eval_steps = [step for step in eval_job["steps"] if isinstance(step, dict)]
    credential_step = _workflow_step(
        eval_job,
        "Validate credentials and invocation limit",
    )
    user_sim_step = _workflow_step(
        eval_job,
        "Run dynamic ADK User Simulation evals",
    )
    artifact_steps = [
        step
        for step in eval_steps
        if str(step.get("uses", "")).startswith("actions/upload-artifact@")
    ]
    job_run_script = "\n".join(
        str(step["run"]) for step in eval_steps if "run" in step
    )

    # Assert
    assert workflow["name"] == "ADK User Simulation Eval"
    assert set(trigger) == {"workflow_dispatch"}
    assert "pull_request" not in trigger
    assert "push" not in trigger
    assert "schedule" not in trigger
    assert "eval-user-sim" in jobs
    assert "quality" not in jobs
    assert "eval-basic" not in jobs
    assert eval_job_env["MAX_ALLOWED_INVOCATIONS"] == "12"
    assert (
        config["user_simulator_config"]["max_allowed_invocations"]
        <= int(eval_job_env["MAX_ALLOWED_INVOCATIONS"])
    )
    assert eval_job_env["GOOGLE_CLOUD_PROJECT"] == (
        "${{ secrets.GOOGLE_CLOUD_PROJECT }}"
    )
    assert credential_step["env"]["GOOGLE_APPLICATION_CREDENTIALS_JSON"] == (
        "${{ secrets.GOOGLE_APPLICATION_CREDENTIALS_JSON }}"
    )
    assert "uv run --locked adk eval_set create" in user_sim_step["run"]
    assert "uv run --locked adk eval_set add_eval_case" in user_sim_step["run"]
    assert "uv run --locked adk eval" in user_sim_step["run"]
    assert (
        "--config_file_path orchestrator_demo/evals/user_sim_eval_config.json"
        in user_sim_step["run"]
    )
    assert "tee .ai-native/eval-user-sim/eval-user-sim.log" in user_sim_step["run"]
    assert "secrets.GOOGLE_APPLICATION_CREDENTIALS_JSON" not in job_run_script
    assert "secrets.GOOGLE_CLOUD_PROJECT" not in job_run_script
    assert not re.search(
        r"\becho\b[^\n]*\$\{?GOOGLE_APPLICATION_CREDENTIALS_JSON\}?",
        job_run_script,
    )
    assert not re.search(
        r"\bcat\b[^\n]*GOOGLE_APPLICATION_CREDENTIALS",
        job_run_script,
    )
    assert len(artifact_steps) == 1
    assert artifact_steps[0]["if"] == "${{ always() }}"
    assert artifact_steps[0]["with"]["name"] == "eval-user-sim-results"
    assert artifact_steps[0]["with"]["path"] == ".ai-native/eval-user-sim/"
    assert artifact_steps[0]["with"]["include-hidden-files"] is True
    assert "eval-user-sim" in normalized_readme
    assert ".github/workflows/eval-user-sim.yml" in readme
    assert "workflow_dispatch" in normalized_readme
    assert "google_cloud_project" in normalized_readme
    assert "google_application_credentials_json" in normalized_readme
    assert "max_allowed_invocations" in normalized_readme
    assert "not run on pull requests" in normalized_readme
    assert "artifact" in normalized_readme


def test_readme_no_longer_documents_custom_http_runtime_paths() -> None:
    # Arrange
    readme_path = ROOT / "README.md"
    prohibited_markers = PROHIBITED_README_RUNTIME_MARKERS

    # Act
    readme = readme_path.read_text(encoding="utf-8")
    documented_markers = [
        marker for marker in prohibited_markers if marker in readme
    ]

    # Assert
    assert documented_markers == []


def test_readme_shell_commands_use_uv_only_project_execution() -> None:
    # Arrange
    readme_path = ROOT / "README.md"

    # Act
    readme = readme_path.read_text(encoding="utf-8")
    command_lines = _readme_command_lines(readme)
    non_uv_commands = sorted(
        command for command in command_lines if not command.startswith("uv ")
    )

    # Assert
    assert non_uv_commands == []


def test_uv_lock_contains_declared_project_dependencies() -> None:
    # Arrange
    lock_path = ROOT / "uv.lock"

    # Act
    lock_contents = lock_path.read_text(encoding="utf-8")
    missing_locked_packages = sorted(
        package
        for package in RUNTIME_DEPENDENCIES | DEV_DEPENDENCIES
        if f'name = "{package}"' not in lock_contents
    )

    # Assert
    assert missing_locked_packages == []


def test_a2a_sdk_lock_stays_adk_compatible() -> None:
    # Arrange
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    lockfile = tomllib.loads((ROOT / "uv.lock").read_text(encoding="utf-8"))

    # Act
    declared_a2a_constraints = [
        requirement
        for requirement in pyproject["project"]["dependencies"]
        if _dependency_name(requirement) == "a2a-sdk"
    ]
    locked_a2a_versions = [
        package["version"]
        for package in lockfile["package"]
        if package["name"] == "a2a-sdk"
    ]

    # Assert
    assert declared_a2a_constraints == ["a2a-sdk>=0.3.4,<0.4"]
    assert len(locked_a2a_versions) == 1
    assert (0, 3, 4) <= _version_tuple(locked_a2a_versions[0]) < (0, 4, 0)


def test_gitignore_protects_runtime_secrets_and_local_venv() -> None:
    # Arrange
    gitignore_path = ROOT / ".gitignore"

    # Act
    ignore_rules = {
        line.strip()
        for line in gitignore_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    }

    # Assert
    assert ".env" in ignore_rules
    assert ".env.*" in ignore_rules
    assert "!.env.example" in ignore_rules
    assert ".adk/" in ignore_rules
    assert ".venv/" in ignore_rules
    assert "__pycache__/" in ignore_rules
    assert "*.py[cod]" in ignore_rules


def test_dockerignore_excludes_runtime_secrets_and_local_venv() -> None:
    # Arrange
    dockerignore_path = ROOT / ".dockerignore"

    # Act
    ignore_lines = [
        line.strip()
        for line in dockerignore_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    ignore_rules = set(ignore_lines)

    # Assert
    assert ".env" in ignore_rules
    assert ".env.*" in ignore_rules
    assert "!.env.example" in ignore_rules
    assert ignore_lines.index("!.env.example") > ignore_lines.index(".env.*")
    assert ".venv/" in ignore_rules


def test_no_prohibited_dependency_manager_files_exist() -> None:
    # Arrange
    prohibited_files = PROHIBITED_DEPENDENCY_MANAGER_FILES

    # Act
    present_files = sorted(
        name for name in prohibited_files if (ROOT / name).exists()
    )
    tracked_paths = subprocess.run(
        ["git", "ls-files"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    tracked_virtualenv_paths = sorted(
        path for path in tracked_paths if path == ".venv" or path.startswith(".venv/")
    )

    # Assert
    assert present_files == []
    assert tracked_virtualenv_paths == []


def test_no_generated_python_bytecode_is_tracked() -> None:
    # Act
    tracked_paths = subprocess.run(
        ["git", "ls-files"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    tracked_bytecode_paths = sorted(
        path
        for path in tracked_paths
        if path == "__pycache__"
        or path.startswith("__pycache__/")
        or "/__pycache__/" in path
        or path.endswith((".pyc", ".pyo"))
    )

    # Assert
    assert tracked_bytecode_paths == []
