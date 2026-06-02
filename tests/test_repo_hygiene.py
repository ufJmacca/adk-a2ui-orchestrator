from pathlib import Path
import re
import subprocess
import tomllib


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

PROHIBITED_DEPENDENCY_MANAGER_FILES = {
    "requirements.txt",
    "poetry.lock",
    "Pipfile",
    "Pipfile.lock",
    "conda.yaml",
    "environment.yml",
}


def _dependency_name(requirement: str) -> str:
    return re.split(r"[\[<>=!~; ]", requirement, maxsplit=1)[0].lower()


def _version_tuple(version: str) -> tuple[int, int, int]:
    major, minor, patch = version.split(".")[:3]
    return int(major), int(minor), int(patch)


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
    assert ".venv/" in ignore_rules
    assert "__pycache__/" in ignore_rules
    assert "*.py[cod]" in ignore_rules


def test_dockerignore_excludes_runtime_secrets_and_local_venv() -> None:
    # Arrange
    dockerignore_path = ROOT / ".dockerignore"

    # Act
    ignore_rules = {
        line.strip()
        for line in dockerignore_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    }

    # Assert
    assert ".env" in ignore_rules
    assert ".env.*" in ignore_rules
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
