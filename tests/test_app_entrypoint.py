import os
from pathlib import Path
import subprocess
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _add_repository_to_pythonpath(env: dict[str, str]) -> None:
    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        str(REPOSITORY_ROOT)
        if not existing_pythonpath
        else f"{REPOSITORY_ROOT}{os.pathsep}{existing_pythonpath}"
    )


def test_app_module_entrypoint_runs_with_required_runtime_env() -> None:
    # Arrange
    env = os.environ.copy()
    env["OPENROUTER_API_KEY"] = "test-openrouter-key"
    env["LLM_MODEL"] = "openrouter/test/model"

    # Act
    result = subprocess.run(
        [sys.executable, "-m", "orchestrator_demo.app"],
        cwd=REPOSITORY_ROOT,
        env=env,
        text=True,
        capture_output=True,
        timeout=10,
        check=False,
    )

    # Assert
    assert result.returncode == 0
    assert "runtime configuration validated" in result.stdout
    assert "No module named orchestrator_demo.app.__main__" not in result.stderr
    assert "test-openrouter-key" not in result.stdout
    assert "test-openrouter-key" not in result.stderr


def test_app_module_entrypoint_loads_local_dotenv_before_required_env_check(
    tmp_path: Path,
) -> None:
    # Arrange
    env = os.environ.copy()
    env.pop("OPENROUTER_API_KEY", None)
    env.pop("LLM_MODEL", None)
    _add_repository_to_pythonpath(env)
    dotenv_key = "dotenv-openrouter-key"
    (tmp_path / ".env").write_text(
        f"OPENROUTER_API_KEY={dotenv_key}\nLLM_MODEL=openrouter/test/model\n",
        encoding="utf-8",
    )

    # Act
    result = subprocess.run(
        [sys.executable, "-m", "orchestrator_demo.app"],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        timeout=10,
        check=False,
    )

    # Assert
    assert result.returncode == 0
    assert "runtime configuration validated" in result.stdout
    assert dotenv_key not in result.stdout
    assert dotenv_key not in result.stderr


def test_app_module_entrypoint_fails_fast_without_required_runtime_env(
    tmp_path: Path,
) -> None:
    # Arrange
    env = os.environ.copy()
    env.pop("OPENROUTER_API_KEY", None)
    env.pop("LLM_MODEL", None)
    _add_repository_to_pythonpath(env)

    # Act
    result = subprocess.run(
        [sys.executable, "-m", "orchestrator_demo.app"],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        timeout=10,
        check=False,
    )

    # Assert
    assert result.returncode == 2
    assert "Configuration error" in result.stderr
    assert "OPENROUTER_API_KEY" in result.stderr
    assert "LLM_MODEL" in result.stderr
    assert "test-openrouter-key" not in result.stderr
