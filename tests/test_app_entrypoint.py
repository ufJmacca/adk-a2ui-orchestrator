import os
from pathlib import Path
import select
import subprocess
import sys
import time


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _add_repository_to_pythonpath(env: dict[str, str]) -> None:
    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        str(REPOSITORY_ROOT)
        if not existing_pythonpath
        else f"{REPOSITORY_ROOT}{os.pathsep}{existing_pythonpath}"
    )


def _start_app_process(
    *,
    cwd: Path,
    env: dict[str, str],
) -> subprocess.Popen[str]:
    env["ORCHESTRATOR_APP_HOST"] = "127.0.0.1"
    env["ORCHESTRATOR_APP_PORT"] = "0"
    env["PYTHONUNBUFFERED"] = "1"
    return subprocess.Popen(
        [sys.executable, "-m", "orchestrator_demo.app"],
        cwd=cwd,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def _read_startup_line(process: subprocess.Popen[str]) -> str:
    assert process.stdout is not None
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        ready, _, _ = select.select([process.stdout], [], [], 0.1)
        if ready:
            return process.stdout.readline()
        if process.poll() is not None:
            break

    stderr = process.stderr.read() if process.stderr is not None else ""
    raise AssertionError(
        "app module did not print a startup line before exiting; "
        f"returncode={process.poll()} stderr={stderr}"
    )


def _terminate_process(process: subprocess.Popen[str]) -> tuple[str, str]:
    process.terminate()
    try:
        stdout, stderr = process.communicate(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        stdout, stderr = process.communicate(timeout=10)
    return stdout, stderr


def test_app_module_entrypoint_starts_local_http_app_with_required_runtime_env() -> None:
    # Arrange
    env = os.environ.copy()
    env["OPENROUTER_API_KEY"] = "test-openrouter-key"
    env["LLM_MODEL"] = "openrouter/test/model"

    # Act
    process = _start_app_process(cwd=REPOSITORY_ROOT, env=env)
    try:
        startup_line = _read_startup_line(process)

        # Assert
        assert process.poll() is None
        assert "Local orchestrator app listening on http://127.0.0.1:" in startup_line
        assert "test-openrouter-key" not in startup_line
    finally:
        stdout, stderr = _terminate_process(process)
    assert "No module named orchestrator_demo.app.__main__" not in stderr
    assert "test-openrouter-key" not in stdout
    assert "test-openrouter-key" not in stderr


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
    process = _start_app_process(cwd=tmp_path, env=env)
    try:
        startup_line = _read_startup_line(process)

        # Assert
        assert process.poll() is None
        assert "Local orchestrator app listening on http://127.0.0.1:" in startup_line
        assert dotenv_key not in startup_line
    finally:
        stdout, stderr = _terminate_process(process)
    assert dotenv_key not in stdout
    assert dotenv_key not in stderr


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
