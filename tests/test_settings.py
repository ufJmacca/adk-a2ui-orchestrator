import os

import pytest
from pydantic import SecretStr


def test_settings_load_required_and_optional_openrouter_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Arrange
    monkeypatch.setenv("OPENROUTER_API_KEY", "unit-test-openrouter-key")
    monkeypatch.setenv("LLM_MODEL", "openrouter/example/model")
    monkeypatch.setenv("OPENROUTER_API_BASE", "https://example.test/api/v1")
    monkeypatch.setenv("OR_APP_NAME", "relationship-manager-demo")
    monkeypatch.setenv("OR_SITE_URL", "https://relationship-manager.example.test")

    from orchestrator_demo.app.settings import load_settings

    # Act
    settings = load_settings(env_file=None)

    # Assert
    assert settings.openrouter_api_key.get_secret_value() == "unit-test-openrouter-key"
    assert settings.llm_model == "openrouter/example/model"
    assert settings.openrouter_api_base == "https://example.test/api/v1"
    assert settings.or_app_name == "relationship-manager-demo"
    assert settings.or_site_url == "https://relationship-manager.example.test"


def test_missing_required_settings_fail_fast_with_redacted_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Arrange
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("LLM_MODEL", raising=False)

    from orchestrator_demo.app.settings import ConfigurationError, load_settings

    # Act / Assert
    with pytest.raises(ConfigurationError) as exc_info:
        load_settings(env_file=None)

    message = str(exc_info.value)
    assert "Missing required runtime configuration" in message
    assert "OPENROUTER_API_KEY" in message
    assert "LLM_MODEL" in message
    assert "openrouter_api_key" not in message


def test_litellm_environment_setup_uses_isolated_mapping() -> None:
    # Arrange
    from orchestrator_demo.app.bootstrap_llm import configure_litellm_environment
    from orchestrator_demo.app.settings import Settings

    settings = Settings(
        openrouter_api_key=SecretStr("unit-test-openrouter-key"),
        llm_model="openrouter/example/model",
        openrouter_api_base="https://example.test/api/v1",
        or_app_name="relationship-manager-demo",
        or_site_url="https://relationship-manager.example.test",
    )
    isolated_environ: dict[str, str] = {}

    # Act
    configure_litellm_environment(settings, environ=isolated_environ)

    # Assert
    assert isolated_environ == {
        "OPENROUTER_API_KEY": "unit-test-openrouter-key",
        "LLM_MODEL": "openrouter/example/model",
        "OPENROUTER_API_BASE": "https://example.test/api/v1",
        "OR_APP_NAME": "relationship-manager-demo",
        "OR_SITE_URL": "https://relationship-manager.example.test",
    }
    assert os.environ.get("OPENROUTER_API_KEY") != "unit-test-openrouter-key"


def test_litellm_model_builder_accepts_injected_factory() -> None:
    # Arrange
    from orchestrator_demo.app.bootstrap_llm import build_litellm_model
    from orchestrator_demo.app.settings import Settings

    built_models: list[str] = []
    settings = Settings(
        openrouter_api_key=SecretStr("unit-test-openrouter-key"),
        llm_model="openrouter/example/model",
    )

    def fake_model_factory(model: str) -> dict[str, str]:
        built_models.append(model)
        return {"model": model}

    # Act
    model = build_litellm_model(
        settings,
        model_factory=fake_model_factory,
        environ={},
    )

    # Assert
    assert model == {"model": "openrouter/example/model"}
    assert built_models == ["openrouter/example/model"]


def test_litellm_model_builder_can_bypass_environment_configuration() -> None:
    # Arrange
    from orchestrator_demo.app.bootstrap_llm import build_litellm_model
    from orchestrator_demo.app.settings import Settings

    built_models: list[str] = []
    settings = Settings(
        openrouter_api_key=SecretStr("unit-test-openrouter-key"),
        llm_model="openrouter/example/model",
        openrouter_api_base="https://example.test/api/v1",
        or_app_name="relationship-manager-demo",
        or_site_url="https://relationship-manager.example.test",
    )
    isolated_environ = {"UNRELATED_SETTING": "preserved"}

    def fake_model_factory(model: str) -> dict[str, str]:
        built_models.append(model)
        return {"model": model}

    # Act
    model = build_litellm_model(
        settings,
        configure_environment=False,
        environ=isolated_environ,
        model_factory=fake_model_factory,
    )

    # Assert
    assert model == {"model": "openrouter/example/model"}
    assert built_models == ["openrouter/example/model"]
    assert isolated_environ == {"UNRELATED_SETTING": "preserved"}
    assert "OPENROUTER_API_KEY" not in isolated_environ
    assert "LLM_MODEL" not in isolated_environ
    assert "OPENROUTER_API_BASE" not in isolated_environ
