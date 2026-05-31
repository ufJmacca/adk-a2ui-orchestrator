import logging

import pytest


SECRET_SENTINEL = "secret-value-that-must-not-leak"


def test_settings_repr_json_and_diagnostics_redact_secret(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Arrange
    monkeypatch.setenv("OPENROUTER_API_KEY", SECRET_SENTINEL)
    monkeypatch.setenv("LLM_MODEL", "openrouter/example/model")

    from orchestrator_demo.app.settings import load_settings

    # Act
    settings = load_settings(env_file=None)
    rendered_outputs = [
        repr(settings),
        str(settings),
        settings.model_dump_json(),
        repr(settings.redacted_diagnostics()),
    ]

    # Assert
    assert all(SECRET_SENTINEL not in rendered for rendered in rendered_outputs)
    assert any("**********" in rendered for rendered in rendered_outputs)


def test_configuration_error_and_logs_do_not_expose_present_secret(
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Arrange
    monkeypatch.setenv("OPENROUTER_API_KEY", SECRET_SENTINEL)
    monkeypatch.delenv("LLM_MODEL", raising=False)

    from orchestrator_demo.app.settings import ConfigurationError, load_settings

    # Act
    with caplog.at_level(logging.DEBUG):
        with pytest.raises(ConfigurationError) as exc_info:
            load_settings(env_file=None)

    # Assert
    assert "LLM_MODEL" in str(exc_info.value)
    assert SECRET_SENTINEL not in str(exc_info.value)
    assert SECRET_SENTINEL not in repr(exc_info.value)
    assert exc_info.value.__context__ is None
    assert exc_info.value.__cause__ is None
    assert all(SECRET_SENTINEL not in record.getMessage() for record in caplog.records)
    assert SECRET_SENTINEL not in caplog.text
