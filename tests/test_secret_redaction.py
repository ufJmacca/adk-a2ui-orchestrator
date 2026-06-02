import logging

import pytest

from orchestrator_demo.a2a_support.transport import A2UI_MIME_TYPE, DataPart, TextPart
from orchestrator_demo.a2ui_support.schema_manager import (
    A2UI_VERSION,
    BASIC_CATALOG_ID,
)
from orchestrator_demo.a2ui_support.validation import validate_outbound_a2ui


SECRET_SENTINEL = "secret-value-that-must-not-leak"
OPENROUTER_SECRET = "sk-or-v1-redaction-secret-should-not-leak"


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


def test_audit_redaction_recursively_scrubs_secret_keys_values_and_bytes() -> None:
    # Arrange
    from orchestrator_demo.app.logging import redact_for_audit

    unsafe_payload = {
        "descriptor": {
            "input_schema": {
                "type": "object",
                "properties": {
                    "apiToken": {"const": OPENROUTER_SECRET},
                },
            },
        },
        "headers": {
            "Authorization": f"Bearer {OPENROUTER_SECRET}",
        },
        "environment": f"OPENROUTER_API_KEY={OPENROUTER_SECRET}",
        "raw": f"api_key={OPENROUTER_SECRET}".encode(),
        "nested": [
            {"password": "not-for-logs"},
            f"credential={OPENROUTER_SECRET}",
        ],
    }

    # Act
    redacted = redact_for_audit(unsafe_payload)
    rendered = repr(redacted)

    # Assert
    assert OPENROUTER_SECRET not in rendered
    assert "apiToken" not in rendered
    assert "Authorization" not in rendered
    assert "OPENROUTER_API_KEY" not in rendered
    assert "password" not in rendered
    assert "<redacted-secret>" in rendered
    assert "<redacted-key>" in rendered


def test_a2ui_payload_and_transport_envelope_secret_diagnostics_are_redacted() -> None:
    # Arrange
    unsafe_part = DataPart(
        data={
            "version": A2UI_VERSION,
            "createSurface": {
                "surfaceId": "surface_secret_rejection",
                "catalogId": BASIC_CATALOG_ID,
            },
            "OPENROUTER_API_KEY": OPENROUTER_SECRET,
        },
        metadata={
            "mimeType": A2UI_MIME_TYPE,
            "Authorization": f"Bearer {OPENROUTER_SECRET}",
        },
    )

    # Act
    result = validate_outbound_a2ui(unsafe_part)
    diagnostic = result.renderer_part.metadata.get("developerDiagnostic")
    rendered_diagnostic = repr(diagnostic)

    # Assert
    assert result.valid is False
    assert isinstance(result.renderer_part, TextPart)
    assert diagnostic is not None
    assert OPENROUTER_SECRET not in rendered_diagnostic
    assert "OPENROUTER_API_KEY" not in rendered_diagnostic
    assert "Authorization" not in rendered_diagnostic
    assert result.renderer_part.text == (
        "A2UI rendering unavailable. The generated UI payload failed "
        "validation and was not emitted to the renderer."
    )
