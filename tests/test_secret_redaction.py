import logging
import json

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


def test_audit_redaction_scrubs_truncated_pem_private_key() -> None:
    # Arrange
    from orchestrator_demo.app.logging import redact_for_audit

    leaked_value = (
        "-----BEGIN EC PRIVATE KEY-----\n"
        "MHcCAQEEILtruncatedKeyMaterialForDiagnostics"
    )

    # Act
    redacted = redact_for_audit(
        {
            "diagnostic": f"agent config failed with {leaked_value}",
            "raw": leaked_value.encode(),
        }
    )
    rendered = repr(redacted)

    # Assert
    assert "-----BEGIN EC PRIVATE KEY-----" not in rendered
    assert "MHcCAQEEILtruncatedKeyMaterialForDiagnostics" not in rendered
    assert "<redacted-secret>" in rendered


def test_audit_redaction_scrubs_secret_in_invalid_utf8_bytes() -> None:
    # Arrange
    from orchestrator_demo.app.logging import redact_for_audit

    leaked_bytes = b"diagnostic sk-or-v1-byte-secret-should-not-leak\xff"

    # Act
    redacted = redact_for_audit({"raw": leaked_bytes})
    rendered = repr(redacted)

    # Assert
    assert leaked_bytes not in redacted.values()
    assert "sk-or-v1-byte-secret-should-not-leak" not in rendered
    assert "<redacted-secret>" in rendered


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


def test_a2ui_transport_metadata_invalid_utf8_byte_secret_is_rejected() -> None:
    # Arrange
    leaked_secret = "sk-or-v1-byte-secret-should-not-leak"
    unsafe_part = DataPart(
        data={
            "version": A2UI_VERSION,
            "createSurface": {
                "surfaceId": "surface_byte_secret_rejection",
                "catalogId": BASIC_CATALOG_ID,
            },
        },
        metadata={
            "mimeType": A2UI_MIME_TYPE,
            "note": f"{leaked_secret}\xff".encode("latin-1"),
        },
    )

    # Act
    result = validate_outbound_a2ui(unsafe_part)
    rendered_result = repr(
        {
            "validation_errors": result.validation_errors,
            "renderer_part": result.renderer_part.model_dump(mode="python"),
        }
    )

    # Assert
    assert result.valid is False
    assert isinstance(result.renderer_part, TextPart)
    assert leaked_secret not in rendered_result
    assert "<redacted-secret>" in rendered_result
    assert "A2UI metadata contains secret-like value" in rendered_result


@pytest.mark.parametrize(
    "secret_field_name",
    ["apiToken", "accessKey", "privateKey", "Authorization"],
)
def test_registry_descriptor_secret_field_diagnostics_are_redacted(
    secret_field_name: str,
) -> None:
    # Arrange
    from orchestrator_demo.registry.descriptors import (
        DescriptorValidationError,
        validate_agent_descriptors,
    )

    raw_descriptors = [
        {
            "agent_id": "internal_knowledge",
            "display_name": "Internal Knowledge Agent",
            "capabilities": ["crm notes"],
            "input_schema": {
                "type": "object",
                "properties": {
                    secret_field_name: {"type": "string"},
                },
            },
            "output_schema": {"type": "object"},
            "a2ui_catalogs": ["basic"],
            "routing_examples": ["Summarize notes."],
            "execution_mode": "local_llm",
        }
    ]

    # Act / Assert
    with pytest.raises(DescriptorValidationError) as exc_info:
        validate_agent_descriptors(raw_descriptors)

    rendered_diagnostic = str(exc_info.value)
    assert "secret-like field" in rendered_diagnostic
    assert "input_schema.properties.<redacted>" in rendered_diagnostic
    assert secret_field_name not in rendered_diagnostic


def test_adk_tool_error_payload_redacts_secret_and_traceback_text() -> None:
    # Arrange
    from orchestrator_demo.orchestrator.approval_state import PlanMutationError
    from orchestrator_demo.orchestrator.response_payloads import build_error_response

    leaked_secret = "sk-or-v1-response-secret-should-not-leak"
    unsafe_error = PlanMutationError(
        "approval failed: api_key="
        f"{leaked_secret}\nTraceback (most recent call last):\n"
        "  File '/tmp/secret/path.py', line 1, in <module>"
    )

    # Act
    payload = build_error_response(unsafe_error)
    rendered = json.dumps(payload, sort_keys=True)

    # Assert
    assert payload["status"] == "error"
    assert payload["error"]["code"] == "invalid_plan_mutation"
    assert leaked_secret not in rendered
    assert "Traceback" not in rendered
    assert "/tmp/secret/path.py" not in rendered
