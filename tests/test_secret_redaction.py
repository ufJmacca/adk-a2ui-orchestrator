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


def test_audit_log_redacts_invalid_utf8_byte_payloads(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Arrange
    from orchestrator_demo.app.logging import (
        AUDIT_LOGGER_NAME,
        REDACTED_SECRET,
        log_audit_event,
    )

    byte_secret = "sk-or-v1-byte-secret-should-not-leak"
    bytearray_secret = "sk-or-v1-bytearray-secret-should-not-leak"

    # Act
    with caplog.at_level(logging.INFO, logger=AUDIT_LOGGER_NAME):
        log_audit_event(
            "invalid_utf8_byte_payload",
            {
                "raw_bytes": f"api_key={byte_secret}".encode() + b"\xff",
                "raw_bytearray": bytearray(
                    f"token={bytearray_secret}".encode() + b"\xff"
                ),
            },
        )

    # Assert
    audit_records = [
        record
        for record in caplog.records
        if getattr(record, "audit_event", None) == "invalid_utf8_byte_payload"
    ]
    assert len(audit_records) == 1
    rendered_payload = repr(getattr(audit_records[0], "event_payload"))
    assert byte_secret not in rendered_payload
    assert bytearray_secret not in rendered_payload
    assert "raw_bytes" in rendered_payload
    assert "raw_bytearray" in rendered_payload
    assert REDACTED_SECRET in rendered_payload


def test_audit_log_redacts_values_after_embedded_secret_like_text_keys(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Arrange
    from orchestrator_demo.app.logging import (
        AUDIT_LOGGER_NAME,
        REDACTED_SECRET,
        log_audit_event,
    )

    leaked_api_token = "abcdefghi"
    leaked_auth_token = "jklmnopqr"

    # Act
    with caplog.at_level(logging.INFO, logger=AUDIT_LOGGER_NAME):
        log_audit_event(
            "free_text_secret_key_value",
            {
                "msg": f"payload.apiToken: {leaked_api_token}",
                "rationale": f"diagnostic.authToken={leaked_auth_token}",
            },
        )

    # Assert
    audit_records = [
        record
        for record in caplog.records
        if getattr(record, "audit_event", None) == "free_text_secret_key_value"
    ]
    assert len(audit_records) == 1
    rendered_payload = repr(getattr(audit_records[0], "event_payload"))
    assert leaked_api_token not in rendered_payload
    assert leaked_auth_token not in rendered_payload
    assert "apiToken" not in rendered_payload
    assert "authToken" not in rendered_payload
    assert REDACTED_SECRET in rendered_payload


def test_audit_log_redacts_natural_language_credential_phrases(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Arrange
    from orchestrator_demo.app.logging import (
        AUDIT_LOGGER_NAME,
        REDACTED_SECRET,
        log_audit_event,
    )

    pasted_password = "hunter2"
    pasted_token = "abcdefghi"

    # Act
    with caplog.at_level(logging.INFO, logger=AUDIT_LOGGER_NAME):
        log_audit_event(
            "free_text_natural_language_secret",
            {
                "rejection_reason": f"Do not proceed; password is {pasted_password}",
                "llm_rationale": f"Escalate because token is {pasted_token}",
            },
        )

    # Assert
    audit_records = [
        record
        for record in caplog.records
        if getattr(record, "audit_event", None)
        == "free_text_natural_language_secret"
    ]
    assert len(audit_records) == 1
    rendered_payload = repr(getattr(audit_records[0], "event_payload"))
    assert pasted_password not in rendered_payload
    assert pasted_token not in rendered_payload
    assert "Do not proceed" in rendered_payload
    assert "Escalate because" in rendered_payload
    assert REDACTED_SECRET in rendered_payload


def test_audit_redaction_scrubs_complete_pem_private_key_blocks() -> None:
    # Arrange
    from orchestrator_demo.app.logging import redact_for_audit

    pem_private_key = (
        "-----BEGIN RSA PRIVATE KEY-----\n"
        "MIIEpAIBAAKCAQEAuPrivateKeyBodyMustNotLeak\n"
        "x9m8anotherLineThatMustNotLeak\n"
        "-----END RSA PRIVATE KEY-----"
    )

    # Act
    redacted = redact_for_audit(
        {"diagnostic": f"before context\n{pem_private_key}\nafter context"}
    )
    rendered = repr(redacted)

    # Assert
    assert "before context" in rendered
    assert "after context" in rendered
    assert "BEGIN RSA PRIVATE KEY" not in rendered
    assert "MIIEpAIBAAKCAQEAuPrivateKeyBodyMustNotLeak" not in rendered
    assert "x9m8anotherLineThatMustNotLeak" not in rendered
    assert "END RSA PRIVATE KEY" not in rendered
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


def test_graph_specialist_failure_omits_secret_bearing_exception_text(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Arrange
    from orchestrator_demo.contracts import ExecutionPlan, PlanStep
    from orchestrator_demo.orchestrator.graph_runtime import (
        AdkGraphRuntime,
        GraphRuntimeError,
    )

    plan = ExecutionPlan(
        plan_id="plan_secret_failure",
        objective="Run a step that raises provider headers.",
        detected_intents=["credit_risk"],
        selected_agents=["credit_risk"],
        steps=[
            PlanStep(
                step_id="step_credit_risk",
                agent_id="credit_risk",
                instruction="Assess credit risk.",
                expected_output="Credit risk themes.",
            )
        ],
        approval_surface_id="surface_plan_secret_failure",
    )

    def failing_handler(_request: object) -> None:
        raise RuntimeError(f"Authorization: Bearer {OPENROUTER_SECRET}")

    runtime = AdkGraphRuntime(specialist_handlers={"credit_risk": failing_handler})

    # Act
    with caplog.at_level(logging.ERROR):
        with pytest.raises(GraphRuntimeError) as exc_info:
            runtime.execute(plan)

    failure = exc_info.value
    rendered_failure = repr(
        {
            "message": str(failure),
            "cause": str(failure.__cause__) if failure.__cause__ else None,
            "cause_context": (
                repr(failure.__cause__.__context__)
                if failure.__cause__ and failure.__cause__.__context__
                else None
            ),
            "events": [
                event.model_dump(mode="json") for event in failure.status_events
            ],
        }
    )

    # Assert
    assert failure.__cause__ is not None
    assert OPENROUTER_SECRET not in rendered_failure
    assert OPENROUTER_SECRET not in caplog.text
    assert "Authorization" not in rendered_failure
    assert "Authorization" not in caplog.text
    assert "Bearer" not in rendered_failure
    assert "Bearer" not in caplog.text
    assert failure.status_events[-1].message == (
        "Approved plan step step_credit_risk failed during execution: RuntimeError."
    )
