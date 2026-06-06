import logging
import json
from typing import Any

import pytest
from google.genai import types

from orchestrator_demo.a2a_support.transport import A2UI_MIME_TYPE, DataPart, TextPart
from orchestrator_demo.a2ui_support.schema_manager import (
    A2UI_VERSION,
    BASIC_CATALOG_ID,
)
from orchestrator_demo.a2ui_support.validation import validate_outbound_a2ui
from orchestrator_demo.contracts import (
    ExecutionPlan,
    IntentSuggestion,
    LlmIntentAssessment,
    PlanStep,
    RoutingDecision,
    StatusEvent,
)
from orchestrator_demo.orchestrator.request_context import RequestContext


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


def test_adk_tool_response_payload_redacts_secret_like_structured_fields() -> None:
    # Arrange
    from orchestrator_demo.a2ui_support.secret_safety import REDACTED_SECRET
    from orchestrator_demo.orchestrator.response_payloads import build_request_response
    from orchestrator_demo.orchestrator.service import OrchestratorRequestResult

    leaked_secret = "sk-or-v1-tool-response-secret-should-not-leak"
    user_input = (
        "Prepare me for tomorrow's meeting with ABC Manufacturing "
        f"using api_key={leaked_secret}"
    )
    decision = RoutingDecision(
        path="plan_required",
        selected_agent=None,
        confidence=0.91,
        reason=f"Planner diagnostic Authorization: Bearer {leaked_secret}",
    )
    context = RequestContext(
        user_input=user_input,
        slm_suggestion=IntentSuggestion(intent="meeting_prep", confidence=0.82),
        llm_assessment=LlmIntentAssessment(
            intents=["meeting_prep"],
            confidence=0.91,
            complexity="complex",
            rationale=f"Need multiple specialists for api_key={leaked_secret}",
            required_agents=["internal_knowledge", "synthesis"],
        ),
        decision=decision,
        plan_scope_id="tool_response_redaction",
    )
    plan = ExecutionPlan(
        plan_id="plan_meeting_prep_tool_response_redaction",
        objective=user_input,
        detected_intents=["meeting_prep"],
        selected_agents=["internal_knowledge", "synthesis"],
        steps=[
            PlanStep(
                step_id="step_internal_knowledge",
                agent_id="internal_knowledge",
                instruction=f"Review internal notes without emitting {leaked_secret}",
                expected_output="Internal customer context.",
            )
        ],
        data_source_categories=["internal_crm"],
        risk_notes=[f"Do not log Authorization: Bearer {leaked_secret}"],
        approval_surface_id="surface_plan_meeting_prep_tool_response_redaction",
    )
    result = OrchestratorRequestResult(
        path="plan_required",
        decision=decision,
        context=context,
        approval_plan=plan,
        status_events=(
            StatusEvent(
                event_id="event_tool_response_redaction",
                graph_id="graph_tool_response_redaction",
                plan_id=plan.plan_id,
                status="graph_created",
                message=f"Prepared draft with api_key={leaked_secret}",
            ),
        ),
        final_artifacts={
            "diagnostic": f"Authorization: Bearer {leaked_secret}",
        },
    )

    # Act
    payload = build_request_response(result)
    rendered = json.dumps(payload, sort_keys=True)

    # Assert
    assert payload["status"] == "plan_required"
    assert payload["planId"] == plan.plan_id
    assert leaked_secret not in rendered
    assert "Authorization" not in rendered
    assert REDACTED_SECRET in rendered


def test_adk_tool_response_payload_preserves_contract_ids_with_secret_marker_words() -> None:
    # Arrange
    from orchestrator_demo.orchestrator.response_payloads import build_request_response
    from orchestrator_demo.orchestrator.service import OrchestratorRequestResult

    decision = RoutingDecision(
        path="plan_required",
        selected_agent="credential_review",
        confidence=0.91,
        reason="Needs structured review.",
    )
    context = RequestContext(
        user_input="Prepare a customer review.",
        slm_suggestion=IntentSuggestion(intent="meeting_prep", confidence=0.82),
        llm_assessment=LlmIntentAssessment(
            intents=["meeting_prep"],
            confidence=0.91,
            complexity="complex",
            rationale="Needs multiple specialists.",
            required_agents=["credential_review", "token_analysis"],
        ),
        decision=decision,
        plan_scope_id="credential_review",
    )
    plan = ExecutionPlan(
        plan_id="plan_credential_review",
        objective="Prepare a customer review.",
        detected_intents=["meeting_prep"],
        selected_agents=["credential_review", "token_analysis"],
        steps=[
            PlanStep(
                step_id="step_credential_review",
                agent_id="credential_review",
                instruction="Review customer context.",
                expected_output="Customer context.",
            ),
            PlanStep(
                step_id="step_token_follow_up",
                agent_id="token_analysis",
                instruction="Prepare follow-up actions.",
                expected_output="Follow-up actions.",
                depends_on=["step_credential_review"],
            ),
        ],
        data_source_categories=["credential_context"],
        risk_notes=["Confirm before final outreach."],
        approval_surface_id="surface_plan_credential_review",
    )
    a2ui_part = DataPart(
        data={
            "version": A2UI_VERSION,
            "updateComponents": {
                "surfaceId": "surface_plan_credential_review",
                "components": [
                    {
                        "component": "Column",
                        "id": "root",
                        "children": [
                            "component_plan_credential_review_summary",
                            "component_step_token_follow_up",
                            "component_token_review_card",
                        ],
                    },
                    {
                        "component": "Text",
                        "id": "component_plan_credential_review_summary",
                        "text": "Review this draft.",
                    },
                    {
                        "component": "Text",
                        "id": "component_step_token_follow_up",
                        "text": "Prepare follow-up actions.",
                    },
                    {
                        "component": "Card",
                        "id": "component_token_review_card",
                        "child": "component_token_review_card_body",
                    },
                    {
                        "component": "Text",
                        "id": "component_token_review_card_body",
                        "text": "Confirm the plan before execution.",
                    },
                ],
            },
        },
        metadata={"mimeType": A2UI_MIME_TYPE},
    )
    result = OrchestratorRequestResult(
        path="plan_required",
        decision=decision,
        context=context,
        approval_plan=plan,
        a2ui_parts=(a2ui_part,),
    )

    # Act
    payload = build_request_response(result)
    approve_action = next(
        action
        for action in payload["nextActions"]
        if action["toolName"] == "approve_orchestrator_plan"
    )
    a2ui_payload = payload["a2uiParts"][0]["data"]
    components = a2ui_payload["updateComponents"]["components"]

    # Assert
    assert payload["planId"] == "plan_credential_review"
    assert payload["approvalSurfaceId"] == "surface_plan_credential_review"
    assert payload["selectedAgents"] == ["credential_review", "token_analysis"]
    assert payload["stepIds"] == ["step_credential_review", "step_token_follow_up"]
    assert payload["plan"]["steps"][0]["stepId"] == "step_credential_review"
    assert payload["plan"]["steps"][0]["agentId"] == "credential_review"
    assert payload["plan"]["steps"][1]["dependsOn"] == ["step_credential_review"]
    assert approve_action["planId"] == "plan_credential_review"
    assert approve_action["approvalSurfaceId"] == "surface_plan_credential_review"
    assert approve_action["approvedStepIds"] == [
        "step_credential_review",
        "step_token_follow_up",
    ]
    assert a2ui_payload["updateComponents"]["surfaceId"] == (
        "surface_plan_credential_review"
    )
    assert components[0]["id"] == "root"
    assert components[0]["children"] == [
        "component_plan_credential_review_summary",
        "component_step_token_follow_up",
        "component_token_review_card",
    ]
    assert components[3]["child"] == "component_token_review_card_body"
    assert validate_outbound_a2ui(payload["a2uiParts"]).valid is True
    assert "<redacted-key>" not in json.dumps(payload, sort_keys=True)


def test_response_redaction_preserves_plan_graph_routing_identifiers() -> None:
    # Arrange
    from orchestrator_demo.a2ui_support.response_redaction import (
        redacted_response_json_safe,
    )
    from orchestrator_demo.a2ui_support.secret_safety import REDACTED_SECRET

    leaked_secret = "sk-or-v1-route-redaction-secret-should-not-leak"
    payload = {
        "plan": {
            "steps": [
                {
                    "stepId": "step_token_review",
                    "condition": "token_review_complete",
                    "parallelGroup": "parallel_token_review",
                }
            ]
        },
        "statusEvents": [
            {
                "details": {
                    "parallel_group": "parallel_credential_token_review",
                    "route": "token_review_complete",
                    "routes": ["credential_token_review"],
                    "selectedRoute": "credential_token_review",
                    "graphRoute": "policy_token_route",
                    "routeCondition": "token_review_complete",
                    "edgeRoutes": ["edge_token_review"],
                    "diagnostic": f"provider failed with {leaked_secret}",
                }
            }
        ],
    }

    # Act
    redacted = redacted_response_json_safe(payload)
    rendered = json.dumps(redacted, sort_keys=True)

    # Assert
    assert redacted["plan"]["steps"][0]["condition"] == "token_review_complete"
    assert redacted["plan"]["steps"][0]["parallelGroup"] == "parallel_token_review"
    details = redacted["statusEvents"][0]["details"]
    assert details["parallel_group"] == "parallel_credential_token_review"
    assert details["route"] == "token_review_complete"
    assert details["routes"] == ["credential_token_review"]
    assert details["selectedRoute"] == "credential_token_review"
    assert details["graphRoute"] == "policy_token_route"
    assert details["routeCondition"] == "token_review_complete"
    assert details["edgeRoutes"] == ["edge_token_review"]
    assert leaked_secret not in rendered
    assert REDACTED_SECRET in rendered
    assert "<redacted-key>" not in rendered


def test_response_redaction_redacts_secret_like_field_child_values() -> None:
    # Arrange
    from orchestrator_demo.a2ui_support.response_redaction import (
        redacted_response_json_safe,
    )
    from orchestrator_demo.a2ui_support.secret_safety import (
        REDACTED_KEY,
        REDACTED_SECRET,
    )

    payload = {
        "structuredOutput": {
            "password": "hunter2",
        },
        "artifacts": [
            {
                "apiKey": "abc123xyz",
            }
        ],
        "safeText": "Review customer credentials without exposing secrets.",
    }

    # Act
    redacted = redacted_response_json_safe(payload)
    rendered = json.dumps(redacted, sort_keys=True)

    # Assert
    assert redacted["structuredOutput"][REDACTED_KEY] == REDACTED_SECRET
    assert redacted["artifacts"][0][REDACTED_KEY] == REDACTED_SECRET
    assert "hunter2" not in rendered
    assert "abc123xyz" not in rendered


def test_response_redaction_preserves_benign_user_facing_marker_words() -> None:
    # Arrange
    from orchestrator_demo.a2ui_support.response_redaction import (
        redacted_response_json_safe,
    )
    from orchestrator_demo.a2ui_support.secret_safety import REDACTED_SECRET

    leaked_secret = "sk-or-v1-free-text-secret-should-not-leak"
    payload = {
        "plan": {
            "objective": "Review customer credentials and Token analysis.",
            "steps": [
                {
                    "instruction": (
                        "Confirm credentials context before token follow-up."
                    ),
                    "expectedOutput": "Credential summary.",
                }
            ],
        },
        "specialistResponses": [
            {
                "content": (
                    "Token analysis completed for customer credentials. "
                    f"Do not expose api_key={leaked_secret}"
                )
            }
        ],
        "a2uiParts": [
            {
                "data": {
                    "updateComponents": {
                        "components": [
                            {
                                "component": "Text",
                                "id": "component_customer_credentials",
                                "text": "Token analysis for customer credentials.",
                            }
                        ]
                    }
                }
            }
        ],
    }

    # Act
    redacted = redacted_response_json_safe(payload)
    rendered = json.dumps(redacted, sort_keys=True)

    # Assert
    assert redacted["plan"]["objective"] == (
        "Review customer credentials and Token analysis."
    )
    assert redacted["plan"]["steps"][0]["instruction"] == (
        "Confirm credentials context before token follow-up."
    )
    assert redacted["plan"]["steps"][0]["expectedOutput"] == "Credential summary."
    assert redacted["specialistResponses"][0]["content"] == (
        f"Token analysis completed for customer credentials. "
        f"Do not expose {REDACTED_SECRET}"
    )
    component = redacted["a2uiParts"][0]["data"]["updateComponents"]["components"][0]
    assert component["text"] == "Token analysis for customer credentials."
    assert leaked_secret not in rendered
    assert "<redacted-key>" not in rendered


def test_adk_tool_response_payload_redacts_secret_values_in_contract_ids() -> None:
    # Arrange
    from orchestrator_demo.a2ui_support.secret_safety import REDACTED_SECRET
    from orchestrator_demo.orchestrator.response_payloads import build_request_response
    from orchestrator_demo.orchestrator.service import OrchestratorRequestResult

    leaked_secret = "sk-or-v1-contract-id-secret-should-not-leak"
    decision = RoutingDecision(
        path="plan_required",
        selected_agent=None,
        confidence=0.91,
        reason="Needs structured review.",
    )
    context = RequestContext(
        user_input="Prepare a customer review.",
        slm_suggestion=IntentSuggestion(intent="meeting_prep", confidence=0.82),
        llm_assessment=LlmIntentAssessment(
            intents=["meeting_prep"],
            confidence=0.91,
            complexity="complex",
            rationale="Needs multiple specialists.",
            required_agents=["internal_knowledge", "synthesis"],
        ),
        decision=decision,
        plan_scope_id="contract_id_redaction",
    )
    plan = ExecutionPlan(
        plan_id=f"plan_{leaked_secret}",
        objective="Prepare a customer review.",
        detected_intents=["meeting_prep"],
        selected_agents=["internal_knowledge", "synthesis"],
        steps=[
            PlanStep(
                step_id="step_internal_knowledge",
                agent_id="internal_knowledge",
                instruction="Review customer context.",
                expected_output="Customer context.",
            )
        ],
        data_source_categories=["internal_crm"],
        risk_notes=["Confirm before final outreach."],
        approval_surface_id=f"surface_plan_{leaked_secret}",
    )
    result = OrchestratorRequestResult(
        path="plan_required",
        decision=decision,
        context=context,
        approval_plan=plan,
    )

    # Act
    payload = build_request_response(result)
    rendered = json.dumps(payload, sort_keys=True)

    # Assert
    assert leaked_secret not in rendered
    assert REDACTED_SECRET in rendered


def test_adk_artifact_document_redacts_secret_like_payload_values() -> None:
    # Arrange
    from orchestrator_demo.orchestrator.artifacts import build_artifact_document

    leaked_secret = "sk-or-v1-artifact-document-secret-should-not-leak"

    # Act
    document = build_artifact_document(
        {
            "status": "approved",
            "path": "approved",
            "planId": "plan_secret_document",
            "artifacts": {
                "final_response": {
                    "agent_id": "synthesis",
                    "structured_output": {
                        "diagnostic": f"Authorization: Bearer {leaked_secret}",
                    },
                },
            },
            "statusEvents": [
                {
                    "message": f"completed with api_key={leaked_secret}",
                }
            ],
        }
    )
    rendered = json.dumps(document, sort_keys=True)

    # Assert
    assert leaked_secret not in rendered
    assert "Authorization" not in rendered
    assert "<redacted-secret>" in rendered


@pytest.mark.asyncio
async def test_saved_adk_artifact_part_redacts_secret_like_payload_values() -> None:
    # Arrange
    from orchestrator_demo.orchestrator.artifacts import save_response_artifact

    class FakeToolContext:
        def __init__(self) -> None:
            self.saved_artifacts: list[dict[str, Any]] = []

        async def save_artifact(
            self,
            filename: str,
            artifact: Any,
            custom_metadata: dict[str, Any] | None = None,
        ) -> int:
            self.saved_artifacts.append(
                {
                    "filename": filename,
                    "artifact": artifact,
                    "customMetadata": custom_metadata,
                    "version": 0,
                }
            )
            return 0

    leaked_secret = "sk-or-v1-saved-artifact-secret-should-not-leak"
    tool_context = FakeToolContext()

    # Act
    artifact_ref = await save_response_artifact(
        tool_context,
        filename="orchestrator_latest_result.json",
        response={
            "status": "approved",
            "path": "approved",
            "planId": "plan_review_saved",
            "artifacts": {
                "final_response": {
                    "agent_id": "synthesis",
                    "structured_output": {
                        "diagnostic": f"Authorization: Bearer {leaked_secret}",
                    },
                },
            },
            "statusEvents": [
                {
                    "message": f"completed with api_key={leaked_secret}",
                }
            ],
        },
        document_type="approved_result",
        plan_id="plan_review_saved",
    )

    # Assert
    assert artifact_ref == {
        "filename": "orchestrator_latest_result.json",
        "version": 0,
        "mimeType": "application/json",
        "documentType": "approved_result",
        "planId": "plan_review_saved",
    }
    assert len(tool_context.saved_artifacts) == 1
    saved_artifact = tool_context.saved_artifacts[0]["artifact"]
    assert isinstance(saved_artifact, types.Part)
    assert saved_artifact.text is not None
    assert saved_artifact.inline_data is None
    assert leaked_secret not in saved_artifact.text
    saved_document = json.loads(saved_artifact.text)
    rendered_saved_document = json.dumps(saved_document, sort_keys=True)
    assert leaked_secret not in rendered_saved_document
    assert "Authorization" not in rendered_saved_document
    assert "<redacted-secret>" in rendered_saved_document
