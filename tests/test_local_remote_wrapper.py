import pytest
from pydantic import BaseModel

from orchestrator_demo.contracts import SpecialistRequest


REMOTE_COMPATIBLE_AGENT_IDS = {"internal_knowledge", "product_opportunity"}


def _openai_style_key(suffix: str) -> str:
    return "sk-" + suffix


def _openrouter_key(suffix: str) -> str:
    return _openai_style_key(f"or-{suffix}")


def _bearer_token(token: str) -> str:
    return "Bearer " + token


def _github_token() -> str:
    return "g" + "hp_" + "abcdefghijklmnopqrstuvwxyz123456"


def _aws_access_key() -> str:
    return "AK" + "IA" + "IOSFODNN7EXAMPLE"


def _google_api_key() -> str:
    return "AI" + "za" + "SyDabcdefghijklmnopqrstuvwxyz12"


def _slack_bot_token() -> str:
    return "xo" + "xb-" + "123456789012-abcdefghijklmnop"


def _jwt() -> str:
    return ".".join(
        [
            "eyJ" + "hbGciOiJIUzI1NiJ9",
            "eyJ" + "zdWIiOiIxMjM0NTY3ODkwIn0",
            "SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c",
        ]
    )


class Settings(BaseModel):
    api_key: str
    model: str
    note: str


def _request_for(agent_id: str, *, context: dict[str, object] | None = None) -> SpecialistRequest:
    return SpecialistRequest(
        request_id=f"request_{agent_id}",
        user_input="Summarize synthetic business banking context.",
        agent_id=agent_id,
        context=context or {"customer": "ABC Manufacturing", "scope": "synthetic_demo"},
    )


def test_default_local_remote_wrappers_include_required_specialists() -> None:
    # Arrange
    from orchestrator_demo.a2a_support.local_remote_wrapper import (
        LocalRemoteAgentWrapper,
        build_default_local_remote_wrappers,
    )
    from orchestrator_demo.agents import build_default_specialists

    specialists = build_default_specialists()

    # Act
    wrappers = build_default_local_remote_wrappers(specialists)

    # Assert
    assert set(wrappers) == REMOTE_COMPATIBLE_AGENT_IDS
    assert {
        agent_id
        for agent_id, wrapper in wrappers.items()
        if isinstance(wrapper, LocalRemoteAgentWrapper)
    } == REMOTE_COMPATIBLE_AGENT_IDS
    assert {wrapper.agent_id for wrapper in wrappers.values()} == REMOTE_COMPATIBLE_AGENT_IDS


def test_default_local_remote_wrappers_respect_explicit_empty_specialist_map() -> None:
    # Arrange
    from orchestrator_demo.a2a_support.local_remote_wrapper import (
        build_default_local_remote_wrappers,
    )

    # Act
    wrappers = build_default_local_remote_wrappers({})

    # Assert
    assert wrappers == {}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("agent_id", "agent_module", "agent_class_name"),
    [
        (
            "internal_knowledge",
            "orchestrator_demo.agents.internal_knowledge",
            "InternalKnowledgeAgent",
        ),
        (
            "product_opportunity",
            "orchestrator_demo.agents.product_opportunity",
            "ProductOpportunityAgent",
        ),
    ],
)
async def test_wrapped_specialists_match_direct_local_specialist_responses(
    agent_id: str,
    agent_module: str,
    agent_class_name: str,
) -> None:
    # Arrange
    import importlib

    from orchestrator_demo.a2a_support.local_remote_wrapper import LocalRemoteAgentWrapper
    from orchestrator_demo.a2a_support.remote_agent_adapter import RemoteA2AAgentAdapter

    agent_class = getattr(importlib.import_module(agent_module), agent_class_name)
    direct_agent = agent_class()
    wrapped_local_agent = agent_class()
    wrapper = LocalRemoteAgentWrapper(wrapped_local_agent)
    request = _request_for(agent_id)

    # Act
    direct_response = await direct_agent.handle(request)
    wrapped_response = await wrapper.run(request)

    # Assert
    assert isinstance(wrapper, RemoteA2AAgentAdapter)
    assert wrapped_response == direct_response
    assert wrapped_local_agent.calls == [request]
    assert wrapped_local_agent.calls[0] is request


@pytest.mark.asyncio
async def test_wrapper_preserves_safe_user_action_payload_for_local_specialist() -> None:
    # Arrange
    from orchestrator_demo.a2a_support.local_remote_wrapper import LocalRemoteAgentWrapper
    from orchestrator_demo.agents.product_opportunity import ProductOpportunityAgent

    local_agent = ProductOpportunityAgent()
    wrapper = LocalRemoteAgentWrapper(local_agent)
    user_action_payload = {
        "userAction": {
            "actionId": "action_product_detail",
            "type": "specialist_action",
            "surfaceId": "surface_product_opportunity_request_product_opportunity",
            "payload": {
                "selectedProduct": "treasury_services",
                "filters": ["cash_visibility", "controls"],
                "metadata": {"renderer": "basic-demo"},
            },
        }
    }

    # Act
    response = await wrapper.handle_user_action(user_action_payload)

    # Assert
    assert response.agent_id == "product_opportunity"
    assert local_agent.call_count == 1
    forwarded_request = local_agent.calls[0]
    assert forwarded_request.agent_id == "product_opportunity"
    assert forwarded_request.context["user_action_payload"] == user_action_payload
    assert forwarded_request.context["user_action"] == user_action_payload["userAction"]


@pytest.mark.asyncio
async def test_wrapped_a2ui_specialist_response_payload_is_data_part_compatible() -> None:
    # Arrange
    from a2a.types import DataPart as SdkDataPart

    from orchestrator_demo.a2a_support.local_remote_wrapper import LocalRemoteAgentWrapper
    from orchestrator_demo.a2a_support.part_converters import a2ui_data_parts_from_payload
    from orchestrator_demo.agents.product_opportunity import ProductOpportunityAgent

    wrapper = LocalRemoteAgentWrapper(ProductOpportunityAgent())
    request = _request_for("product_opportunity")

    # Act
    response = await wrapper.run(request)

    # Assert
    assert response.a2ui_payload is not None
    data_parts = a2ui_data_parts_from_payload(response.a2ui_payload)
    assert [part.data for part in data_parts] == response.a2ui_payload
    for part in data_parts:
        assert "a2ui" not in part.data
        SdkDataPart(data=part.data, metadata=part.metadata)
    assert data_parts[0].data["createSurface"]["surfaceId"] == response.surface_id


@pytest.mark.asyncio
async def test_wrapper_redacts_secret_like_values_before_forwarding() -> None:
    # Arrange
    from orchestrator_demo.a2a_support.local_remote_wrapper import LocalRemoteAgentWrapper
    from orchestrator_demo.agents.internal_knowledge import InternalKnowledgeAgent

    local_agent = InternalKnowledgeAgent()
    wrapper = LocalRemoteAgentWrapper(local_agent)
    request = _request_for(
        "internal_knowledge",
        context={
            "customer": "ABC Manufacturing",
            "api_key": _openrouter_key("secret-value"),
            "nested": {
                "authorization": _bearer_token("secret-token"),
                "safe_note": "keep this business context",
                "freeform_note": f"renderer supplied api_key={_openrouter_key('inline-secret')}",
            },
        },
    )
    user_action_payload = {
        "userAction": {
            "actionId": "action_secret_check",
            "type": "specialist_action",
            "surfaceId": "surface_internal_knowledge_notes",
            "payload": {
                "apiKey": _openrouter_key("renderer-secret"),
                "note": f"inline credential token={_openrouter_key('renderer-inline')}",
                "selectedNote": "treasury follow-up",
            },
        }
    }

    # Act
    await wrapper.run(request)
    event_response = await wrapper.handle_user_action(user_action_payload)

    # Assert
    run_request = local_agent.calls[0]
    action_request = local_agent.calls[1]
    assert request.context["api_key"] == _openrouter_key("secret-value")
    assert run_request.context == {
        "customer": "ABC Manufacturing",
        "api_key": "[REDACTED]",
        "nested": {
            "authorization": "[REDACTED]",
            "safe_note": "keep this business context",
            "freeform_note": "[REDACTED]",
        },
    }
    assert action_request.context["user_action_payload"]["userAction"]["payload"] == {
        "apiKey": "[REDACTED]",
        "note": "[REDACTED]",
        "selectedNote": "treasury follow-up",
    }
    serialized_calls = [call.model_dump_json() for call in local_agent.calls]
    serialized_response = event_response.model_dump_json()
    assert _openrouter_key("secret-value") not in serialized_calls[0]
    assert _bearer_token("secret-token") not in serialized_calls[0]
    assert _openrouter_key("inline-secret") not in serialized_calls[0]
    assert _openrouter_key("renderer-secret") not in serialized_calls[1]
    assert _openrouter_key("renderer-inline") not in serialized_calls[1]
    assert _openrouter_key("renderer-secret") not in serialized_response
    assert _openrouter_key("renderer-inline") not in serialized_response


@pytest.mark.asyncio
async def test_wrapper_redacts_byte_like_secret_values_before_forwarding() -> None:
    # Arrange
    from orchestrator_demo.a2a_support.local_remote_wrapper import LocalRemoteAgentWrapper
    from orchestrator_demo.agents.internal_knowledge import InternalKnowledgeAgent

    context_bytes_secret = _openrouter_key("byte-context-secret")
    context_bytearray_secret = _github_token()
    event_bytes_secret = _openrouter_key("byte-event-secret")
    event_bytearray_secret = _bearer_token("bytearray-event-secret")
    local_agent = InternalKnowledgeAgent()
    wrapper = LocalRemoteAgentWrapper(local_agent)
    request = _request_for(
        "internal_knowledge",
        context={
            "customer": "ABC Manufacturing",
            "attachment": context_bytes_secret.encode("utf-8"),
            "renderer_blob": bytearray(context_bytearray_secret.encode("utf-8")),
            "safe_bytes": b"synthetic-policy-reference",
        },
    )
    user_action_payload = {
        "userAction": {
            "actionId": "action_byte_secret_check",
            "type": "specialist_action",
            "surfaceId": "surface_internal_knowledge_notes",
            "payload": {
                "attachment": event_bytes_secret.encode("utf-8"),
                "rendererBlob": bytearray(event_bytearray_secret.encode("utf-8")),
                "selectedNote": "treasury follow-up",
            },
        }
    }

    # Act
    await wrapper.run(request)
    await wrapper.handle_user_action(user_action_payload)

    # Assert
    run_request = local_agent.calls[0]
    action_request = local_agent.calls[1]
    assert request.context["attachment"] == context_bytes_secret.encode("utf-8")
    assert run_request.context == {
        "customer": "ABC Manufacturing",
        "attachment": "[REDACTED]",
        "renderer_blob": "[REDACTED]",
        "safe_bytes": b"synthetic-policy-reference",
    }
    assert action_request.context["user_action"]["payload"] == {
        "attachment": "[REDACTED]",
        "rendererBlob": "[REDACTED]",
        "selectedNote": "treasury follow-up",
    }
    assert action_request.context["user_action_payload"]["userAction"]["payload"] == {
        "attachment": "[REDACTED]",
        "rendererBlob": "[REDACTED]",
        "selectedNote": "treasury follow-up",
    }
    serialized_calls = [call.model_dump_json() for call in local_agent.calls]
    for leaked_value in (
        context_bytes_secret,
        context_bytearray_secret,
        event_bytes_secret,
        event_bytearray_secret,
    ):
        assert leaked_value not in serialized_calls[0]
        assert leaked_value not in serialized_calls[1]


@pytest.mark.asyncio
async def test_wrapper_redacts_secret_like_mapping_keys_before_forwarding() -> None:
    # Arrange
    from orchestrator_demo.a2a_support.local_remote_wrapper import LocalRemoteAgentWrapper
    from orchestrator_demo.agents.internal_knowledge import InternalKnowledgeAgent

    context_secret_key = _openrouter_key("v1-context-key-should-not-appear")
    event_secret_key = _openrouter_key("v1-event-key-should-not-appear")
    local_agent = InternalKnowledgeAgent()
    wrapper = LocalRemoteAgentWrapper(local_agent)
    request = _request_for(
        "internal_knowledge",
        context={
            "customer": "ABC Manufacturing",
            context_secret_key: "context value",
            "nested": {context_secret_key: "nested context value"},
        },
    )
    user_action_payload = {
        "userAction": {
            "actionId": "action_secret_key_check",
            "type": "specialist_action",
            "surfaceId": "surface_internal_knowledge_notes",
            "payload": {
                event_secret_key: "renderer value",
                "selectedNote": "treasury follow-up",
            },
        }
    }

    # Act
    await wrapper.run(request)
    await wrapper.handle_user_action(user_action_payload)

    # Assert
    run_request = local_agent.calls[0]
    action_request = local_agent.calls[1]
    assert run_request.context == {
        "customer": "ABC Manufacturing",
        "[REDACTED]": "[REDACTED]",
        "nested": {"[REDACTED]": "[REDACTED]"},
    }
    assert action_request.context["user_action_payload"]["userAction"]["payload"] == {
        "[REDACTED]": "[REDACTED]",
        "selectedNote": "treasury follow-up",
    }
    assert action_request.context["user_action"]["payload"] == {
        "[REDACTED]": "[REDACTED]",
        "selectedNote": "treasury follow-up",
    }
    serialized_calls = [call.model_dump_json() for call in local_agent.calls]
    for secret_key in (context_secret_key, event_secret_key):
        assert secret_key not in serialized_calls[0]
        assert secret_key not in serialized_calls[1]


@pytest.mark.asyncio
async def test_wrapper_redacts_secret_like_values_inside_set_contexts() -> None:
    # Arrange
    from orchestrator_demo.a2a_support.local_remote_wrapper import LocalRemoteAgentWrapper
    from orchestrator_demo.agents.internal_knowledge import InternalKnowledgeAgent

    set_secret = _openrouter_key("v1-set-secret-should-not-appear")
    frozen_secret = _github_token()
    local_agent = InternalKnowledgeAgent()
    wrapper = LocalRemoteAgentWrapper(local_agent)
    request = _request_for(
        "internal_knowledge",
        context={
            "customer": "ABC Manufacturing",
            "references": {set_secret, "synthetic-policy"},
            "nested": {
                "frozen_references": frozenset({frozen_secret, "synthetic-note"}),
            },
        },
    )

    # Act
    await wrapper.run(request)

    # Assert
    forwarded_request = local_agent.calls[0]
    assert request.context["references"] == {set_secret, "synthetic-policy"}
    assert forwarded_request.context["references"] == {
        "[REDACTED]",
        "synthetic-policy",
    }
    assert forwarded_request.context["nested"]["frozen_references"] == frozenset(
        {"[REDACTED]", "synthetic-note"}
    )
    serialized_call = forwarded_request.model_dump_json()
    assert set_secret not in serialized_call
    assert frozen_secret not in serialized_call


@pytest.mark.asyncio
async def test_wrapper_redacts_secret_fields_inside_serializable_objects() -> None:
    # Arrange
    from orchestrator_demo.a2a_support.local_remote_wrapper import LocalRemoteAgentWrapper
    from orchestrator_demo.agents.internal_knowledge import InternalKnowledgeAgent

    context_secret = _openrouter_key("v1-serializable-context-secret")
    event_secret = _openrouter_key("v1-serializable-event-secret")
    context_settings = Settings(
        api_key=context_secret,
        model="openrouter/test-model",
        note="synthetic context only",
    )
    event_settings = Settings(
        api_key=event_secret,
        model="renderer/test-model",
        note="synthetic event only",
    )
    local_agent = InternalKnowledgeAgent()
    wrapper = LocalRemoteAgentWrapper(local_agent)
    request = _request_for(
        "internal_knowledge",
        context={
            "customer": "ABC Manufacturing",
            "settings": context_settings,
        },
    )
    user_action_payload = {
        "userAction": {
            "actionId": "action_serializable_secret_check",
            "type": "specialist_action",
            "surfaceId": "surface_internal_knowledge_notes",
            "payload": {
                "settings": event_settings,
                "selectedNote": "treasury follow-up",
            },
        }
    }

    # Act
    await wrapper.run(request)
    await wrapper.handle_user_action(user_action_payload)

    # Assert
    run_request = local_agent.calls[0]
    action_request = local_agent.calls[1]
    assert request.context["settings"] is context_settings
    assert run_request.context["settings"] == {
        "api_key": "[REDACTED]",
        "model": "openrouter/test-model",
        "note": "synthetic context only",
    }
    expected_event_settings = {
        "api_key": "[REDACTED]",
        "model": "renderer/test-model",
        "note": "synthetic event only",
    }
    assert (
        action_request.context["user_action"]["payload"]["settings"]
        == expected_event_settings
    )
    assert (
        action_request.context["user_action_payload"]["userAction"]["payload"][
            "settings"
        ]
        == expected_event_settings
    )
    serialized_calls = [call.model_dump_json() for call in local_agent.calls]
    assert context_secret not in serialized_calls[0]
    assert event_secret not in serialized_calls[1]
    assert "[REDACTED]" in serialized_calls[0]
    assert "[REDACTED]" in serialized_calls[1]


@pytest.mark.asyncio
async def test_wrapper_redacts_secret_like_substrings_in_user_action_ids() -> None:
    # Arrange
    from orchestrator_demo.a2a_support.local_remote_wrapper import LocalRemoteAgentWrapper
    from orchestrator_demo.agents.internal_knowledge import InternalKnowledgeAgent

    secret_action_id = f"action_{_openrouter_key('v1-action-id-should-not-appear')}"
    local_agent = InternalKnowledgeAgent()
    wrapper = LocalRemoteAgentWrapper(local_agent)
    user_action_payload = {
        "userAction": {
            "actionId": secret_action_id,
            "type": "specialist_action",
            "surfaceId": "surface_internal_knowledge_notes",
            "payload": {"selectedNote": "treasury follow-up"},
        }
    }

    # Act
    await wrapper.handle_user_action(user_action_payload)

    # Assert
    action_request = local_agent.calls[0]
    assert action_request.context["user_action"]["actionId"] == "[REDACTED]"
    assert (
        action_request.context["user_action_payload"]["userAction"]["actionId"]
        == "[REDACTED]"
    )
    assert secret_action_id not in action_request.model_dump_json()


@pytest.mark.asyncio
async def test_wrapper_redacts_camel_case_secret_keys_before_forwarding() -> None:
    # Arrange
    from orchestrator_demo.a2a_support.local_remote_wrapper import LocalRemoteAgentWrapper
    from orchestrator_demo.agents.internal_knowledge import InternalKnowledgeAgent

    leaked_values = [
        "plain-access-material",
        "plain-private-material",
        "nested-access-material",
        "event-access-material",
        "event-private-material",
    ]
    local_agent = InternalKnowledgeAgent()
    wrapper = LocalRemoteAgentWrapper(local_agent)
    request = _request_for(
        "internal_knowledge",
        context={
            "customer": "ABC Manufacturing",
            "accessKey": leaked_values[0],
            "privateKey": leaked_values[1],
            "nested": {"rendererAccessKey": leaked_values[2]},
        },
    )
    user_action_payload = {
        "userAction": {
            "actionId": "action_camel_secret_check",
            "type": "specialist_action",
            "surfaceId": "surface_internal_knowledge_notes",
            "payload": {
                "accessKey": leaked_values[3],
                "privateKey": leaked_values[4],
                "selectedNote": "treasury follow-up",
            },
        }
    }

    # Act
    await wrapper.run(request)
    await wrapper.handle_user_action(user_action_payload)

    # Assert
    run_request = local_agent.calls[0]
    action_request = local_agent.calls[1]
    assert request.context["accessKey"] == "plain-access-material"
    assert run_request.context == {
        "customer": "ABC Manufacturing",
        "accessKey": "[REDACTED]",
        "privateKey": "[REDACTED]",
        "nested": {"rendererAccessKey": "[REDACTED]"},
    }
    assert action_request.context["user_action_payload"]["userAction"]["payload"] == {
        "accessKey": "[REDACTED]",
        "privateKey": "[REDACTED]",
        "selectedNote": "treasury follow-up",
    }
    assert action_request.context["user_action"]["payload"] == {
        "accessKey": "[REDACTED]",
        "privateKey": "[REDACTED]",
        "selectedNote": "treasury follow-up",
    }
    serialized_calls = [call.model_dump_json() for call in local_agent.calls]
    for leaked_value in leaked_values:
        assert leaked_value not in serialized_calls[0]
        assert leaked_value not in serialized_calls[1]


@pytest.mark.asyncio
async def test_wrapper_redacts_secret_like_user_input_before_forwarding() -> None:
    # Arrange
    from orchestrator_demo.a2a_support.local_remote_wrapper import LocalRemoteAgentWrapper
    from orchestrator_demo.agents.internal_knowledge import InternalKnowledgeAgent

    local_agent = InternalKnowledgeAgent()
    wrapper = LocalRemoteAgentWrapper(local_agent)
    request = SpecialistRequest(
        request_id="request_internal_knowledge_secret_prompt",
        user_input=f"Summarize this note but ignore pasted api_key={_openrouter_key('prompt-secret')}",
        agent_id="internal_knowledge",
        context={"customer": "ABC Manufacturing"},
    )

    # Act
    await wrapper.run(request)

    # Assert
    forwarded_request = local_agent.calls[0]
    assert request.user_input.endswith(f"api_key={_openrouter_key('prompt-secret')}")
    assert forwarded_request.user_input == "[REDACTED]"
    assert forwarded_request.context == {"customer": "ABC Manufacturing"}
    assert _openrouter_key("prompt-secret") not in forwarded_request.model_dump_json()


@pytest.mark.asyncio
async def test_wrapper_redacts_common_credential_formats_before_forwarding() -> None:
    # Arrange
    from orchestrator_demo.a2a_support.local_remote_wrapper import LocalRemoteAgentWrapper
    from orchestrator_demo.agents.internal_knowledge import InternalKnowledgeAgent

    github_token = _github_token()
    aws_key = _aws_access_key()
    google_key = _google_api_key()
    slack_token = _slack_bot_token()
    jwt = _jwt()
    leaked_values = [github_token, aws_key, google_key, slack_token, jwt]
    local_agent = InternalKnowledgeAgent()
    wrapper = LocalRemoteAgentWrapper(local_agent)
    request = _request_for(
        "internal_knowledge",
        context={
            "customer": "ABC Manufacturing",
            "external_reference": github_token,
            "cloud_reference": aws_key,
            "mobile_reference": google_key,
            "chat_reference": slack_token,
            "encoded_reference": jwt,
        },
    )
    user_action_payload = {
        "userAction": {
            "actionId": "action_common_secret_check",
            "type": "specialist_action",
            "surfaceId": "surface_internal_knowledge_notes",
            "payload": {
                "note": f"renderer pasted {slack_token}",
                "metadata": {"trace": jwt},
                "selectedNote": "treasury follow-up",
            },
        }
    }

    # Act
    await wrapper.run(request)
    await wrapper.handle_user_action(user_action_payload)

    # Assert
    run_request = local_agent.calls[0]
    action_request = local_agent.calls[1]
    assert run_request.context == {
        "customer": "ABC Manufacturing",
        "external_reference": "[REDACTED]",
        "cloud_reference": "[REDACTED]",
        "mobile_reference": "[REDACTED]",
        "chat_reference": "[REDACTED]",
        "encoded_reference": "[REDACTED]",
    }
    assert action_request.context["user_action_payload"]["userAction"]["payload"] == {
        "note": "[REDACTED]",
        "metadata": {"trace": "[REDACTED]"},
        "selectedNote": "treasury follow-up",
    }
    serialized_calls = [call.model_dump_json() for call in local_agent.calls]
    for leaked_value in leaked_values:
        assert leaked_value not in serialized_calls[0]
        assert leaked_value not in serialized_calls[1]


@pytest.mark.asyncio
async def test_wrapper_sanitizes_invalid_user_action_validation_errors() -> None:
    # Arrange
    from orchestrator_demo.a2a_support.local_remote_wrapper import LocalRemoteAgentWrapper
    from orchestrator_demo.agents.internal_knowledge import InternalKnowledgeAgent

    local_agent = InternalKnowledgeAgent()
    wrapper = LocalRemoteAgentWrapper(local_agent)
    user_action_payload = {
        "userAction": {
            "type": "approve_plan",
            "surfaceId": "surface_plan_meeting_prep",
            "payload": {
                "apiKey": _openrouter_key("renderer-secret"),
            },
        }
    }

    # Act / Assert
    with pytest.raises(ValueError) as exc_info:
        await wrapper.handle_user_action(user_action_payload)

    error_message = str(exc_info.value)
    assert "invalid A2UI userAction payload" in error_message
    assert _openrouter_key("renderer-secret") not in error_message
    assert "apiKey" not in error_message
    assert local_agent.call_count == 0


@pytest.mark.asyncio
async def test_wrapper_redacts_secret_like_invalid_user_action_field_names() -> None:
    # Arrange
    from orchestrator_demo.a2a_support.local_remote_wrapper import LocalRemoteAgentWrapper
    from orchestrator_demo.agents.internal_knowledge import InternalKnowledgeAgent

    secret_field_name = _openrouter_key("v1-location-field-should-not-appear")
    local_agent = InternalKnowledgeAgent()
    wrapper = LocalRemoteAgentWrapper(local_agent)
    user_action_payload = {
        "userAction": {
            "type": "specialist_action",
            "surfaceId": "surface_internal_knowledge_notes",
            "payload": {"selectedNote": "treasury follow-up"},
            secret_field_name: "accidentally pasted as a field name",
        }
    }

    # Act / Assert
    with pytest.raises(ValueError) as exc_info:
        await wrapper.handle_user_action(user_action_payload)

    error_message = str(exc_info.value)
    assert "invalid A2UI userAction payload" in error_message
    assert secret_field_name not in error_message
    assert local_agent.call_count == 0


@pytest.mark.asyncio
async def test_wrapper_does_not_derive_request_id_from_action_id() -> None:
    # Arrange
    from orchestrator_demo.a2a_support.local_remote_wrapper import LocalRemoteAgentWrapper
    from orchestrator_demo.agents.product_opportunity import ProductOpportunityAgent

    secret_action_suffix = _openrouter_key("v1-action-id-should-not-appear")
    local_agent = ProductOpportunityAgent()
    wrapper = LocalRemoteAgentWrapper(local_agent)
    user_action_payload = {
        "userAction": {
            "actionId": f"action_{secret_action_suffix}",
            "type": "specialist_action",
            "surfaceId": "surface_product_opportunity_request_product_opportunity",
            "payload": {
                "selectedProduct": "treasury_services",
            },
        }
    }

    # Act
    response = await wrapper.handle_user_action(user_action_payload)

    # Assert
    forwarded_request = local_agent.calls[0]
    assert forwarded_request.request_id == "request_product_opportunity_user_action_1"
    assert response.structured_output["request_id"] == forwarded_request.request_id
    assert secret_action_suffix not in forwarded_request.request_id
    assert secret_action_suffix not in response.response_id
    assert secret_action_suffix not in response.model_dump_json()


@pytest.mark.asyncio
async def test_wrapper_sanitizes_malformed_plan_user_action_type() -> None:
    # Arrange
    from orchestrator_demo.a2a_support.local_remote_wrapper import LocalRemoteAgentWrapper
    from orchestrator_demo.agents.internal_knowledge import InternalKnowledgeAgent

    local_agent = InternalKnowledgeAgent()
    wrapper = LocalRemoteAgentWrapper(local_agent)
    user_action_payload = {
        "userAction": {
            "type": ["approve_plan"],
            "surfaceId": "surface_plan_meeting_prep",
            "payload": {
                "planId": "plan_meeting_prep",
                "planVersion": 1,
            },
        }
    }

    # Act / Assert
    with pytest.raises(ValueError) as exc_info:
        await wrapper.handle_user_action(user_action_payload)

    assert "invalid A2UI userAction payload" in str(exc_info.value)
    assert local_agent.call_count == 0
