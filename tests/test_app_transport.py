from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor
import json
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen


@contextmanager
def _running_server(app: Any | None = None) -> Iterator[str]:
    from orchestrator_demo.app.server import create_server

    server = create_server(host="127.0.0.1", port=0, app=app)
    server.start_in_thread()
    try:
        yield server.base_url
    finally:
        server.stop()


def _request_json(
    base_url: str,
    path: str,
    *,
    method: str = "GET",
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    request = Request(
        f"{base_url}{path}",
        data=body,
        method=method,
        headers={"content-type": "application/json"},
    )
    try:
        with urlopen(request, timeout=10) as response:
            assert response.status == 200
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        detail = exc.read().decode("utf-8")
        raise AssertionError(f"{method} {path} failed: {exc.code} {detail}") from exc


def _request_text(base_url: str, path: str) -> tuple[str, str]:
    request = Request(
        f"{base_url}{path}",
        method="GET",
        headers={"accept": "text/event-stream"},
    )
    try:
        with urlopen(request, timeout=10) as response:
            assert response.status == 200
            content_type = response.headers.get("content-type", "")
            return response.read().decode("utf-8"), content_type
    except HTTPError as exc:
        detail = exc.read().decode("utf-8")
        raise AssertionError(f"GET {path} failed: {exc.code} {detail}") from exc


def _request_status(base_url: str, path: str) -> tuple[int, dict[str, Any]]:
    request = Request(f"{base_url}{path}", method="GET")
    try:
        with urlopen(request, timeout=10) as response:
            body = json.loads(response.read().decode("utf-8"))
            return response.status, body
    except HTTPError as exc:
        body = json.loads(exc.read().decode("utf-8"))
        return exc.code, body


def _sse_status_payloads(stream_body: str) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    current_event: str | None = None
    for line in stream_body.splitlines():
        if line.startswith("event: "):
            current_event = line.removeprefix("event: ")
        if line.startswith("data: ") and current_event == "status":
            events.append(json.loads(line.removeprefix("data: ")))
            current_event = None
    return events


def test_foreground_server_stop_closes_without_same_thread_shutdown() -> None:
    # Arrange
    from orchestrator_demo.app.server import LocalHttpServer

    class FakeHttpd:
        server_address = ("127.0.0.1", 8000)

        def __init__(self) -> None:
            self.shutdown_called = False
            self.server_close_called = False

        def shutdown(self) -> None:
            self.shutdown_called = True

        def server_close(self) -> None:
            self.server_close_called = True

    httpd = FakeHttpd()
    server = LocalHttpServer(httpd)  # type: ignore[arg-type]

    # Act
    server.stop()

    # Assert
    assert httpd.shutdown_called is False
    assert httpd.server_close_called is True


def test_static_directory_requests_return_not_found() -> None:
    # Arrange
    with _running_server() as base_url:
        # Act
        status, body = _request_status(base_url, "/static/")

        # Assert
        assert status == 404
        assert body == {"error": "not found"}


def _approve_event(response: dict[str, Any]) -> dict[str, Any]:
    plan = response["approvalPlan"]
    step_ids = [step["step_id"] for step in plan["steps"]]
    return {
        "userAction": {
            "type": "approve_plan",
            "surfaceId": plan["approval_surface_id"],
            "payload": {
                "planId": plan["plan_id"],
                "editedPlanVersion": plan["plan_version"],
                "approvedStepIds": step_ids,
            },
        }
    }


def _reject_event(response: dict[str, Any]) -> dict[str, Any]:
    plan = response["approvalPlan"]
    return {
        "userAction": {
            "type": "reject_plan",
            "surfaceId": plan["approval_surface_id"],
            "payload": {
                "planId": plan["plan_id"],
                "editedPlanVersion": plan["plan_version"],
                "reason": "Too broad; focus on credit risk only.",
            },
        }
    }


def _add_instruction_event(response: dict[str, Any]) -> dict[str, Any]:
    plan = response["approvalPlan"]
    return {
        "userAction": {
            "type": "add_instruction",
            "surfaceId": plan["approval_surface_id"],
            "payload": {
                "planId": plan["plan_id"],
                "editedPlanVersion": plan["plan_version"],
                "stepId": plan["steps"][0]["step_id"],
                "instruction": "Prioritize covenant follow-ups.",
            },
        }
    }


def _a2ui_surface_ids(response: dict[str, Any]) -> list[str]:
    surface_ids: list[str] = []
    for part in response.get("a2uiParts", []):
        data = part["data"]
        if "createSurface" in data:
            surface_ids.append(data["createSurface"]["surfaceId"])
        if "updateComponents" in data:
            surface_ids.append(data["updateComponents"]["surfaceId"])
    return surface_ids


def test_module_entrypoint_wires_configured_litellm_model_into_runtime(
    monkeypatch: Any,
) -> None:
    # Arrange
    from orchestrator_demo.app import __main__ as app_main

    class FakeModel:
        def __init__(self) -> None:
            self.prompts: list[str] = []

        async def generate_content_async(self, prompt: str) -> dict[str, Any]:
            self.prompts.append(prompt)
            return {
                "intents": ["meeting_prep", "relationship_summary"],
                "confidence": 0.99,
                "complexity": "complex",
                "required_agents": [
                    "relationship_summary",
                    "internal_knowledge",
                    "industry_research",
                    "synthesis",
                ],
                "rationale": "Fake configured LiteLLM model assessment.",
            }

    fake_model = FakeModel()
    monkeypatch.setattr(app_main, "build_litellm_model", lambda: fake_model)

    # Act
    app = app_main.build_runtime_app()
    response = app.submit_request({"input": "Use the configured runtime model."})

    # Assert
    assert len(fake_model.prompts) == 1
    assert "Use the configured runtime model." in fake_model.prompts[0]
    assert response["path"] == "plan_required"


def test_http_app_exposes_request_status_and_artifact_endpoints() -> None:
    # Arrange
    with _running_server() as base_url:
        request_payload = {
            "input": "Prepare me for tomorrow's meeting with ABC Manufacturing."
        }

        # Act
        submitted = _request_json(
            base_url,
            "/api/request",
            method="POST",
            payload=request_payload,
        )
        status_snapshot = _request_json(base_url, "/api/status")

        # Assert
        assert submitted["taskId"].startswith("task_")
        assert submitted["contextId"].startswith("ctx_")
        assert submitted["path"] == "plan_required"
        assert submitted["approvalPlan"]["selected_agents"] == [
            "relationship_summary",
            "internal_knowledge",
            "industry_research",
            "synthesis",
        ]
        assert len(submitted["a2uiParts"]) == 2
        assert _a2ui_surface_ids(submitted) == [
            submitted["approvalPlan"]["approval_surface_id"],
            submitted["approvalPlan"]["approval_surface_id"],
        ]
        assert submitted["artifacts"] == {}
        assert status_snapshot["statusEvents"][-1] == {
            "status": "approval_required",
            "message": "Plan approval is pending.",
            "taskId": submitted["taskId"],
            "planId": submitted["approvalPlan"]["plan_id"],
        }


def test_http_request_and_artifacts_drop_rejected_specialist_a2ui_payload() -> None:
    # Arrange
    from orchestrator_demo.agents import build_default_specialists
    from orchestrator_demo.app.server import LocalOrchestratorApp
    from orchestrator_demo.contracts import (
        IntentSuggestion,
        LlmIntentAssessment,
        SpecialistResponse,
    )
    from orchestrator_demo.orchestrator.service import OrchestratorService

    secret_value = "sk-testsecret1234567890"

    class SecretA2uiSpecialist:
        agent_id = "product_opportunity"
        call_count = 0

        async def handle(self, _request: Any) -> SpecialistResponse:
            self.call_count += 1
            return SpecialistResponse(
                response_id="response_product_opportunity_secret_a2ui",
                agent_id=self.agent_id,
                content="Product Opportunity Agent: product fit summary.",
                structured_output={"summary": "product fit summary"},
                a2ui_payload={
                    "version": "v0.9",
                    "updateComponents": {
                        "surfaceId": "surface_product_secret",
                        "components": [
                            {
                                "component": "Text",
                                "id": "root",
                                "text": "Unsafe recommendation details.",
                            }
                        ],
                    },
                    "OPENROUTER_API_KEY": secret_value,
                },
                surface_id="surface_product_secret",
            )

    class StaticSlmIntentClient:
        async def classify(self, _user_input: str) -> IntentSuggestion:
            return IntentSuggestion(intent="product_opportunity", confidence=0.95)

    class StaticIntentClassifier:
        async def assess(
            self,
            _user_input: str,
            _slm_suggestion: IntentSuggestion,
            *,
            available_agents: Any = None,
        ) -> LlmIntentAssessment:
            return LlmIntentAssessment(
                intents=["product_opportunity"],
                confidence=0.95,
                complexity="simple",
                required_agents=["product_opportunity"],
                rationale="Injected single-agent assessment.",
            )

    specialists = build_default_specialists()
    specialists["product_opportunity"] = SecretA2uiSpecialist()
    service = OrchestratorService(
        specialists=specialists,
        slm_client=StaticSlmIntentClient(),
        intent_classifier=StaticIntentClassifier(),
    )

    with _running_server(LocalOrchestratorApp(service=service)) as base_url:
        # Act
        submitted = _request_json(
            base_url,
            "/api/request",
            method="POST",
            payload={"input": "Suggest product opportunities."},
        )
        artifacts = _request_json(base_url, "/api/artifacts")

        # Assert
        assert submitted["path"] == "direct"
        assert submitted["a2uiParts"][0]["type"] == "text"
        assert submitted["a2uiParts"][0]["metadata"]["developerDiagnostic"][
            "fallback"
        ] == "text"
        assert submitted["specialistResponses"][0]["a2ui_payload"] is None
        assert artifacts["artifacts"]["final_response"]["a2ui_payload"] is None
        serialized = json.dumps({"submitted": submitted, "artifacts": artifacts})
        assert "OPENROUTER_API_KEY" not in serialized
        assert secret_value not in serialized


def test_transport_ids_are_unique_under_concurrent_request_allocation() -> None:
    # Arrange
    from orchestrator_demo.app.server import LocalOrchestratorApp

    app = LocalOrchestratorApp()

    # Act
    with ThreadPoolExecutor(max_workers=16) as executor:
        ids = list(executor.map(lambda _index: app._next_transport_ids(), range(1000)))

    # Assert
    task_ids = [task_id for task_id, _context_id in ids]
    context_ids = [context_id for _task_id, context_id in ids]
    assert len(set(task_ids)) == len(task_ids)
    assert len(set(context_ids)) == len(context_ids)


def test_http_app_documents_endpoint_contract_and_streams_status_events() -> None:
    # Arrange
    with _running_server() as base_url:
        submitted = _request_json(
            base_url,
            "/api/request",
            method="POST",
            payload={
                "input": "Prepare me for tomorrow's meeting with ABC Manufacturing."
            },
        )

        # Act
        contract = _request_json(base_url, "/api")
        stream_body, content_type = _request_text(base_url, "/api/status/stream")
        streamed_statuses = _sse_status_payloads(stream_body)

        # Assert
        assert contract["endpoints"] == {
            "submit_request": {
                "method": "POST",
                "path": "/api/request",
                "requestContentType": "application/json",
                "responseContentType": "application/json",
            },
            "submit_user_action": {
                "method": "POST",
                "path": "/api/user-action",
                "requestContentType": "application/json",
                "responseContentType": "application/json",
            },
            "fetch_status_list": {
                "method": "GET",
                "path": "/api/status",
                "responseContentType": "application/json",
            },
            "fetch_status_stream": {
                "method": "GET",
                "path": "/api/status/stream",
                "responseContentType": "text/event-stream",
            },
            "fetch_latest_artifacts": {
                "method": "GET",
                "path": "/api/artifacts",
                "responseContentType": "application/json",
            },
            "renderer": {
                "method": "GET",
                "path": "/",
                "responseContentType": "text/html",
            },
        }
        assert content_type.startswith("text/event-stream")
        assert stream_body.startswith("event: status\ndata: ")
        assert streamed_statuses[-1] == {
            "status": "approval_required",
            "message": "Plan approval is pending.",
            "taskId": submitted["taskId"],
            "planId": submitted["approvalPlan"]["plan_id"],
        }


def test_user_action_endpoint_applies_edit_reject_and_approve_events() -> None:
    # Arrange / Act / Assert: edit keeps the plan mutable and refreshes A2UI.
    with _running_server() as base_url:
        submitted = _request_json(
            base_url,
            "/api/request",
            method="POST",
            payload={
                "input": "Prepare me for tomorrow's meeting with ABC Manufacturing."
            },
        )
        edited = _request_json(
            base_url,
            "/api/user-action",
            method="POST",
            payload=_add_instruction_event(submitted),
        )
        assert edited["status"] == "draft_updated"
        assert edited["approvalResult"]["planId"] == submitted["approvalPlan"]["plan_id"]
        assert edited["a2uiParts"]
        assert _request_json(base_url, "/api/status")["statusEvents"][-1]["status"] == (
            "draft_updated"
        )

    # Arrange / Act / Assert: reject is captured through structured userAction.
    with _running_server() as base_url:
        submitted = _request_json(
            base_url,
            "/api/request",
            method="POST",
            payload={
                "input": "Prepare me for tomorrow's meeting with ABC Manufacturing."
            },
        )
        rejected = _request_json(
            base_url,
            "/api/user-action",
            method="POST",
            payload=_reject_event(submitted),
        )
        assert rejected["status"] == "rejected"
        assert rejected["approvalResult"]["reason"] == (
            "Too broad; focus on credit risk only."
        )

    # Arrange / Act / Assert: approve executes the dynamic graph and exposes artifacts.
    with _running_server() as base_url:
        submitted = _request_json(
            base_url,
            "/api/request",
            method="POST",
            payload={
                "input": "Prepare me for tomorrow's meeting with ABC Manufacturing."
            },
        )
        approved = _request_json(
            base_url,
            "/api/user-action",
            method="POST",
            payload=_approve_event(submitted),
        )
        artifacts = _request_json(base_url, "/api/artifacts")
        statuses = _request_json(base_url, "/api/status")["statusEvents"]

        assert approved["status"] == "approved"
        assert [event["status"] for event in approved["statusEvents"]] == [
            "plan_approved",
            "graph_created",
            "parallel_branch_started",
            "step_started",
            "step_completed",
            "parallel_branch_completed",
            "parallel_branch_started",
            "step_started",
            "step_completed",
            "parallel_branch_completed",
            "parallel_branch_started",
            "step_started",
            "step_completed",
            "parallel_branch_completed",
            "synthesis_started",
            "step_started",
            "step_completed",
            "final_response_ready",
        ]
        assert artifacts["artifacts"]["final_response"]["agent_id"] == "synthesis"
        assert artifacts["a2uiParts"] == []
        assert statuses[-1]["status"] == "final_response_ready"


def test_user_action_endpoint_returns_structured_graph_failure() -> None:
    # Arrange
    from orchestrator_demo.agents import build_default_specialists
    from orchestrator_demo.app.server import LocalOrchestratorApp
    from orchestrator_demo.orchestrator.service import OrchestratorService

    specialists = build_default_specialists()
    specialists.pop("relationship_summary")
    app = LocalOrchestratorApp(service=OrchestratorService(specialists=specialists))

    with _running_server(app) as base_url:
        submitted = _request_json(
            base_url,
            "/api/request",
            method="POST",
            payload={
                "input": "Prepare me for tomorrow's meeting with ABC Manufacturing."
            },
        )

        # Act
        failed = _request_json(
            base_url,
            "/api/user-action",
            method="POST",
            payload=_approve_event(submitted),
        )
        statuses = _request_json(base_url, "/api/status")["statusEvents"]

        # Assert
        assert failed["status"] == "failed"
        assert failed["approvalResult"]["failureReason"].startswith(
            "no specialist handler registered"
        )
        assert [event["status"] for event in failed["statusEvents"]] == [
            "plan_approved",
            "graph_created",
            "step_failed",
        ]
        assert statuses[-1]["status"] == "step_failed"


def test_downstream_a2ui_surface_is_exposed_and_specialist_action_routes_by_surface() -> None:
    # Arrange
    with _running_server() as base_url:
        submitted = _request_json(
            base_url,
            "/api/request",
            method="POST",
            payload={
                "input": "What product opportunities should I consider for a cafe business?"
            },
        )
        surface_id = next(
            surface_id
            for surface_id in _a2ui_surface_ids(submitted)
            if not surface_id.startswith("surface_plan_")
        )

        # Act
        handled = _request_json(
            base_url,
            "/api/user-action",
            method="POST",
            payload={
                "userAction": {
                    "type": "specialist_action",
                    "surfaceId": surface_id,
                    "payload": {
                        "agentId": "product_opportunity",
                        "action": "show_more_detail",
                    },
                }
            },
        )
        artifacts = _request_json(base_url, "/api/artifacts")
        statuses = _request_json(base_url, "/api/status")["statusEvents"]

        # Assert
        assert submitted["path"] == "direct"
        assert submitted["artifacts"]["final_response"]["agent_id"] == (
            "product_opportunity"
        )
        assert submitted["a2uiParts"][0]["metadata"]["mimeType"] == (
            "application/json+a2ui"
        )
        assert handled["status"] == "routed"
        assert handled["surfaceRouteResult"]["status"] == "routed"
        assert handled["specialistResponses"][0]["agent_id"] == "product_opportunity"
        assert len(handled["a2uiParts"]) == 2
        assert handled["a2uiParts"][0]["metadata"]["mimeType"] == (
            "application/json+a2ui"
        )
        assert artifacts["a2uiParts"] == handled["a2uiParts"]
        assert statuses[-1]["status"] == "routed"


def test_user_action_endpoint_includes_structured_surface_routing_errors() -> None:
    # Arrange
    with _running_server() as base_url:
        # Act
        response = _request_json(
            base_url,
            "/api/user-action",
            method="POST",
            payload={
                "userAction": {
                    "type": "specialist_action",
                    "surfaceId": "surface_unknown",
                    "payload": {"buttonId": "show_more_detail"},
                }
            },
        )

        # Assert
        assert response["status"] == "error"
        assert response["surfaceRouteResult"] == {
            "status": "error",
            "owner": None,
            "error": {
                "code": "unknown_surface",
                "surfaceId": None,
                "message": "No owner is registered for the requested A2UI surface.",
                "ownerInferenceAttempted": False,
            },
        }
