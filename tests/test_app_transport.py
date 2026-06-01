from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
import json
import threading
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest


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


def _request_error_json(
    base_url: str,
    path: str,
    *,
    method: str = "GET",
    payload: dict[str, Any] | None = None,
) -> tuple[int, dict[str, Any]]:
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    request = Request(
        f"{base_url}{path}",
        data=body,
        method=method,
        headers={"content-type": "application/json"},
    )
    try:
        with urlopen(request, timeout=10) as response:
            detail = response.read().decode("utf-8")
            raise AssertionError(
                f"{method} {path} unexpectedly succeeded: {detail}"
            )
    except HTTPError as exc:
        return exc.code, json.loads(exc.read().decode("utf-8"))


class FailOnceSpecialist:
    def __init__(self, delegate: Any) -> None:
        self._delegate = delegate
        self._failed = False

    @property
    def agent_id(self) -> str:
        return self._delegate.agent_id

    @property
    def call_count(self) -> int:
        return self._delegate.call_count

    @property
    def calls(self) -> list[Any]:
        return self._delegate.calls

    async def handle(self, request: Any) -> Any:
        if not self._failed:
            self._failed = True
            raise RuntimeError("transient specialist failure")
        return await self._delegate.handle(request)


class FailNewRequestService:
    def __init__(self, delegate: Any) -> None:
        self._delegate = delegate
        self.fail_next_request = False
        self.failure_message = "transient request failure"

    def clear_renderer_surfaces(self) -> Any:
        return self._delegate.clear_renderer_surfaces()

    def restore_renderer_surfaces(self, owners_by_surface_id: Any) -> None:
        self._delegate.restore_renderer_surfaces(owners_by_surface_id)

    async def handle_user_request(self, user_input: str) -> Any:
        if self.fail_next_request:
            self.fail_next_request = False
            raise RuntimeError(self.failure_message)
        return await self._delegate.handle_user_request(user_input)

    async def handle_user_action(self, payload: Any) -> Any:
        return await self._delegate.handle_user_action(payload)


class DeltaOnlyUserActionAdapter:
    def __init__(self) -> None:
        self.surface_id: str | None = None

    async def handle_user_action(self, _payload: Any) -> dict[str, Any]:
        assert self.surface_id is not None
        return {
            "response_id": "response_product_opportunity_delta",
            "agent_id": "product_opportunity",
            "content": "Product Opportunity Agent: detail updated.",
            "structured_output": {"status": "handled"},
            "surface_id": self.surface_id,
            "a2ui_payload": [
                {
                    "version": "v0.9",
                    "updateComponents": {
                        "surfaceId": self.surface_id,
                        "components": [
                            {
                                "id": "component_product_opportunity_summary",
                                "component": "Text",
                                "text": "Delta detail ready.",
                            }
                        ],
                    },
                }
            ],
        }


class FailingUserActionAdapter:
    async def handle_user_action(self, _payload: Any) -> dict[str, Any]:
        raise ValueError(
            "provider failed with OPENROUTER_API_KEY="
            "sk-live-user-action-secret-123456789"
        )


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


def test_http_app_clears_renderer_parts_when_next_request_has_no_a2ui() -> None:
    # Arrange
    with _running_server() as base_url:
        proposed = _request_json(
            base_url,
            "/api/request",
            method="POST",
            payload={
                "input": (
                    "Prepare me for tomorrow's meeting with ABC Manufacturing."
                )
            },
        )
        assert _request_json(base_url, "/api/artifacts")["a2uiParts"]

        # Act
        submitted = _request_json(
            base_url,
            "/api/request",
            method="POST",
            payload={"input": "Show internal notes from CRM."},
        )
        artifacts = _request_json(base_url, "/api/artifacts")
        replayed = _request_json(
            base_url,
            "/api/user-action",
            method="POST",
            payload=_approve_event(proposed),
        )

        # Assert
        assert submitted["path"] == "direct"
        assert submitted["a2uiParts"] == []
        assert artifacts["artifacts"]["final_response"]["agent_id"] == (
            "internal_knowledge"
        )
        assert artifacts["a2uiParts"] == []
        assert replayed["status"] == "error"
        assert replayed["surfaceRouteResult"]["error"]["code"] == "unknown_surface"


def test_http_app_preserves_surface_ownership_when_new_request_fails() -> None:
    # Arrange
    from orchestrator_demo.app.server import LocalOrchestratorApp
    from orchestrator_demo.orchestrator.service import OrchestratorService

    service = FailNewRequestService(OrchestratorService())
    app = LocalOrchestratorApp(service=service)
    proposed = app.submit_request(
        {"input": "Prepare me for tomorrow's meeting with ABC Manufacturing."}
    )

    # Act
    service.fail_next_request = True
    with pytest.raises(RuntimeError, match="transient request failure"):
        app.submit_request({"input": "Prepare follow-up notes."})
    edited = app.submit_user_action(_add_instruction_event(proposed))

    # Assert
    assert edited["status"] == "draft_updated"
    assert edited["approvalResult"]["planId"] == proposed["approvalPlan"]["plan_id"]
    assert edited["surfaceRouteResult"] is None


def test_request_endpoint_returns_json_for_request_runtime_failure() -> None:
    # Arrange
    from orchestrator_demo.app.server import LocalOrchestratorApp
    from orchestrator_demo.orchestrator.service import OrchestratorService

    service = FailNewRequestService(OrchestratorService())
    service.fail_next_request = True
    service.failure_message = (
        "provider failed with OPENROUTER_API_KEY=sk-live-request-secret-123456789"
    )
    app = LocalOrchestratorApp(service=service)

    with _running_server(app=app) as base_url:
        # Act
        status, failed = _request_error_json(
            base_url,
            "/api/request",
            method="POST",
            payload={"input": "Prepare follow-up notes."},
        )
        retried = _request_json(
            base_url,
            "/api/request",
            method="POST",
            payload={"input": "Prepare follow-up notes."},
        )

        # Assert
        assert status == 500
        assert failed == {
            "status": "error",
            "error": {
                "code": "runtime_error",
                "message": (
                    "Request failed while processing. Retry after checking local "
                    "service logs."
                ),
                "retryable": True,
            },
            "statusEvents": [],
        }
        assert "OPENROUTER_API_KEY" not in json.dumps(failed)
        assert "sk-live-request-secret-123456789" not in json.dumps(failed)
        assert retried["taskId"].startswith("task_")


def test_http_app_retires_approval_surface_when_next_request_has_direct_a2ui() -> None:
    # Arrange
    with _running_server() as base_url:
        proposed = _request_json(
            base_url,
            "/api/request",
            method="POST",
            payload={
                "input": (
                    "Prepare me for tomorrow's meeting with ABC Manufacturing."
                )
            },
        )
        assert _request_json(base_url, "/api/artifacts")["a2uiParts"]

        # Act
        submitted = _request_json(
            base_url,
            "/api/request",
            method="POST",
            payload={
                "input": (
                    "What product opportunities should I consider for a cafe "
                    "business?"
                )
            },
        )
        artifacts = _request_json(base_url, "/api/artifacts")
        replayed = _request_json(
            base_url,
            "/api/user-action",
            method="POST",
            payload=_approve_event(proposed),
        )

        # Assert
        assert submitted["path"] == "direct"
        assert submitted["a2uiParts"]
        assert artifacts["a2uiParts"] == submitted["a2uiParts"]
        assert all(
            not surface_id.startswith("surface_plan_")
            for surface_id in _a2ui_surface_ids(submitted)
        )
        assert replayed["status"] == "error"
        assert replayed["surfaceRouteResult"]["error"]["code"] == "unknown_surface"


def test_static_directories_return_json_404() -> None:
    # Arrange
    with _running_server() as base_url:
        # Act
        status, payload = _request_error_json(base_url, "/static/")

        # Assert
        assert status == 404
        assert payload == {"error": "not found"}


def test_http_server_stop_before_start_does_not_block() -> None:
    # Arrange
    from orchestrator_demo.app.server import create_server

    stopped = threading.Event()

    def stop_unstarted_server() -> None:
        server = create_server(host="127.0.0.1", port=0)
        server.stop()
        stopped.set()

    # Act
    thread = threading.Thread(target=stop_unstarted_server, daemon=True)
    thread.start()
    thread.join(timeout=2)

    # Assert
    assert stopped.is_set()
    assert not thread.is_alive()


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
        assert (
            edited["approvalResult"]["planId"] == submitted["approvalPlan"]["plan_id"]
        )
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
        assert rejected["a2uiParts"] == [
            {
                "type": "data",
                "data": {
                    "version": "v0.9",
                    "deleteSurface": {
                        "surfaceId": submitted["approvalPlan"]["approval_surface_id"],
                    },
                },
                "metadata": {"mimeType": "application/json+a2ui"},
            }
        ]

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
        assert approved["a2uiParts"] == [
            {
                "type": "data",
                "data": {
                    "version": "v0.9",
                    "deleteSurface": {
                        "surfaceId": submitted["approvalPlan"]["approval_surface_id"],
                    },
                },
                "metadata": {"mimeType": "application/json+a2ui"},
            }
        ]
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


def test_user_action_endpoint_returns_json_for_graph_execution_failure() -> None:
    # Arrange
    from orchestrator_demo.agents import build_default_specialists
    from orchestrator_demo.app.server import LocalOrchestratorApp
    from orchestrator_demo.orchestrator.service import OrchestratorService

    specialists = build_default_specialists()
    specialists["relationship_summary"] = FailOnceSpecialist(
        specialists["relationship_summary"]
    )
    app = LocalOrchestratorApp(
        service=OrchestratorService(specialists=specialists)
    )

    with _running_server(app=app) as base_url:
        submitted = _request_json(
            base_url,
            "/api/request",
            method="POST",
            payload={
                "input": "Prepare me for tomorrow's meeting with ABC Manufacturing."
            },
        )

        # Act
        status, failed = _request_error_json(
            base_url,
            "/api/user-action",
            method="POST",
            payload=_approve_event(submitted),
        )
        failure_statuses = _request_json(base_url, "/api/status")["statusEvents"]
        stream_body, _content_type = _request_text(base_url, "/api/status/stream")
        streamed_statuses = _sse_status_payloads(stream_body)
        edited = _request_json(
            base_url,
            "/api/user-action",
            method="POST",
            payload=_add_instruction_event(submitted),
        )

        # Assert
        assert status == 500
        assert failed["status"] == "error"
        assert failed["error"]["code"] == "graph_execution_failed"
        assert failed["error"]["message"] == (
            "Graph execution failed. The draft approval remains retryable."
        )
        assert failed["error"]["retryable"] is True
        assert failed["statusEvents"]
        assert "step_failed" in [
            event["status"] for event in failed["statusEvents"]
        ]
        assert "transient specialist failure" not in json.dumps(failed)
        assert failure_statuses[-len(failed["statusEvents"]) :] == (
            failed["statusEvents"]
        )
        assert streamed_statuses[-len(failed["statusEvents"]) :] == (
            failed["statusEvents"]
        )
        assert edited["status"] == "draft_updated"


def test_user_action_endpoint_maps_stale_approval_version_to_client_error() -> None:
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
        edited = _request_json(
            base_url,
            "/api/user-action",
            method="POST",
            payload=_add_instruction_event(submitted),
        )

        # Act
        status, failed = _request_error_json(
            base_url,
            "/api/user-action",
            method="POST",
            payload=_approve_event(submitted),
        )

        # Assert
        assert edited["status"] == "draft_updated"
        assert status == 409
        assert failed == {
            "status": "error",
            "error": {
                "code": "approval_state_conflict",
                "message": (
                    "User action cannot be applied to the current approval state. "
                    "Refresh the plan and submit a current structured userAction."
                ),
                "retryable": False,
            },
            "statusEvents": [],
        }


def test_user_action_endpoint_sanitizes_specialist_handler_failures() -> None:
    # Arrange
    from orchestrator_demo.app.server import LocalOrchestratorApp
    from orchestrator_demo.orchestrator.service import OrchestratorService

    app = LocalOrchestratorApp(
        service=OrchestratorService(
            specialist_user_action_adapters={
                "product_opportunity": FailingUserActionAdapter(),
            }
        )
    )

    with _running_server(app=app) as base_url:
        submitted = _request_json(
            base_url,
            "/api/request",
            method="POST",
            payload={
                "input": (
                    "What product opportunities should I consider for a cafe "
                    "business?"
                )
            },
        )
        surface_id = next(
            surface_id
            for surface_id in _a2ui_surface_ids(submitted)
            if not surface_id.startswith("surface_plan_")
        )

        # Act
        status, failed = _request_error_json(
            base_url,
            "/api/user-action",
            method="POST",
            payload={
                "userAction": {
                    "type": "specialist_action",
                    "surfaceId": surface_id,
                    "payload": {"action": "show_more_detail"},
                }
            },
        )

        # Assert
        assert status == 500
        assert failed == {
            "status": "error",
            "error": {
                "code": "runtime_error",
                "message": (
                    "Request failed while processing. Retry after checking local "
                    "service logs."
                ),
                "retryable": True,
            },
            "statusEvents": [],
        }
        assert "OPENROUTER_API_KEY" not in json.dumps(failed)
        assert "sk-live-user-action-secret-123456789" not in json.dumps(failed)


def test_artifacts_replay_merges_incremental_specialist_a2ui_updates() -> None:
    # Arrange
    from orchestrator_demo.app.server import LocalOrchestratorApp
    from orchestrator_demo.orchestrator.service import OrchestratorService

    adapter = DeltaOnlyUserActionAdapter()
    app = LocalOrchestratorApp(
        service=OrchestratorService(
            specialist_user_action_adapters={
                "product_opportunity": adapter,
            }
        )
    )

    with _running_server(app=app) as base_url:
        submitted = _request_json(
            base_url,
            "/api/request",
            method="POST",
            payload={
                "input": (
                    "What product opportunities should I consider for a cafe "
                    "business?"
                )
            },
        )
        surface_id = next(
            surface_id
            for surface_id in _a2ui_surface_ids(submitted)
            if not surface_id.startswith("surface_plan_")
        )
        adapter.surface_id = surface_id

        # Act
        _request_json(
            base_url,
            "/api/user-action",
            method="POST",
            payload={
                "userAction": {
                    "type": "specialist_action",
                    "surfaceId": surface_id,
                    "payload": {"action": "show_more_detail"},
                }
            },
        )
        artifacts = _request_json(base_url, "/api/artifacts")

        # Assert
        replay_payloads = [part["data"] for part in artifacts["a2uiParts"]]
        assert replay_payloads[0]["createSurface"]["surfaceId"] == surface_id
        replay_update = next(
            payload["updateComponents"]
            for payload in replay_payloads
            if "updateComponents" in payload
        )
        component_text_by_id = {
            component["id"]: component.get("text")
            for component in replay_update["components"]
        }
        assert "root" in component_text_by_id
        assert component_text_by_id["component_product_opportunity_summary"] == (
            "Delta detail ready."
        )


def test_artifacts_replay_merges_incremental_data_model_path_value_updates() -> None:
    # Arrange
    from orchestrator_demo.app.server import LocalOrchestratorApp

    app = LocalOrchestratorApp()
    surface_id = "surface_data_model_replay"
    metadata = {"mimeType": "application/json+a2ui"}

    # Act
    app._merge_result_a2ui(
        [
            {
                "type": "data",
                "data": {
                    "version": "v0.9",
                    "createSurface": {
                        "surfaceId": surface_id,
                        "catalogId": (
                            "https://a2ui.org/specification/v0_9/"
                            "basic_catalog.json"
                        ),
                    },
                },
                "metadata": metadata,
            },
            {
                "type": "data",
                "data": {
                    "version": "v0.9",
                    "updateDataModel": {
                        "surfaceId": surface_id,
                        "path": "/",
                        "value": {
                            "customer": {"name": "ABC Manufacturing"},
                        },
                    },
                },
                "metadata": metadata,
            },
            {
                "type": "data",
                "data": {
                    "version": "v0.9",
                    "updateDataModel": {
                        "surfaceId": surface_id,
                        "path": "/customer/segment",
                        "value": "commercial",
                    },
                },
                "metadata": metadata,
            },
            {
                "type": "data",
                "data": {
                    "version": "v0.9",
                    "updateDataModel": {
                        "surfaceId": surface_id,
                        "path": "/customer/name",
                        "value": "XYZ Supplies",
                    },
                },
                "metadata": metadata,
            },
            {
                "type": "data",
                "data": {
                    "version": "v0.9",
                    "updateDataModel": {
                        "surfaceId": surface_id,
                        "path": "/customer/segment",
                    },
                },
                "metadata": metadata,
            },
        ]
    )
    artifacts = app.latest_artifacts()

    # Assert
    replay_payloads = [part["data"] for part in artifacts["a2uiParts"]]
    replay_update = next(
        payload["updateDataModel"]
        for payload in replay_payloads
        if "updateDataModel" in payload
    )
    assert replay_update == {
        "surfaceId": surface_id,
        "path": "/",
        "value": {
            "customer": {
                "name": "XYZ Supplies",
            },
        },
    }


def test_downstream_a2ui_surface_is_exposed_and_specialist_action_routes_by_surface() -> (
    None
):
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
        status_snapshot = _request_json(base_url, "/api/status")

        # Assert
        assert submitted["path"] == "direct"
        assert submitted["artifacts"]["final_response"]["agent_id"] == (
            "product_opportunity"
        )
        assert handled["status"] == "routed"
        assert handled["specialistResponses"][0]["agent_id"] == "product_opportunity"
        assert artifacts["a2uiParts"][0]["metadata"]["mimeType"] == (
            "application/json+a2ui"
        )
        assert status_snapshot["statusEvents"][-1] == {
            "status": "routed",
            "message": "A2UI event routed to surface owner.",
            "taskId": None,
            "planId": None,
        }


def test_user_action_endpoint_returns_structured_routing_errors() -> None:
    # Arrange
    with _running_server() as base_url:
        unknown_surface_event = {
            "userAction": {
                "type": "specialist_action",
                "surfaceId": "surface_unknown",
                "payload": {"buttonId": "show_more_detail"},
            }
        }

        # Act
        response = _request_json(
            base_url,
            "/api/user-action",
            method="POST",
            payload=unknown_surface_event,
        )

        # Assert
        assert response["status"] == "error"
        assert response["surfaceRouteResult"]["status"] == "error"
        assert response["surfaceRouteResult"]["error"] == {
            "code": "unknown_surface",
            "surfaceId": None,
            "message": "No owner is registered for the requested A2UI surface.",
            "ownerInferenceAttempted": False,
        }
        assert response["statusEvents"] == [
            {
                "status": "error",
                "message": "No owner is registered for the requested A2UI surface.",
                "taskId": None,
                "planId": None,
                "details": {
                    "code": "unknown_surface",
                    "surfaceId": None,
                    "ownerInferenceAttempted": False,
                },
            }
        ]

        status_snapshot = _request_json(base_url, "/api/status")
        assert status_snapshot["statusEvents"][-1] == response["statusEvents"][0]
