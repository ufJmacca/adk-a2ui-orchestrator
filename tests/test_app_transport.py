from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
import json
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen


@contextmanager
def _running_server() -> Iterator[str]:
    from orchestrator_demo.app.server import create_server

    server = create_server(host="127.0.0.1", port=0)
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
        assert statuses[-1]["status"] == "final_response_ready"


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
