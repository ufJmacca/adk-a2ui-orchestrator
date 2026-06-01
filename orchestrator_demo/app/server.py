"""Local HTTP transport for the orchestrator demo.

The server intentionally uses the Python standard library. The demo only needs
small harness-visible JSON endpoints and static renderer assets, so adding an
ASGI stack would increase the dependency surface without improving the slice.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from dataclasses import is_dataclass
import dataclasses
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading
from typing import Any
from urllib.parse import unquote, urlparse

from pydantic import BaseModel

from orchestrator_demo.a2a_support.transport import DataPart
from orchestrator_demo.contracts import SpecialistResponse, StatusEvent
from orchestrator_demo.orchestrator.service import (
    OrchestratorRequestResult,
    OrchestratorService,
    OrchestratorUserActionResult,
)


STATIC_ROOT = Path(__file__).resolve().parent / "static"


class LocalOrchestratorApp:
    """In-process application state behind the local HTTP endpoints."""

    def __init__(self, service: OrchestratorService | None = None) -> None:
        self._service = service or OrchestratorService()
        self._counter = 0
        self._latest_a2ui_parts: list[dict[str, Any]] = []
        self._latest_artifacts: dict[str, Any] = {}
        self._status_events: list[dict[str, Any]] = []

    def documented_endpoints(self) -> dict[str, dict[str, str]]:
        """Return the local transport contract exposed to the harness."""

        return {
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

    def submit_request(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        """Submit a natural-language user request to the orchestrator."""

        user_input = _required_text(payload, "input", "userInput", "user_input")
        task_id, context_id = self._next_transport_ids()
        result = asyncio.run(self._service.handle_user_request(user_input))
        response = self._request_response_payload(
            result,
            task_id=task_id,
            context_id=context_id,
        )
        self._remember_result_a2ui(response["a2uiParts"])
        self._latest_artifacts = response["artifacts"]
        self._record_request_status(result, task_id=task_id)
        return response

    def submit_user_action(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        """Submit a structured A2UI ``userAction`` event."""

        result = asyncio.run(self._service.handle_user_action(payload))
        response = self._user_action_response_payload(result)
        if response["a2uiParts"]:
            self._remember_result_a2ui(response["a2uiParts"])
        if response["artifacts"]:
            self._latest_artifacts = response["artifacts"]
        self._record_user_action_status(result)
        return response

    def status_list(self) -> dict[str, Any]:
        """Return accumulated app and graph status events."""

        return {"statusEvents": list(self._status_events)}

    def status_stream(self) -> str:
        """Return a small Server-Sent Events snapshot of current statuses."""

        lines: list[str] = []
        for event in self._status_events:
            lines.append("event: status")
            lines.append(f"data: {json.dumps(event, separators=(',', ':'))}")
            lines.append("")
        return "\n".join(lines) + ("\n" if lines else "")

    def latest_artifacts(self) -> dict[str, Any]:
        """Return the latest final artifacts and renderer A2UI parts."""

        return {
            "artifacts": _jsonable(self._latest_artifacts),
            "a2uiParts": list(self._latest_a2ui_parts),
        }

    def _next_transport_ids(self) -> tuple[str, str]:
        self._counter += 1
        return f"task_local_{self._counter}", f"ctx_local_{self._counter}"

    def _request_response_payload(
        self,
        result: OrchestratorRequestResult,
        *,
        task_id: str,
        context_id: str,
    ) -> dict[str, Any]:
        return {
            "taskId": task_id,
            "contextId": context_id,
            "path": result.path,
            "decision": _jsonable(result.decision),
            "approvalPlan": _jsonable(result.approval_plan),
            "approvalResult": _approval_result_payload(result.approval_result),
            "specialistResponses": _jsonable(result.specialist_responses),
            "a2uiParts": _data_parts_payload(result.a2ui_parts),
            "statusEvents": _status_events_payload(result.status_events),
            "artifacts": _jsonable(result.final_artifacts),
        }

    def _user_action_response_payload(
        self,
        result: OrchestratorUserActionResult,
    ) -> dict[str, Any]:
        status = _public_user_action_status(result.status)
        return {
            "status": status,
            "approvalResult": _approval_result_payload(result.approval_result),
            "surfaceRouteResult": _surface_route_payload(result),
            "specialistResponses": _jsonable(result.specialist_responses),
            "a2uiParts": _data_parts_payload(result.a2ui_parts),
            "statusEvents": _status_events_payload(result.status_events),
            "artifacts": _jsonable(result.final_artifacts),
        }

    def _record_request_status(
        self,
        result: OrchestratorRequestResult,
        *,
        task_id: str,
    ) -> None:
        if result.path == "plan_required" and result.approval_plan is not None:
            self._status_events.append(
                {
                    "status": "approval_required",
                    "message": "Plan approval is pending.",
                    "taskId": task_id,
                    "planId": result.approval_plan.plan_id,
                }
            )
            return

        self._status_events.append(
            {
                "status": result.path,
                "message": "Request handled.",
                "taskId": task_id,
                "planId": None,
            }
        )

    def _record_user_action_status(
        self,
        result: OrchestratorUserActionResult,
    ) -> None:
        if result.status_events:
            self._status_events.extend(_status_events_payload(result.status_events))
            return

        plan_id = (
            result.approval_result.plan_id
            if result.approval_result is not None
            else None
        )
        message = _user_action_status_message(result.status)
        self._status_events.append(
            {
                "status": result.status,
                "message": message,
                "taskId": None,
                "planId": plan_id,
            }
        )

    def _remember_result_a2ui(self, a2ui_parts: list[dict[str, Any]]) -> None:
        if a2ui_parts:
            self._latest_a2ui_parts = list(a2ui_parts)


class LocalHttpServer:
    """Lifecycle wrapper for tests, harnesses, and the module entrypoint."""

    def __init__(self, httpd: ThreadingHTTPServer) -> None:
        self._httpd = httpd
        self._thread: threading.Thread | None = None

    @property
    def base_url(self) -> str:
        host, port = self._httpd.server_address[:2]
        return f"http://{host}:{port}"

    def start_in_thread(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._httpd.serve_forever,
            name="orchestrator-demo-http",
            daemon=True,
        )
        self._thread.start()

    def serve_forever(self) -> None:
        self._httpd.serve_forever()

    def stop(self) -> None:
        self._httpd.shutdown()
        self._httpd.server_close()
        if self._thread is not None:
            self._thread.join(timeout=10)
            self._thread = None


class _LocalHttpServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(
        self,
        server_address: tuple[str, int],
        app: LocalOrchestratorApp,
    ) -> None:
        self.app = app
        super().__init__(server_address, _RequestHandler)


class _RequestHandler(BaseHTTPRequestHandler):
    server: _LocalHttpServer

    def log_message(self, _format: str, *_args: Any) -> None:
        """Suppress default request logging to avoid noisy harness output."""

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path == "/api":
            self._write_json({"endpoints": self.server.app.documented_endpoints()})
            return
        if path == "/api/status":
            self._write_json(self.server.app.status_list())
            return
        if path == "/api/status/stream":
            self._write_text(
                self.server.app.status_stream(),
                content_type="text/event-stream; charset=utf-8",
            )
            return
        if path == "/api/artifacts":
            self._write_json(self.server.app.latest_artifacts())
            return
        if path == "/" or path == "/index.html":
            self._write_static_file(STATIC_ROOT / "index.html")
            return
        if path.startswith("/static/"):
            self._write_static_file(_static_path(path))
            return
        self._write_json({"error": "not found"}, status=HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:  # noqa: N802
        try:
            payload = self._read_json_body()
            path = urlparse(self.path).path
            if path == "/api/request":
                self._write_json(self.server.app.submit_request(payload))
                return
            if path == "/api/user-action":
                self._write_json(self.server.app.submit_user_action(payload))
                return
            self._write_json({"error": "not found"}, status=HTTPStatus.NOT_FOUND)
        except ValueError as exc:
            self._write_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)

    def _read_json_body(self) -> dict[str, Any]:
        content_length = int(self.headers.get("content-length", "0"))
        raw_body = self.rfile.read(content_length)
        if not raw_body:
            return {}
        try:
            decoded = json.loads(raw_body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("request body must be valid JSON") from exc
        if not isinstance(decoded, dict):
            raise ValueError("request body must be a JSON object")
        return decoded

    def _write_static_file(self, path: Path) -> None:
        try:
            resolved_path = path.resolve(strict=True)
        except FileNotFoundError:
            self._write_json({"error": "not found"}, status=HTTPStatus.NOT_FOUND)
            return
        if STATIC_ROOT not in resolved_path.parents and resolved_path != STATIC_ROOT:
            self._write_json({"error": "not found"}, status=HTTPStatus.NOT_FOUND)
            return
        content_type = _content_type_for(resolved_path)
        body = resolved_path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("content-type", content_type)
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _write_json(
        self,
        payload: Mapping[str, Any],
        *,
        status: HTTPStatus = HTTPStatus.OK,
    ) -> None:
        body = json.dumps(_jsonable(payload), separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("content-type", "application/json; charset=utf-8")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _write_text(
        self,
        text: str,
        *,
        content_type: str,
        status: HTTPStatus = HTTPStatus.OK,
    ) -> None:
        body = text.encode("utf-8")
        self.send_response(status)
        self.send_header("content-type", content_type)
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def create_server(
    *,
    host: str = "127.0.0.1",
    port: int = 8000,
    app: LocalOrchestratorApp | None = None,
) -> LocalHttpServer:
    """Create a local HTTP server for the orchestrator app."""

    httpd = _LocalHttpServer((host, port), app or LocalOrchestratorApp())
    return LocalHttpServer(httpd)


def _required_text(payload: Mapping[str, Any], *keys: str) -> str:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value
    joined = ", ".join(keys)
    raise ValueError(f"request requires one non-empty text field: {joined}")


def _static_path(path: str) -> Path:
    relative = unquote(path.removeprefix("/static/"))
    return STATIC_ROOT / relative


def _content_type_for(path: Path) -> str:
    if path.suffix == ".html":
        return "text/html; charset=utf-8"
    if path.suffix == ".js":
        return "text/javascript; charset=utf-8"
    if path.suffix == ".css":
        return "text/css; charset=utf-8"
    return "application/octet-stream"


def _data_parts_payload(parts: tuple[DataPart, ...]) -> list[dict[str, Any]]:
    return [part.model_dump(by_alias=True, mode="json") for part in parts]


def _status_events_payload(events: tuple[StatusEvent, ...]) -> list[dict[str, Any]]:
    return [event.model_dump(mode="json") for event in events]


def _approval_result_payload(result: Any) -> dict[str, Any] | None:
    if result is None:
        return None
    return {
        "status": result.status,
        "planId": result.plan_id,
        "planVersion": result.plan_version,
        "approvedPlan": _jsonable(result.approved_plan),
        "reason": result.rejection_reason,
        "graphCreated": result.graph_created,
        "specialistsCalled": result.specialists_called,
    }


def _surface_route_payload(result: OrchestratorUserActionResult) -> dict[str, Any] | None:
    route_result = result.surface_route_result
    if route_result is None:
        return None
    owner = route_result.owner
    return {
        "status": route_result.status,
        "owner": None
        if owner is None
        else {
            "surfaceId": owner.surface_id,
            "ownerType": owner.owner_type,
            "ownerId": owner.owner_id,
            "planId": owner.plan_id,
        },
    }


def _user_action_status_message(status: str) -> str:
    return {
        "draft_updated": "Plan draft updated.",
        "rejected": "Plan rejected.",
        "approved": "Plan approved.",
        "routed": "A2UI event routed to surface owner.",
    }.get(status, "User action handled.")


def _public_user_action_status(status: str) -> str:
    if status == "forwarded":
        return "routed"
    return status


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, SpecialistResponse):
        return value.model_dump(mode="json")
    if is_dataclass(value) and not isinstance(value, type):
        return _jsonable(dataclasses.asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _jsonable(child) for key, child in value.items()}
    if isinstance(value, tuple | list):
        return [_jsonable(child) for child in value]
    return str(value)


__all__ = [
    "LocalHttpServer",
    "LocalOrchestratorApp",
    "create_server",
]
