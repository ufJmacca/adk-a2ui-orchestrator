from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import time
import urllib.error
import urllib.request
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

import httpx
import pytest
from a2a import types as a2a_types
from a2a.client import A2AClient


A2UI_MIME_TYPE = "application/json+a2ui"
DETERMINISTIC_MODEL_ENV = "ORCHESTRATOR_DEMO_DETERMINISTIC_MODEL"
KNOWN_SECRET_VALUE = "sk-or-v1-a2a-integration-secret-should-not-appear"
KNOWN_SECRET_ASSIGNMENT = f"OPENROUTER_API_KEY={KNOWN_SECRET_VALUE}"
KNOWN_SECRET_STRINGS = (KNOWN_SECRET_VALUE, KNOWN_SECRET_ASSIGNMENT)


@dataclass(frozen=True)
class RunningAdkServer:
    base_url: str
    a2a_url: str
    session_db: Path
    artifact_dir: Path


@pytest.fixture
def adk_a2a_server(
    tmp_path: Path,
    unused_tcp_port: int,
    repository_root: Path,
) -> Iterable[RunningAdkServer]:
    host = "127.0.0.1"
    port = unused_tcp_port
    session_db = tmp_path / "sessions.sqlite"
    artifact_dir = tmp_path / "artifacts"
    server_log = tmp_path / "adk-api-server.log"
    base_url = f"http://{host}:{port}"
    card_url = f"{base_url}/a2a/orchestrator/.well-known/agent-card.json"
    env = {
        **os.environ,
        DETERMINISTIC_MODEL_ENV: "1",
        "OPENROUTER_API_KEY": KNOWN_SECRET_VALUE,
        "LLM_MODEL": "openrouter/a2a-integration-test",
        "PYTHONUNBUFFERED": "1",
    }
    command = [
        "uv",
        "run",
        "adk",
        "api_server",
        "--a2a",
        "--with_ui",
        "orchestrator_demo",
        "--host",
        host,
        "--port",
        str(port),
        "--session_service_uri",
        f"sqlite:///{session_db}",
        "--artifact_service_uri",
        f"file://{artifact_dir}",
    ]

    # Arrange
    with server_log.open("w", encoding="utf-8") as output:
        process = subprocess.Popen(
            command,
            cwd=repository_root,
            env=env,
            stdout=output,
            stderr=subprocess.STDOUT,
            text=True,
        )

    try:
        _wait_for_agent_card(card_url, process, server_log)
        yield RunningAdkServer(
            base_url=base_url,
            a2a_url=f"{base_url}/a2a/orchestrator",
            session_db=session_db,
            artifact_dir=artifact_dir,
        )
    finally:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)


@pytest.mark.asyncio
async def test_adk_api_server_a2a_card_and_complex_message_flow(
    adk_a2a_server: RunningAdkServer,
) -> None:
    # Arrange
    async with httpx.AsyncClient(timeout=30) as httpx_client:
        client = A2AClient(httpx_client=httpx_client, url=adk_a2a_server.a2a_url)

        # Act
        card = await client.get_card()
        response = await client.send_message(
            a2a_types.SendMessageRequest(
                id=f"request-{uuid4()}",
                params=a2a_types.MessageSendParams(
                    configuration=a2a_types.MessageSendConfiguration(
                        acceptedOutputModes=[
                            "application/json",
                            A2UI_MIME_TYPE,
                            "text/plain",
                        ],
                        blocking=True,
                    ),
                    message=a2a_types.Message(
                        messageId=f"message-{uuid4()}",
                        role=a2a_types.Role.user,
                        parts=[
                            a2a_types.Part(
                                root=a2a_types.TextPart(
                                    text=(
                                        "Research this prospect and give me risks, "
                                        "opportunities, and talking points."
                                    )
                                )
                            )
                        ],
                    ),
                ),
            )
        )

    # Assert
    assert card.name == "orchestrator"
    assert str(card.url) == "http://127.0.0.1:8000/a2a/orchestrator"
    assert A2UI_MIME_TYPE in card.default_output_modes

    payload = response.root
    assert isinstance(payload, a2a_types.SendMessageSuccessResponse)
    data_parts = _data_parts_from_result(payload.result)
    normalized_response = _orchestrator_function_response(data_parts)

    json.dumps(normalized_response, sort_keys=True)
    assert normalized_response["status"] == "plan_required"
    assert normalized_response["path"] == "plan_required"
    assert normalized_response["planId"] == normalized_response["plan"]["planId"]
    assert normalized_response["approvalSurfaceId"] == (
        normalized_response["plan"]["approvalSurfaceId"]
    )
    assert normalized_response["stepIds"] == [
        step["stepId"] for step in normalized_response["plan"]["steps"]
    ]

    a2ui_data_parts = [
        part
        for part in data_parts
        if (part.metadata or {}).get("mimeType") == A2UI_MIME_TYPE
    ]
    _assert_a2ui_parts_render_returned_approval_plan(
        a2ui_data_parts,
        normalized_response,
    )

    serialized_response = response.model_dump_json(
        by_alias=True,
        exclude_none=True,
    )
    _assert_secret_strings_absent(serialized_response)
    assert adk_a2a_server.session_db.exists()
    assert adk_a2a_server.artifact_dir.is_dir()
    persisted_orchestrator_session = _persisted_orchestrator_session_for_plan(
        adk_a2a_server.session_db,
        normalized_response["planId"],
    )
    _assert_persisted_session_has_pending_draft_plan(
        persisted_orchestrator_session,
        normalized_response,
    )
    _assert_secret_strings_absent(
        json.dumps(persisted_orchestrator_session, sort_keys=True),
    )
    _assert_secret_strings_absent_from_files(
        [adk_a2a_server.session_db.parent, adk_a2a_server.artifact_dir]
    )


def test_default_adk_runtime_store_is_gitignored(repository_root: Path) -> None:
    # Arrange
    gitignore_path = repository_root / ".gitignore"

    # Act
    ignore_rules = {
        line.strip()
        for line in gitignore_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    }

    # Assert
    assert ".adk/" in ignore_rules


def _wait_for_agent_card(
    card_url: str,
    process: subprocess.Popen[str],
    server_log: Path,
) -> None:
    deadline = time.monotonic() + 60
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        if process.poll() is not None:
            pytest.fail(
                "ADK API server exited before A2A card became available.\n"
                f"Exit code: {process.returncode}\n"
                f"Server log:\n{server_log.read_text(encoding='utf-8')}"
            )
        try:
            with urllib.request.urlopen(card_url, timeout=1) as response:
                if response.status == 200:
                    return
        except (OSError, urllib.error.URLError) as exc:
            last_error = exc
        time.sleep(0.25)

    pytest.fail(
        "Timed out waiting for ADK API server A2A card.\n"
        f"Last error: {last_error!r}\n"
        f"Server log:\n{server_log.read_text(encoding='utf-8')}"
    )


def _data_parts_from_result(
    result: a2a_types.Task | a2a_types.Message,
) -> list[a2a_types.DataPart]:
    if isinstance(result, a2a_types.Message):
        return _data_parts_from_message(result)

    data_parts: list[a2a_types.DataPart] = []
    if result.status.message is not None:
        data_parts.extend(_data_parts_from_message(result.status.message))
    for message in result.history or []:
        data_parts.extend(_data_parts_from_message(message))
    for artifact in result.artifacts or []:
        for part in artifact.parts:
            if isinstance(part.root, a2a_types.DataPart):
                data_parts.append(part.root)
    return data_parts


def _data_parts_from_message(message: a2a_types.Message) -> list[a2a_types.DataPart]:
    return [
        part.root
        for part in message.parts
        if isinstance(part.root, a2a_types.DataPart)
    ]


def _orchestrator_function_response(
    data_parts: Iterable[a2a_types.DataPart],
) -> dict[str, Any]:
    for part in data_parts:
        metadata = part.metadata or {}
        if (
            metadata.get("adk_type") == "function_response"
            and part.data.get("name") == "submit_orchestrator_request"
            and isinstance(part.data.get("response"), dict)
        ):
            return part.data["response"]
    pytest.fail("A2A response did not include normalized orchestrator JSON fallback")


def _assert_a2ui_parts_render_returned_approval_plan(
    a2ui_data_parts: list[a2a_types.DataPart],
    normalized_response: dict[str, Any],
) -> None:
    assert a2ui_data_parts
    a2ui_payloads = [
        part.data for part in a2ui_data_parts if isinstance(part.data, dict)
    ]
    assert len(a2ui_payloads) == len(a2ui_data_parts)

    plan_id = normalized_response["planId"]
    plan_version = normalized_response["planVersion"]
    approval_surface_id = normalized_response["approvalSurfaceId"]
    step_ids = normalized_response["stepIds"]

    create_surfaces = [
        payload["createSurface"]
        for payload in a2ui_payloads
        if isinstance(payload.get("createSurface"), dict)
    ]
    assert any(
        create_surface.get("surfaceId") == approval_surface_id
        for create_surface in create_surfaces
    )

    approval_updates = [
        payload["updateComponents"]
        for payload in a2ui_payloads
        if isinstance(payload.get("updateComponents"), dict)
        and payload["updateComponents"].get("surfaceId") == approval_surface_id
    ]
    assert approval_updates

    components = [
        component
        for update in approval_updates
        for component in update.get("components", [])
        if isinstance(component, dict)
    ]
    metadata_texts = [
        component["text"]
        for component in components
        if component.get("component") == "Text"
        and isinstance(component.get("text"), str)
    ]
    assert any(f"planId: {plan_id}" in text for text in metadata_texts)
    assert any(f"planVersion: {plan_version}" in text for text in metadata_texts)
    assert all(
        any(step_id in text for text in metadata_texts)
        for step_id in step_ids
    )

    button_contexts = [
        component["action"]["event"]["context"]
        for component in components
        if component.get("component") == "Button"
        and isinstance(component.get("action"), dict)
        and isinstance(component["action"].get("event"), dict)
        and isinstance(component["action"]["event"].get("context"), dict)
    ]
    assert any(
        context.get("type") == "approve_plan"
        and context.get("surfaceId") == approval_surface_id
        and context.get("payload") == {
            "planId": plan_id,
            "planVersion": plan_version,
            "editedPlanVersion": plan_version,
            "approvedStepIds": step_ids,
        }
        for context in button_contexts
    )
    assert any(
        context.get("type") == "reject_plan"
        and context.get("surfaceId") == approval_surface_id
        and context.get("payload", {}).get("planId") == plan_id
        and context.get("payload", {}).get("planVersion") == plan_version
        for context in button_contexts
    )
    assert all(
        any(
            context.get("payload", {}).get("stepId") == step_id
            and context.get("payload", {}).get("planId") == plan_id
            and context.get("payload", {}).get("editedPlanVersion")
            == plan_version
            for context in button_contexts
        )
        for step_id in step_ids
    )


def _persisted_orchestrator_session_for_plan(
    session_db: Path,
    plan_id: str,
) -> dict[str, Any]:
    for state in _persisted_session_states(session_db):
        orchestrator_session = state.get("orchestrator_session")
        if not isinstance(orchestrator_session, dict):
            continue
        approval_records = orchestrator_session.get("approvalRecords")
        if isinstance(approval_records, dict) and plan_id in approval_records:
            return orchestrator_session

    pytest.fail(
        "Persisted ADK session state did not include orchestrator_session "
        f"for pending plan {plan_id}"
    )


def _persisted_session_states(session_db: Path) -> list[dict[str, Any]]:
    deadline = time.monotonic() + 5
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            with sqlite3.connect(f"file:{session_db}?mode=ro", uri=True) as db:
                states = [
                    json.loads(row[0])
                    for row in db.execute("SELECT state FROM sessions")
                ]
        except (json.JSONDecodeError, sqlite3.Error) as exc:
            last_error = exc
            time.sleep(0.1)
            continue

        if states:
            return states
        time.sleep(0.1)

    pytest.fail(
        "Timed out reading persisted ADK session states from SQLite"
        f"{f': {last_error!r}' if last_error is not None else ''}"
    )


def _assert_persisted_session_has_pending_draft_plan(
    orchestrator_session: dict[str, Any],
    normalized_response: dict[str, Any],
) -> None:
    plan_id = normalized_response["planId"]
    approval_records = orchestrator_session.get("approvalRecords")
    assert isinstance(approval_records, dict)
    approval_record = approval_records.get(plan_id)
    assert isinstance(approval_record, dict)
    assert approval_record["status"] == "draft"

    draft_plan = approval_record.get("draftPlan")
    assert isinstance(draft_plan, dict)
    assert _plan_field(draft_plan, "planId", "plan_id") == plan_id
    assert (
        _plan_field(draft_plan, "approvalSurfaceId", "approval_surface_id")
        == normalized_response["approvalSurfaceId"]
    )
    assert (
        _plan_field(draft_plan, "planVersion", "plan_version")
        == normalized_response["planVersion"]
    )
    persisted_step_ids = [
        _plan_field(step, "stepId", "step_id")
        for step in draft_plan.get("steps", [])
        if isinstance(step, dict)
    ]
    assert persisted_step_ids == normalized_response["stepIds"]


def _plan_field(
    payload: dict[str, Any],
    camel_name: str,
    snake_name: str,
) -> Any:
    if camel_name in payload:
        return payload[camel_name]
    return payload[snake_name]


def _assert_secret_strings_absent(serialized: str) -> None:
    for secret in KNOWN_SECRET_STRINGS:
        assert secret not in serialized


def _assert_secret_strings_absent_from_files(paths: Iterable[Path]) -> None:
    for path in paths:
        files = path.rglob("*") if path.is_dir() else [path]
        for candidate in files:
            if not candidate.is_file():
                continue
            content = candidate.read_bytes()
            for secret in KNOWN_SECRET_STRINGS:
                assert secret.encode("utf-8") not in content
