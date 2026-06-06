"""ADK artifact document and persistence helpers."""

from __future__ import annotations

from collections.abc import Mapping
import json
import re
from typing import Any

from google.genai import types

from orchestrator_demo.a2ui_support.secret_safety import (
    redact_secret_like_values,
    safe_path_component,
)


LATEST_RESULT_ARTIFACT_NAME = "orchestrator_latest_result.json"
ARTIFACT_MIME_TYPE = "application/json"


class ArtifactStorageError(RuntimeError):
    """Raised when ADK artifact persistence fails."""


def plan_execution_artifact_name(plan_id: str) -> str:
    """Return the required approved-plan execution artifact filename."""

    return f"orchestrator_plan_{artifact_plan_id_token(plan_id)}_execution.json"


def artifact_plan_id_token(plan_id: str) -> str:
    """Return the filesystem-safe token used in plan-scoped artifact names."""

    token = re.sub(r"[^A-Za-z0-9_.-]+", "_", safe_path_component(plan_id))
    return token.strip("._-") or "unknown"


def build_artifact_document(response: Mapping[str, Any]) -> dict[str, Any]:
    """Build the JSON document saved through ADK artifact services."""

    document: dict[str, Any] = {
        "status": response.get("status"),
        "path": response.get("path"),
        "artifacts": response.get("artifacts", {}),
        "statusEvents": response.get("statusEvents", []),
    }
    for field_name in ("planId", "planVersion"):
        if response.get(field_name) is not None:
            document[field_name] = response[field_name]
    return redacted_json_safe(document)


async def save_response_artifact(
    tool_context: Any,
    *,
    filename: str,
    response: Mapping[str, Any],
    document_type: str,
    plan_id: str | None = None,
) -> dict[str, Any] | None:
    """Persist a response artifact as a text-backed ADK ``Part``."""

    artifacts = response.get("artifacts")
    if tool_context is None or not artifacts:
        return None

    document = build_artifact_document(response)
    try:
        version = await tool_context.save_artifact(
            filename,
            types.Part(
                text=json.dumps(
                    document,
                    sort_keys=True,
                    separators=(",", ":"),
                )
            ),
            custom_metadata={
                "documentType": document_type,
                "mimeType": ARTIFACT_MIME_TYPE,
            },
        )
    except Exception as exc:
        raise ArtifactStorageError("ADK artifact persistence failed.") from exc

    artifact_ref: dict[str, Any] = {
        "filename": filename,
        "version": version,
        "mimeType": ARTIFACT_MIME_TYPE,
        "documentType": document_type,
    }
    if plan_id is not None:
        artifact_ref["planId"] = safe_path_component(plan_id)
    return redacted_json_safe(artifact_ref)


def redacted_json_safe(value: Any) -> Any:
    """Return a JSON-safe value with secret-like keys and values redacted."""

    if isinstance(value, Mapping):
        return {
            safe_path_component(str(key)): redacted_json_safe(child)
            for key, child in value.items()
        }
    if isinstance(value, list | tuple):
        return [redacted_json_safe(child) for child in value]
    if isinstance(value, set | frozenset):
        return sorted(redacted_json_safe(child) for child in value)
    if isinstance(value, str):
        return redact_secret_like_values(value)
    return value


__all__ = [
    "ARTIFACT_MIME_TYPE",
    "LATEST_RESULT_ARTIFACT_NAME",
    "ArtifactStorageError",
    "artifact_plan_id_token",
    "build_artifact_document",
    "plan_execution_artifact_name",
    "redacted_json_safe",
    "save_response_artifact",
]
