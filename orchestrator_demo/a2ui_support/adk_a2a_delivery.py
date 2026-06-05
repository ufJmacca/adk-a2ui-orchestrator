"""ADK A2A delivery helpers for validated A2UI DataParts."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

from a2a import types as a2a_types
from google.adk.a2a.converters import part_converter
from google.genai import types as genai_types

from orchestrator_demo.a2a_support.transport import DataPart
from orchestrator_demo.a2ui_support.secret_safety import redact_secret_like_values
from orchestrator_demo.a2ui_support.validation import validate_outbound_a2ui


class A2UIA2ADeliveryError(ValueError):
    """Raised when A2UI cannot be safely emitted as A2A bridge parts."""


def adk_a2a_bridge_parts_for_response(
    response: Mapping[str, Any],
    *,
    existing_components_by_surface_id: Mapping[str, Mapping[str, Mapping[str, Any]]]
    | None = None,
) -> list[genai_types.Part]:
    """Return ADK inline-data bridge parts for response ``a2uiParts``."""

    parts = response.get("a2uiParts")
    if parts is None or parts == []:
        return []
    if not isinstance(parts, Sequence) or isinstance(parts, str | bytes | bytearray):
        raise A2UIA2ADeliveryError("a2uiParts must be a list of A2UI DataParts")

    return adk_a2a_bridge_parts_from_data_parts(
        parts,
        existing_components_by_surface_id=existing_components_by_surface_id,
    )


def adk_a2a_bridge_parts_from_data_parts(
    parts: Sequence[Any],
    *,
    existing_components_by_surface_id: Mapping[str, Mapping[str, Mapping[str, Any]]]
    | None = None,
) -> list[genai_types.Part]:
    """Build ADK inline-data bridge parts from validated A2UI DataParts."""

    result = validate_outbound_a2ui(
        list(parts),
        existing_components_by_surface_id=existing_components_by_surface_id,
    )
    if not result.valid:
        errors = "; ".join(
            redact_secret_like_values(error) for error in result.validation_errors
        )
        raise A2UIA2ADeliveryError(f"A2UI payload failed outbound validation: {errors}")

    bridge_parts: list[genai_types.Part] = []
    for part in result.renderer_parts:
        if not isinstance(part, DataPart):
            raise A2UIA2ADeliveryError(
                "A2UI payload validation did not produce DataParts"
            )
        bridge_parts.append(adk_a2a_bridge_part_from_data_part(part))

    if not bridge_parts:
        raise A2UIA2ADeliveryError("A2UI payload produced no DataParts")
    return bridge_parts


def adk_a2a_bridge_part_from_data_part(
    part: DataPart | Mapping[str, Any],
) -> genai_types.Part:
    """Wrap one A2UI DataPart in ADK's A2A inline-data tag format."""

    data_part = DataPart.model_validate(part)
    a2a_data_part = a2a_types.DataPart(
        data=data_part.data,
        metadata=data_part.metadata,
    )
    inline_blob = (
        part_converter.A2A_DATA_PART_START_TAG
        + json.dumps(
            a2a_data_part.model_dump(
                by_alias=True,
                exclude_none=True,
                mode="json",
            ),
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        + part_converter.A2A_DATA_PART_END_TAG
    )
    return genai_types.Part(
        inline_data=genai_types.Blob(
            data=inline_blob,
            mime_type=part_converter.A2A_DATA_PART_TEXT_MIME_TYPE,
        )
    )


__all__ = [
    "A2UIA2ADeliveryError",
    "adk_a2a_bridge_part_from_data_part",
    "adk_a2a_bridge_parts_for_response",
    "adk_a2a_bridge_parts_from_data_parts",
]
