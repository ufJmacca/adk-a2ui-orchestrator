import tomllib
from pathlib import Path
from typing import Any

from a2a import types as a2a_types
import google.adk
import pytest
from google.adk.a2a.converters import part_converter
from google.genai import types as genai_types

from orchestrator_demo.a2ui_support.schema_manager import A2UI_VERSION


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
A2UI_MIME_TYPE = "application/json+a2ui"


def _valid_a2ui_payload() -> dict[str, Any]:
    return {
        "version": A2UI_VERSION,
        "updateComponents": {
            "surfaceId": "surface_bridge",
            "components": [
                {
                    "component": "Text",
                    "id": "root",
                    "text": "Bridge this plan review surface.",
                }
            ],
        },
    }


def _incremental_reference_payload(surface_id: str) -> dict[str, Any]:
    return {
        "version": A2UI_VERSION,
        "updateComponents": {
            "surfaceId": surface_id,
            "components": [
                {
                    "component": "Table",
                    "id": "component_table",
                    "columns": [{"key": "name", "label": "Name"}],
                    "rows": [{"name": "ABC Manufacturing"}],
                },
                {
                    "component": "Column",
                    "id": "root",
                    "children": ["component_existing_summary"],
                },
            ],
        },
    }


def test_google_adk_2_1_0_is_locked() -> None:
    # Arrange
    pyproject = tomllib.loads(
        (REPOSITORY_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )
    lockfile = tomllib.loads((REPOSITORY_ROOT / "uv.lock").read_text(encoding="utf-8"))

    # Act
    declared_requirements = [
        requirement
        for requirement in pyproject["project"]["dependencies"]
        if requirement.split("==", maxsplit=1)[0] == "google-adk"
    ]
    locked_versions = [
        package["version"]
        for package in lockfile["package"]
        if package["name"] == "google-adk"
    ]

    # Assert
    assert declared_requirements == ["google-adk==2.1.0"]
    assert locked_versions == ["2.1.0"]
    assert google.adk.__version__ == "2.1.0"


def test_adk_tagged_inline_data_part_converts_to_a2a_data_part() -> None:
    # Arrange
    a2ui_payload = {
        "type": "ApprovalPlan",
        "planId": "plan_characterization",
        "planVersion": 1,
    }
    source_data_part = a2a_types.DataPart(
        data=a2ui_payload,
        metadata={"mimeType": A2UI_MIME_TYPE},
    )
    inline_blob = (
        part_converter.A2A_DATA_PART_START_TAG
        + source_data_part.model_dump_json(
            by_alias=True,
            exclude_none=True,
        ).encode("utf-8")
        + part_converter.A2A_DATA_PART_END_TAG
    )
    genai_part = genai_types.Part(
        inline_data=genai_types.Blob(
            data=inline_blob,
            mime_type=part_converter.A2A_DATA_PART_TEXT_MIME_TYPE,
        )
    )

    # Act
    converted_part = part_converter.convert_genai_part_to_a2a_part(genai_part)

    # Assert
    assert converted_part is not None
    data_part = converted_part.root
    assert isinstance(data_part, a2a_types.DataPart)
    assert data_part.kind == "data"
    assert data_part.metadata == {"mimeType": A2UI_MIME_TYPE}
    assert data_part.data == a2ui_payload


def test_adk_a2a_delivery_builds_inline_bridge_parts_from_validated_data_parts() -> (
    None
):
    # Arrange
    from orchestrator_demo.a2ui_support.adk_a2a_delivery import (
        adk_a2a_bridge_parts_from_data_parts,
    )

    source_data_part = {
        "type": "data",
        "data": _valid_a2ui_payload(),
        "metadata": {"mimeType": A2UI_MIME_TYPE},
    }

    # Act
    bridge_parts = adk_a2a_bridge_parts_from_data_parts([source_data_part])

    # Assert
    assert len(bridge_parts) == 1
    bridge_part = bridge_parts[0]
    assert isinstance(bridge_part, genai_types.Part)
    assert bridge_part.function_response is None
    assert bridge_part.inline_data is not None
    assert bridge_part.inline_data.mime_type == (
        part_converter.A2A_DATA_PART_TEXT_MIME_TYPE
    )
    assert bridge_part.inline_data.data.startswith(
        part_converter.A2A_DATA_PART_START_TAG
    )
    assert bridge_part.inline_data.data.endswith(part_converter.A2A_DATA_PART_END_TAG)

    converted_part = part_converter.convert_genai_part_to_a2a_part(bridge_part)
    assert converted_part is not None
    data_part = converted_part.root
    assert isinstance(data_part, a2a_types.DataPart)
    assert data_part.kind == "data"
    assert data_part.metadata == {"mimeType": A2UI_MIME_TYPE}
    assert data_part.data == source_data_part["data"]


def test_adk_a2a_delivery_uses_existing_surface_context_for_incremental_parts() -> None:
    # Arrange
    from orchestrator_demo.a2ui_support.adk_a2a_delivery import (
        adk_a2a_bridge_parts_from_data_parts,
    )

    surface_id = "surface_incremental_bridge"
    source_data_part = {
        "type": "data",
        "data": _incremental_reference_payload(surface_id),
        "metadata": {"mimeType": A2UI_MIME_TYPE},
    }
    existing_components = {
        surface_id: {
            "component_existing_summary": {
                "component": "Text",
                "id": "component_existing_summary",
                "text": "Existing summary.",
            }
        }
    }

    # Act
    bridge_parts = adk_a2a_bridge_parts_from_data_parts(
        [source_data_part],
        existing_components_by_surface_id=existing_components,
    )

    # Assert
    assert len(bridge_parts) == 1
    converted_part = part_converter.convert_genai_part_to_a2a_part(bridge_parts[0])
    assert converted_part is not None
    data_part = converted_part.root
    assert isinstance(data_part, a2a_types.DataPart)
    assert data_part.metadata == {"mimeType": A2UI_MIME_TYPE}
    assert data_part.data == source_data_part["data"]


def test_adk_a2a_delivery_rejects_secret_bearing_data_part_without_bridge_output() -> (
    None
):
    # Arrange
    from orchestrator_demo.a2ui_support.adk_a2a_delivery import (
        A2UIA2ADeliveryError,
        adk_a2a_bridge_parts_from_data_parts,
    )

    leaked_secret = "sk-secretbridge123456"
    secret_bearing_data_part = {
        "type": "data",
        "data": {
            "version": A2UI_VERSION,
            "updateComponents": {
                "surfaceId": "surface_secret_rejection",
                "components": [
                    {
                        "component": "Text",
                        "id": "root",
                        "text": f"Never emit api_key={leaked_secret}",
                    }
                ],
            },
        },
        "metadata": {"mimeType": A2UI_MIME_TYPE},
    }
    bridge_parts: list[genai_types.Part] = []

    # Act
    with pytest.raises(A2UIA2ADeliveryError) as exc_info:
        bridge_parts = adk_a2a_bridge_parts_from_data_parts([secret_bearing_data_part])

    # Assert
    error_text = str(exc_info.value)
    assert bridge_parts == []
    assert leaked_secret not in error_text
    assert "<redacted-secret>" in error_text
    assert "failed outbound validation" in error_text
    assert [
        part_converter.convert_genai_part_to_a2a_part(part) for part in bridge_parts
    ] == []
