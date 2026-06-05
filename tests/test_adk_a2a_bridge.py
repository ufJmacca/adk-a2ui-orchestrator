from pathlib import Path
import tomllib

from a2a import types as a2a_types
import google.adk
from google.adk.a2a.converters import part_converter
from google.genai import types as genai_types


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
A2UI_MIME_TYPE = "application/json+a2ui"


def test_google_adk_2_1_0_is_locked() -> None:
    # Arrange
    pyproject = tomllib.loads(
        (REPOSITORY_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )
    lockfile = tomllib.loads(
        (REPOSITORY_ROOT / "uv.lock").read_text(encoding="utf-8")
    )

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
