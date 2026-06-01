"""Outbound A2UI validation and renderer-safe fallback handling."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from pydantic import ValidationError

from orchestrator_demo.a2a_support.transport import (
    A2UI_MIME_TYPE,
    DataPart,
    TextPart,
)
from orchestrator_demo.a2ui_support.schema_manager import (
    validate_basic_catalog_payload,
)


RepairCallback = Callable[[dict[str, Any], list[str]], dict[str, Any]]


@dataclass(frozen=True)
class A2UIValidationResult:
    """Result of preparing one A2UI payload for renderer emission."""

    valid: bool
    renderer_part: DataPart | TextPart
    validation_errors: list[str]
    repaired: bool = False


def validate_outbound_a2ui(
    candidate: DataPart | Mapping[str, Any],
    *,
    repair: RepairCallback | None = None,
) -> A2UIValidationResult:
    """Parse, schema-validate, repair once, and return a renderer-safe part."""

    parsed = _parse_candidate(candidate)
    if parsed.errors:
        return _fallback_result(parsed.errors, repair_attempted=False)

    assert parsed.payload is not None
    validation_errors = validate_basic_catalog_payload(parsed.payload)
    if not validation_errors:
        return A2UIValidationResult(
            valid=True,
            renderer_part=parsed.part,
            validation_errors=[],
            repaired=False,
        )

    if repair is None:
        return _fallback_result(validation_errors, repair_attempted=False)

    try:
        repaired_payload = repair(dict(parsed.payload), list(validation_errors))
    except Exception as exc:
        return _fallback_result(
            [
                *validation_errors,
                f"repair callback failed: {type(exc).__name__}",
            ],
            repair_attempted=True,
        )

    repaired = _parse_candidate(repaired_payload)
    if repaired.errors:
        return _fallback_result(repaired.errors, repair_attempted=True)

    assert repaired.payload is not None
    repaired_errors = validate_basic_catalog_payload(repaired.payload)
    if repaired_errors:
        return _fallback_result(repaired_errors, repair_attempted=True)

    return A2UIValidationResult(
        valid=True,
        renderer_part=repaired.part,
        validation_errors=[],
        repaired=True,
    )


@dataclass(frozen=True)
class _ParsedCandidate:
    part: DataPart
    payload: dict[str, Any] | None
    errors: list[str]


def _parse_candidate(candidate: DataPart | Mapping[str, Any]) -> _ParsedCandidate:
    if isinstance(candidate, DataPart):
        return _ParsedCandidate(part=candidate, payload=dict(candidate.data), errors=[])

    if not isinstance(candidate, Mapping):
        return _invalid_parse("A2UI candidate must be a data part or object")

    candidate_dict = dict(candidate)
    if _looks_like_data_part(candidate_dict):
        try:
            part = DataPart.model_validate(candidate_dict)
        except ValidationError as exc:
            return _invalid_parse(_validation_error_summary(exc))
        return _ParsedCandidate(part=part, payload=dict(part.data), errors=[])

    try:
        part = DataPart(
            data=candidate_dict,
            metadata={"mimeType": A2UI_MIME_TYPE},
        )
    except ValidationError as exc:
        return _invalid_parse(_validation_error_summary(exc))

    return _ParsedCandidate(part=part, payload=dict(part.data), errors=[])


def _looks_like_data_part(candidate: Mapping[str, Any]) -> bool:
    return candidate.get("type") == "data" or (
        "data" in candidate and "metadata" in candidate
    )


def _invalid_parse(error: str) -> _ParsedCandidate:
    placeholder = DataPart(
        data={"catalog": "basic", "surfaceId": "surface_invalid_a2ui"},
        metadata={"mimeType": A2UI_MIME_TYPE},
    )
    return _ParsedCandidate(part=placeholder, payload=None, errors=[error])


def _fallback_result(
    validation_errors: list[str],
    *,
    repair_attempted: bool,
) -> A2UIValidationResult:
    diagnostic = {
        "fallback": "text",
        "validationErrors": validation_errors,
        "repairAttempted": repair_attempted,
    }
    return A2UIValidationResult(
        valid=False,
        renderer_part=TextPart(
            text=(
                "A2UI rendering unavailable. The generated UI payload failed "
                "validation and was not emitted to the renderer."
            ),
            metadata={"developerDiagnostic": diagnostic},
        ),
        validation_errors=validation_errors,
        repaired=False,
    )


def _validation_error_summary(exc: ValidationError) -> str:
    errors = exc.errors()
    if not errors:
        return str(exc)

    first_error = errors[0]
    location = ".".join(str(part) for part in first_error.get("loc", ()))
    message = first_error.get("msg", str(exc))
    if location:
        return f"{location}: {message}"
    return str(message)


__all__ = [
    "A2UIValidationResult",
    "RepairCallback",
    "validate_outbound_a2ui",
]
