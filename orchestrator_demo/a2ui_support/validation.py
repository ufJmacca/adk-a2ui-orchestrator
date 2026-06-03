"""Outbound A2UI validation and renderer-safe fallback handling."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import dataclass, field
import json
from typing import Any

from pydantic import ValidationError

from orchestrator_demo.a2a_support.transport import (
    A2UI_MIME_TYPE,
    DataPart,
    TextPart,
)
from orchestrator_demo.a2ui_support.schema_manager import (
    DELETE_SURFACE_MESSAGE,
    UPDATE_COMPONENTS_MESSAGE,
    validate_basic_catalog_payload,
)
from orchestrator_demo.a2ui_support.secret_safety import (
    REDACTED_SECRET,
    is_secret_like_key,
    is_secret_like_value,
    redact_secret_like_values,
    safe_path_component,
)


RepairCallback = Callable[[dict[str, Any], list[str]], dict[str, Any]]


@dataclass(frozen=True)
class A2UIValidationResult:
    """Result of preparing one A2UI payload for renderer emission."""

    valid: bool
    renderer_part: DataPart | TextPart
    validation_errors: list[str]
    repaired: bool = False
    renderer_parts: tuple[DataPart | TextPart, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not self.renderer_parts:
            object.__setattr__(self, "renderer_parts", (self.renderer_part,))


def validate_outbound_a2ui(
    candidate: Any,
    *,
    existing_components_by_surface_id: Mapping[
        str, Mapping[str, Mapping[str, Any]]
    ]
    | None = None,
    repair: RepairCallback | None = None,
) -> A2UIValidationResult:
    """Parse, schema-validate, repair once, and return a renderer-safe part."""

    candidate, parse_errors = _parse_serialized_candidate(candidate)
    if parse_errors:
        return _fallback_result(parse_errors, repair_attempted=False)

    if isinstance(candidate, list):
        return _validate_candidate_list(
            candidate,
            existing_components_by_surface_id=existing_components_by_surface_id,
            repair=repair,
        )

    return _validate_single_candidate(
        candidate,
        existing_components_by_surface_id=existing_components_by_surface_id,
        repair=repair,
    )


def _validate_candidate_list(
    candidates: list[Any],
    *,
    existing_components_by_surface_id: Mapping[
        str, Mapping[str, Mapping[str, Any]]
    ]
    | None,
    repair: RepairCallback | None,
) -> A2UIValidationResult:
    if not candidates:
        return _fallback_result(
            ["A2UI payload list must contain at least one envelope"],
            repair_attempted=False,
        )

    renderer_parts: list[DataPart | TextPart] = []
    validation_errors: list[str] = []
    repaired = False
    repair_attempted = False
    staged_components = clone_surface_component_graphs(
        existing_components_by_surface_id
    )
    for index, candidate in enumerate(candidates):
        result = _validate_single_candidate(
            candidate,
            existing_components_by_surface_id=staged_components,
            repair=repair,
        )
        if not result.valid:
            if isinstance(result.renderer_part, TextPart):
                diagnostic = result.renderer_part.metadata.get("developerDiagnostic")
                if isinstance(diagnostic, Mapping):
                    repair_attempted = repair_attempted or (
                        diagnostic.get("repairAttempted") is True
                    )
            validation_errors.extend(
                f"payload[{index}]: {error}" for error in result.validation_errors
            )
            continue

        repaired = repaired or result.repaired
        renderer_parts.extend(result.renderer_parts)
        for part in result.renderer_parts:
            if isinstance(part, DataPart):
                apply_validated_a2ui_component_graph(staged_components, part.data)

    if validation_errors:
        return _fallback_result(
            validation_errors,
            repair_attempted=repair_attempted,
        )

    return A2UIValidationResult(
        valid=True,
        renderer_part=renderer_parts[0],
        renderer_parts=tuple(renderer_parts),
        validation_errors=[],
        repaired=repaired,
    )


def _validate_single_candidate(
    candidate: Any,
    *,
    existing_components_by_surface_id: Mapping[
        str, Mapping[str, Mapping[str, Any]]
    ]
    | None,
    repair: RepairCallback | None,
) -> A2UIValidationResult:
    parsed = _parse_candidate(candidate)
    if parsed.errors:
        return _fallback_result(parsed.errors, repair_attempted=False)

    assert parsed.payload is not None
    validation_errors = validate_basic_catalog_payload(
        parsed.payload,
        existing_components_by_surface_id=existing_components_by_surface_id,
    )
    if not validation_errors:
        secret_errors = _secret_safety_errors(parsed.payload, parsed.part.metadata)
        if secret_errors:
            return _fallback_result(secret_errors, repair_attempted=False)

        return A2UIValidationResult(
            valid=True,
            renderer_part=parsed.part,
            validation_errors=[],
            repaired=False,
        )

    secret_errors = _secret_safety_errors(parsed.payload, parsed.part.metadata)
    if secret_errors:
        return _fallback_result(
            [*validation_errors, *secret_errors],
            repair_attempted=False,
        )

    if repair is None:
        return _fallback_result(validation_errors, repair_attempted=False)

    try:
        repaired_payload = repair(
            dict(parsed.payload),
            _sanitize_validation_errors(validation_errors),
        )
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
    repaired_errors = validate_basic_catalog_payload(
        repaired.payload,
        existing_components_by_surface_id=existing_components_by_surface_id,
    )
    if repaired_errors:
        return _fallback_result(repaired_errors, repair_attempted=True)

    secret_errors = _secret_safety_errors(repaired.payload, repaired.part.metadata)
    if secret_errors:
        return _fallback_result(secret_errors, repair_attempted=True)

    return A2UIValidationResult(
        valid=True,
        renderer_part=repaired.part,
        validation_errors=[],
        repaired=True,
    )


def _parse_serialized_candidate(candidate: Any) -> tuple[Any, list[str]]:
    if isinstance(candidate, str | bytes | bytearray):
        try:
            return json.loads(candidate), []
        except UnicodeDecodeError:
            return candidate, ["A2UI payload bytes must be valid UTF-8 JSON"]
        except json.JSONDecodeError as exc:
            return candidate, [f"A2UI payload must be valid JSON: {exc.msg}"]

    return candidate, []


SurfaceComponentGraphs = dict[str, dict[str, dict[str, Any]]]


def clone_surface_component_graphs(
    components_by_surface_id: Mapping[str, Mapping[str, Mapping[str, Any]]] | None,
) -> SurfaceComponentGraphs:
    """Return a mutable copy of registered renderer components by surface."""

    if components_by_surface_id is None:
        return {}
    return {
        surface_id: {
            component_id: deepcopy(dict(component))
            for component_id, component in components.items()
            if isinstance(component_id, str) and isinstance(component, Mapping)
        }
        for surface_id, components in components_by_surface_id.items()
        if isinstance(surface_id, str) and isinstance(components, Mapping)
    }


def apply_validated_a2ui_component_graph(
    components_by_surface_id: SurfaceComponentGraphs,
    payload: Mapping[str, Any],
) -> None:
    """Apply a validated A2UI surface component update to a staged graph."""

    delete_surface = payload.get(DELETE_SURFACE_MESSAGE)
    if isinstance(delete_surface, Mapping):
        surface_id = delete_surface.get("surfaceId")
        if isinstance(surface_id, str):
            components_by_surface_id.pop(surface_id, None)

    update_components = payload.get(UPDATE_COMPONENTS_MESSAGE)
    if not isinstance(update_components, Mapping):
        return

    surface_id = update_components.get("surfaceId")
    components = update_components.get("components")
    if not isinstance(surface_id, str) or not isinstance(components, list):
        return

    surface_components = components_by_surface_id.setdefault(surface_id, {})
    if _is_full_component_replacement(update_components):
        surface_components.clear()

    for component in components:
        if not isinstance(component, Mapping):
            continue
        component_id = component.get("id")
        if not isinstance(component_id, str) or not component_id:
            continue
        surface_components[component_id] = deepcopy(dict(component))


def _is_full_component_replacement(message: Mapping[str, Any]) -> bool:
    return (
        message.get("replace") is True
        or message.get("fullReplacement") is True
        or message.get("mode") == "replace"
    )


@dataclass(frozen=True)
class _ParsedCandidate:
    part: DataPart
    payload: dict[str, Any] | None
    errors: list[str]


def _parse_candidate(candidate: Any) -> _ParsedCandidate:
    if isinstance(candidate, str | bytes | bytearray):
        try:
            candidate = json.loads(candidate)
        except UnicodeDecodeError:
            return _invalid_parse("A2UI payload bytes must be valid UTF-8 JSON")
        except json.JSONDecodeError as exc:
            return _invalid_parse(f"A2UI payload must be valid JSON: {exc.msg}")

    if isinstance(candidate, DataPart):
        try:
            part = DataPart.model_validate(
                candidate.model_dump(by_alias=True, mode="python")
            )
        except ValidationError as exc:
            return _invalid_parse(_validation_error_summary(exc))
        return _ParsedCandidate(part=part, payload=dict(part.data), errors=[])

    if not isinstance(candidate, Mapping):
        try:
            part = DataPart.model_validate(candidate)
        except ValidationError as exc:
            return _invalid_parse(_validation_error_summary(exc))
        return _ParsedCandidate(part=part, payload=dict(part.data), errors=[])

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
    return candidate.get("type") == "data" or candidate.get("kind") == "data" or (
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
    safe_validation_errors = _sanitize_validation_errors(validation_errors)
    diagnostic = {
        "fallback": "text",
        "validationErrors": safe_validation_errors,
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
        validation_errors=safe_validation_errors,
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


def _sanitize_validation_errors(validation_errors: list[str]) -> list[str]:
    return [redact_secret_like_values(error) for error in validation_errors]


def _redact_secret_like_values(message: str) -> str:
    return redact_secret_like_values(message)


def _secret_safety_errors(
    payload: Mapping[str, Any],
    metadata: Mapping[str, Any],
) -> list[str]:
    errors: list[str] = []
    _collect_secret_safety_errors(payload, "payload", "A2UI payload", errors)
    _collect_secret_safety_errors(metadata, "metadata", "A2UI metadata", errors)
    return errors


def _collect_secret_safety_errors(
    value: Any,
    path: str,
    subject: str,
    errors: list[str],
) -> None:
    if isinstance(value, str):
        if is_secret_like_value(value):
            errors.append(
                f"{subject} contains secret-like value at {path}: {REDACTED_SECRET}"
            )
        return

    if isinstance(value, (bytes, bytearray)):
        try:
            value_text = bytes(value).decode("utf-8")
        except UnicodeDecodeError:
            return
        if is_secret_like_value(value_text):
            errors.append(
                f"{subject} contains secret-like value at {path}: {REDACTED_SECRET}"
            )
        return

    if isinstance(value, Mapping):
        for key, child_value in value.items():
            key_text = str(key)
            child_path = f"{path}.{safe_path_component(key_text)}"
            if is_secret_like_key(key_text):
                errors.append(f"{subject} contains secret-like key at {child_path}")
            _collect_secret_safety_errors(child_value, child_path, subject, errors)
        return

    if isinstance(value, list | tuple):
        for index, child_value in enumerate(value):
            _collect_secret_safety_errors(
                child_value,
                f"{path}[{index}]",
                subject,
                errors,
            )

__all__ = [
    "A2UIValidationResult",
    "RepairCallback",
    "SurfaceComponentGraphs",
    "apply_validated_a2ui_component_graph",
    "clone_surface_component_graphs",
    "validate_outbound_a2ui",
]
