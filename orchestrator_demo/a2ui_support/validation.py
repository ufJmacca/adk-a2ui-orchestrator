"""Outbound A2UI validation and renderer-safe fallback handling."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
import json
import re
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

_REDACTED_SECRET = "<redacted-secret>"
_REDACTED_KEY = "<redacted-key>"
_QUOTED_TOKEN_PATTERN = re.compile(r"(['\"])([^'\"]{1,120})(\1)")
_SECRET_FIELD_MARKERS = (
    "api_key",
    "apikey",
    "access_key",
    "secret",
    "password",
    "token",
    "authorization",
    "credential",
    "private_key",
    "openrouter_api_key",
)
_SECRET_VALUE_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----",
        r"(?<![A-Za-z0-9])sk-[A-Za-z0-9][A-Za-z0-9_-]{8,}",
        r"(?<![A-Za-z0-9])(?:ghp|gho|ghu|ghs|github_pat)_[A-Za-z0-9_]{20,}",
        r"(?<![A-Za-z0-9])xox[baprs]-[A-Za-z0-9-]{10,}",
        r"(?<![A-Za-z0-9])(?:AKIA|ASIA)[A-Z0-9]{16}",
        r"(?<![A-Za-z0-9])AIza[0-9A-Za-z_-]{20,}",
        r"(?<![A-Za-z0-9])eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}",
        r"(?<![A-Za-z0-9])authorization\b\s*[:=]\s*(?:bearer|basic)\s+[A-Za-z0-9._~+/=-]{6,}",
        r"(?<![A-Za-z0-9])bearer\s+[A-Za-z0-9._~+/=-]{10,}",
        r"(?<![A-Za-z0-9])(?:api[_-]?key|access[_-]?key|private[_-]?key|secret|password|token|credential)\b\s*[:=]\s*\S{6,}",
    )
)


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
    repair: RepairCallback | None = None,
) -> A2UIValidationResult:
    """Parse, schema-validate, repair once, and return a renderer-safe part."""

    candidate, parse_errors = _parse_serialized_candidate(candidate)
    if parse_errors:
        return _fallback_result(parse_errors, repair_attempted=False)

    if isinstance(candidate, list):
        return _validate_candidate_list(candidate, repair=repair)

    return _validate_single_candidate(candidate, repair=repair)


def _validate_candidate_list(
    candidates: list[Any],
    *,
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
    for index, candidate in enumerate(candidates):
        result = _validate_single_candidate(candidate, repair=repair)
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
    repair: RepairCallback | None,
) -> A2UIValidationResult:
    parsed = _parse_candidate(candidate)
    if parsed.errors:
        return _fallback_result(parsed.errors, repair_attempted=False)

    assert parsed.payload is not None
    validation_errors = validate_basic_catalog_payload(parsed.payload)
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
    repaired_errors = validate_basic_catalog_payload(repaired.payload)
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
        return _redact_secret_like_values(str(exc))

    first_error = errors[0]
    location = ".".join(
        _safe_path_component(str(part)) for part in first_error.get("loc", ())
    )
    message = _redact_secret_like_values(first_error.get("msg", str(exc)))
    if location:
        return f"{location}: {message}"
    return str(message)


def _sanitize_validation_errors(validation_errors: list[str]) -> list[str]:
    return [_redact_secret_like_values(error) for error in validation_errors]


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
        if _is_secret_like_value(value):
            errors.append(
                f"{subject} contains secret-like value at {path}: {_REDACTED_SECRET}"
            )
        return

    if isinstance(value, (bytes, bytearray)):
        try:
            value_text = bytes(value).decode("utf-8")
        except UnicodeDecodeError:
            return
        if _is_secret_like_value(value_text):
            errors.append(
                f"{subject} contains secret-like value at {path}: {_REDACTED_SECRET}"
            )
        return

    if isinstance(value, Mapping):
        for key, child_value in value.items():
            key_text = str(key)
            child_path = f"{path}.{_safe_path_component(key_text)}"
            if _is_secret_like_key(key_text):
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


def _is_secret_like_key(key: str) -> bool:
    return _is_secret_like_field_name(key) or _is_secret_like_value(key)


def _is_secret_like_field_name(field_name: str) -> bool:
    normalized = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", field_name)
    normalized = normalized.lower().replace("-", "_")
    compact_normalized = normalized.replace("_", "")
    return any(
        marker in normalized or marker.replace("_", "") in compact_normalized
        for marker in _SECRET_FIELD_MARKERS
    )


def _is_secret_like_value(value: str) -> bool:
    stripped_value = value.strip()
    return bool(stripped_value) and any(
        pattern.search(stripped_value) for pattern in _SECRET_VALUE_PATTERNS
    )


def _safe_path_component(value: str) -> str:
    if _is_secret_like_key(value):
        return _REDACTED_KEY
    return value


def _redact_secret_like_values(message: str) -> str:
    redacted = str(message)
    for pattern in _SECRET_VALUE_PATTERNS:
        redacted = pattern.sub(_REDACTED_SECRET, redacted)
    return _QUOTED_TOKEN_PATTERN.sub(_redact_quoted_secret_key, redacted)


def _redact_quoted_secret_key(match: re.Match[str]) -> str:
    quote = match.group(1)
    token = match.group(2)
    if _is_secret_like_key(token):
        return f"{quote}{_REDACTED_KEY}{quote}"
    return match.group(0)


__all__ = [
    "A2UIValidationResult",
    "RepairCallback",
    "validate_outbound_a2ui",
]
