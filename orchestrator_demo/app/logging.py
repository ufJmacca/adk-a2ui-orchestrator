"""Structured audit logging with recursive secret redaction."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
import logging as py_logging
import re
from typing import Any


AUDIT_LOGGER_NAME = "orchestrator_demo.audit"
REDACTED_SECRET = "<redacted-secret>"
REDACTED_KEY = "<redacted-key>"

_SECRET_FIELD_MARKERS = (
    "api_key",
    "apikey",
    "access_key",
    "secret",
    "password",
    "token",
    "credential",
    "authorization",
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
_SECRET_PATH_SEGMENT_PATTERN = re.compile(
    r"(?<=[.\[\"'])"
    r"([A-Za-z0-9_-]*(?:api[_-]?key|apikey|access[_-]?key|private[_-]?key|"
    r"openrouter[_-]?api[_-]?key|authorization|credential|password|token|secret)"
    r"[A-Za-z0-9_-]*)",
    re.IGNORECASE,
)


def audit_logger() -> py_logging.Logger:
    """Return the process-wide audit logger."""

    return py_logging.getLogger(AUDIT_LOGGER_NAME)


def log_audit_event(
    event_name: str,
    payload: Mapping[str, Any] | None = None,
    *,
    level: int = py_logging.INFO,
    logger: py_logging.Logger | None = None,
) -> None:
    """Emit one structured audit event with a redacted payload."""

    safe_payload = redact_for_audit(dict(payload or {}))
    target_logger = logger or audit_logger()
    target_logger.log(
        level,
        "audit event: %s",
        event_name,
        extra={
            "audit_event": event_name,
            "event_payload": safe_payload,
        },
    )


def redact_for_audit(value: Any) -> Any:
    """Return a recursively redacted value suitable for logs and diagnostics."""

    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return redact_for_audit(model_dump(by_alias=True, mode="json"))

    if is_dataclass(value) and not isinstance(value, type):
        return redact_for_audit(asdict(value))

    if isinstance(value, Mapping):
        redacted: dict[str, Any] = {}
        for key, child_value in value.items():
            key_text = str(key)
            if _is_secret_like_key(key_text):
                redacted[REDACTED_KEY] = REDACTED_SECRET
                continue
            redacted[_redact_text(key_text)] = redact_for_audit(child_value)
        return redacted

    if isinstance(value, list):
        return [redact_for_audit(item) for item in value]

    if isinstance(value, tuple):
        return tuple(redact_for_audit(item) for item in value)

    if isinstance(value, set):
        return {redact_for_audit(item) for item in value}

    if isinstance(value, frozenset):
        return frozenset(redact_for_audit(item) for item in value)

    if isinstance(value, bytes | bytearray):
        try:
            text_value = bytes(value).decode("utf-8")
        except UnicodeDecodeError:
            return value
        redacted_text = _redact_text(text_value)
        return REDACTED_SECRET if redacted_text != text_value else redacted_text

    if isinstance(value, str):
        return _redact_text(value)

    return value


def redact_text_for_audit(value: Any) -> str:
    """Redact a text diagnostic without dropping non-secret context."""

    return str(redact_for_audit(str(value)))


def _redact_text(value: str) -> str:
    redacted = str(value)
    for pattern in _SECRET_VALUE_PATTERNS:
        redacted = pattern.sub(REDACTED_SECRET, redacted)
    return _SECRET_PATH_SEGMENT_PATTERN.sub(REDACTED_KEY, redacted)


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


__all__ = [
    "AUDIT_LOGGER_NAME",
    "REDACTED_KEY",
    "REDACTED_SECRET",
    "audit_logger",
    "log_audit_event",
    "redact_for_audit",
    "redact_text_for_audit",
]
