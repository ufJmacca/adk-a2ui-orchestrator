"""Shared redaction helpers for renderer-visible A2UI diagnostics."""

from __future__ import annotations

import re


REDACTED_SECRET = "<redacted-secret>"
REDACTED_KEY = "<redacted-key>"

_QUOTED_TOKEN_PATTERN = re.compile(r"(['\"])([^'\"]{1,120})(\1)")
_SECRET_ASSIGNMENT_PATTERN = re.compile(
    r"(?<![A-Za-z0-9])"
    r"(?:[A-Za-z0-9]+[_-])*"
    r"(?:api[_-]?key|access[_-]?key|private[_-]?key|secret|password|token|credential)"
    r"(?:[_-][A-Za-z0-9]+)*"
    r"\b\s*[:=]\s*\S{6,}",
    re.IGNORECASE,
)
_SECRET_KEY_TOKEN_PATTERN = re.compile(
    r"(?<![A-Za-z0-9_.-])"
    r"([A-Za-z0-9_-]*"
    r"(?:api[_-]?key|access[_-]?key|private[_-]?key|secret|password|token|"
    r"credential|authorization)"
    r"[A-Za-z0-9_-]*)"
    r"(?![A-Za-z0-9_.-])",
    re.IGNORECASE,
)
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


def is_secret_like_key(key: str) -> bool:
    return is_secret_like_field_name(key) or is_secret_like_value(key)


def is_secret_like_field_name(field_name: str) -> bool:
    normalized = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", field_name)
    normalized = normalized.lower().replace("-", "_")
    compact_normalized = normalized.replace("_", "")
    return any(
        marker in normalized or marker.replace("_", "") in compact_normalized
        for marker in _SECRET_FIELD_MARKERS
    )


def is_secret_like_value(value: str) -> bool:
    stripped_value = value.strip()
    return bool(stripped_value) and any(
        pattern.search(stripped_value) for pattern in _SECRET_VALUE_PATTERNS
    )


def safe_path_component(value: str) -> str:
    if is_secret_like_key(value):
        return REDACTED_KEY
    return value


def redact_secret_like_values(message: str) -> str:
    redacted = str(message)
    redacted = _SECRET_ASSIGNMENT_PATTERN.sub(REDACTED_SECRET, redacted)
    for pattern in _SECRET_VALUE_PATTERNS:
        redacted = pattern.sub(REDACTED_SECRET, redacted)
    redacted = _QUOTED_TOKEN_PATTERN.sub(_redact_quoted_secret_key, redacted)
    return _SECRET_KEY_TOKEN_PATTERN.sub(_redact_unquoted_secret_key, redacted)


def _redact_quoted_secret_key(match: re.Match[str]) -> str:
    quote = match.group(1)
    token = match.group(2)
    if is_secret_like_key(token):
        return f"{quote}{REDACTED_KEY}{quote}"
    return match.group(0)


def _redact_unquoted_secret_key(match: re.Match[str]) -> str:
    token = match.group(1)
    normalized = token.casefold()
    if normalized in {
        REDACTED_SECRET.strip("<>"),
        REDACTED_KEY.strip("<>"),
        "secret-like",
        "secret-bearing",
    }:
        return match.group(0)
    if is_secret_like_key(token):
        return REDACTED_KEY
    return match.group(0)


__all__ = [
    "REDACTED_KEY",
    "REDACTED_SECRET",
    "is_secret_like_key",
    "is_secret_like_value",
    "redact_secret_like_values",
    "safe_path_component",
]
