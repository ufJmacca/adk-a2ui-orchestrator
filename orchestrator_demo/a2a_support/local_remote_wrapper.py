"""Local specialist wrappers that match the future remote A2A adapter shape."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
from typing import Any
import re

from orchestrator_demo.agents import build_default_specialists
from orchestrator_demo.agents.base import SpecialistAgent
from orchestrator_demo.a2a_support.remote_agent_adapter import UserActionPayload
from orchestrator_demo.a2ui_support.event_parser import (
    PlanUserActionParseError,
    StructuredUserActionRequiredError,
    parse_user_action,
)
from orchestrator_demo.contracts import SpecialistRequest, SpecialistResponse, UserAction
from orchestrator_demo.registry.descriptors import (
    SECRET_VALUE_PATTERNS as DESCRIPTOR_SECRET_VALUE_PATTERNS,
)


DEFAULT_LOCAL_REMOTE_AGENT_IDS = frozenset(
    {"internal_knowledge", "product_opportunity"}
)
REDACTED_SECRET_VALUE = "[REDACTED]"
_SECRET_KEY_FRAGMENTS = (
    "access_key",
    "accesskey",
    "api_key",
    "apikey",
    "authorization",
    "credential",
    "openrouter",
    "password",
    "private_key",
    "privatekey",
    "secret",
    "token",
)
_SECRET_VALUE_PATTERNS = (
    *DESCRIPTOR_SECRET_VALUE_PATTERNS,
    re.compile(r"(?i)(?<![A-Za-z0-9])sk-[A-Za-z0-9][A-Za-z0-9_-]{8,}"),
    re.compile(
        r"(?i)(?<![A-Za-z0-9])(?:ghp|gho|ghu|ghs|github_pat)_[A-Za-z0-9_]{20,}"
    ),
    re.compile(r"(?i)(?<![A-Za-z0-9])xox[baprs]-[A-Za-z0-9-]{10,}"),
    re.compile(r"(?i)(?<![A-Za-z0-9])(?:AKIA|ASIA)[A-Z0-9]{16}"),
    re.compile(r"(?i)(?<![A-Za-z0-9])AIza[0-9A-Za-z_-]{20,}"),
    re.compile(
        r"(?i)(?<![A-Za-z0-9])eyJ[A-Za-z0-9_-]{10,}"
        r"\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}"
    ),
    re.compile(r"(?i)(?<![A-Za-z0-9])bearer\s+[A-Za-z0-9._~+/=-]{6,}"),
    re.compile(
        r"(?i)(?<![A-Za-z0-9])authorization\b\s*[:=]\s*bearer\s+\S{6,}"
    ),
    re.compile(
        r"(?i)(?<![A-Za-z0-9])"
        r"(?:api[_-]?key|access[_-]?key|private[_-]?key|secret|password|token|credential)"
        r"\b\s*[:=]\s*\S{6,}"
    ),
    re.compile(r"(?i)\b(?:authorization|openrouter)\b\s*[:=]\s*\S{6,}"),
    re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]{6,}"),
)


class LocalRemoteAgentWrapper:
    """Expose a local specialist through the remote-compatible adapter contract."""

    def __init__(self, local_agent: SpecialistAgent) -> None:
        self._local_agent = local_agent
        self._user_action_request_count = 0

    @property
    def agent_id(self) -> str:
        return self._local_agent.agent_id

    @property
    def call_count(self) -> int:
        return self._local_agent.call_count

    @property
    def calls(self) -> list[SpecialistRequest]:
        return self._local_agent.calls

    async def run(self, request: SpecialistRequest) -> SpecialistResponse:
        forwarded_request = _request_without_secrets(request)
        return await self._local_agent.handle(forwarded_request)

    async def handle(self, request: SpecialistRequest) -> SpecialistResponse:
        """Compatibility alias for callers still using the local specialist shape."""

        return await self.run(request)

    async def handle_user_action(
        self,
        user_action: UserActionPayload,
    ) -> SpecialistResponse:
        parsed_action = _validated_user_action(user_action)
        original_payload = _jsonable_payload(user_action)
        action_payload = parsed_action.model_dump(
            by_alias=True,
            mode="json",
            exclude_none=True,
        )
        request = SpecialistRequest(
            request_id=self._next_user_action_request_id(),
            user_input=(
                f"A2UI user action {parsed_action.type} on "
                f"{parsed_action.surface_id}."
            ),
            agent_id=self.agent_id,
            context={
                "event_type": "userAction",
                "user_action": _redact_secrets(action_payload),
                "user_action_payload": _redact_secrets(original_payload),
            },
        )

        return await self.run(request)

    def _next_user_action_request_id(self) -> str:
        self._user_action_request_count += 1
        suffix = f"{self.agent_id}_user_action_{self._user_action_request_count}"
        return _stable_id("request", suffix)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._local_agent, name)


def build_default_local_remote_wrappers(
    specialists: Mapping[str, SpecialistAgent] | None = None,
) -> dict[str, LocalRemoteAgentWrapper]:
    """Wrap the demo specialists configured as local A2A-compatible agents."""

    if specialists is None:
        specialists = build_default_specialists()

    return {
        agent_id: LocalRemoteAgentWrapper(specialists[agent_id])
        for agent_id in sorted(DEFAULT_LOCAL_REMOTE_AGENT_IDS)
        if agent_id in specialists
    }


def _request_without_secrets(request: SpecialistRequest) -> SpecialistRequest:
    redacted_user_input = _redact_secrets(request.user_input)
    redacted_context = _redact_secrets(request.context)
    if (
        redacted_user_input == request.user_input
        and redacted_context == request.context
    ):
        return request

    return request.model_copy(
        update={
            "user_input": redacted_user_input,
            "context": redacted_context,
        },
        deep=True,
    )


def _jsonable_payload(user_action: UserActionPayload) -> dict[str, Any]:
    if isinstance(user_action, UserAction):
        return user_action.model_dump(by_alias=True, mode="json", exclude_none=True)

    return dict(user_action)


def _validated_user_action(user_action: UserActionPayload) -> UserAction:
    try:
        return parse_user_action(user_action)
    except (PlanUserActionParseError, StructuredUserActionRequiredError) as exc:
        safe_error = _redact_secrets(str(exc))
        suffix = f": {safe_error}" if safe_error else ""
        raise ValueError(f"invalid A2UI userAction payload{suffix}") from None


def _stable_id(prefix: str, suffix: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_-]+", "_", suffix).strip("_")
    if not normalized or not normalized[0].isalnum():
        normalized = f"generated_{normalized}"

    return f"{prefix}_{normalized}"


def _redact_secrets(value: Any) -> Any:
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return _redact_secrets(model_dump(by_alias=True, mode="json"))

    if is_dataclass(value) and not isinstance(value, type):
        return _redact_secrets(asdict(value))

    if isinstance(value, Mapping):
        redacted: dict[str, Any] = {}
        for key, child_value in value.items():
            string_key = str(key)
            if _looks_like_secret_value(string_key):
                redacted[REDACTED_SECRET_VALUE] = REDACTED_SECRET_VALUE
            elif _is_secret_key(string_key):
                redacted[string_key] = REDACTED_SECRET_VALUE
            else:
                redacted[string_key] = _redact_secrets(child_value)
        return redacted

    if isinstance(value, list):
        return [_redact_secrets(item) for item in value]

    if isinstance(value, tuple):
        return tuple(_redact_secrets(item) for item in value)

    if isinstance(value, set):
        return {_redact_secrets(item) for item in value}

    if isinstance(value, frozenset):
        return frozenset(_redact_secrets(item) for item in value)

    if isinstance(value, (bytes, bytearray)):
        try:
            decoded_value = bytes(value).decode("utf-8")
        except UnicodeDecodeError:
            return value
        if _looks_like_secret_value(decoded_value):
            return REDACTED_SECRET_VALUE
        return value

    if isinstance(value, str) and _looks_like_secret_value(value):
        return REDACTED_SECRET_VALUE

    return value


def _is_secret_key(key: str) -> bool:
    camel_split = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", key)
    normalized = re.sub(r"[^a-z0-9]+", "_", camel_split.lower()).strip("_")
    return any(fragment in normalized for fragment in _SECRET_KEY_FRAGMENTS)


def _looks_like_secret_value(value: str) -> bool:
    return any(pattern.search(value) is not None for pattern in _SECRET_VALUE_PATTERNS)


__all__ = [
    "DEFAULT_LOCAL_REMOTE_AGENT_IDS",
    "LocalRemoteAgentWrapper",
    "REDACTED_SECRET_VALUE",
    "build_default_local_remote_wrappers",
]
