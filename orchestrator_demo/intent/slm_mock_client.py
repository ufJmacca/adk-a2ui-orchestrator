"""Replaceable SLM intent client implementations for local demos."""

from collections.abc import Mapping
from typing import Any, Protocol, runtime_checkable

from orchestrator_demo.contracts import IntentName, IntentSuggestion


IntentOverride = IntentSuggestion | Mapping[str, Any] | tuple[IntentName, float]


@runtime_checkable
class SlmIntentClient(Protocol):
    """Abstraction for lightweight external intent classification."""

    async def classify(self, user_input: str) -> IntentSuggestion:
        """Return the SLM intent suggestion for a user request."""
        ...


class MockSlmIntentClient:
    """Deterministic local SLM mock with exact-input test overrides."""

    def __init__(self, overrides: Mapping[str, IntentOverride] | None = None) -> None:
        self._overrides = dict(overrides or {})

    async def classify(self, user_input: str) -> IntentSuggestion:
        if user_input in self._overrides:
            return _coerce_intent_suggestion(self._overrides[user_input])

        text = user_input.casefold()

        if "internal notes" in text or "crm" in text:
            return IntentSuggestion(intent="internal_knowledge", confidence=0.90)

        if "meeting" in text or "prepare" in text:
            return IntentSuggestion(intent="meeting_prep", confidence=0.82)

        if "web search" in text or "public information" in text:
            return IntentSuggestion(intent="web_search", confidence=0.86)

        if "industry" in text or "sector" in text:
            return IntentSuggestion(intent="industry_research", confidence=0.88)

        if "product opportunities" in text or "product opportunity" in text:
            return IntentSuggestion(intent="product_opportunity", confidence=0.87)

        if "prospect" in text:
            return IntentSuggestion(intent="prospect_research", confidence=0.78)

        if "credit" in text or "risk" in text:
            return IntentSuggestion(intent="credit_risk", confidence=0.80)

        if "relationship history" in text or "relationship summary" in text:
            return IntentSuggestion(intent="relationship_summary", confidence=0.84)

        if "compliance" in text or "policy" in text:
            return IntentSuggestion(intent="compliance_policy", confidence=0.83)

        if "data quality" in text or "missing data" in text:
            return IntentSuggestion(intent="data_quality", confidence=0.81)

        if "research" in text:
            return IntentSuggestion(intent="prospect_research", confidence=0.78)

        return IntentSuggestion(intent="unknown", confidence=0.35)


def _coerce_intent_suggestion(override: IntentOverride) -> IntentSuggestion:
    if isinstance(override, IntentSuggestion):
        return override

    if isinstance(override, tuple):
        intent, confidence = override
        return IntentSuggestion(intent=intent, confidence=confidence)

    return IntentSuggestion.model_validate(override)


__all__ = ["IntentOverride", "MockSlmIntentClient", "SlmIntentClient"]
