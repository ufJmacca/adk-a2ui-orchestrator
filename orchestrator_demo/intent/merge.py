"""Confidence merge support for intent routing."""

from orchestrator_demo.contracts import IntentSuggestion, LlmIntentAssessment


SIMPLE_DIRECT_ROUTE_THRESHOLD = 0.85
SLM_CONFIDENCE_WEIGHT = 0.4
LLM_CONFIDENCE_WEIGHT = 0.6


def merge_intent_confidence(
    slm_suggestion: IntentSuggestion,
    llm_assessment: LlmIntentAssessment,
) -> float:
    """Merge SLM and LLM confidence using the PRD's fixed weighting."""

    return (
        SLM_CONFIDENCE_WEIGHT * slm_suggestion.confidence
        + LLM_CONFIDENCE_WEIGHT * llm_assessment.confidence
    )


__all__ = [
    "LLM_CONFIDENCE_WEIGHT",
    "SIMPLE_DIRECT_ROUTE_THRESHOLD",
    "SLM_CONFIDENCE_WEIGHT",
    "merge_intent_confidence",
]
