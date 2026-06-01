import pytest

from orchestrator_demo.contracts import IntentSuggestion, LlmIntentAssessment
from orchestrator_demo.intent.merge import (
    SIMPLE_DIRECT_ROUTE_THRESHOLD,
    merge_intent_confidence,
)


def test_direct_route_threshold_is_prd_value() -> None:
    # Arrange
    expected_threshold = 0.85

    # Act
    threshold = SIMPLE_DIRECT_ROUTE_THRESHOLD

    # Assert
    assert threshold == expected_threshold


@pytest.mark.parametrize(
    ("slm_confidence", "llm_confidence", "expected_confidence"),
    [
        (0.0, 0.0, 0.0),
        (1.0, 1.0, 1.0),
        (0.9, 0.9, 0.9),
        (0.62, 0.91, 0.794),
        (0.3, 0.8, 0.6),
    ],
)
def test_merge_uses_fixed_slm_llm_weighting(
    slm_confidence: float,
    llm_confidence: float,
    expected_confidence: float,
) -> None:
    # Arrange
    slm = IntentSuggestion(intent="internal_knowledge", confidence=slm_confidence)
    llm = LlmIntentAssessment(
        intents=["meeting_prep"],
        confidence=llm_confidence,
        complexity="complex",
        required_agents=["meeting_prep", "synthesis"],
        rationale="Meeting preparation requires multiple steps.",
    )

    # Act
    merged_confidence = merge_intent_confidence(slm, llm)

    # Assert
    assert merged_confidence == pytest.approx(expected_confidence)


def test_disagreement_merge_keeps_low_slm_suggestion_from_direct_confidence() -> None:
    # Arrange
    slm = IntentSuggestion(intent="internal_knowledge", confidence=0.62)
    llm = LlmIntentAssessment(
        intents=[
            "meeting_prep",
            "relationship_summary",
            "internal_knowledge",
            "industry_research",
        ],
        confidence=0.91,
        complexity="complex",
        required_agents=[
            "relationship_summary",
            "internal_knowledge",
            "industry_research",
            "synthesis",
        ],
        rationale="The request implies meeting preparation across multiple sources.",
    )

    # Act
    merged_confidence = merge_intent_confidence(slm, llm)

    # Assert
    assert merged_confidence == pytest.approx(0.794)
    assert merged_confidence < SIMPLE_DIRECT_ROUTE_THRESHOLD
