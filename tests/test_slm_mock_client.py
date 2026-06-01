import pytest

from orchestrator_demo.contracts import IntentSuggestion
from orchestrator_demo.intent.slm_mock_client import (
    MockSlmIntentClient,
    SlmIntentClient,
)


@pytest.mark.asyncio
async def test_slm_client_abstraction_exposes_async_classify() -> None:
    # Arrange
    client: SlmIntentClient = MockSlmIntentClient()

    # Act
    suggestion = await client.classify(
        "Summarize the internal notes for ABC Manufacturing."
    )

    # Assert
    assert isinstance(suggestion, IntentSuggestion)
    assert suggestion.intent == "internal_knowledge"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("user_input", "expected_intent", "expected_confidence"),
    [
        (
            "Prepare me for tomorrow's meeting with ABC Manufacturing.",
            "meeting_prep",
            0.82,
        ),
        (
            "Research this prospect and give me risks, opportunities, and talking points.",
            "prospect_research",
            0.78,
        ),
        (
            "Summarize the internal notes for ABC Manufacturing.",
            "internal_knowledge",
            0.90,
        ),
        (
            "Give me a quick overview of the manufacturing industry.",
            "industry_research",
            0.88,
        ),
        (
            "Research the manufacturing industry.",
            "industry_research",
            0.88,
        ),
        (
            "What product opportunities should I consider for a cafe business?",
            "product_opportunity",
            0.87,
        ),
        (
            "Research product opportunities for a cafe.",
            "product_opportunity",
            0.87,
        ),
        (
            "Pull together what I need before seeing ABC Manufacturing tomorrow.",
            "unknown",
            0.35,
        ),
    ],
)
async def test_mock_slm_returns_deterministic_demo_phrase_suggestions(
    user_input: str,
    expected_intent: str,
    expected_confidence: float,
) -> None:
    # Arrange
    client = MockSlmIntentClient()

    # Act
    suggestion = await client.classify(user_input)

    # Assert
    assert suggestion.model_dump() == {
        "intent": expected_intent,
        "confidence": expected_confidence,
    }


@pytest.mark.asyncio
async def test_mock_slm_supports_low_confidence_override() -> None:
    # Arrange
    user_input = "Summarize the internal notes for ABC Manufacturing."
    client = MockSlmIntentClient(
        overrides={
            user_input: IntentSuggestion(intent="internal_knowledge", confidence=0.41)
        }
    )

    # Act
    suggestion = await client.classify(user_input)

    # Assert
    assert suggestion.model_dump() == {
        "intent": "internal_knowledge",
        "confidence": 0.41,
    }


@pytest.mark.asyncio
async def test_mock_slm_supports_wrong_slm_override() -> None:
    # Arrange
    user_input = "Pull together what I need before seeing ABC Manufacturing tomorrow."
    client = MockSlmIntentClient(
        overrides={
            user_input: {"intent": "internal_knowledge", "confidence": 0.62}
        }
    )

    # Act
    suggestion = await client.classify(user_input)

    # Assert
    assert suggestion.model_dump() == {
        "intent": "internal_knowledge",
        "confidence": 0.62,
    }


@pytest.mark.asyncio
async def test_mock_slm_supports_tuple_override() -> None:
    # Arrange
    user_input = "Summarize the internal notes for ABC Manufacturing."
    client = MockSlmIntentClient(overrides={user_input: ("meeting_prep", 0.74)})

    # Act
    suggestion = await client.classify(user_input)

    # Assert
    assert suggestion.model_dump() == {
        "intent": "meeting_prep",
        "confidence": 0.74,
    }
