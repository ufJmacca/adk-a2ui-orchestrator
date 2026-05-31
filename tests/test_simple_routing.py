import pytest

from orchestrator_demo.contracts import (
    AgentDescriptor,
    IntentSuggestion,
    LlmIntentAssessment,
)
from orchestrator_demo.intent.classifier import DeterministicIntentClassifier
from orchestrator_demo.intent.slm_mock_client import MockSlmIntentClient
from orchestrator_demo.orchestrator.router import RequestRouter
from orchestrator_demo.registry.agent_registry import AgentRegistry


class _RecordingSlmClient:
    def __init__(self, suggestion: IntentSuggestion) -> None:
        self._suggestion = suggestion
        self.calls: list[str] = []

    async def classify(self, user_input: str) -> IntentSuggestion:
        self.calls.append(user_input)
        return self._suggestion


class _RecordingClassifier:
    def __init__(
        self,
        assessment: LlmIntentAssessment,
        slm_client: _RecordingSlmClient,
    ) -> None:
        self._assessment = assessment
        self._slm_client = slm_client
        self.assess_calls: list[tuple[str, IntentSuggestion]] = []
        self.slm_call_counts_seen: list[int] = []

    async def assess(
        self,
        user_input: str,
        slm_suggestion: IntentSuggestion,
        *,
        available_agents: object | None = None,
    ) -> LlmIntentAssessment:
        del available_agents
        self.assess_calls.append((user_input, slm_suggestion))
        self.slm_call_counts_seen.append(len(self._slm_client.calls))
        return self._assessment


class _StaticRegistry:
    def __init__(self, descriptors: list[AgentDescriptor]) -> None:
        self._descriptors = descriptors

    def descriptors(self) -> list[AgentDescriptor]:
        return list(self._descriptors)


@pytest.mark.asyncio
async def test_slm_called_exactly_once_before_route_decision() -> None:
    # Arrange
    user_input = "Summarize the internal notes for ABC Manufacturing."
    slm_suggestion = IntentSuggestion(intent="internal_knowledge", confidence=0.9)
    slm_client = _RecordingSlmClient(slm_suggestion)
    classifier = _RecordingClassifier(
        LlmIntentAssessment(
            intents=["internal_knowledge"],
            confidence=0.93,
            complexity="simple",
            required_agents=["internal_knowledge"],
            rationale="Internal notes require one specialist.",
        ),
        slm_client,
    )
    router = RequestRouter(
        slm_client=slm_client,
        intent_classifier=classifier,
        registry=AgentRegistry.from_default_config(),
    )

    # Act
    context = await router.route_request(user_input)

    # Assert
    assert slm_client.calls == [user_input]
    assert classifier.assess_calls == [(user_input, slm_suggestion)]
    assert classifier.slm_call_counts_seen == [1]
    assert context.decision.path == "direct"


@pytest.mark.asyncio
async def test_simple_route_above_threshold_selects_single_agent_directly() -> None:
    # Arrange
    user_input = "Summarize the internal notes for ABC Manufacturing."
    router = RequestRouter(
        slm_client=MockSlmIntentClient(),
        intent_classifier=DeterministicIntentClassifier(),
        registry=AgentRegistry.from_default_config(),
    )

    # Act
    context = await router.route_request(user_input)

    # Assert
    assert context.decision.path == "direct"
    assert context.decision.selected_agent == "internal_knowledge"
    assert context.decision.confidence >= 0.85
    assert context.llm_assessment.intents == ["internal_knowledge"]
    assert context.llm_assessment.required_agents == ["internal_knowledge"]


@pytest.mark.asyncio
async def test_unavailable_classifier_agent_returns_clarification_required() -> None:
    # Arrange
    user_input = "Summarize the internal notes for ABC Manufacturing."
    default_registry = AgentRegistry.from_default_config()
    available_descriptors = [
        descriptor
        for descriptor in default_registry.descriptors()
        if descriptor.agent_id != "internal_knowledge"
    ]
    slm_client = _RecordingSlmClient(
        IntentSuggestion(intent="internal_knowledge", confidence=0.9)
    )
    router = RequestRouter(
        slm_client=slm_client,
        intent_classifier=DeterministicIntentClassifier(),
        registry=_StaticRegistry(available_descriptors),
    )

    # Act
    context = await router.route_request(user_input)

    # Assert
    assert slm_client.calls == [user_input]
    assert context.llm_assessment.required_agents == ["internal_knowledge"]
    assert context.decision.path == "clarification_required"
    assert context.decision.selected_agent is None
    assert "unavailable" in context.decision.reason.casefold()


@pytest.mark.asyncio
async def test_high_confidence_single_agent_complex_assessment_requires_plan() -> None:
    # Arrange
    user_input = "Summarize the internal notes and prepare a follow-up workflow."
    slm_suggestion = IntentSuggestion(intent="internal_knowledge", confidence=0.95)
    slm_client = _RecordingSlmClient(slm_suggestion)
    classifier = _RecordingClassifier(
        LlmIntentAssessment(
            intents=["internal_knowledge"],
            confidence=0.96,
            complexity="complex",
            required_agents=["internal_knowledge"],
            rationale="The request asks for a workflow, so approval is required.",
        ),
        slm_client,
    )
    router = RequestRouter(
        slm_client=slm_client,
        intent_classifier=classifier,
        registry=AgentRegistry.from_default_config(),
    )

    # Act
    context = await router.route_request(user_input)

    # Assert
    assert context.decision.confidence >= 0.85
    assert context.llm_assessment.intents == ["internal_knowledge"]
    assert context.llm_assessment.required_agents == ["internal_knowledge"]
    assert context.llm_assessment.complexity == "complex"
    assert context.decision.path == "plan_required"
    assert context.decision.selected_agent is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("intents", "required_agents"),
    [
        (
            ["internal_knowledge", "relationship_summary"],
            ["internal_knowledge"],
        ),
        (
            ["internal_knowledge"],
            ["internal_knowledge", "relationship_summary"],
        ),
    ],
)
async def test_high_confidence_simple_assessment_with_multiple_intents_or_agents_requires_plan(
    intents: list[str],
    required_agents: list[str],
) -> None:
    # Arrange
    user_input = "Summarize the internal notes and relationship history."
    slm_suggestion = IntentSuggestion(intent="internal_knowledge", confidence=0.95)
    slm_client = _RecordingSlmClient(slm_suggestion)
    classifier = _RecordingClassifier(
        LlmIntentAssessment(
            intents=intents,
            confidence=0.96,
            complexity="simple",
            required_agents=required_agents,
            rationale="The assessment is high confidence but not single-owner.",
        ),
        slm_client,
    )
    router = RequestRouter(
        slm_client=slm_client,
        intent_classifier=classifier,
        registry=AgentRegistry.from_default_config(),
    )

    # Act
    context = await router.route_request(user_input)

    # Assert
    assert context.decision.confidence >= 0.85
    assert context.llm_assessment.complexity == "simple"
    assert context.decision.path == "plan_required"
    assert context.decision.selected_agent is None


@pytest.mark.asyncio
async def test_low_confidence_simple_intent_routes_to_plan() -> None:
    # Arrange
    user_input = "Review data quality for ABC Manufacturing."
    router = RequestRouter(
        slm_client=MockSlmIntentClient(),
        intent_classifier=DeterministicIntentClassifier(),
        registry=AgentRegistry.from_default_config(),
    )

    # Act
    context = await router.route_request(user_input)

    # Assert
    assert context.llm_assessment.intents == ["data_quality"]
    assert context.llm_assessment.required_agents == ["data_quality"]
    assert context.decision.confidence < 0.85
    assert context.decision.path == "plan_required"
    assert context.decision.selected_agent is None


@pytest.mark.asyncio
async def test_sensitive_single_agent_request_requires_plan_not_direct_route() -> None:
    # Arrange
    user_input = "Flag the credit risks for ABC Manufacturing."
    slm_client = _RecordingSlmClient(
        IntentSuggestion(intent="credit_risk", confidence=0.95)
    )
    classifier = _RecordingClassifier(
        LlmIntentAssessment(
            intents=["credit_risk"],
            confidence=0.96,
            complexity="simple",
            required_agents=["credit_risk"],
            rationale="A focused credit risk request has regulated-output sensitivity.",
        ),
        slm_client,
    )
    router = RequestRouter(
        slm_client=slm_client,
        intent_classifier=classifier,
        registry=AgentRegistry.from_default_config(),
    )

    # Act
    context = await router.route_request(user_input)

    # Assert
    assert context.decision.confidence >= 0.85
    assert context.decision.path == "plan_required"
    assert context.decision.selected_agent is None
    assert "sensitive" in context.decision.reason.casefold()
