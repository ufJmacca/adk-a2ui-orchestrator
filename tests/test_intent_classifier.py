import json
from pathlib import Path
from typing import Any

import pytest

from orchestrator_demo.contracts import IntentSuggestion, LlmIntentAssessment
from orchestrator_demo.intent.classifier import (
    DeterministicIntentClassifier,
    LiteLlmIntentClassifier,
)
from orchestrator_demo.registry.agent_registry import AgentRegistry


GOLDEN_INTENTS_PATH = Path(__file__).with_name("golden_intents.json")


def _golden_cases() -> list[dict[str, Any]]:
    return json.loads(GOLDEN_INTENTS_PATH.read_text())


@pytest.mark.asyncio
@pytest.mark.parametrize("case", _golden_cases(), ids=lambda case: case["name"])
async def test_deterministic_classifier_covers_demo_intent_scenarios(
    case: dict[str, Any],
) -> None:
    # Arrange
    classifier = DeterministicIntentClassifier()
    registry = AgentRegistry.from_default_config()
    slm_suggestion = IntentSuggestion.model_validate(case["slm"])

    # Act
    assessment = await classifier.assess(
        case["input"],
        slm_suggestion,
        available_agents=registry.descriptors(),
    )

    # Assert
    assert isinstance(assessment, LlmIntentAssessment)
    assert assessment.intents == case["expected_intents"]
    assert assessment.complexity == case["expected_complexity"]
    assert assessment.required_agents == case["expected_required_agents"]
    assert assessment.confidence == pytest.approx(case["expected_confidence"])
    assert case["expected_rationale_contains"] in assessment.rationale.casefold()


@pytest.mark.asyncio
async def test_classifier_treats_slm_result_as_non_binding_suggestion() -> None:
    # Arrange
    classifier = DeterministicIntentClassifier()
    slm_suggestion = IntentSuggestion(intent="internal_knowledge", confidence=0.62)

    # Act
    assessment = await classifier.assess(
        "Pull together what I need before seeing ABC Manufacturing tomorrow.",
        slm_suggestion,
    )

    # Assert
    assert assessment.complexity == "complex"
    assert assessment.intents == [
        "meeting_prep",
        "relationship_summary",
        "internal_knowledge",
        "industry_research",
    ]
    assert assessment.required_agents == [
        "relationship_summary",
        "internal_knowledge",
        "industry_research",
        "synthesis",
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("user_input", "slm_suggestion", "expected_intents", "expected_required_agents"),
    [
        (
            (
                "Prepare me for tomorrow's meeting with ABC Manufacturing and "
                "include loan compliance guardrails."
            ),
            IntentSuggestion(intent="meeting_prep", confidence=0.82),
            [
                "meeting_prep",
                "relationship_summary",
                "internal_knowledge",
                "industry_research",
                "credit_risk",
                "compliance_policy",
            ],
            [
                "relationship_summary",
                "internal_knowledge",
                "industry_research",
                "credit_risk",
                "compliance_policy",
                "synthesis",
            ],
        ),
        (
            (
                "Research this prospect for credit and compliance risks, "
                "opportunities, and talking points."
            ),
            IntentSuggestion(intent="prospect_research", confidence=0.78),
            [
                "prospect_research",
                "web_search",
                "industry_research",
                "product_opportunity",
                "credit_risk",
                "compliance_policy",
            ],
            [
                "web_search",
                "industry_research",
                "product_opportunity",
                "credit_risk",
                "compliance_policy",
                "synthesis",
            ],
        ),
    ],
)
async def test_classifier_preserves_primary_workflow_with_sensitive_guardrails(
    user_input: str,
    slm_suggestion: IntentSuggestion,
    expected_intents: list[str],
    expected_required_agents: list[str],
) -> None:
    # Arrange
    classifier = DeterministicIntentClassifier()

    # Act
    assessment = await classifier.assess(
        user_input,
        slm_suggestion,
        available_agents=AgentRegistry.from_default_config().descriptors(),
    )

    # Assert
    assert assessment.intents == expected_intents
    assert assessment.required_agents == expected_required_agents
    assert assessment.complexity == "complex"
    assert "guardrails" in assessment.rationale.casefold()


@pytest.mark.asyncio
async def test_ambiguous_input_returns_low_confidence_unknown_assessment() -> None:
    # Arrange
    classifier = DeterministicIntentClassifier()
    slm_suggestion = IntentSuggestion(intent="unknown", confidence=0.35)

    # Act
    assessment = await classifier.assess("Help me with ABC.", slm_suggestion)

    # Assert
    assert assessment.intents == ["unknown"]
    assert assessment.required_agents == ["data_quality"]
    assert assessment.complexity == "complex"
    assert assessment.confidence < 0.85
    assert "ambiguous" in assessment.rationale.casefold()


@pytest.mark.asyncio
async def test_deterministic_classifier_requires_available_agents() -> None:
    # Arrange
    classifier = DeterministicIntentClassifier()
    slm_suggestion = IntentSuggestion(intent="internal_knowledge", confidence=0.9)
    registry = AgentRegistry.from_default_config()
    available_agents = [
        descriptor
        for descriptor in registry.descriptors()
        if descriptor.agent_id != "internal_knowledge"
    ]

    # Act / Assert
    with pytest.raises(ValueError, match="unavailable agents: internal_knowledge"):
        await classifier.assess(
            "Summarize the internal notes for ABC Manufacturing.",
            slm_suggestion,
            available_agents=available_agents,
        )


class _FakeLiteLlmModel:
    async def generate_content_async(self, prompt: str) -> str:
        assert "Summarize the internal notes for ABC Manufacturing." in prompt

        return json.dumps(
            {
                "intents": ["internal_knowledge"],
                "confidence": 0.94,
                "complexity": "simple",
                "required_agents": ["internal_knowledge"],
                "rationale": "Fake model assessed an internal notes request.",
            }
        )


class _FakeAdkLiteLlmModel:
    __module__ = "google.adk.models.lite_llm"

    def __init__(self) -> None:
        self.request: Any | None = None

    def generate_content_async(self, request: Any) -> Any:
        self.request = request

        async def stream() -> Any:
            yield json.dumps(
                {
                    "intents": ["internal_knowledge"],
                    "confidence": 0.94,
                    "complexity": "simple",
                    "required_agents": ["internal_knowledge"],
                    "rationale": "Fake ADK model assessed an internal notes request.",
                }
            )

        return stream()


@pytest.mark.asyncio
async def test_litellm_classifier_uses_injected_model_without_live_credentials() -> None:
    # Arrange
    classifier = LiteLlmIntentClassifier(model=_FakeLiteLlmModel())
    slm_suggestion = IntentSuggestion(intent="internal_knowledge", confidence=0.9)

    # Act
    assessment = await classifier.assess(
        "Summarize the internal notes for ABC Manufacturing.",
        slm_suggestion,
    )

    # Assert
    assert assessment == LlmIntentAssessment(
        intents=["internal_knowledge"],
        confidence=0.94,
        complexity="simple",
        required_agents=["internal_knowledge"],
        rationale="Fake model assessed an internal notes request.",
    )


@pytest.mark.asyncio
async def test_litellm_classifier_passes_llm_request_to_injected_adk_model() -> None:
    # Arrange
    model = _FakeAdkLiteLlmModel()
    classifier = LiteLlmIntentClassifier(model=model)
    slm_suggestion = IntentSuggestion(intent="internal_knowledge", confidence=0.9)

    # Act
    assessment = await classifier.assess(
        "Summarize the internal notes for ABC Manufacturing.",
        slm_suggestion,
    )

    # Assert
    assert model.request is not None
    assert type(model.request).__name__ == "LlmRequest"
    assert "Summarize the internal notes" in model.request.contents[0].parts[0].text
    assert assessment == LlmIntentAssessment(
        intents=["internal_knowledge"],
        confidence=0.94,
        complexity="simple",
        required_agents=["internal_knowledge"],
        rationale="Fake ADK model assessed an internal notes request.",
    )


@pytest.mark.asyncio
async def test_litellm_classifier_validates_settings_before_importing_litellm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Arrange
    import importlib.machinery
    import sys

    from orchestrator_demo.app.settings import ConfigurationError

    events: list[str] = []

    class FakeLiteLlmLoader:
        def create_module(self, _spec: Any) -> None:
            return None

        def exec_module(self, module: Any) -> None:
            events.append("litellm_import")

    class FakeLiteLlmFinder:
        def find_spec(
            self,
            fullname: str,
            _path: Any = None,
            _target: Any = None,
        ) -> Any:
            if fullname == "litellm":
                return importlib.machinery.ModuleSpec(fullname, FakeLiteLlmLoader())

            return None

    def fake_load_settings() -> None:
        events.append("load_settings")
        raise ConfigurationError("Missing required runtime configuration")

    monkeypatch.delitem(sys.modules, "litellm", raising=False)
    monkeypatch.setattr(sys, "meta_path", [FakeLiteLlmFinder(), *sys.meta_path])
    monkeypatch.setattr("orchestrator_demo.app.settings.load_settings", fake_load_settings)
    classifier = LiteLlmIntentClassifier()
    slm_suggestion = IntentSuggestion(intent="internal_knowledge", confidence=0.9)

    # Act / Assert
    with pytest.raises(ConfigurationError, match="Missing required runtime configuration"):
        await classifier.assess(
            "Summarize the internal notes for ABC Manufacturing.",
            slm_suggestion,
        )

    assert events == ["load_settings"]
    assert "litellm" not in sys.modules


@pytest.mark.asyncio
async def test_litellm_classifier_default_path_uses_litellm_without_live_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Arrange
    from orchestrator_demo.app.settings import Settings

    settings = Settings(
        OPENROUTER_API_KEY="unit-test-openrouter-key",
        LLM_MODEL="openrouter/unit-test/model",
    )
    configured_settings: list[Settings] = []
    captured_completion_kwargs: dict[str, Any] = {}

    def fake_load_settings() -> Settings:
        return settings

    def fake_configure_litellm_environment(configured: Settings) -> None:
        configured_settings.append(configured)

    async def fake_acompletion(**kwargs: Any) -> dict[str, Any]:
        captured_completion_kwargs.update(kwargs)
        return {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "intents": ["internal_knowledge"],
                                "confidence": 0.94,
                                "complexity": "simple",
                                "required_agents": ["internal_knowledge"],
                                "rationale": (
                                    "LiteLLM assessed an internal notes request."
                                ),
                            }
                        )
                    }
                }
            ]
        }

    monkeypatch.setattr("orchestrator_demo.app.settings.load_settings", fake_load_settings)
    monkeypatch.setattr(
        "orchestrator_demo.app.bootstrap_llm.configure_litellm_environment",
        fake_configure_litellm_environment,
    )
    monkeypatch.setattr("litellm.acompletion", fake_acompletion)
    classifier = LiteLlmIntentClassifier()
    slm_suggestion = IntentSuggestion(intent="internal_knowledge", confidence=0.9)

    # Act
    assessment = await classifier.assess(
        "Summarize the internal notes for ABC Manufacturing.",
        slm_suggestion,
    )

    # Assert
    assert configured_settings == [settings]
    assert captured_completion_kwargs["model"] == "openrouter/unit-test/model"
    assert captured_completion_kwargs["response_format"] == {"type": "json_object"}
    assert captured_completion_kwargs["messages"][0]["role"] == "system"
    assert captured_completion_kwargs["messages"][1]["role"] == "user"
    assert "Summarize the internal notes" in captured_completion_kwargs["messages"][1][
        "content"
    ]
    assert assessment == LlmIntentAssessment(
        intents=["internal_knowledge"],
        confidence=0.94,
        complexity="simple",
        required_agents=["internal_knowledge"],
        rationale="LiteLLM assessed an internal notes request.",
    )
