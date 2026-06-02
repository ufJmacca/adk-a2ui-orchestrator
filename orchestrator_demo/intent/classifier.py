"""LLM intent assessment abstractions.

The deterministic classifier is the default test/demo implementation. The
LiteLLM-backed classifier uses the same contract and can be wired into app
entrypoints without forcing tests to carry live model credentials.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping, Sequence
import inspect
import json
import textwrap
import unicodedata
from typing import Any, Protocol, cast, runtime_checkable

from orchestrator_demo.contracts import (
    AgentDescriptor,
    IntentName,
    IntentSuggestion,
    LlmIntentAssessment,
)


CompletionCallable = Callable[[str], Awaitable[str | Mapping[str, Any]]]
_ASSESSMENT_PAYLOAD_KEYS = frozenset(
    {"intents", "confidence", "complexity", "required_agents", "rationale"}
)
_SENSITIVE_GUARDRAIL_INTENTS: tuple[IntentName, ...] = (
    "credit_risk",
    "compliance_policy",
)
_SENSITIVE_GUARDRAIL_AGENTS = ("credit_risk", "compliance_policy")
_SYNTHESIS_AGENT_ID = "synthesis"


class ClassifierUnavailableAgentsError(ValueError):
    """Raised when an assessment selects agents outside the current registry."""

    def __init__(
        self,
        assessment: LlmIntentAssessment,
        unavailable_agent_ids: Sequence[str],
    ) -> None:
        self.assessment = assessment
        self.unavailable_agent_ids = list(unavailable_agent_ids)
        unavailable = ", ".join(self.unavailable_agent_ids)
        super().__init__(
            f"classifier assessment requires unavailable agents: {unavailable}"
        )


@runtime_checkable
class IntentClassifier(Protocol):
    """Common contract for deterministic and model-backed assessment."""

    async def assess(
        self,
        user_input: str,
        slm_suggestion: IntentSuggestion,
        *,
        available_agents: Sequence[AgentDescriptor] | None = None,
    ) -> LlmIntentAssessment:
        """Return an enhanced intent assessment for a user request."""
        ...


class DeterministicIntentClassifier:
    """Rule-based classifier covering the local demo scenarios."""

    async def assess(
        self,
        user_input: str,
        slm_suggestion: IntentSuggestion,
        *,
        available_agents: Sequence[AgentDescriptor] | None = None,
    ) -> LlmIntentAssessment:
        del slm_suggestion

        text = _normalize_text(user_input)

        if _is_prospect_research(text):
            intents, required_agents, rationale = _add_sensitive_guardrails_if_needed(
                text=text,
                intents=[
                    "prospect_research",
                    "web_search",
                    "industry_research",
                    "product_opportunity",
                    "credit_risk",
                ],
                required_agents=[
                    "web_search",
                    "industry_research",
                    "product_opportunity",
                    "credit_risk",
                    "synthesis",
                ],
                rationale=(
                    "Prospect research requires public research, sector context, "
                    "risk review, opportunity analysis, and synthesis."
                ),
            )
            assessment = _assessment(
                intents=intents,
                confidence=0.90,
                complexity="complex",
                required_agents=required_agents,
                rationale=rationale,
            )
        elif _is_relationship_and_industry_comparison(text):
            intents, required_agents, rationale = _add_sensitive_guardrails_if_needed(
                text=text,
                intents=[
                    "relationship_summary",
                    "industry_research",
                    "credit_risk",
                    "meeting_prep",
                ],
                required_agents=[
                    "relationship_summary",
                    "industry_research",
                    "credit_risk",
                    "synthesis",
                ],
                rationale=(
                    "Comparing relationship history with industry risks requires "
                    "multiple sources and synthesis."
                ),
            )
            assessment = _assessment(
                intents=intents,
                confidence=0.88,
                complexity="complex",
                required_agents=required_agents,
                rationale=rationale,
            )
        elif _is_meeting_prep(text):
            intents, required_agents, rationale = _add_sensitive_guardrails_if_needed(
                text=text,
                intents=[
                    "meeting_prep",
                    "relationship_summary",
                    "internal_knowledge",
                    "industry_research",
                ],
                required_agents=[
                    "relationship_summary",
                    "internal_knowledge",
                    "industry_research",
                    "synthesis",
                ],
                rationale=(
                    "Meeting preparation requires relationship context, internal "
                    "notes, industry context, and final synthesis."
                ),
            )
            assessment = _assessment(
                intents=intents,
                confidence=0.91,
                complexity="complex",
                required_agents=required_agents,
                rationale=rationale,
            )
        elif _is_sensitive_credit_or_compliance(text):
            assessment = _assessment(
                intents=["credit_risk", "compliance_policy"],
                confidence=0.87,
                complexity="complex",
                required_agents=["credit_risk", "compliance_policy", "synthesis"],
                rationale=(
                    "Sensitive credit, risk, or compliance language requires "
                    "review before producing an RM-facing answer."
                ),
            )
        elif _is_standalone_risk_request(text):
            assessment = _assessment(
                intents=["credit_risk"],
                confidence=0.89,
                complexity="simple",
                required_agents=["credit_risk"],
                rationale=(
                    "Standalone customer risk language requires guarded "
                    "credit-risk review before producing an RM-facing answer."
                ),
            )
        elif _contains_any(text, ("internal notes", "crm", "relationship notes")):
            assessment = _assessment(
                intents=["internal_knowledge"],
                confidence=0.93,
                complexity="simple",
                required_agents=["internal_knowledge"],
                rationale=(
                    "The request asks for internal notes and can be handled by "
                    "one internal knowledge specialist."
                ),
            )
        elif _contains_any(text, ("product opportunities", "product opportunity")):
            assessment = _assessment(
                intents=["product_opportunity"],
                confidence=0.90,
                complexity="simple",
                required_agents=["product_opportunity"],
                rationale=(
                    "A single product opportunity specialist can answer this "
                    "focused product opportunity question."
                ),
            )
        elif _contains_any(text, ("web search", "public information", "recent public")):
            assessment = _assessment(
                intents=["web_search"],
                confidence=0.89,
                complexity="simple",
                required_agents=["web_search"],
                rationale=(
                    "The request asks for public information and can be routed "
                    "to one web search specialist."
                ),
            )
        elif _is_industry_research(text):
            assessment = _assessment(
                intents=["industry_research"],
                confidence=0.90,
                complexity="simple",
                required_agents=["industry_research"],
                rationale=(
                    "A quick industry overview is a single-intent request for "
                    "the industry research specialist."
                ),
            )
        elif _contains_any(text, ("relationship history", "relationship summary")):
            assessment = _assessment(
                intents=["relationship_summary"],
                confidence=0.88,
                complexity="simple",
                required_agents=["relationship_summary"],
                rationale=(
                    "The request asks for one relationship summary from a "
                    "single specialist."
                ),
            )
        elif _contains_any(text, ("data quality", "missing data", "stale context")):
            assessment = _assessment(
                intents=["data_quality"],
                confidence=0.84,
                complexity="simple",
                required_agents=["data_quality"],
                rationale=(
                    "The request is focused on data quality and can be handled "
                    "by one specialist."
                ),
            )
        else:
            assessment = _assessment(
                intents=["unknown"],
                confidence=0.42,
                complexity="complex",
                required_agents=["data_quality"],
                rationale=(
                    "The request is ambiguous and does not identify a safe "
                    "single owner agent."
                ),
            )

        _require_available_agents(assessment, available_agents)
        return assessment


class LiteLlmIntentClassifier:
    """LiteLLM-backed assessment using the same classifier contract."""

    def __init__(
        self,
        *,
        model: Any | None = None,
        completion: CompletionCallable | None = None,
        model_name: str | None = None,
    ) -> None:
        self._model = model
        self._completion = completion
        self._model_name = model_name

    async def assess(
        self,
        user_input: str,
        slm_suggestion: IntentSuggestion,
        *,
        available_agents: Sequence[AgentDescriptor] | None = None,
    ) -> LlmIntentAssessment:
        prompt = _build_classifier_prompt(user_input, slm_suggestion, available_agents)
        raw_response = await self._complete(prompt)
        assessment = LlmIntentAssessment.model_validate(
            _coerce_assessment_payload(raw_response)
        )
        _require_available_agents(assessment, available_agents)

        return assessment

    async def _complete(self, prompt: str) -> Any:
        if self._completion is not None:
            return await self._completion(prompt)

        if self._model is not None:
            return await _complete_with_injected_model(self._model, prompt)

        return await _complete_with_litellm(prompt, self._model_name)


def _assessment(
    *,
    intents: list[IntentName],
    confidence: float,
    complexity: str,
    required_agents: list[str],
    rationale: str,
) -> LlmIntentAssessment:
    return LlmIntentAssessment(
        intents=intents,
        confidence=confidence,
        complexity=cast(Any, complexity),
        required_agents=required_agents,
        rationale=rationale,
    )


def _add_sensitive_guardrails_if_needed(
    *,
    text: str,
    intents: list[IntentName],
    required_agents: list[str],
    rationale: str,
) -> tuple[list[IntentName], list[str], str]:
    if not _is_sensitive_credit_or_compliance(text):
        return intents, required_agents, rationale

    return (
        _dedupe_intents([*intents, *_SENSITIVE_GUARDRAIL_INTENTS]),
        _append_agents_before_synthesis(
            required_agents,
            _SENSITIVE_GUARDRAIL_AGENTS,
        ),
        (
            f"{rationale} Sensitive credit, loan, or compliance language adds "
            "credit and policy guardrails."
        ),
    )


def _append_agents_before_synthesis(
    agent_ids: Sequence[str],
    additions: Sequence[str],
) -> list[str]:
    ordered_agent_ids = [
        agent_id for agent_id in agent_ids if agent_id != _SYNTHESIS_AGENT_ID
    ]
    for agent_id in additions:
        if agent_id not in ordered_agent_ids:
            ordered_agent_ids.append(agent_id)
    if _SYNTHESIS_AGENT_ID in agent_ids:
        ordered_agent_ids.append(_SYNTHESIS_AGENT_ID)

    return ordered_agent_ids


def _dedupe_intents(values: Sequence[IntentName]) -> list[IntentName]:
    deduped: list[IntentName] = []
    for value in values:
        if value not in deduped:
            deduped.append(value)

    return deduped


def _normalize_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text)
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii")
    return " ".join(ascii_text.casefold().split())


def _is_prospect_research(text: str) -> bool:
    if "prospect" in text:
        return True

    return (
        "research" in text
        and not _is_product_opportunity(text)
        and _contains_any(text, ("risks", "opportunities", "talking points"))
    )


def _is_meeting_prep(text: str) -> bool:
    return _contains_any(text, ("meeting", "prepare me", "talking points")) or (
        _contains_any(text, ("before seeing", "seeing"))
        and _contains_any(text, ("tomorrow", "pull together", "what i need"))
    )


def _is_sensitive_credit_or_compliance(text: str) -> bool:
    return _contains_any(
        text,
        (
            "approved",
            "approval",
            "credit",
            "loan",
            "covenant",
            "repayment",
            "risk rating",
            "compliance",
            "policy",
            "regulated",
            "unsupported claim",
            "advisory",
        ),
    )


def _is_relationship_and_industry_comparison(text: str) -> bool:
    return (
        _contains_any(text, ("compare", "relationship history"))
        and _contains_any(text, ("industry", "sector", "risk"))
        and _contains_any(text, ("priorities", "suggest", "recommend"))
    )


def _is_product_opportunity(text: str) -> bool:
    return _contains_any(text, ("product opportunities", "product opportunity"))


def _is_industry_research(text: str) -> bool:
    return _contains_any(
        text,
        (
            "industry",
            "sector",
            "market risk",
            "market risks",
            "retail trade",
            "retail risk",
            "retail risks",
        ),
    )


def _is_standalone_risk_request(text: str) -> bool:
    if _is_industry_research(text):
        return False

    return _contains_word(text, {"risk", "risks"})


def _contains_any(text: str, needles: Sequence[str]) -> bool:
    return any(needle in text for needle in needles)


def _contains_word(text: str, words: set[str]) -> bool:
    tokens = {
        token.strip(".,;:!?()[]{}")
        for token in text.split()
    }
    return bool(tokens.intersection(words))


def _require_available_agents(
    assessment: LlmIntentAssessment,
    available_agents: Sequence[AgentDescriptor] | None,
) -> None:
    if available_agents is None:
        return

    available_agent_ids = {descriptor.agent_id for descriptor in available_agents}
    unavailable_agent_ids = [
        agent_id
        for agent_id in assessment.required_agents
        if agent_id not in available_agent_ids
    ]
    if unavailable_agent_ids:
        raise ClassifierUnavailableAgentsError(assessment, unavailable_agent_ids)


def _build_classifier_prompt(
    user_input: str,
    slm_suggestion: IntentSuggestion,
    available_agents: Sequence[AgentDescriptor] | None,
) -> str:
    available_agent_ids = (
        [descriptor.agent_id for descriptor in available_agents]
        if available_agents is not None
        else None
    )

    return textwrap.dedent(
        f"""
        Assess the business banking relationship-manager request.

        Return only JSON matching:
        {{
          "intents": ["meeting_prep"],
          "confidence": 0.91,
          "complexity": "simple|complex",
          "required_agents": ["relationship_summary"],
          "rationale": "short reason"
        }}

        User request: {user_input}
        SLM suggestion: {slm_suggestion.model_dump_json()}
        Available agents: {json.dumps(available_agent_ids)}
        """
    ).strip()


async def _complete_with_injected_model(model: Any, prompt: str) -> Any:
    if hasattr(model, "generate_content_async"):
        method = model.generate_content_async
        if _is_adk_generate_content_model(model, method):
            request = _build_adk_llm_request(prompt)
            return await _resolve_model_result(method(request))

        try:
            return await _resolve_model_result(method(prompt))
        except (AttributeError, TypeError):
            request = _build_adk_llm_request(prompt)
            return await _resolve_model_result(method(request))

    if hasattr(model, "ainvoke"):
        return await _resolve_model_result(model.ainvoke(prompt))

    if callable(model):
        return await _resolve_model_result(model(prompt))

    raise TypeError("model must expose generate_content_async, ainvoke, or be callable")


def _is_adk_generate_content_model(model: Any, method: Any) -> bool:
    model_type = type(model)
    if model_type.__module__.startswith("google.adk."):
        return True

    try:
        signature = inspect.signature(method)
    except (TypeError, ValueError):
        return False

    for parameter in signature.parameters.values():
        if parameter.name in {"llm_request", "request"}:
            return True

        annotation = parameter.annotation
        if annotation is inspect.Signature.empty:
            continue

        if "LlmRequest" in str(annotation):
            return True

    return False


async def _complete_with_litellm(prompt: str, model_name: str | None) -> Any:
    from orchestrator_demo.app.bootstrap_llm import configure_litellm_environment
    from orchestrator_demo.app.settings import load_settings

    settings = load_settings()
    configure_litellm_environment(settings)
    from litellm import acompletion

    resolved_model_name = model_name if model_name is not None else settings.llm_model
    return await acompletion(
        model=resolved_model_name,
        messages=[
            {
                "role": "system",
                "content": (
                    "You classify business banking relationship-manager requests. "
                    "Return only valid JSON."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        response_format={"type": "json_object"},
    )


async def _resolve_model_result(result: Any) -> Any:
    if inspect.isasyncgen(result):
        chunks: list[str] = []
        async for chunk in result:
            chunks.append(_text_from_response_chunk(chunk))
        return "".join(chunks)

    if inspect.isawaitable(result):
        return await result

    return result


def _build_adk_llm_request(prompt: str) -> Any:
    from google.adk.models.llm_request import LlmRequest
    from google.genai import types

    return LlmRequest(
        contents=[
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=prompt)],
            )
        ]
    )


def _text_from_response_chunk(chunk: Any) -> str:
    if isinstance(chunk, str):
        return chunk

    error_message = getattr(chunk, "error_message", None)
    if error_message:
        raise RuntimeError("LiteLLM classifier model returned an error")

    content = getattr(chunk, "content", None)
    parts = getattr(content, "parts", None)
    if parts is None:
        return ""

    return "".join(
        part_text
        for part in parts
        if isinstance((part_text := getattr(part, "text", None)), str)
    )


def _coerce_assessment_payload(raw_response: Any) -> Mapping[str, Any]:
    if isinstance(raw_response, Mapping):
        if "choices" in raw_response:
            return _coerce_assessment_payload(_choice_content(raw_response["choices"]))
        return raw_response

    if isinstance(raw_response, str):
        return _extract_json_object(raw_response)

    choices = getattr(raw_response, "choices", None)
    if choices is not None:
        return _coerce_assessment_payload(_choice_content(choices))

    content = getattr(raw_response, "content", None)
    if isinstance(content, str):
        return _extract_json_object(content)

    raise TypeError("model response did not contain a JSON assessment")


def _choice_content(choices: Any) -> Any:
    first_choice = choices[0]
    if isinstance(first_choice, Mapping):
        message = first_choice.get("message", {})
        if isinstance(message, Mapping):
            return message.get("content", "")
        return ""

    message = getattr(first_choice, "message", None)
    return getattr(message, "content", "")


def _extract_json_object(text: str) -> Mapping[str, Any]:
    decoder = json.JSONDecoder()
    start = text.find("{")
    last_candidate: Mapping[str, Any] | None = None
    last_object: Mapping[str, Any] | None = None
    while start != -1:
        try:
            payload, _ = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            start = text.find("{", start + 1)
            continue

        if isinstance(payload, Mapping):
            last_object = payload
            if _ASSESSMENT_PAYLOAD_KEYS.issubset(payload.keys()):
                last_candidate = payload

        start = text.find("{", start + 1)

    if last_candidate is not None:
        return last_candidate

    if last_object is not None:
        return last_object

    raise ValueError("model response did not include a JSON object")


__all__ = [
    "ClassifierUnavailableAgentsError",
    "CompletionCallable",
    "DeterministicIntentClassifier",
    "IntentClassifier",
    "LiteLlmIntentClassifier",
]
