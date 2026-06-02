"""Routing engine for direct specialist calls versus approved plan workflows."""

from __future__ import annotations

from collections.abc import Sequence

from orchestrator_demo.contracts import (
    AgentDescriptor,
    IntentName,
    LlmIntentAssessment,
    RoutingDecision,
)
from orchestrator_demo.intent.classifier import (
    ClassifierUnavailableAgentsError,
    IntentClassifier,
)
from orchestrator_demo.intent.merge import (
    SIMPLE_DIRECT_ROUTE_THRESHOLD,
    merge_intent_confidence,
)
from orchestrator_demo.intent.slm_mock_client import SlmIntentClient
from orchestrator_demo.orchestrator.request_context import RequestContext
from orchestrator_demo.registry.agent_registry import AgentRegistry


SENSITIVE_INTENTS: set[IntentName] = {"credit_risk", "compliance_policy"}
SENSITIVE_AGENTS = {"credit_risk", "compliance_policy"}
SYNTHESIS_AGENT_ID = "synthesis"


class RequestRouter:
    """Classify requests and decide whether execution needs approval first."""

    def __init__(
        self,
        *,
        slm_client: SlmIntentClient,
        intent_classifier: IntentClassifier,
        registry: AgentRegistry,
        direct_route_threshold: float = SIMPLE_DIRECT_ROUTE_THRESHOLD,
    ) -> None:
        self._slm_client = slm_client
        self._intent_classifier = intent_classifier
        self._registry = registry
        self._direct_route_threshold = direct_route_threshold

    async def route_request(self, user_input: str) -> RequestContext:
        slm_suggestion = await self._slm_client.classify(user_input)
        available_agents = self._registry.descriptors()
        try:
            llm_assessment = await self._intent_classifier.assess(
                user_input,
                slm_suggestion,
                available_agents=available_agents,
            )
        except ClassifierUnavailableAgentsError as exc:
            llm_assessment = exc.assessment
        confidence = merge_intent_confidence(slm_suggestion, llm_assessment)
        decision = self._decide(llm_assessment, available_agents, confidence)

        return RequestContext(
            user_input=user_input,
            slm_suggestion=slm_suggestion,
            llm_assessment=llm_assessment,
            decision=decision,
        )

    def _decide(
        self,
        llm_assessment: LlmIntentAssessment,
        available_agents: Sequence[AgentDescriptor],
        confidence: float,
    ) -> RoutingDecision:
        unavailable_agents = _unavailable_required_agents(
            llm_assessment,
            available_agents,
        )
        unavailable_sensitive_agents = [
            agent_id for agent_id in unavailable_agents if agent_id in SENSITIVE_AGENTS
        ]
        if unavailable_sensitive_agents or (
            unavailable_agents
            and not _can_form_partial_plan(
                llm_assessment,
                available_agents,
            )
        ):
            unavailable = ", ".join(unavailable_sensitive_agents or unavailable_agents)
            return RoutingDecision(
                path="clarification_required",
                selected_agent=None,
                confidence=confidence,
                reason=(
                    "A safe route or plan cannot be formed because required "
                    f"agents are unavailable: {unavailable}."
                ),
            )

        if _is_direct_route_candidate(
            llm_assessment,
            confidence,
            direct_route_threshold=self._direct_route_threshold,
        ):
            return RoutingDecision(
                path="direct",
                selected_agent=llm_assessment.required_agents[0],
                confidence=confidence,
                reason="Single intent, single agent, non-sensitive, high confidence.",
            )

        return RoutingDecision(
            path="plan_required",
            selected_agent=None,
            confidence=confidence,
            reason=_plan_required_reason(
                llm_assessment,
                confidence,
                direct_route_threshold=self._direct_route_threshold,
            ),
        )


def _is_direct_route_candidate(
    assessment: LlmIntentAssessment,
    confidence: float,
    *,
    direct_route_threshold: float,
) -> bool:
    return (
        assessment.complexity == "simple"
        and len(assessment.intents) == 1
        and assessment.intents[0] != "unknown"
        and len(assessment.required_agents) == 1
        and SYNTHESIS_AGENT_ID not in assessment.required_agents
        and confidence >= direct_route_threshold
        and not _is_sensitive(assessment)
    )


def _is_sensitive(assessment: LlmIntentAssessment) -> bool:
    return bool(
        SENSITIVE_INTENTS.intersection(assessment.intents)
        or SENSITIVE_AGENTS.intersection(assessment.required_agents)
    )


def _unavailable_required_agents(
    assessment: LlmIntentAssessment,
    available_agents: Sequence[AgentDescriptor],
) -> list[str]:
    available_agent_ids = {descriptor.agent_id for descriptor in available_agents}
    required_agent_ids = _required_agent_ids_for_availability(
        assessment,
        available_agent_ids=available_agent_ids,
    )
    return [
        agent_id
        for agent_id in required_agent_ids
        if agent_id not in available_agent_ids
    ]


def _required_agent_ids_for_availability(
    assessment: LlmIntentAssessment,
    *,
    available_agent_ids: set[str],
) -> list[str]:
    required_agent_ids = _dedupe(assessment.required_agents)
    selected_workstreams = [
        agent_id
        for agent_id in required_agent_ids
        if agent_id in available_agent_ids and agent_id != SYNTHESIS_AGENT_ID
    ]
    requires_synthesis = (
        assessment.complexity == "complex"
        or len(selected_workstreams) > 1
        or SYNTHESIS_AGENT_ID in required_agent_ids
    )

    if requires_synthesis and SYNTHESIS_AGENT_ID not in required_agent_ids:
        required_agent_ids.append(SYNTHESIS_AGENT_ID)

    return required_agent_ids


def _can_form_partial_plan(
    assessment: LlmIntentAssessment,
    available_agents: Sequence[AgentDescriptor],
) -> bool:
    available_agent_ids = {descriptor.agent_id for descriptor in available_agents}
    required_agent_ids = _required_agent_ids_for_availability(
        assessment,
        available_agent_ids=available_agent_ids,
    )
    selected_workstream_ids = [
        agent_id
        for agent_id in required_agent_ids
        if agent_id in available_agent_ids and agent_id != SYNTHESIS_AGENT_ID
    ]
    if not selected_workstream_ids:
        return False

    requires_synthesis = SYNTHESIS_AGENT_ID in required_agent_ids
    return not (
        requires_synthesis and SYNTHESIS_AGENT_ID not in available_agent_ids
    )


def _plan_required_reason(
    assessment: LlmIntentAssessment,
    confidence: float,
    *,
    direct_route_threshold: float,
) -> str:
    reasons: list[str] = []

    if assessment.complexity == "complex":
        reasons.append("complex or multi-step")
    if len(assessment.intents) > 1 or len(assessment.required_agents) > 1:
        reasons.append("multi-intent or multi-agent")
    if SYNTHESIS_AGENT_ID in assessment.required_agents:
        reasons.append("requires synthesis")
    if confidence < direct_route_threshold:
        reasons.append("below direct-route confidence threshold")
    if _is_sensitive(assessment):
        reasons.append("sensitive credit, risk, compliance, or advisory path")
    if "unknown" in assessment.intents:
        reasons.append("ambiguous but routable")

    if not reasons:
        reasons.append("does not satisfy direct-route requirements")

    return f"Plan approval required: {', '.join(reasons)}."



def _dedupe(values: Sequence[str]) -> list[str]:
    deduped: list[str] = []
    for value in values:
        if value not in deduped:
            deduped.append(value)

    return deduped


__all__ = [
    "RequestRouter",
    "SENSITIVE_AGENTS",
    "SENSITIVE_INTENTS",
]
