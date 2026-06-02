"""Routing engine for direct specialist calls versus approved plan workflows."""

from __future__ import annotations

from collections.abc import Sequence

from orchestrator_demo.app.logging import log_audit_event
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
    LLM_CONFIDENCE_WEIGHT,
    SIMPLE_DIRECT_ROUTE_THRESHOLD,
    SLM_CONFIDENCE_WEIGHT,
    merge_intent_confidence,
)
from orchestrator_demo.intent.slm_mock_client import SlmIntentClient
from orchestrator_demo.orchestrator.planner import SYNTHESIS_AGENT_ID
from orchestrator_demo.orchestrator.request_context import RequestContext
from orchestrator_demo.registry.agent_registry import AgentRegistry


SENSITIVE_INTENTS: set[IntentName] = {"credit_risk", "compliance_policy"}
SENSITIVE_AGENTS = {"credit_risk", "compliance_policy"}


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
        log_audit_event(
            "slm_suggestion",
            {
                "intent": slm_suggestion.intent,
                "confidence": slm_suggestion.confidence,
            },
        )
        available_agents = self._registry.descriptors()
        try:
            llm_assessment = await self._intent_classifier.assess(
                user_input,
                slm_suggestion,
                available_agents=available_agents,
            )
        except ClassifierUnavailableAgentsError as exc:
            llm_assessment = exc.assessment
        log_audit_event(
            "llm_assessment",
            {
                "intents": list(llm_assessment.intents),
                "confidence": llm_assessment.confidence,
                "complexity": llm_assessment.complexity,
                "required_agent_ids": list(llm_assessment.required_agents),
                "rationale": llm_assessment.rationale,
            },
        )
        confidence = merge_intent_confidence(slm_suggestion, llm_assessment)
        log_audit_event(
            "merge_decision",
            {
                "slm_confidence": slm_suggestion.confidence,
                "llm_confidence": llm_assessment.confidence,
                "slm_weight": SLM_CONFIDENCE_WEIGHT,
                "llm_weight": LLM_CONFIDENCE_WEIGHT,
                "final_confidence": confidence,
                "direct_route_threshold": self._direct_route_threshold,
            },
        )
        decision = self._decide(llm_assessment, available_agents, confidence)
        log_audit_event(
            "route_decision",
            {
                "path": decision.path,
                "selected_agent": decision.selected_agent,
                "confidence": decision.confidence,
                "reason": decision.reason,
            },
        )

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
        if unavailable_agents:
            unavailable = ", ".join(unavailable_agents)
            return RoutingDecision(
                path="clarification_required",
                selected_agent=None,
                confidence=confidence,
                reason=(
                    "A safe route or plan cannot be formed because required "
                    f"agents are unavailable: {unavailable}."
                ),
            )

        if _has_only_synthesis_workstream(llm_assessment):
            return RoutingDecision(
                path="clarification_required",
                selected_agent=None,
                confidence=confidence,
                reason=(
                    "A safe route or plan cannot be formed because the assessment "
                    "selected synthesis without any upstream specialist workstream."
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
    unique_intents = _dedupe(assessment.intents)
    unique_required_agents = _dedupe(assessment.required_agents)
    return (
        assessment.complexity == "simple"
        and len(unique_intents) == 1
        and unique_intents[0] != "unknown"
        and len(unique_required_agents) == 1
        and SYNTHESIS_AGENT_ID not in unique_required_agents
        and confidence >= direct_route_threshold
        and not _is_sensitive(assessment)
    )


def _is_sensitive(assessment: LlmIntentAssessment) -> bool:
    return bool(
        SENSITIVE_INTENTS.intersection(assessment.intents)
        or SENSITIVE_AGENTS.intersection(assessment.required_agents)
    )


def _has_only_synthesis_workstream(assessment: LlmIntentAssessment) -> bool:
    return not any(
        agent_id != SYNTHESIS_AGENT_ID for agent_id in assessment.required_agents
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


def _dedupe(values: Sequence[str]) -> list[str]:
    deduped: list[str] = []
    for value in values:
        if value not in deduped:
            deduped.append(value)

    return deduped


def _plan_required_reason(
    assessment: LlmIntentAssessment,
    confidence: float,
    *,
    direct_route_threshold: float,
) -> str:
    reasons: list[str] = []
    unique_intents = _dedupe(assessment.intents)
    unique_required_agents = _dedupe(assessment.required_agents)

    if assessment.complexity == "complex":
        reasons.append("complex or multi-step")
    if len(unique_intents) > 1 or len(unique_required_agents) > 1:
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


__all__ = [
    "RequestRouter",
    "SENSITIVE_AGENTS",
    "SENSITIVE_INTENTS",
]
