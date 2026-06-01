"""Local specialist agent implementations."""

from orchestrator_demo.agents.base import SpecialistAgent
from orchestrator_demo.agents.compliance_policy import CompliancePolicyAgent
from orchestrator_demo.agents.credit_risk import CreditRiskAgent
from orchestrator_demo.agents.data_quality import DataQualityAgent
from orchestrator_demo.agents.industry_research import IndustryResearchAgent
from orchestrator_demo.agents.internal_knowledge import InternalKnowledgeAgent
from orchestrator_demo.agents.meeting_prep import MeetingPrepAgent
from orchestrator_demo.agents.product_opportunity import ProductOpportunityAgent
from orchestrator_demo.agents.relationship_summary import RelationshipSummaryAgent
from orchestrator_demo.agents.synthesis import SynthesisAgent
from orchestrator_demo.agents.web_search import WebSearchAgent


def build_default_specialists() -> dict[str, SpecialistAgent]:
    """Create fresh local synthetic specialists keyed by agent id."""

    specialists: list[SpecialistAgent] = [
        IndustryResearchAgent(),
        WebSearchAgent(),
        InternalKnowledgeAgent(),
        CreditRiskAgent(),
        RelationshipSummaryAgent(),
        ProductOpportunityAgent(),
        CompliancePolicyAgent(),
        DataQualityAgent(),
        MeetingPrepAgent(),
        SynthesisAgent(),
    ]
    return {specialist.agent_id: specialist for specialist in specialists}


__all__ = [
    "CompliancePolicyAgent",
    "CreditRiskAgent",
    "DataQualityAgent",
    "IndustryResearchAgent",
    "InternalKnowledgeAgent",
    "MeetingPrepAgent",
    "ProductOpportunityAgent",
    "RelationshipSummaryAgent",
    "SpecialistAgent",
    "SynthesisAgent",
    "WebSearchAgent",
    "build_default_specialists",
]
