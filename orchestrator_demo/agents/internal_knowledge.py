"""Synthetic internal knowledge specialist."""

from orchestrator_demo.agents.base import (
    SpecialistProfile,
    SyntheticSpecialistAgent,
    citation,
)


class InternalKnowledgeAgent(SyntheticSpecialistAgent):
    def __init__(self) -> None:
        super().__init__(
            profile=SpecialistProfile(
                agent_id="internal_knowledge",
                display_name="Internal Knowledge Agent",
                summary="Summarizes synthetic CRM notes and policy snippets.",
                findings=(
                    "Demo notes show a recent treasury-services discussion.",
                    "Open follow-ups include cash-flow timing and card program needs.",
                    "Relationship context is intentionally synthetic and local-only.",
                ),
                citations=(
                    citation("internal_knowledge", "Synthetic CRM relationship notes"),
                ),
                data_source_categories=("synthetic_crm", "synthetic_policy_notes"),
            )
        )


__all__ = ["InternalKnowledgeAgent"]
