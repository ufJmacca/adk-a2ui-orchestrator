"""Synthetic relationship summary specialist."""

from orchestrator_demo.agents.base import (
    SpecialistProfile,
    SyntheticSpecialistAgent,
    citation,
)


class RelationshipSummaryAgent(SyntheticSpecialistAgent):
    def __init__(self) -> None:
        super().__init__(
            profile=SpecialistProfile(
                agent_id="relationship_summary",
                display_name="Relationship Summary Agent",
                summary="Builds a concise synthetic relationship history.",
                findings=(
                    "Demo history includes deposits, treasury interest, and check-ins.",
                    "Key contacts are represented as synthetic role placeholders.",
                    "Prior meeting notes show follow-up questions for liquidity needs.",
                ),
                citations=(
                    citation("relationship_summary", "Synthetic relationship timeline"),
                ),
                data_source_categories=("synthetic_relationship_records",),
            )
        )


__all__ = ["RelationshipSummaryAgent"]
