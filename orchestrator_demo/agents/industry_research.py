"""Synthetic industry research specialist."""

from orchestrator_demo.agents.base import (
    SpecialistProfile,
    SyntheticSpecialistAgent,
    citation,
)


class IndustryResearchAgent(SyntheticSpecialistAgent):
    def __init__(self) -> None:
        super().__init__(
            profile=SpecialistProfile(
                agent_id="industry_research",
                display_name="Industry Research Agent",
                summary="Provides a sector overview using deterministic demo trends.",
                findings=(
                    "Manufacturing demand indicators are mixed in the demo dataset.",
                    "Input costs and skilled-labor availability remain watch items.",
                    "Stable receivables practices are a useful relationship question.",
                ),
                citations=(
                    citation("industry_research", "Synthetic sector outlook notes"),
                ),
                data_source_categories=("synthetic_industry_notes",),
            )
        )


__all__ = ["IndustryResearchAgent"]
