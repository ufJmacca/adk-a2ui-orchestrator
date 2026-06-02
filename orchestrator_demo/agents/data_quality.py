"""Synthetic data quality specialist."""

from orchestrator_demo.agents.base import (
    DEFAULT_CAVEAT,
    SpecialistProfile,
    SyntheticSpecialistAgent,
    citation,
)


class DataQualityAgent(SyntheticSpecialistAgent):
    def __init__(self) -> None:
        super().__init__(
            profile=SpecialistProfile(
                agent_id="data_quality",
                display_name="Data Quality Agent",
                summary="Highlights missing, stale, or weak synthetic evidence.",
                findings=(
                    "Current financial statement date is absent in the demo context.",
                    "Some relationship notes are intentionally generic placeholders.",
                    "Evidence gaps should be resolved before customer-facing use.",
                ),
                citations=(
                    citation("data_quality", "Synthetic data quality checklist"),
                ),
                caveats=(
                    DEFAULT_CAVEAT,
                    "Missing-data flags are quality prompts, not final conclusions.",
                ),
                data_source_categories=("synthetic_quality_checks",),
            )
        )


__all__ = ["DataQualityAgent"]
