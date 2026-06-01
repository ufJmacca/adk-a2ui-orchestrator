"""Synthetic credit risk specialist."""

from orchestrator_demo.agents.base import (
    DEFAULT_CAVEAT,
    SpecialistProfile,
    SyntheticSpecialistAgent,
    citation,
)


class CreditRiskAgent(SyntheticSpecialistAgent):
    def __init__(self) -> None:
        super().__init__(
            profile=SpecialistProfile(
                agent_id="credit_risk",
                display_name="Credit Risk Agent",
                summary="Flags non-binding credit themes from synthetic inputs.",
                findings=(
                    "Review working-capital seasonality before lending discussion.",
                    "Ask for current financial statements and covenant context.",
                    "Treat repayment indicators as prompts for banker review.",
                ),
                citations=(
                    citation("credit_risk", "Synthetic credit monitoring notes"),
                ),
                caveats=(
                    DEFAULT_CAVEAT,
                    "Credit observations are screening prompts, not underwriting.",
                ),
                data_source_categories=("synthetic_credit_notes",),
            )
        )


__all__ = ["CreditRiskAgent"]
