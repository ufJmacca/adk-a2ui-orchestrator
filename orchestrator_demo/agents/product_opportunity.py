"""Synthetic product opportunity specialist."""

from orchestrator_demo.agents.base import (
    DEFAULT_CAVEAT,
    SpecialistProfile,
    SyntheticSpecialistAgent,
    citation,
)


class ProductOpportunityAgent(SyntheticSpecialistAgent):
    def __init__(self) -> None:
        super().__init__(
            profile=SpecialistProfile(
                agent_id="product_opportunity",
                display_name="Product Opportunity Agent",
                summary="Suggests non-binding product conversation themes.",
                findings=(
                    "Discuss treasury services for cash visibility and controls.",
                    "Consider merchant services if payment acceptance is relevant.",
                    "Frame lending topics as discovery questions for banker review.",
                ),
                citations=(
                    citation("product_opportunity", "Synthetic product fit matrix"),
                ),
                caveats=(
                    DEFAULT_CAVEAT,
                    "Product ideas are advisory prompts, not suitability findings.",
                ),
                data_source_categories=("synthetic_product_matrix",),
                emits_a2ui=True,
            )
        )


__all__ = ["ProductOpportunityAgent"]
