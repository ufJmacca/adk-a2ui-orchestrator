"""Synthetic synthesis specialist."""

from orchestrator_demo.agents.base import (
    SpecialistProfile,
    SyntheticSpecialistAgent,
    citation,
)


class SynthesisAgent(SyntheticSpecialistAgent):
    def __init__(self) -> None:
        super().__init__(
            profile=SpecialistProfile(
                agent_id="synthesis",
                display_name="Synthesis Agent",
                summary="Combines specialist outputs into an RM-ready response.",
                findings=(
                    "Group findings into context, opportunities, risks, and actions.",
                    "Separate verified facts from synthetic assumptions.",
                    "Carry forward caveats from upstream specialists.",
                ),
                citations=(
                    citation("synthesis", "Synthetic synthesis assembly guide"),
                ),
                data_source_categories=("synthetic_specialist_outputs",),
            )
        )


__all__ = ["SynthesisAgent"]
