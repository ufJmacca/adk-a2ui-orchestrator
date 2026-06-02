"""Synthetic public web research specialist."""

from orchestrator_demo.agents.base import (
    SpecialistProfile,
    SyntheticSpecialistAgent,
    citation,
)


class WebSearchAgent(SyntheticSpecialistAgent):
    def __init__(self) -> None:
        super().__init__(
            profile=SpecialistProfile(
                agent_id="web_search",
                display_name="Web Search Agent",
                summary="Summarizes mocked public-source signals for demo research.",
                findings=(
                    "The mocked public scan highlights expansion and supplier updates.",
                    "No live web data is used in this local deterministic response.",
                    "Public-source claims should be checked against current sources.",
                ),
                citations=(
                    citation("web_search", "Synthetic public-source digest"),
                ),
                data_source_categories=("mocked_public_web",),
            )
        )


__all__ = ["WebSearchAgent"]
