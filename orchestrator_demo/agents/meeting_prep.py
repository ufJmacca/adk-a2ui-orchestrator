"""Synthetic meeting preparation specialist."""

from orchestrator_demo.agents.base import (
    SpecialistProfile,
    SyntheticSpecialistAgent,
    citation,
)


class MeetingPrepAgent(SyntheticSpecialistAgent):
    def __init__(self) -> None:
        super().__init__(
            profile=SpecialistProfile(
                agent_id="meeting_prep",
                display_name="Meeting Prep Agent",
                summary="Creates a synthetic RM meeting brief outline.",
                findings=(
                    "Open with relationship context and recent follow-ups.",
                    "Ask discovery questions about liquidity and operating cycle.",
                    "Close with next steps that require banker confirmation.",
                ),
                citations=(
                    citation("meeting_prep", "Synthetic meeting-prep playbook"),
                ),
                data_source_categories=("synthetic_meeting_notes",),
                emits_a2ui=True,
            )
        )


__all__ = ["MeetingPrepAgent"]
