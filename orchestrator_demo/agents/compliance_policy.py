"""Synthetic compliance and policy specialist."""

from orchestrator_demo.agents.base import (
    DEFAULT_CAVEAT,
    SpecialistProfile,
    SyntheticSpecialistAgent,
    citation,
)


class CompliancePolicyAgent(SyntheticSpecialistAgent):
    def __init__(self) -> None:
        super().__init__(
            profile=SpecialistProfile(
                agent_id="compliance_policy",
                display_name="Compliance / Policy Agent",
                summary="Reviews language for unsupported or regulated claims.",
                findings=(
                    "Use source-backed wording for credit-sensitive statements.",
                    "Keep recommendations framed as discussion prompts.",
                    "Escalate policy-sensitive claims to the proper internal review.",
                ),
                citations=(
                    citation("compliance_policy", "Synthetic policy guidance excerpt"),
                ),
                caveats=(
                    DEFAULT_CAVEAT,
                    "Policy notes are guardrails, not legal or compliance advice.",
                ),
                data_source_categories=("synthetic_policy_notes",),
            )
        )


__all__ = ["CompliancePolicyAgent"]
