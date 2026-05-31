"""Common interface and helpers for local synthetic specialist agents."""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any, Protocol

from orchestrator_demo.contracts import SpecialistRequest, SpecialistResponse


DEFAULT_CAVEAT = (
    "Synthetic demo information only; verify source material before customer use. "
    "This is not a binding credit, risk, compliance, or advisory decision."
)
DEFAULT_RISK_CONTROLS = {
    "binding_decision": False,
    "requires_human_review": True,
}


class SpecialistAgent(Protocol):
    """Minimal async interface shared by local and future remote-compatible agents."""

    call_count: int
    calls: list[SpecialistRequest]

    @property
    def agent_id(self) -> str:
        """Stable specialist id used for routing and response ownership."""

    async def handle(self, request: SpecialistRequest) -> SpecialistResponse:
        """Handle a specialist request and return a structured response envelope."""


@dataclass(frozen=True)
class SyntheticCitation:
    """A provenance reference to deterministic synthetic demo material."""

    source_id: str
    title: str
    uri: str

    def as_dict(self) -> dict[str, str]:
        return {
            "source_id": self.source_id,
            "title": self.title,
            "uri": self.uri,
        }


@dataclass(frozen=True)
class SpecialistProfile:
    """Static response profile for a deterministic synthetic specialist."""

    agent_id: str
    display_name: str
    summary: str
    findings: tuple[str, ...]
    citations: tuple[SyntheticCitation, ...]
    caveats: tuple[str, ...] = (DEFAULT_CAVEAT,)
    data_source_categories: tuple[str, ...] = ()
    emits_a2ui: bool = False


@dataclass
class SyntheticSpecialistAgent:
    """Deterministic local specialist with observable call counts."""

    profile: SpecialistProfile
    call_count: int = 0
    calls: list[SpecialistRequest] = field(default_factory=list)

    @property
    def agent_id(self) -> str:
        return self.profile.agent_id

    @property
    def display_name(self) -> str:
        return self.profile.display_name

    async def handle(self, request: SpecialistRequest) -> SpecialistResponse:
        if request.agent_id != self.agent_id:
            raise ValueError(
                f"{self.agent_id} cannot handle request for {request.agent_id}"
            )

        self.call_count += 1
        self.calls.append(request)

        structured_output = self._structured_output(request)
        surface_id = (
            _stable_contract_id("surface", self.agent_id, request.request_id)
            if self.profile.emits_a2ui
            else None
        )
        a2ui_payload = (
            self._a2ui_payload(surface_id, structured_output)
            if surface_id is not None
            else None
        )

        return SpecialistResponse(
            response_id=_stable_contract_id("response", self.agent_id, request.request_id),
            agent_id=self.agent_id,
            content=self._content(),
            structured_output=structured_output,
            a2ui_payload=a2ui_payload,
            surface_id=surface_id,
        )

    def _content(self) -> str:
        findings = " ".join(self.profile.findings)
        return (
            f"{self.display_name}: {self.profile.summary} "
            f"{findings} Synthetic demo data only."
        )

    def _structured_output(self, request: SpecialistRequest) -> dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "request_id": request.request_id,
            "summary": self.profile.summary,
            "findings": list(self.profile.findings),
            "citations": [
                citation.as_dict() for citation in self.profile.citations
            ],
            "provenance": {
                "data_classification": "synthetic_demo",
                "customer_data": "synthetic_only",
                "generated_by": self.agent_id,
                "source_policy": "no real customer data",
            },
            "data_source_categories": list(self.profile.data_source_categories),
            "caveats": list(self.profile.caveats),
            "risk_controls": dict(DEFAULT_RISK_CONTROLS),
        }

    def _a2ui_payload(
        self,
        surface_id: str,
        structured_output: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "surfaceId": surface_id,
            "catalog": "basic",
            "components": [
                {
                    "type": "card",
                    "id": _stable_contract_id("component", self.agent_id, "summary"),
                    "title": self.display_name,
                    "body": structured_output["summary"],
                },
                {
                    "type": "button",
                    "id": _stable_contract_id("component", self.agent_id, "details"),
                    "label": "Show more detail",
                    "action": {
                        "type": "specialist_action",
                        "surfaceId": surface_id,
                        "payload": {"agentId": self.agent_id},
                    },
                },
            ],
            "metadata": {"ownerAgentId": self.agent_id},
        }


def citation(agent_id: str, title: str) -> SyntheticCitation:
    return SyntheticCitation(
        source_id=f"synthetic_{agent_id}",
        title=title,
        uri=f"synthetic://business-banking/{agent_id}",
    )


def _stable_contract_id(prefix: str, *parts: str) -> str:
    raw_suffix = "_".join(parts)
    suffix = re.sub(r"[^A-Za-z0-9_-]+", "_", raw_suffix).strip("_")
    if not suffix or not suffix[0].isalnum():
        suffix = f"generated_{suffix}"
    return f"{prefix}_{suffix}"


__all__ = [
    "SpecialistAgent",
    "SpecialistProfile",
    "SyntheticCitation",
    "SyntheticSpecialistAgent",
    "citation",
]
