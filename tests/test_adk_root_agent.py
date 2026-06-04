import sys
from pathlib import Path

import pytest

from google.adk.agents.base_agent import BaseAgent

from orchestrator_demo.orchestrator.agent import (
    AdkOrchestratorAdapter,
    build_root_agent,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def test_build_root_agent_returns_adk_agent_with_orchestrator_tools() -> None:
    agent = build_root_agent(
        adapter=AdkOrchestratorAdapter(),
        model="gemini-2.0-flash",
    )

    assert isinstance(agent, BaseAgent)
    assert agent.name == "orchestrator"
    assert {tool.name for tool in agent.tools} == {
        "submit_orchestrator_request",
        "approve_orchestrator_plan",
        "reject_orchestrator_plan",
    }


@pytest.mark.asyncio
async def test_adk_adapter_submits_complex_request_and_approves_plan() -> None:
    adapter = AdkOrchestratorAdapter()

    submitted = await adapter.submit_orchestrator_request(
        "Prepare me for tomorrow's meeting with ABC Manufacturing."
    )

    assert submitted["path"] == "plan_required"
    plan = submitted["approvalPlan"]
    assert plan["approval_surface_id"].startswith("surface_plan_")
    assert plan["selected_agents"] == [
        "relationship_summary",
        "internal_knowledge",
        "industry_research",
        "synthesis",
    ]

    approved = await adapter.approve_orchestrator_plan(
        plan["plan_id"],
        plan["approval_surface_id"],
        [step["step_id"] for step in plan["steps"]],
    )

    assert approved["status"] == "approved"
    assert approved["approvalResult"]["graphCreated"] is True
    assert approved["approvalResult"]["specialistsCalled"] is True
    assert approved["artifacts"]["final_response"]["agent_id"] == "synthesis"


@pytest.mark.asyncio
async def test_adk_adapter_rejects_pending_plan_without_execution() -> None:
    adapter = AdkOrchestratorAdapter()
    submitted = await adapter.submit_orchestrator_request(
        "Research this prospect and give me risks, opportunities, and talking points."
    )
    plan = submitted["approvalPlan"]

    rejected = await adapter.reject_orchestrator_plan(
        plan["plan_id"],
        plan["approval_surface_id"],
        "Do not run this workflow.",
    )

    assert rejected["status"] == "rejected"
    assert rejected["approvalResult"]["graphCreated"] is False
    assert rejected["approvalResult"]["specialistsCalled"] is False
    assert rejected["approvalResult"]["reason"] == "Do not run this workflow."


def test_adk_loader_finds_orchestrator_root_agent(monkeypatch) -> None:
    from google.adk.cli.utils.agent_loader import AgentLoader

    monkeypatch.setenv("OPENROUTER_API_KEY", "unit-test-openrouter-key")
    monkeypatch.setenv("LLM_MODEL", "openrouter/unit-test/model")
    monkeypatch.syspath_prepend(str(REPOSITORY_ROOT))
    for module_name in ("orchestrator.agent", "orchestrator"):
        sys.modules.pop(module_name, None)

    loaded = AgentLoader(str(REPOSITORY_ROOT / "orchestrator_demo")).load_agent(
        "orchestrator"
    )

    assert isinstance(loaded, BaseAgent)
    assert loaded.name == "orchestrator"
