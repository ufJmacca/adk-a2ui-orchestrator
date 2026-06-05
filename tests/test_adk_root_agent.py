import inspect
import sys
from pathlib import Path
from typing import Any

import pytest

from google.adk.agents.base_agent import BaseAgent
from google.adk.apps.app import App

from orchestrator_demo.orchestrator.agent import (
    AdkOrchestratorAdapter,
    build_root_agent,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


class FakeToolContext:
    def __init__(self) -> None:
        self.state: dict[str, Any] = {}
        self.saved_artifacts: list[dict[str, Any]] = []

    async def save_artifact(
        self,
        filename: str,
        artifact: Any,
        custom_metadata: dict[str, Any] | None = None,
    ) -> int:
        version = len(
            [
                saved
                for saved in self.saved_artifacts
                if saved["filename"] == filename
            ]
        )
        self.saved_artifacts.append(
            {
                "filename": filename,
                "artifact": artifact,
                "customMetadata": custom_metadata,
                "version": version,
            }
        )
        return version


def test_build_root_agent_returns_adk_agent_with_orchestrator_tools() -> None:
    agent = build_root_agent(
        adapter=AdkOrchestratorAdapter(),
        model="gemini-2.0-flash",
    )

    assert isinstance(agent, BaseAgent)
    assert agent.name == "orchestrator"
    assert {tool.name for tool in agent.tools} == {
        "submit_orchestrator_request",
        "add_plan_instruction",
        "remove_plan_step",
        "replace_plan_agent",
        "reorder_plan_steps",
        "approve_orchestrator_plan",
        "reject_orchestrator_plan",
    }
    submit_tool = next(
        tool for tool in agent.tools if tool.name == "submit_orchestrator_request"
    )
    assert list(inspect.signature(submit_tool.func).parameters) == [
        "user_input",
        "tool_context",
    ]


@pytest.mark.asyncio
async def test_adk_adapter_submits_complex_request_and_approves_plan() -> None:
    adapter = AdkOrchestratorAdapter()
    tool_context = FakeToolContext()

    submitted = await adapter.submit_orchestrator_request(
        "Prepare me for tomorrow's meeting with ABC Manufacturing.",
        tool_context=tool_context,
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
        edited_plan_version=plan["plan_version"],
        tool_context=tool_context,
    )

    assert approved["status"] == "approved"
    assert approved["approvalResult"]["graphCreated"] is True
    assert approved["approvalResult"]["specialistsCalled"] is True
    assert approved["artifacts"]["final_response"]["agent_id"] == "synthesis"


@pytest.mark.asyncio
async def test_adk_adapter_rejects_pending_plan_without_execution() -> None:
    adapter = AdkOrchestratorAdapter()
    tool_context = FakeToolContext()
    submitted = await adapter.submit_orchestrator_request(
        "Research this prospect and give me risks, opportunities, and talking points.",
        tool_context=tool_context,
    )
    plan = submitted["approvalPlan"]

    rejected = await adapter.reject_orchestrator_plan(
        plan["plan_id"],
        plan["approval_surface_id"],
        "Do not run this workflow.",
        tool_context=tool_context,
    )

    assert rejected["status"] == "rejected"
    assert rejected["approvalResult"]["graphCreated"] is False
    assert rejected["approvalResult"]["specialistsCalled"] is False
    assert rejected["approvalResult"]["reason"] == "Do not run this workflow."


def test_adk_loader_finds_orchestrator_root_agent(monkeypatch) -> None:
    from google.adk.cli.utils.agent_loader import AgentLoader

    # Arrange
    monkeypatch.setenv("OPENROUTER_API_KEY", "unit-test-openrouter-key")
    monkeypatch.setenv("LLM_MODEL", "openrouter/unit-test/model")
    monkeypatch.syspath_prepend(str(REPOSITORY_ROOT))
    for module_name in ("orchestrator.agent", "orchestrator"):
        monkeypatch.delitem(sys.modules, module_name, raising=False)

    # Act
    loaded = AgentLoader(str(REPOSITORY_ROOT / "orchestrator_demo")).load_agent(
        "orchestrator"
    )

    # Assert
    if isinstance(loaded, App):
        assert loaded.name == "orchestrator"
        assert isinstance(loaded.root_agent, BaseAgent)
        assert loaded.root_agent.name == "orchestrator"
    else:
        assert isinstance(loaded, BaseAgent)
        assert loaded.name == "orchestrator"


def test_adk_loader_finds_exported_app_with_root_agent(
    monkeypatch, tmp_path
) -> None:
    from google.adk.cli.utils.agent_loader import AgentLoader

    # Arrange
    agent_package = tmp_path / "orchestrator"
    agent_package.mkdir()
    (agent_package / "__init__.py").write_text(
        "\n".join(
            [
                "from google.adk.agents import Agent",
                "from google.adk.apps.app import App",
                "",
                'root_agent = Agent(name="orchestrator", model="gemini-2.0-flash")',
                'app = App(name="orchestrator", root_agent=root_agent)',
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    for module_name in ("orchestrator.agent", "orchestrator"):
        monkeypatch.delitem(sys.modules, module_name, raising=False)

    # Act
    loaded = AgentLoader(str(tmp_path)).load_agent("orchestrator")

    # Assert
    assert isinstance(loaded, App)
    assert loaded.name == "orchestrator"
    assert isinstance(loaded.root_agent, BaseAgent)
    assert loaded.root_agent.name == "orchestrator"
