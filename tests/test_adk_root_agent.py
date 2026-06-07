import inspect
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from google.adk.agents.base_agent import BaseAgent
from google.adk.apps.app import App

from orchestrator_demo.a2ui_support.adk_a2a_plugin import A2uiA2AProtocolPlugin
from orchestrator_demo.orchestrator.agent import (
    AdkOrchestratorAdapter,
    build_root_agent,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
A2UI_MIME_TYPE = "application/json+a2ui"
EXPECTED_TOOL_NAMES = {
    "submit_orchestrator_request",
    "add_plan_instruction",
    "remove_plan_step",
    "replace_plan_agent",
    "reorder_plan_steps",
    "approve_orchestrator_plan",
    "reject_orchestrator_plan",
}


def assert_exact_stable_tools(agent: BaseAgent) -> None:
    tools = list(agent.tools)

    assert len(tools) == len(EXPECTED_TOOL_NAMES)
    assert {tool.name for tool in tools} == EXPECTED_TOOL_NAMES


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
    # Arrange
    agent = build_root_agent(
        adapter=AdkOrchestratorAdapter(),
        model="gemini-2.0-flash",
    )

    # Assert
    assert isinstance(agent, BaseAgent)
    assert agent.name == "orchestrator"
    assert_exact_stable_tools(agent)
    submit_tool = next(
        tool for tool in agent.tools if tool.name == "submit_orchestrator_request"
    )
    assert list(inspect.signature(submit_tool.func).parameters) == [
        "user_input",
        "tool_context",
    ]


def test_agent_module_exports_root_agent_and_app(monkeypatch) -> None:
    # Arrange
    import orchestrator_demo.orchestrator.agent as agent_module

    monkeypatch.setenv("OPENROUTER_API_KEY", "unit-test-openrouter-key")
    monkeypatch.setenv("LLM_MODEL", "openrouter/unit-test/model")

    # Act
    root_agent = agent_module.root_agent
    app = agent_module.app

    # Assert
    assert {"root_agent", "app"} <= set(agent_module.__all__)
    assert isinstance(root_agent, BaseAgent)
    assert_exact_stable_tools(root_agent)
    assert isinstance(app, App)
    assert app.name == "orchestrator"
    assert app.root_agent is root_agent
    assert any(isinstance(plugin, A2uiA2AProtocolPlugin) for plugin in app.plugins)


def test_orchestrator_package_lazy_exports_resolve_in_fresh_process() -> None:
    # Arrange
    env = {
        **os.environ,
        "ORCHESTRATOR_DEMO_DETERMINISTIC_MODEL": "1",
        "ORCHESTRATOR_DEMO_ADK_EVAL_MODE": "1",
    }
    env.pop("OPENROUTER_API_KEY", None)
    code = """
import json
import sys

import orchestrator_demo.orchestrator as o

agent_module_name = "orchestrator_demo.orchestrator.agent"
before = {
    "agent_in_sys_modules": agent_module_name in sys.modules,
    "export_attrs": sorted(
        name for name in ("agent", "app", "root_agent") if name in o.__dict__
    ),
    "all": sorted(o.__all__),
}

assert before["agent_in_sys_modules"] is False, before
assert before["export_attrs"] == [], before
assert before["all"] == ["agent", "app", "root_agent"], before

agent = o.agent
root_agent = o.root_agent
app = o.app

assert agent.__name__ == agent_module_name
assert agent.root_agent is root_agent
assert root_agent.name == "orchestrator"
assert app.name == "orchestrator"
assert app.root_agent is root_agent
print(json.dumps({"agent": agent.__name__, "app": app.name, "root_agent": root_agent.name}))
"""

    # Act
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPOSITORY_ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    # Assert
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout) == {
        "agent": "orchestrator_demo.orchestrator.agent",
        "app": "orchestrator",
        "root_agent": "orchestrator",
    }


def test_root_agent_instruction_requires_tools_and_structured_approval() -> None:
    # Arrange
    agent = build_root_agent(
        adapter=AdkOrchestratorAdapter(),
        model="gemini-2.0-flash",
    )

    # Act
    instruction = agent.instruction

    # Assert
    assert "call tools" in instruction
    assert "explicit structured approval" in instruction
    assert "natural-language approval" in instruction
    assert "must not execute" in instruction
    assert "planId" in instruction
    assert "planVersion" in instruction
    assert "step ids" in instruction


def test_orchestrator_agent_card_enables_adk_a2a_discovery() -> None:
    from a2a.types import AgentCard

    agent_dir = REPOSITORY_ROOT / "orchestrator_demo" / "orchestrator"
    card_path = agent_dir / "agent.json"

    assert card_path.is_file()
    card = AgentCard(**json.loads(card_path.read_text(encoding="utf-8")))

    assert card.name == "orchestrator"
    assert str(card.url) == "http://127.0.0.1:8000/a2a/orchestrator"
    assert card.capabilities.streaming is True
    assert card.capabilities.state_transition_history is True
    assert card.default_input_modes == ["text/plain"]
    assert set(card.default_output_modes) == {
        "application/json",
        A2UI_MIME_TYPE,
        "text/plain",
    }
    assert {
        "submit_orchestrator_request",
        "review_orchestrator_a2ui_plan",
        "edit_orchestrator_plan",
        "approve_orchestrator_plan",
        "reject_orchestrator_plan",
    } <= {skill.id for skill in card.skills}


def test_adk_api_server_registers_orchestrator_a2a_routes(monkeypatch) -> None:
    from google.adk.cli.fast_api import get_fast_api_app

    monkeypatch.chdir(REPOSITORY_ROOT)
    monkeypatch.setenv("OPENROUTER_API_KEY", "unit-test-openrouter-key")
    monkeypatch.setenv("LLM_MODEL", "openrouter/unit-test/model")

    app = get_fast_api_app(
        agents_dir="orchestrator_demo",
        web=True,
        a2a=True,
        host="0.0.0.0",
        port=8000,
        use_local_storage=False,
    )

    paths = {route.path for route in app.routes}
    assert "/a2a/orchestrator" in paths
    assert "/a2a/orchestrator/.well-known/agent-card.json" in paths


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
    assert isinstance(loaded, App)
    assert loaded.name == "orchestrator"
    assert isinstance(loaded.root_agent, BaseAgent)
    assert loaded.root_agent.name == "orchestrator"
    assert_exact_stable_tools(loaded.root_agent)
    assert any(isinstance(plugin, A2uiA2AProtocolPlugin) for plugin in loaded.plugins)


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
