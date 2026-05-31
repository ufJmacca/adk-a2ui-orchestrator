import importlib
import logging
import os
import py_compile
import sys
from pathlib import Path

import pytest

from orchestrator_demo.contracts import ExecutionPlan, PlanStep


SPEC_REQUIRED_SPECIALIST_AGENT_IDS = {
    "industry_research",
    "web_search",
    "internal_knowledge",
    "credit_risk",
    "relationship_summary",
    "product_opportunity",
    "compliance_policy",
    "data_quality",
    "meeting_prep",
    "synthesis",
}


def _descriptor_source(agent_id: str, *, display_name: str | None = None) -> str:
    display_name = display_name or agent_id.replace("_", " ").title()
    return f"""AgentDescriptor(
        agent_id={agent_id!r},
        display_name={display_name!r},
        capabilities=["business banking support"],
        input_schema={{"type": "object"}},
        output_schema={{"type": "object"}},
        a2ui_catalogs=["basic"],
        routing_examples=["Handle a {agent_id} request."],
        execution_mode="local_llm",
    )"""


def _write_config(path: Path, descriptor_sources: list[str]) -> None:
    path.write_text(
        "from orchestrator_demo.contracts import AgentDescriptor\n\n"
        "AVAILABLE_AGENTS = [\n"
        + ",\n".join(descriptor_sources)
        + "\n]\n",
        encoding="utf-8",
    )


def _registry_from(path: Path):
    from orchestrator_demo.registry.agent_registry import AgentRegistry

    return AgentRegistry.from_config_path(path)


def test_default_agent_config_registers_all_required_specialist_roles() -> None:
    # Arrange
    from orchestrator_demo.registry.agent_config import AVAILABLE_AGENTS

    # Act
    configured_agent_ids = {descriptor.agent_id for descriptor in AVAILABLE_AGENTS}

    # Assert
    assert SPEC_REQUIRED_SPECIALIST_AGENT_IDS <= configured_agent_ids


def test_registry_reload_adds_agent_and_logs_added_agent(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Arrange
    config_path = tmp_path / "agent_config.py"
    _write_config(
        config_path,
        [
            _descriptor_source("internal_knowledge"),
            _descriptor_source("synthesis"),
        ],
    )
    registry = _registry_from(config_path)
    _write_config(
        config_path,
        [
            _descriptor_source("internal_knowledge"),
            _descriptor_source("industry_research"),
            _descriptor_source("synthesis"),
        ],
    )

    # Act
    with caplog.at_level(logging.INFO, logger="orchestrator_demo.registry.agent_registry"):
        registry.reload()

    # Assert
    assert registry.agent_ids() == [
        "industry_research",
        "internal_knowledge",
        "synthesis",
    ]
    assert "added=industry_research" in caplog.text


def test_registry_reload_removes_agent_and_makes_it_unavailable_for_new_plans(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Arrange
    config_path = tmp_path / "agent_config.py"
    _write_config(
        config_path,
        [
            _descriptor_source("internal_knowledge"),
            _descriptor_source("relationship_summary"),
            _descriptor_source("synthesis"),
        ],
    )
    registry = _registry_from(config_path)
    _write_config(
        config_path,
        [
            _descriptor_source("relationship_summary"),
            _descriptor_source("synthesis"),
        ],
    )

    # Act
    with caplog.at_level(logging.INFO, logger="orchestrator_demo.registry.agent_registry"):
        registry.reload()

    # Assert
    assert registry.get("internal_knowledge") is None
    assert registry.agent_ids() == ["relationship_summary", "synthesis"]
    assert "removed=internal_knowledge" in caplog.text


def test_module_reload_rejects_missing_available_agents_without_publishing_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Arrange
    from orchestrator_demo.registry.agent_registry import (
        AgentRegistry,
        RegistryValidationError,
    )

    module_name = "agent_config_missing_available_agents"
    config_path = tmp_path / f"{module_name}.py"
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delitem(sys.modules, module_name, raising=False)
    _write_config(config_path, [_descriptor_source("internal_knowledge")])
    registry = AgentRegistry(config_module=module_name)
    accepted_module = sys.modules[module_name]
    config_path.write_text(
        "from orchestrator_demo.contracts import AgentDescriptor\n\n",
        encoding="utf-8",
    )

    # Act / Assert
    with pytest.raises(RegistryValidationError) as exc_info:
        registry.reload()

    assert "registry config must define AVAILABLE_AGENTS" in str(exc_info.value)
    assert sys.modules[module_name] is accepted_module
    assert importlib.import_module(module_name) is accepted_module
    assert [
        descriptor.agent_id for descriptor in accepted_module.AVAILABLE_AGENTS
    ] == ["internal_knowledge"]
    assert registry.agent_ids() == ["internal_knowledge"]


def test_module_reload_rejects_invalid_descriptor_without_publishing_module_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Arrange
    from orchestrator_demo.registry.agent_registry import (
        AgentRegistry,
        RegistryValidationError,
    )

    module_name = "agent_config_invalid_descriptor"
    config_path = tmp_path / f"{module_name}.py"
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delitem(sys.modules, module_name, raising=False)
    _write_config(config_path, [_descriptor_source("internal_knowledge")])
    registry = AgentRegistry(config_module=module_name)
    accepted_module = sys.modules[module_name]
    config_path.write_text(
        "AVAILABLE_AGENTS = [\n"
        "    {\n"
        "        'agent_id': 'rejected_agent',\n"
        "        'display_name': 'Rejected Agent',\n"
        "        'capabilities': ['business banking support'],\n"
        "        'input_schema': {'type': 'object'},\n"
        "        'output_schema': {'type': 'object'},\n"
        "        'a2ui_catalogs': ['basic'],\n"
        "        'routing_examples': ['Handle a rejected request.'],\n"
        "        'execution_mode': 'local_llm',\n"
        "        'unsupported_field': 'must-not-publish',\n"
        "    }\n"
        "]\n",
        encoding="utf-8",
    )

    # Act / Assert
    with pytest.raises(RegistryValidationError) as exc_info:
        registry.reload()

    assert "unsupported_field" in str(exc_info.value)
    assert sys.modules[module_name] is accepted_module
    assert importlib.import_module(module_name) is accepted_module
    assert [
        descriptor.agent_id for descriptor in accepted_module.AVAILABLE_AGENTS
    ] == ["internal_knowledge"]
    assert registry.agent_ids() == ["internal_knowledge"]


def test_module_reload_observes_same_size_rapid_config_edit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Arrange
    from orchestrator_demo.registry.agent_registry import AgentRegistry

    module_name = "agent_config_same_size_edit"
    config_path = tmp_path / f"{module_name}.py"
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delitem(sys.modules, module_name, raising=False)

    first_source = (
        "from orchestrator_demo.contracts import AgentDescriptor\n\n"
        "AVAILABLE_AGENTS = [\n"
        f"{_descriptor_source('alpha_agent')}\n"
        "]\n"
    )
    second_source = (
        "from orchestrator_demo.contracts import AgentDescriptor\n\n"
        "AVAILABLE_AGENTS = [\n"
        f"{_descriptor_source('bravo_agent')}\n"
        "]\n"
    )
    assert len(first_source) == len(second_source)
    config_path.write_text(first_source, encoding="utf-8")
    fixed_mtime = 1_700_000_000
    os.utime(config_path, (fixed_mtime, fixed_mtime))
    py_compile.compile(str(config_path), doraise=True)
    registry = AgentRegistry(config_module=module_name)
    config_path.write_text(second_source, encoding="utf-8")
    os.utime(config_path, (fixed_mtime, fixed_mtime))

    # Act
    registry.reload()

    # Assert
    assert registry.agent_ids() == ["bravo_agent"]


def test_invalid_reload_keeps_previous_registry_and_redacts_secret_like_values(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Arrange
    from orchestrator_demo.registry.agent_registry import RegistryValidationError

    leaked_value = "sk-live-should-not-appear"
    config_path = tmp_path / "agent_config.py"
    _write_config(
        config_path,
        [
            _descriptor_source(
                "internal_knowledge",
                display_name="Original Internal Knowledge Agent",
            )
        ],
    )
    registry = _registry_from(config_path)
    previous_descriptors = {
        descriptor.agent_id: descriptor.model_dump()
        for descriptor in registry.descriptors()
    }
    config_path.write_text(
        "AVAILABLE_AGENTS = [\n"
        "    {\n"
        "        'agent_id': 'internal_knowledge',\n"
        "        'display_name': 'Mutated Internal Knowledge Agent',\n"
        "        'capabilities': ['crm notes'],\n"
        "        'input_schema': {'type': 'object'},\n"
        "        'output_schema': {'type': 'object'},\n"
        "        'a2ui_catalogs': ['basic'],\n"
        "        'routing_examples': ['Summarize notes.'],\n"
        "        'execution_mode': 'local_llm',\n"
        f"        'api_key': {leaked_value!r},\n"
        "    }\n"
        "]\n",
        encoding="utf-8",
    )

    # Act
    with caplog.at_level(logging.ERROR, logger="orchestrator_demo.registry.agent_registry"):
        with pytest.raises(RegistryValidationError) as exc_info:
            registry.reload()

    # Assert
    assert registry.agent_ids() == ["internal_knowledge"]
    assert {
        descriptor.agent_id: descriptor.model_dump()
        for descriptor in registry.descriptors()
    } == previous_descriptors
    assert "secret-like" in str(exc_info.value)
    assert "api_key" in str(exc_info.value)
    assert "agent registry reload rejected" in caplog.text
    assert "secret-like" in caplog.text
    assert leaked_value not in str(exc_info.value)
    assert leaked_value not in caplog.text


@pytest.mark.parametrize("schema_field", ["input_schema", "output_schema"])
def test_registry_rejects_invalid_input_and_output_schemas(
    tmp_path: Path,
    schema_field: str,
) -> None:
    # Arrange
    from orchestrator_demo.registry.agent_registry import RegistryValidationError

    config_path = tmp_path / "agent_config.py"
    invalid_schema_value = "definitely_not_a_json_schema_type"
    input_schema = "{'type': 'object'}"
    output_schema = "{'type': 'object'}"
    if schema_field == "input_schema":
        input_schema = f"{{'type': {invalid_schema_value!r}}}"
    else:
        output_schema = f"{{'type': {invalid_schema_value!r}}}"

    config_path.write_text(
        "from orchestrator_demo.contracts import AgentDescriptor\n\n"
        "AVAILABLE_AGENTS = [\n"
        "    AgentDescriptor(\n"
        "        agent_id='internal_knowledge',\n"
        "        display_name='Internal Knowledge Agent',\n"
        "        capabilities=['crm notes'],\n"
        f"        input_schema={input_schema},\n"
        f"        output_schema={output_schema},\n"
        "        a2ui_catalogs=['basic'],\n"
        "        routing_examples=['Summarize notes.'],\n"
        "        execution_mode='local_llm',\n"
        "    )\n"
        "]\n",
        encoding="utf-8",
    )

    # Act / Assert
    with pytest.raises(RegistryValidationError) as exc_info:
        _registry_from(config_path)

    assert schema_field in str(exc_info.value)
    assert invalid_schema_value in str(exc_info.value)


def test_approved_plan_references_to_unavailable_agents_are_detected_at_execution_time(
    tmp_path: Path,
) -> None:
    # Arrange
    from orchestrator_demo.registry.agent_registry import UnavailableAgentError

    config_path = tmp_path / "agent_config.py"
    _write_config(config_path, [_descriptor_source("internal_knowledge")])
    registry = _registry_from(config_path)
    approved_plan = ExecutionPlan(
        plan_id="plan_meeting_prep",
        objective="Prepare for a customer meeting.",
        detected_intents=["meeting_prep"],
        selected_agents=["internal_knowledge", "synthesis"],
        steps=[
            PlanStep(
                step_id="step_internal_notes",
                agent_id="internal_knowledge",
                instruction="Summarize CRM notes.",
                expected_output="Internal notes summary.",
            ),
            PlanStep(
                step_id="step_synthesis",
                agent_id="synthesis",
                instruction="Prepare the meeting brief.",
                depends_on=["step_internal_notes"],
                expected_output="RM-ready meeting brief.",
            ),
        ],
    )

    # Act
    unavailable_agents = registry.find_unavailable_plan_agents(approved_plan)

    # Assert
    assert unavailable_agents == ["synthesis"]
    with pytest.raises(UnavailableAgentError) as exc_info:
        registry.require_plan_agents_available(approved_plan)
    assert "synthesis" in str(exc_info.value)
