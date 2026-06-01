import importlib
import logging
import os
import py_compile
import sys
import traceback
import warnings
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


def test_module_registry_reload_reads_source_when_bytecode_cache_is_stale(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Arrange
    from orchestrator_demo.registry.agent_registry import AgentRegistry

    package_name = "registry_reload_case"
    module_name = f"{package_name}.agent_config"
    package_dir = tmp_path / package_name
    package_dir.mkdir()
    (package_dir / "__init__.py").write_text("", encoding="utf-8")
    config_path = package_dir / "agent_config.py"

    monkeypatch.syspath_prepend(str(tmp_path))
    sys.modules.pop(module_name, None)
    sys.modules.pop(package_name, None)

    _write_config(config_path, [_descriptor_source("agent_alpha")])
    initial_stat = config_path.stat()
    imported_module = importlib.import_module(module_name)
    assert imported_module.AVAILABLE_AGENTS[0].agent_id == "agent_alpha"

    registry = AgentRegistry(config_module=module_name)
    assert registry.agent_ids() == ["agent_alpha"]

    _write_config(config_path, [_descriptor_source("agent_bravo")])
    assert config_path.stat().st_size == initial_stat.st_size
    os.utime(
        config_path,
        ns=(initial_stat.st_atime_ns, initial_stat.st_mtime_ns),
    )

    # Act
    registry.reload()

    # Assert
    assert registry.agent_ids() == ["agent_bravo"]
    parent_package = importlib.import_module(package_name)
    assert parent_package.agent_config.AVAILABLE_AGENTS[0].agent_id == "agent_bravo"
    assert sys.modules[module_name] is parent_package.agent_config


def test_module_registry_invalid_reload_restores_parent_package_submodule(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Arrange
    from orchestrator_demo.registry.agent_registry import (
        AgentRegistry,
        RegistryValidationError,
    )

    package_name = "registry_reload_rollback_case"
    module_name = f"{package_name}.agent_config"
    package_dir = tmp_path / package_name
    package_dir.mkdir()
    (package_dir / "__init__.py").write_text("", encoding="utf-8")
    config_path = package_dir / "agent_config.py"

    monkeypatch.syspath_prepend(str(tmp_path))
    sys.modules.pop(module_name, None)
    sys.modules.pop(package_name, None)

    _write_config(config_path, [_descriptor_source("agent_alpha")])
    importlib.import_module(module_name)
    registry = AgentRegistry(config_module=module_name)

    _write_config(config_path, [_descriptor_source("agent_bravo")])
    registry.reload()

    config_path.write_text(
        "from orchestrator_demo.contracts import AgentDescriptor\n\n"
        "AVAILABLE_AGENTS = [\n"
        "    AgentDescriptor(\n"
        "        agent_id='agent_charlie',\n"
        "        display_name='Agent Charlie',\n"
        "        capabilities=['crm notes'],\n"
        "        input_schema={'type': 'object', 'required': ['api_key']},\n"
        "        output_schema={'type': 'object'},\n"
        "        a2ui_catalogs=['basic'],\n"
        "        routing_examples=['Summarize notes.'],\n"
        "        execution_mode='local_llm',\n"
        "    )\n"
        "]\n",
        encoding="utf-8",
    )

    # Act / Assert
    with pytest.raises(RegistryValidationError):
        registry.reload()

    parent_package = importlib.import_module(package_name)
    assert registry.agent_ids() == ["agent_bravo"]
    assert parent_package.agent_config.AVAILABLE_AGENTS[0].agent_id == "agent_bravo"
    assert sys.modules[module_name] is parent_package.agent_config


def test_module_registry_unexpected_validation_error_restores_published_module(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Arrange
    from orchestrator_demo.registry.agent_registry import (
        AgentRegistry,
        RegistryValidationError,
    )

    package_name = "registry_reload_unexpected_rollback_case"
    module_name = f"{package_name}.agent_config"
    package_dir = tmp_path / package_name
    package_dir.mkdir()
    (package_dir / "__init__.py").write_text("", encoding="utf-8")
    config_path = package_dir / "agent_config.py"

    monkeypatch.syspath_prepend(str(tmp_path))
    sys.modules.pop(module_name, None)
    sys.modules.pop(package_name, None)

    _write_config(config_path, [_descriptor_source("agent_alpha")])
    registry = AgentRegistry(config_module=module_name)
    accepted_module = sys.modules[module_name]
    parent_package = importlib.import_module(package_name)
    assert parent_package.agent_config is accepted_module

    config_path.write_text(
        "from collections.abc import Sequence\n\n"
        "class RaisingAgents(Sequence):\n"
        "    def __len__(self):\n"
        "        return 1\n\n"
        "    def __getitem__(self, index):\n"
        "        raise RuntimeError('must-not-publish')\n\n"
        "    def __iter__(self):\n"
        "        raise RuntimeError('must-not-publish')\n\n"
        "AVAILABLE_AGENTS = RaisingAgents()\n",
        encoding="utf-8",
    )

    # Act / Assert
    with caplog.at_level(logging.ERROR, logger="orchestrator_demo.registry.agent_registry"):
        with pytest.raises(RegistryValidationError) as exc_info:
            registry.reload()

    parent_package = importlib.import_module(package_name)
    assert "failed to validate registry config: RuntimeError" in str(exc_info.value)
    assert "must-not-publish" not in str(exc_info.value)
    assert "must-not-publish" not in caplog.text
    assert registry.agent_ids() == ["agent_alpha"]
    assert sys.modules[module_name] is accepted_module
    assert importlib.import_module(module_name) is accepted_module
    assert parent_package.agent_config is accepted_module
    assert [
        descriptor.agent_id for descriptor in accepted_module.AVAILABLE_AGENTS
    ] == ["agent_alpha"]


def test_module_registry_reload_executes_config_with_import_compatible_module_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Arrange
    from orchestrator_demo.registry.agent_registry import AgentRegistry

    package_name = "registry_reload_dataclass_case"
    module_name = f"{package_name}.agent_config"
    package_dir = tmp_path / package_name
    package_dir.mkdir()
    (package_dir / "__init__.py").write_text("", encoding="utf-8")
    config_path = package_dir / "agent_config.py"

    monkeypatch.syspath_prepend(str(tmp_path))
    sys.modules.pop(module_name, None)
    sys.modules.pop(package_name, None)

    config_path.write_text(
        "from __future__ import annotations\n\n"
        "from dataclasses import dataclass\n\n"
        "from orchestrator_demo.contracts import AgentDescriptor\n\n"
        "@dataclass\n"
        "class DescriptorFactory:\n"
        "    agent_id: str\n\n"
        "    def build(self) -> AgentDescriptor:\n"
        "        return AgentDescriptor(\n"
        "            agent_id=self.agent_id,\n"
        "            display_name='Internal Knowledge Agent',\n"
        "            capabilities=['crm notes'],\n"
        "            input_schema={'type': 'object'},\n"
        "            output_schema={'type': 'object'},\n"
        "            a2ui_catalogs=['basic'],\n"
        "            routing_examples=['Summarize notes.'],\n"
        "            execution_mode='local_llm',\n"
        "        )\n\n"
        "AVAILABLE_AGENTS = [DescriptorFactory('internal_knowledge').build()]\n",
        encoding="utf-8",
    )

    # Act
    registry = AgentRegistry(config_module=module_name)

    # Assert
    assert registry.agent_ids() == ["internal_knowledge"]
    assert sys.modules[module_name].AVAILABLE_AGENTS[0].agent_id == "internal_knowledge"


def test_module_registry_compile_does_not_inherit_future_annotation_flags(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Arrange
    from orchestrator_demo.contracts import AgentDescriptor
    from orchestrator_demo.registry.agent_registry import AgentRegistry

    package_name = "registry_reload_annotation_case"
    module_name = f"{package_name}.agent_config"
    package_dir = tmp_path / package_name
    package_dir.mkdir()
    (package_dir / "__init__.py").write_text("", encoding="utf-8")
    config_path = package_dir / "agent_config.py"

    monkeypatch.syspath_prepend(str(tmp_path))
    sys.modules.pop(module_name, None)
    sys.modules.pop(package_name, None)

    config_path.write_text(
        "from orchestrator_demo.contracts import AgentDescriptor\n\n"
        "def build_descriptor() -> AgentDescriptor:\n"
        "    return AgentDescriptor(\n"
        "        agent_id='internal_knowledge',\n"
        "        display_name='Internal Knowledge Agent',\n"
        "        capabilities=['crm notes'],\n"
        "        input_schema={'type': 'object'},\n"
        "        output_schema={'type': 'object'},\n"
        "        a2ui_catalogs=['basic'],\n"
        "        routing_examples=['Summarize notes.'],\n"
        "        execution_mode='local_llm',\n"
        "    )\n\n"
        "if build_descriptor.__annotations__['return'] is not AgentDescriptor:\n"
        "    raise RuntimeError('runtime annotations were not preserved')\n\n"
        "AVAILABLE_AGENTS = [build_descriptor()]\n",
        encoding="utf-8",
    )

    # Act
    registry = AgentRegistry(config_module=module_name)

    # Assert
    assert registry.agent_ids() == ["internal_knowledge"]
    assert (
        sys.modules[module_name].build_descriptor.__annotations__["return"]
        is AgentDescriptor
    )

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


@pytest.mark.parametrize(
    ("display_name", "routing_examples", "input_schema", "expected_path"),
    [
        pytest.param(
            "leaked_value",
            "['Summarize notes.']",
            "{'type': 'object'}",
            "AVAILABLE_AGENTS[0].display_name",
            id="display-name",
        ),
        pytest.param(
            "'Internal Knowledge Agent'",
            "[leaked_value]",
            "{'type': 'object'}",
            "AVAILABLE_AGENTS[0].routing_examples[0]",
            id="routing-example",
        ),
        pytest.param(
            "'Internal Knowledge Agent'",
            "['Summarize notes.']",
            "{'type': 'object', 'description': leaked_value}",
            "AVAILABLE_AGENTS[0].input_schema.description",
            id="schema-description",
        ),
        pytest.param(
            "'Internal Knowledge Agent'",
            "['Summarize notes.']",
            (
                "{"
                "'type': 'object', "
                "'properties': {"
                "    'status': {"
                "        'type': 'string', "
                "        'enum': ['active', leaked_value],"
                "    },"
                "},"
                "}"
            ),
            "AVAILABLE_AGENTS[0].input_schema.properties.status.enum[1]",
            id="schema-enum",
        ),
    ],
)
def test_invalid_reload_rejects_secret_like_string_values_without_leaking_them(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    display_name: str,
    routing_examples: str,
    input_schema: str,
    expected_path: str,
) -> None:
    # Arrange
    from orchestrator_demo.registry.agent_registry import RegistryValidationError

    leaked_value = "sk-live-string-value-should-not-appear"
    config_path = tmp_path / "agent_config.py"
    _write_config(config_path, [_descriptor_source("internal_knowledge")])
    registry = _registry_from(config_path)
    previous_descriptors = {
        descriptor.agent_id: descriptor.model_dump()
        for descriptor in registry.descriptors()
    }
    config_path.write_text(
        "from orchestrator_demo.contracts import AgentDescriptor\n\n"
        f"leaked_value = {leaked_value!r}\n\n"
        "AVAILABLE_AGENTS = [\n"
        "    AgentDescriptor(\n"
        "        agent_id='internal_knowledge',\n"
        f"        display_name={display_name},\n"
        "        capabilities=['crm notes'],\n"
        f"        input_schema={input_schema},\n"
        "        output_schema={'type': 'object'},\n"
        "        a2ui_catalogs=['basic'],\n"
        f"        routing_examples={routing_examples},\n"
        "        execution_mode='local_llm',\n"
        "    )\n"
        "]\n",
        encoding="utf-8",
    )

    # Act / Assert
    with caplog.at_level(logging.ERROR, logger="orchestrator_demo.registry.agent_registry"):
        with pytest.raises(RegistryValidationError) as exc_info:
            registry.reload()

    assert {
        descriptor.agent_id: descriptor.model_dump()
        for descriptor in registry.descriptors()
    } == previous_descriptors
    assert "secret-like value" in str(exc_info.value)
    assert expected_path in str(exc_info.value)
    assert "agent registry reload rejected" in caplog.text
    assert leaked_value not in str(exc_info.value)
    assert leaked_value not in caplog.text


@pytest.mark.parametrize(
    ("schema_source", "expected_error", "expected_path", "leaked_value"),
    [
        pytest.param(
            "{'type': 'object', 'propertyNames': {'type': 'unsupported'}}",
            "invalid JSON-schema type",
            "input_schema.propertyNames.type",
            None,
            id="unsupported-type",
        ),
        pytest.param(
            "{'type': 'object', 'propertyNames': {"
            "'type': 'object', 'required': ['api_key']"
            "}}",
            "secret-like",
            "input_schema.propertyNames.required[0]",
            "api_key",
            id="secret-like-required-name",
        ),
        pytest.param(
            "{'type': 'object', 'propertyNames': {'type': "
            "'sk-live-property-names-type-should-not-appear'"
            "}}",
            "invalid JSON-schema type",
            "input_schema.propertyNames.type",
            "sk-live-property-names-type-should-not-appear",
            id="secret-like-type-value",
        ),
    ],
)
def test_invalid_reload_validates_property_names_schema_without_leak(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    schema_source: str,
    expected_error: str,
    expected_path: str,
    leaked_value: str | None,
) -> None:
    # Arrange
    from orchestrator_demo.registry.agent_registry import RegistryValidationError

    config_path = tmp_path / "agent_config.py"
    _write_config(config_path, [_descriptor_source("internal_knowledge")])
    registry = _registry_from(config_path)
    previous_descriptors = {
        descriptor.agent_id: descriptor.model_dump()
        for descriptor in registry.descriptors()
    }
    config_path.write_text(
        "from orchestrator_demo.contracts import AgentDescriptor\n\n"
        "AVAILABLE_AGENTS = [\n"
        "    AgentDescriptor(\n"
        "        agent_id='internal_knowledge',\n"
        "        display_name='Mutated Internal Knowledge Agent',\n"
        "        capabilities=['crm notes'],\n"
        f"        input_schema={schema_source},\n"
        "        output_schema={'type': 'object'},\n"
        "        a2ui_catalogs=['basic'],\n"
        "        routing_examples=['Summarize notes.'],\n"
        "        execution_mode='local_llm',\n"
        "    )\n"
        "]\n",
        encoding="utf-8",
    )

    # Act / Assert
    with caplog.at_level(logging.ERROR, logger="orchestrator_demo.registry.agent_registry"):
        with pytest.raises(RegistryValidationError) as exc_info:
            registry.reload()

    current_descriptors = {
        descriptor.agent_id: descriptor.model_dump()
        for descriptor in registry.descriptors()
    }
    assert current_descriptors == previous_descriptors
    assert expected_error in str(exc_info.value)
    assert expected_path in str(exc_info.value)
    assert "agent registry reload rejected" in caplog.text
    if leaked_value is not None:
        assert leaked_value not in repr(current_descriptors)
        assert leaked_value not in str(exc_info.value)
        assert leaked_value not in caplog.text


@pytest.mark.parametrize(
    ("config_prelude", "descriptor_source", "expected_path"),
    [
        pytest.param(
            "",
            (
                "{"
                "    'agent_id': b'sk-live-coerced-agent-secret',"
                "    'display_name': 'Mutated Internal Knowledge Agent',"
                "    'capabilities': ['crm notes'],"
                "    'input_schema': {'type': 'object'},"
                "    'output_schema': {'type': 'object'},"
                "    'a2ui_catalogs': ['basic'],"
                "    'routing_examples': ['Summarize notes.'],"
                "    'execution_mode': 'local_llm',"
                "}"
            ),
            "AVAILABLE_AGENTS[0].agent_id",
            id="coerced-bytes-agent-id",
        ),
        pytest.param(
            "",
            (
                "{"
                "    'agent_id': 'internal_knowledge',"
                "    'display_name': 'Mutated Internal Knowledge Agent',"
                "    'capabilities': {b'sk-live-set-capability-secret'},"
                "    'input_schema': {'type': 'object'},"
                "    'output_schema': {'type': 'object'},"
                "    'a2ui_catalogs': ['basic'],"
                "    'routing_examples': ['Summarize notes.'],"
                "    'execution_mode': 'local_llm',"
                "}"
            ),
            "AVAILABLE_AGENTS[0].capabilities[0]",
            id="set-capability",
        ),
        pytest.param(
            (
                "from enum import Enum\n\n"
                "class AgentIds(Enum):\n"
                "    INTERNAL = 'sk-live-enum-agent-secret'\n\n"
            ),
            (
                "{"
                "    'agent_id': AgentIds.INTERNAL,"
                "    'display_name': 'Mutated Internal Knowledge Agent',"
                "    'capabilities': ['crm notes'],"
                "    'input_schema': {'type': 'object'},"
                "    'output_schema': {'type': 'object'},"
                "    'a2ui_catalogs': ['basic'],"
                "    'routing_examples': ['Summarize notes.'],"
                "    'execution_mode': 'local_llm',"
                "}"
            ),
            "AVAILABLE_AGENTS[0].agent_id",
            id="post-coercion-enum-agent-id",
        ),
    ],
)
def test_invalid_reload_rejects_coerced_secret_like_values_without_leaking_them(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    config_prelude: str,
    descriptor_source: str,
    expected_path: str,
) -> None:
    # Arrange
    from orchestrator_demo.registry.agent_registry import RegistryValidationError

    config_path = tmp_path / "agent_config.py"
    _write_config(config_path, [_descriptor_source("internal_knowledge")])
    registry = _registry_from(config_path)
    previous_descriptors = {
        descriptor.agent_id: descriptor.model_dump()
        for descriptor in registry.descriptors()
    }
    config_path.write_text(
        f"{config_prelude}"
        "AVAILABLE_AGENTS = [\n"
        f"    {descriptor_source}\n"
        "]\n",
        encoding="utf-8",
    )

    # Act / Assert
    with caplog.at_level(logging.ERROR, logger="orchestrator_demo.registry.agent_registry"):
        with pytest.raises(RegistryValidationError) as exc_info:
            registry.reload()

    assert {
        descriptor.agent_id: descriptor.model_dump()
        for descriptor in registry.descriptors()
    } == previous_descriptors
    assert "secret-like value" in str(exc_info.value)
    assert expected_path in str(exc_info.value)
    assert "agent registry reload rejected" in caplog.text
    assert "sk-live" not in str(exc_info.value)
    assert "sk-live" not in caplog.text


def test_invalid_reload_revalidates_mutated_agent_descriptor_instances(
    tmp_path: Path,
) -> None:
    # Arrange
    from orchestrator_demo.registry.agent_registry import RegistryValidationError

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
        "from orchestrator_demo.contracts import AgentDescriptor\n\n"
        "descriptor = AgentDescriptor(\n"
        "    agent_id='internal_knowledge',\n"
        "    display_name='Mutated Internal Knowledge Agent',\n"
        "    capabilities=['crm notes'],\n"
        "    input_schema={'type': 'object'},\n"
        "    output_schema={'type': 'object'},\n"
        "    a2ui_catalogs=['basic'],\n"
        "    routing_examples=['Summarize notes.'],\n"
        "    execution_mode='local_llm',\n"
        ")\n"
        "descriptor.execution_mode = 'serverless'\n"
        "AVAILABLE_AGENTS = [descriptor]\n",
        encoding="utf-8",
    )

    # Act / Assert
    with pytest.raises(RegistryValidationError) as exc_info:
        registry.reload()

    assert registry.agent_ids() == ["internal_knowledge"]
    assert {
        descriptor.agent_id: descriptor.model_dump()
        for descriptor in registry.descriptors()
    } == previous_descriptors
    assert "execution_mode" in str(exc_info.value)


def test_invalid_reload_suppresses_pydantic_dump_warnings_for_mutated_descriptor_instances(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Arrange
    from orchestrator_demo.registry.agent_registry import RegistryValidationError

    leaked_value = "sk-live-mutated-schema-should-not-appear"
    config_path = tmp_path / "agent_config.py"
    _write_config(config_path, [_descriptor_source("internal_knowledge")])
    registry = _registry_from(config_path)
    config_path.write_text(
        "from orchestrator_demo.contracts import AgentDescriptor\n\n"
        "descriptor = AgentDescriptor(\n"
        "    agent_id='internal_knowledge',\n"
        "    display_name='Internal Knowledge Agent',\n"
        "    capabilities=['crm notes'],\n"
        "    input_schema={'type': 'object'},\n"
        "    output_schema={'type': 'object'},\n"
        "    a2ui_catalogs=['basic'],\n"
        "    routing_examples=['Summarize notes.'],\n"
        "    execution_mode='local_llm',\n"
        ")\n"
        f"descriptor.input_schema = {leaked_value!r}\n"
        "AVAILABLE_AGENTS = [descriptor]\n",
        encoding="utf-8",
    )

    # Act / Assert
    with warnings.catch_warnings(record=True) as captured_warnings:
        warnings.simplefilter("always")
        with caplog.at_level(logging.ERROR, logger="orchestrator_demo.registry.agent_registry"):
            with pytest.raises(RegistryValidationError) as exc_info:
                registry.reload()

    warning_text = "\n".join(str(warning.message) for warning in captured_warnings)
    assert registry.agent_ids() == ["internal_knowledge"]
    assert "input_schema" in str(exc_info.value)
    assert "Pydantic serializer warnings" not in warning_text
    assert leaked_value not in warning_text
    assert leaked_value not in str(exc_info.value)
    assert leaked_value not in caplog.text


@pytest.mark.parametrize("schema_field", ["input_schema", "output_schema"])
def test_registry_rejects_invalid_input_and_output_schemas(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    schema_field: str,
) -> None:
    # Arrange
    from orchestrator_demo.registry.agent_registry import RegistryValidationError

    config_path = tmp_path / "agent_config.py"
    leaked_value = "sk-live-schema-type-should-not-appear"
    input_schema = "{'type': 'object'}"
    output_schema = "{'type': 'object'}"
    if schema_field == "input_schema":
        input_schema = f"{{'type': {leaked_value!r}}}"
    else:
        output_schema = f"{{'type': {leaked_value!r}}}"

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
    with caplog.at_level(logging.ERROR, logger="orchestrator_demo.registry.agent_registry"):
        with pytest.raises(RegistryValidationError) as exc_info:
            _registry_from(config_path)

    assert schema_field in str(exc_info.value)
    assert "invalid JSON-schema type" in str(exc_info.value)
    assert leaked_value not in str(exc_info.value)
    assert leaked_value not in caplog.text


@pytest.mark.parametrize("schema_field", ["input_schema", "output_schema"])
def test_registry_reports_non_secret_invalid_schema_type_values(
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


def test_registry_rejects_schema_type_arrays_without_leaking_values(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Arrange
    from orchestrator_demo.registry.agent_registry import RegistryValidationError

    leaked_value = "sk-live-array-type-should-not-appear"
    config_path = tmp_path / "agent_config.py"
    config_path.write_text(
        "from orchestrator_demo.contracts import AgentDescriptor\n\n"
        "AVAILABLE_AGENTS = [\n"
        "    AgentDescriptor(\n"
        "        agent_id='internal_knowledge',\n"
        "        display_name='Internal Knowledge Agent',\n"
        "        capabilities=['crm notes'],\n"
        f"        input_schema={{'type': ['object', {leaked_value!r}]}},\n"
        "        output_schema={'type': 'object'},\n"
        "        a2ui_catalogs=['basic'],\n"
        "        routing_examples=['Summarize notes.'],\n"
        "        execution_mode='local_llm',\n"
        "    )\n"
        "]\n",
        encoding="utf-8",
    )

    # Act / Assert
    with caplog.at_level(logging.ERROR, logger="orchestrator_demo.registry.agent_registry"):
        with pytest.raises(RegistryValidationError) as exc_info:
            _registry_from(config_path)

    assert "input_schema.type" in str(exc_info.value)
    assert "invalid JSON-schema type" in str(exc_info.value)
    assert leaked_value not in str(exc_info.value)
    assert leaked_value not in caplog.text


@pytest.mark.parametrize("secret_field", ["accessKey", "privateKey"])
def test_registry_rejects_camel_case_secret_fields_in_schema_metadata(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    secret_field: str,
) -> None:
    # Arrange
    from orchestrator_demo.registry.agent_registry import RegistryValidationError

    leaked_value = "sk-live-camel-case-secret-should-not-appear"
    config_path = tmp_path / "agent_config.py"
    config_path.write_text(
        "AVAILABLE_AGENTS = [\n"
        "    {\n"
        "        'agent_id': 'internal_knowledge',\n"
        "        'display_name': 'Internal Knowledge Agent',\n"
        "        'capabilities': ['crm notes'],\n"
        "        'input_schema': {\n"
        "            'type': 'object',\n"
        "            'properties': {\n"
        "                'account': {\n"
        "                    'type': 'string',\n"
        f"                    {secret_field!r}: {leaked_value!r},\n"
        "                },\n"
        "            },\n"
        "        },\n"
        "        'output_schema': {'type': 'object'},\n"
        "        'a2ui_catalogs': ['basic'],\n"
        "        'routing_examples': ['Summarize notes.'],\n"
        "        'execution_mode': 'local_llm',\n"
        "    }\n"
        "]\n",
        encoding="utf-8",
    )

    # Act / Assert
    with caplog.at_level(logging.ERROR, logger="orchestrator_demo.registry.agent_registry"):
        with pytest.raises(RegistryValidationError) as exc_info:
            _registry_from(config_path)

    assert "secret-like" in str(exc_info.value)
    assert secret_field in str(exc_info.value)
    assert leaked_value not in str(exc_info.value)
    assert leaked_value not in caplog.text


@pytest.mark.parametrize("schema_field", ["input_schema", "output_schema"])
def test_registry_rejects_secret_like_required_schema_fields(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    schema_field: str,
) -> None:
    # Arrange
    from orchestrator_demo.registry.agent_registry import RegistryValidationError

    input_schema = "{'type': 'object'}"
    output_schema = "{'type': 'object'}"
    schema_with_secret_required = (
        "{"
        "'type': 'object', "
        "'properties': {"
        "    'profile': {"
        "        'type': 'object', "
        "        'required': ['api_key'], "
        "        'properties': {'account_id': {'type': 'string'}},"
        "    },"
        "},"
        "}"
    )
    if schema_field == "input_schema":
        input_schema = schema_with_secret_required
    else:
        output_schema = schema_with_secret_required

    config_path = tmp_path / "agent_config.py"
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
    with caplog.at_level(logging.ERROR, logger="orchestrator_demo.registry.agent_registry"):
        with pytest.raises(RegistryValidationError) as exc_info:
            _registry_from(config_path)

    assert "secret-like" in str(exc_info.value)
    assert f"{schema_field}.properties.profile.required[0]" in str(exc_info.value)
    assert "secret-like" in caplog.text


@pytest.mark.parametrize(
    ("schema_source", "expected_path"),
    [
        pytest.param(
            (
                "{"
                "'type': 'object', "
                "'allOf': ["
                "    {'type': 'object', 'required': ['api_key']}"
                "],"
                "}"
            ),
            "input_schema.allOf[0].required[0]",
            id="allOf",
        ),
        pytest.param(
            (
                "{"
                "'type': 'object', "
                "'$defs': {"
                "    'request_payload': {'type': 'object', 'required': ['api_key']}"
                "},"
                "}"
            ),
            "input_schema.$defs.request_payload.required[0]",
            id="$defs",
        ),
        pytest.param(
            (
                "{"
                "'type': 'object', "
                "'additionalProperties': {"
                "    'type': 'object', "
                "    'required': ['api_key']"
                "},"
                "}"
            ),
            "input_schema.additionalProperties.required[0]",
            id="additionalProperties",
        ),
        pytest.param(
            (
                "{"
                "'type': 'object', "
                "'not': {'type': 'object', 'required': ['api_key']},"
                "}"
            ),
            "input_schema.not.required[0]",
            id="not",
        ),
        pytest.param(
            (
                "{"
                "'type': 'object', "
                "'if': {'type': 'object', 'required': ['api_key']},"
                "}"
            ),
            "input_schema.if.required[0]",
            id="if",
        ),
        pytest.param(
            (
                "{"
                "'type': 'object', "
                "'then': {'type': 'object', 'required': ['api_key']},"
                "}"
            ),
            "input_schema.then.required[0]",
            id="then",
        ),
        pytest.param(
            (
                "{"
                "'type': 'object', "
                "'else': {'type': 'object', 'required': ['api_key']},"
                "}"
            ),
            "input_schema.else.required[0]",
            id="else",
        ),
        pytest.param(
            (
                "{"
                "'type': 'array', "
                "'contains': {'type': 'object', 'required': ['api_key']},"
                "}"
            ),
            "input_schema.contains.required[0]",
            id="contains",
        ),
        pytest.param(
            (
                "{"
                "'type': 'array', "
                "'prefixItems': ["
                "    {'type': 'object', 'required': ['api_key']}"
                "],"
                "}"
            ),
            "input_schema.prefixItems[0].required[0]",
            id="prefixItems",
        ),
        pytest.param(
            (
                "{"
                "'type': 'array', "
                "'unevaluatedItems': {'type': 'object', 'required': ['api_key']},"
                "}"
            ),
            "input_schema.unevaluatedItems.required[0]",
            id="unevaluatedItems",
        ),
        pytest.param(
            (
                "{"
                "'type': 'object', "
                "'unevaluatedProperties': {"
                "    'type': 'object', "
                "    'required': ['api_key']"
                "},"
                "}"
            ),
            "input_schema.unevaluatedProperties.required[0]",
            id="unevaluatedProperties",
        ),
    ],
)
def test_invalid_reload_rejects_secret_like_required_fields_in_schema_containers(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    schema_source: str,
    expected_path: str,
) -> None:
    # Arrange
    from orchestrator_demo.registry.agent_registry import RegistryValidationError

    config_path = tmp_path / "agent_config.py"
    _write_config(config_path, [_descriptor_source("internal_knowledge")])
    registry = _registry_from(config_path)
    previous_descriptors = {
        descriptor.agent_id: descriptor.model_dump()
        for descriptor in registry.descriptors()
    }
    config_path.write_text(
        "from orchestrator_demo.contracts import AgentDescriptor\n\n"
        "AVAILABLE_AGENTS = [\n"
        "    AgentDescriptor(\n"
        "        agent_id='internal_knowledge',\n"
        "        display_name='Mutated Internal Knowledge Agent',\n"
        "        capabilities=['crm notes'],\n"
        f"        input_schema={schema_source},\n"
        "        output_schema={'type': 'object'},\n"
        "        a2ui_catalogs=['basic'],\n"
        "        routing_examples=['Summarize notes.'],\n"
        "        execution_mode='local_llm',\n"
        "    )\n"
        "]\n",
        encoding="utf-8",
    )

    # Act / Assert
    with caplog.at_level(logging.ERROR, logger="orchestrator_demo.registry.agent_registry"):
        with pytest.raises(RegistryValidationError) as exc_info:
            registry.reload()

    assert {
        descriptor.agent_id: descriptor.model_dump()
        for descriptor in registry.descriptors()
    } == previous_descriptors
    assert "secret-like" in str(exc_info.value)
    assert expected_path in str(exc_info.value)
    assert "agent registry reload rejected" in caplog.text


@pytest.mark.parametrize(
    ("schema_source", "expected_path"),
    [
        pytest.param(
            "{'type': 'array', 'unevaluatedItems': ['not', 'a', 'schema']}",
            "input_schema.unevaluatedItems",
            id="unevaluatedItems",
        ),
        pytest.param(
            "{'type': 'object', 'unevaluatedProperties': ['not', 'a', 'schema']}",
            "input_schema.unevaluatedProperties",
            id="unevaluatedProperties",
        ),
    ],
)
def test_invalid_reload_rejects_invalid_unevaluated_schema_containers(
    tmp_path: Path,
    schema_source: str,
    expected_path: str,
) -> None:
    # Arrange
    from orchestrator_demo.registry.agent_registry import RegistryValidationError

    config_path = tmp_path / "agent_config.py"
    _write_config(config_path, [_descriptor_source("internal_knowledge")])
    registry = _registry_from(config_path)
    previous_descriptors = {
        descriptor.agent_id: descriptor.model_dump()
        for descriptor in registry.descriptors()
    }
    config_path.write_text(
        "from orchestrator_demo.contracts import AgentDescriptor\n\n"
        "AVAILABLE_AGENTS = [\n"
        "    AgentDescriptor(\n"
        "        agent_id='internal_knowledge',\n"
        "        display_name='Mutated Internal Knowledge Agent',\n"
        "        capabilities=['crm notes'],\n"
        f"        input_schema={schema_source},\n"
        "        output_schema={'type': 'object'},\n"
        "        a2ui_catalogs=['basic'],\n"
        "        routing_examples=['Summarize notes.'],\n"
        "        execution_mode='local_llm',\n"
        "    )\n"
        "]\n",
        encoding="utf-8",
    )

    # Act / Assert
    with pytest.raises(RegistryValidationError) as exc_info:
        registry.reload()

    assert {
        descriptor.agent_id: descriptor.model_dump()
        for descriptor in registry.descriptors()
    } == previous_descriptors
    assert expected_path in str(exc_info.value)
    assert "must be a JSON-schema object" in str(exc_info.value)


@pytest.mark.parametrize(
    ("schema_source", "expected_path"),
    [
        pytest.param(
            (
                "{"
                "'type': 'object', "
                "'dependentRequired': {'customer_id': ['api_key']},"
                "}"
            ),
            "input_schema.dependentRequired.customer_id[0]",
            id="top-level",
        ),
        pytest.param(
            (
                "{"
                "'type': 'object', "
                "'properties': {"
                "    'profile': {"
                "        'type': 'object', "
                "        'dependentRequired': {'customer_id': ['api_key']},"
                "    },"
                "},"
                "}"
            ),
            "input_schema.properties.profile.dependentRequired.customer_id[0]",
            id="property",
        ),
        pytest.param(
            (
                "{"
                "'type': 'object', "
                "'allOf': ["
                "    {"
                "        'type': 'object', "
                "        'dependentRequired': {'customer_id': ['api_key']}"
                "    }"
                "],"
                "}"
            ),
            "input_schema.allOf[0].dependentRequired.customer_id[0]",
            id="allOf",
        ),
    ],
)
def test_invalid_reload_rejects_secret_like_dependent_required_fields(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    schema_source: str,
    expected_path: str,
) -> None:
    # Arrange
    from orchestrator_demo.registry.agent_registry import RegistryValidationError

    config_path = tmp_path / "agent_config.py"
    _write_config(config_path, [_descriptor_source("internal_knowledge")])
    registry = _registry_from(config_path)
    previous_descriptors = {
        descriptor.agent_id: descriptor.model_dump()
        for descriptor in registry.descriptors()
    }
    config_path.write_text(
        "from orchestrator_demo.contracts import AgentDescriptor\n\n"
        "AVAILABLE_AGENTS = [\n"
        "    AgentDescriptor(\n"
        "        agent_id='internal_knowledge',\n"
        "        display_name='Mutated Internal Knowledge Agent',\n"
        "        capabilities=['crm notes'],\n"
        f"        input_schema={schema_source},\n"
        "        output_schema={'type': 'object'},\n"
        "        a2ui_catalogs=['basic'],\n"
        "        routing_examples=['Summarize notes.'],\n"
        "        execution_mode='local_llm',\n"
        "    )\n"
        "]\n",
        encoding="utf-8",
    )

    # Act / Assert
    with caplog.at_level(logging.ERROR, logger="orchestrator_demo.registry.agent_registry"):
        with pytest.raises(RegistryValidationError) as exc_info:
            registry.reload()

    assert {
        descriptor.agent_id: descriptor.model_dump()
        for descriptor in registry.descriptors()
    } == previous_descriptors
    assert "secret-like" in str(exc_info.value)
    assert expected_path in str(exc_info.value)
    assert "agent registry reload rejected" in caplog.text


@pytest.mark.parametrize(
    ("schema_source", "expected_path", "expected_error"),
    [
        pytest.param(
            (
                "{"
                "'type': 'object', "
                "'dependentSchemas': {"
                "    'customer': {'type': 'object', 'required': ['api_key']}"
                "},"
                "}"
            ),
            "input_schema.dependentSchemas.customer.required[0]",
            "secret-like",
            id="dependent-schemas-secret-required",
        ),
        pytest.param(
            (
                "{"
                "'type': 'object', "
                "'patternProperties': {"
                "    '^customer': {'type': 'object', 'required': ['api_key']}"
                "},"
                "}"
            ),
            "input_schema.patternProperties.^customer.required[0]",
            "secret-like",
            id="pattern-properties-secret-required",
        ),
        pytest.param(
            (
                "{"
                "'type': 'object', "
                "'dependentSchemas': {'customer': {'type': 'unsupported'}},"
                "}"
            ),
            "input_schema.dependentSchemas.customer.type",
            "invalid JSON-schema type",
            id="dependent-schemas-invalid-type",
        ),
        pytest.param(
            (
                "{"
                "'type': 'object', "
                "'patternProperties': {'^customer': ['not', 'a', 'schema']},"
                "}"
            ),
            "input_schema.patternProperties.^customer",
            "must be a JSON-schema object",
            id="pattern-properties-invalid-schema",
        ),
    ],
)
def test_invalid_reload_validates_schema_valued_maps(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    schema_source: str,
    expected_path: str,
    expected_error: str,
) -> None:
    # Arrange
    from orchestrator_demo.registry.agent_registry import RegistryValidationError

    config_path = tmp_path / "agent_config.py"
    _write_config(config_path, [_descriptor_source("internal_knowledge")])
    registry = _registry_from(config_path)
    previous_descriptors = {
        descriptor.agent_id: descriptor.model_dump()
        for descriptor in registry.descriptors()
    }
    config_path.write_text(
        "from orchestrator_demo.contracts import AgentDescriptor\n\n"
        "AVAILABLE_AGENTS = [\n"
        "    AgentDescriptor(\n"
        "        agent_id='internal_knowledge',\n"
        "        display_name='Mutated Internal Knowledge Agent',\n"
        "        capabilities=['crm notes'],\n"
        f"        input_schema={schema_source},\n"
        "        output_schema={'type': 'object'},\n"
        "        a2ui_catalogs=['basic'],\n"
        "        routing_examples=['Summarize notes.'],\n"
        "        execution_mode='local_llm',\n"
        "    )\n"
        "]\n",
        encoding="utf-8",
    )

    # Act / Assert
    with caplog.at_level(logging.ERROR, logger="orchestrator_demo.registry.agent_registry"):
        with pytest.raises(RegistryValidationError) as exc_info:
            registry.reload()

    assert {
        descriptor.agent_id: descriptor.model_dump()
        for descriptor in registry.descriptors()
    } == previous_descriptors
    assert expected_error in str(exc_info.value)
    assert expected_path in str(exc_info.value)
    assert "agent registry reload rejected" in caplog.text


def test_registry_reload_accepts_generator_backed_descriptor_fields(
    tmp_path: Path,
) -> None:
    # Arrange
    config_path = tmp_path / "agent_config.py"
    _write_config(config_path, [_descriptor_source("internal_knowledge")])
    registry = _registry_from(config_path)
    config_path.write_text(
        "def one_shot(value):\n"
        "    yield value\n\n"
        "AVAILABLE_AGENTS = [\n"
        "    {\n"
        "        'agent_id': 'internal_knowledge',\n"
        "        'display_name': 'Internal Knowledge Agent',\n"
        "        'capabilities': one_shot('crm notes'),\n"
        "        'input_schema': {'type': 'object'},\n"
        "        'output_schema': {'type': 'object'},\n"
        "        'a2ui_catalogs': ['basic'],\n"
        "        'routing_examples': one_shot('Summarize notes.'),\n"
        "        'execution_mode': 'local_llm',\n"
        "    }\n"
        "]\n",
        encoding="utf-8",
    )

    # Act
    registry.reload()

    # Assert
    descriptor = registry.get("internal_knowledge")
    assert descriptor is not None
    assert descriptor.capabilities == ["crm notes"]
    assert descriptor.routing_examples == ["Summarize notes."]


@pytest.mark.parametrize(
    "container_name",
    ["$defs", "definitions", "dependentSchemas", "patternProperties"],
)
def test_invalid_reload_rejects_secret_like_schema_map_keys(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    container_name: str,
) -> None:
    # Arrange
    from orchestrator_demo.registry.agent_registry import RegistryValidationError

    leaked_key = "sk-live-map-key-should-not-appear"
    config_path = tmp_path / "agent_config.py"
    _write_config(config_path, [_descriptor_source("internal_knowledge")])
    registry = _registry_from(config_path)
    previous_descriptors = {
        descriptor.agent_id: descriptor.model_dump()
        for descriptor in registry.descriptors()
    }
    schema_source = (
        "{"
        "'type': 'object', "
        f"{container_name!r}: {{{leaked_key!r}: {{'type': 'object'}}}},"
        "}"
    )
    config_path.write_text(
        "from orchestrator_demo.contracts import AgentDescriptor\n\n"
        "AVAILABLE_AGENTS = [\n"
        "    AgentDescriptor(\n"
        "        agent_id='internal_knowledge',\n"
        "        display_name='Mutated Internal Knowledge Agent',\n"
        "        capabilities=['crm notes'],\n"
        f"        input_schema={schema_source},\n"
        "        output_schema={'type': 'object'},\n"
        "        a2ui_catalogs=['basic'],\n"
        "        routing_examples=['Summarize notes.'],\n"
        "        execution_mode='local_llm',\n"
        "    )\n"
        "]\n",
        encoding="utf-8",
    )

    # Act / Assert
    with caplog.at_level(logging.ERROR, logger="orchestrator_demo.registry.agent_registry"):
        with pytest.raises(RegistryValidationError) as exc_info:
            registry.reload()

    assert {
        descriptor.agent_id: descriptor.model_dump()
        for descriptor in registry.descriptors()
    } == previous_descriptors
    assert "secret-like schema map key" in str(exc_info.value)
    assert f"input_schema.{container_name}.<redacted>" in str(exc_info.value)
    assert "agent registry reload rejected" in caplog.text
    assert leaked_key not in str(exc_info.value)
    assert leaked_key not in caplog.text


def test_invalid_reload_rejects_secret_like_top_level_mapping_keys_without_leaking_them(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Arrange
    from orchestrator_demo.registry.agent_registry import RegistryValidationError

    leaked_key = "sk-live-top-level-map-key-should-not-appear"
    config_path = tmp_path / "agent_config.py"
    _write_config(config_path, [_descriptor_source("internal_knowledge")])
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
        f"        {leaked_key!r}: 'unused',\n"
        "    }\n"
        "]\n",
        encoding="utf-8",
    )

    # Act / Assert
    with caplog.at_level(logging.ERROR, logger="orchestrator_demo.registry.agent_registry"):
        with pytest.raises(RegistryValidationError) as exc_info:
            registry.reload()

    assert {
        descriptor.agent_id: descriptor.model_dump()
        for descriptor in registry.descriptors()
    } == previous_descriptors
    assert "secret-like mapping key" in str(exc_info.value)
    assert "AVAILABLE_AGENTS[0].<redacted>" in str(exc_info.value)
    assert "agent registry reload rejected" in caplog.text
    assert leaked_key not in str(exc_info.value)
    assert leaked_key not in caplog.text


def test_invalid_reload_rejects_secret_like_schema_property_keys_without_leaking_them(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Arrange
    from orchestrator_demo.registry.agent_registry import RegistryValidationError

    leaked_key = "sk-live-schema-property-key-should-not-appear"
    config_path = tmp_path / "agent_config.py"
    _write_config(config_path, [_descriptor_source("internal_knowledge")])
    registry = _registry_from(config_path)
    previous_descriptors = {
        descriptor.agent_id: descriptor.model_dump()
        for descriptor in registry.descriptors()
    }
    config_path.write_text(
        "from orchestrator_demo.contracts import AgentDescriptor\n\n"
        "AVAILABLE_AGENTS = [\n"
        "    AgentDescriptor(\n"
        "        agent_id='internal_knowledge',\n"
        "        display_name='Mutated Internal Knowledge Agent',\n"
        "        capabilities=['crm notes'],\n"
        "        input_schema={\n"
        "            'type': 'object',\n"
        "            'properties': {\n"
        f"                {leaked_key!r}: {{'type': 'string'}},\n"
        "            },\n"
        "        },\n"
        "        output_schema={'type': 'object'},\n"
        "        a2ui_catalogs=['basic'],\n"
        "        routing_examples=['Summarize notes.'],\n"
        "        execution_mode='local_llm',\n"
        "    )\n"
        "]\n",
        encoding="utf-8",
    )

    # Act / Assert
    with caplog.at_level(logging.ERROR, logger="orchestrator_demo.registry.agent_registry"):
        with pytest.raises(RegistryValidationError) as exc_info:
            registry.reload()

    assert {
        descriptor.agent_id: descriptor.model_dump()
        for descriptor in registry.descriptors()
    } == previous_descriptors
    assert "secret-like schema map key" in str(exc_info.value)
    assert "input_schema.properties.<redacted>" in str(exc_info.value)
    assert "agent registry reload rejected" in caplog.text
    assert leaked_key not in str(exc_info.value)
    assert leaked_key not in caplog.text


@pytest.mark.parametrize(
    "dependent_fields_source",
    [
        pytest.param("['account_id']", id="valid-list"),
        pytest.param("'account_id'", id="malformed-list"),
    ],
)
def test_invalid_reload_rejects_secret_like_dependent_required_keys_without_leaking_them(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    dependent_fields_source: str,
) -> None:
    # Arrange
    from orchestrator_demo.registry.agent_registry import RegistryValidationError

    leaked_key = "sk-live-abcdefghijklmnop"
    config_path = tmp_path / "agent_config.py"
    _write_config(config_path, [_descriptor_source("internal_knowledge")])
    registry = _registry_from(config_path)
    previous_descriptors = {
        descriptor.agent_id: descriptor.model_dump()
        for descriptor in registry.descriptors()
    }
    schema_source = (
        "{"
        "'type': 'object', "
        "'dependentRequired': {"
        f"{leaked_key!r}: {dependent_fields_source}"
        "},"
        "}"
    )
    config_path.write_text(
        "from orchestrator_demo.contracts import AgentDescriptor\n\n"
        "AVAILABLE_AGENTS = [\n"
        "    AgentDescriptor(\n"
        "        agent_id='internal_knowledge',\n"
        "        display_name='Mutated Internal Knowledge Agent',\n"
        "        capabilities=['crm notes'],\n"
        f"        input_schema={schema_source},\n"
        "        output_schema={'type': 'object'},\n"
        "        a2ui_catalogs=['basic'],\n"
        "        routing_examples=['Summarize notes.'],\n"
        "        execution_mode='local_llm',\n"
        "    )\n"
        "]\n",
        encoding="utf-8",
    )

    # Act / Assert
    with caplog.at_level(logging.ERROR, logger="orchestrator_demo.registry.agent_registry"):
        with pytest.raises(RegistryValidationError) as exc_info:
            registry.reload()

    assert {
        descriptor.agent_id: descriptor.model_dump()
        for descriptor in registry.descriptors()
    } == previous_descriptors
    assert "secret-like schema map key" in str(exc_info.value)
    assert "input_schema.dependentRequired.<redacted>" in str(exc_info.value)
    assert "agent registry reload rejected" in caplog.text
    assert leaked_key not in str(exc_info.value)
    assert leaked_key not in caplog.text


@pytest.mark.parametrize("schema_field", ["input_schema", "output_schema"])
def test_registry_rejects_empty_schema_type_arrays(
    tmp_path: Path,
    schema_field: str,
) -> None:
    # Arrange
    from orchestrator_demo.registry.agent_registry import RegistryValidationError

    input_schema = "{'type': 'object'}"
    output_schema = "{'type': 'object'}"
    if schema_field == "input_schema":
        input_schema = "{'type': []}"
    else:
        output_schema = "{'type': []}"

    config_path = tmp_path / "agent_config.py"
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

    assert f"{schema_field}.type" in str(exc_info.value)
    assert "invalid JSON-schema type" in str(exc_info.value)


def test_registry_accepts_non_empty_schema_type_arrays_with_supported_types(
    tmp_path: Path,
) -> None:
    # Arrange
    config_path = tmp_path / "agent_config.py"
    config_path.write_text(
        "from orchestrator_demo.contracts import AgentDescriptor\n\n"
        "AVAILABLE_AGENTS = [\n"
        "    AgentDescriptor(\n"
        "        agent_id='internal_knowledge',\n"
        "        display_name='Internal Knowledge Agent',\n"
        "        capabilities=['crm notes'],\n"
        "        input_schema={'type': ['object', 'null']},\n"
        "        output_schema={'type': ['object']},\n"
        "        a2ui_catalogs=['basic'],\n"
        "        routing_examples=['Summarize notes.'],\n"
        "        execution_mode='local_llm',\n"
        "    )\n"
        "]\n",
        encoding="utf-8",
    )

    # Act
    registry = _registry_from(config_path)

    # Assert
    assert registry.agent_ids() == ["internal_knowledge"]


def test_registry_accepts_ref_and_composition_only_property_schemas(
    tmp_path: Path,
) -> None:
    # Arrange
    config_path = tmp_path / "agent_config.py"
    config_path.write_text(
        "from orchestrator_demo.contracts import AgentDescriptor\n\n"
        "AVAILABLE_AGENTS = [\n"
        "    AgentDescriptor(\n"
        "        agent_id='internal_knowledge',\n"
        "        display_name='Internal Knowledge Agent',\n"
        "        capabilities=['crm notes'],\n"
        "        input_schema={\n"
        "            'type': 'object',\n"
        "            '$defs': {\n"
        "                'CustomerProfile': {\n"
        "                    'type': 'object',\n"
        "                    'properties': {'relationship_id': {'type': 'string'}},\n"
        "                },\n"
        "            },\n"
        "            'properties': {\n"
        "                'profile': {'$ref': '#/$defs/CustomerProfile'},\n"
        "                'notes': {\n"
        "                    'anyOf': [{'type': 'string'}, {'type': 'null'}]\n"
        "                },\n"
        "            },\n"
        "        },\n"
        "        output_schema={\n"
        "            'type': 'object',\n"
        "            'properties': {\n"
        "                'summary': {\n"
        "                    'anyOf': [{'type': 'string'}, {'type': 'null'}]\n"
        "                },\n"
        "            },\n"
        "        },\n"
        "        a2ui_catalogs=['basic'],\n"
        "        routing_examples=['Summarize notes.'],\n"
        "        execution_mode='local_llm',\n"
        "    )\n"
        "]\n",
        encoding="utf-8",
    )

    # Act
    registry = _registry_from(config_path)

    # Assert
    assert registry.agent_ids() == ["internal_knowledge"]


def test_registry_validation_traceback_redacts_pydantic_input_values(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Arrange
    from orchestrator_demo.registry.agent_registry import RegistryValidationError

    leaked_value = "sk-live-mode-should-not-appear"
    config_path = tmp_path / "agent_config.py"
    config_path.write_text(
        "AVAILABLE_AGENTS = [\n"
        "    {\n"
        "        'agent_id': 'internal_knowledge',\n"
        "        'display_name': 'Internal Knowledge Agent',\n"
        "        'capabilities': ['crm notes'],\n"
        "        'input_schema': {'type': 'object'},\n"
        "        'output_schema': {'type': 'object'},\n"
        "        'a2ui_catalogs': ['basic'],\n"
        "        'routing_examples': ['Summarize notes.'],\n"
        f"        'execution_mode': {leaked_value!r},\n"
        "    }\n"
        "]\n",
        encoding="utf-8",
    )

    # Act / Assert
    with caplog.at_level(logging.ERROR, logger="orchestrator_demo.registry.agent_registry"):
        with pytest.raises(RegistryValidationError) as exc_info:
            _registry_from(config_path)

    traceback_text = "".join(
        traceback.format_exception(
            type(exc_info.value),
            exc_info.value,
            exc_info.value.__traceback__,
        )
    )
    assert "execution_mode" in str(exc_info.value)
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__suppress_context__ is True
    assert leaked_value not in str(exc_info.value)
    assert leaked_value not in caplog.text
    assert leaked_value not in traceback_text


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
