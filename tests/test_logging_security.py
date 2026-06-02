from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pytest

from orchestrator_demo.orchestrator.service import OrchestratorService


AUDIT_LOGGER_NAME = "orchestrator_demo.audit"
SECRET_VALUE = "sk-or-v1-audit-secret-should-not-appear"


def _audit_records(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    return [
        record
        for record in caplog.records
        if getattr(record, "audit_event", None) is not None
    ]


def _audit_payloads_by_event(
    caplog: pytest.LogCaptureFixture,
) -> dict[str, list[dict[str, Any]]]:
    payloads: dict[str, list[dict[str, Any]]] = {}
    for record in _audit_records(caplog):
        payload = getattr(record, "event_payload")
        assert isinstance(payload, dict)
        payloads.setdefault(str(record.audit_event), []).append(payload)
    return payloads


def _descriptor_source(agent_id: str) -> str:
    display_name = agent_id.replace("_", " ").title()
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


def _approve_event(plan_id: str, surface_id: str, step_ids: list[str]) -> dict[str, Any]:
    return {
        "userAction": {
            "type": "approve_plan",
            "surfaceId": surface_id,
            "payload": {
                "planId": plan_id,
                "editedPlanVersion": 1,
                "approvedStepIds": step_ids,
            },
        }
    }


def _reject_event(
    plan_id: str,
    surface_id: str,
    *,
    plan_version: int,
) -> dict[str, Any]:
    return {
        "userAction": {
            "type": "reject_plan",
            "surfaceId": surface_id,
            "payload": {
                "planId": plan_id,
                "editedPlanVersion": plan_version,
                "reason": "Too broad; focus on credit risk only.",
            },
        }
    }


def _add_instruction_event(
    plan_id: str,
    surface_id: str,
    *,
    step_id: str,
) -> dict[str, Any]:
    return {
        "userAction": {
            "type": "add_instruction",
            "surfaceId": surface_id,
            "payload": {
                "planId": plan_id,
                "editedPlanVersion": 1,
                "stepId": step_id,
                "instruction": "Prioritize covenant follow-ups.",
            },
        }
    }


@pytest.mark.asyncio
async def test_audit_logs_cover_routing_plan_approval_graph_and_ui_event(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Arrange
    service = OrchestratorService()

    # Act
    with caplog.at_level(logging.INFO, logger=AUDIT_LOGGER_NAME):
        proposed = await service.handle_user_request(
            "Prepare me for tomorrow's meeting with ABC Manufacturing."
        )
        assert proposed.approval_plan is not None
        await service.handle_user_action(
            _approve_event(
                proposed.approval_plan.plan_id,
                proposed.approval_plan.approval_surface_id or "",
                [step.step_id for step in proposed.approval_plan.steps],
            )
        )

        editable = await service.handle_user_request(
            "Research this prospect and give me risks, opportunities, and talking points."
        )
        assert editable.approval_plan is not None
        first_step_id = editable.approval_plan.steps[0].step_id
        edited = await service.handle_user_action(
            _add_instruction_event(
                editable.approval_plan.plan_id,
                editable.approval_plan.approval_surface_id or "",
                step_id=first_step_id,
            )
        )
        await service.handle_user_action(
            _reject_event(
                editable.approval_plan.plan_id,
                editable.approval_plan.approval_surface_id or "",
                plan_version=edited.approval_result.plan_version
                if edited.approval_result is not None
                else 2,
            )
        )

        direct = await service.handle_user_request(
            "What product opportunities should I consider for a cafe business?"
        )
        surface_id = direct.specialist_responses[0].surface_id
        assert surface_id is not None
        await service.handle_user_action(
            {
                "userAction": {
                    "type": "specialist_action",
                    "surfaceId": surface_id,
                    "payload": {"action": "show_more_detail"},
                }
            }
        )

    # Assert
    payloads = _audit_payloads_by_event(caplog)
    assert set(payloads) >= {
        "slm_suggestion",
        "llm_assessment",
        "merge_decision",
        "route_decision",
        "plan_proposed",
        "approval_approved",
        "approval_edited",
        "approval_rejected",
        "graph_created",
        "graph_execution_started",
        "graph_execution_completed",
        "ui_event_routed",
    }
    assert payloads["route_decision"][0]["path"] == "plan_required"
    assert payloads["plan_proposed"][0]["selected_agent_ids"] == [
        "relationship_summary",
        "internal_knowledge",
        "industry_research",
        "synthesis",
    ]
    assert payloads["approval_approved"][0]["graph_created"] is True
    assert payloads["approval_edited"][0]["status"] == "draft_updated"
    assert payloads["approval_rejected"][0]["status"] == "rejected"
    assert payloads["graph_created"][0]["graph_id"].startswith("graph_")
    assert payloads["graph_execution_completed"][0]["response_count"] == 4
    assert payloads["ui_event_routed"][-1]["status"] == "forwarded"
    assert payloads["ui_event_routed"][-1]["owner_id"] == "product_opportunity"
    assert payloads["ui_event_routed"][-1]["owner_inference_attempted"] is False


def test_registry_reload_audit_log_is_structured_and_redacted(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Arrange
    from orchestrator_demo.registry.agent_registry import AgentRegistry

    config_path = tmp_path / "agent_config.py"
    _write_config(
        config_path,
        [
            _descriptor_source("internal_knowledge"),
            _descriptor_source("synthesis"),
        ],
    )
    registry = AgentRegistry.from_config_path(config_path)
    _write_config(
        config_path,
        [
            _descriptor_source("internal_knowledge"),
            _descriptor_source("industry_research"),
            _descriptor_source("synthesis"),
        ],
    )

    # Act
    with caplog.at_level(logging.INFO, logger=AUDIT_LOGGER_NAME):
        registry.reload()

    # Assert
    payloads = _audit_payloads_by_event(caplog)
    assert payloads["registry_reload"][-1] == {
        "source": str(config_path),
        "added_agent_ids": ["industry_research"],
        "removed_agent_ids": [],
        "total_agents": 3,
    }


def test_rejected_registry_descriptor_audit_log_redacts_secret_like_fields(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Arrange
    from orchestrator_demo.registry.agent_registry import (
        AgentRegistry,
        RegistryValidationError,
    )

    config_path = tmp_path / "agent_config.py"
    config_path.write_text(
        "from orchestrator_demo.contracts import AgentDescriptor\n\n"
        "AVAILABLE_AGENTS = [\n"
        "    AgentDescriptor(\n"
        "        agent_id='internal_knowledge',\n"
        "        display_name='Internal Knowledge',\n"
        "        capabilities=['business banking support'],\n"
        "        input_schema={'type': 'object', 'properties': {\n"
        f"            'apiToken': {{'type': 'string', 'const': {SECRET_VALUE!r}}}\n"
        "        }},\n"
        "        output_schema={'type': 'object'},\n"
        "        a2ui_catalogs=['basic'],\n"
        "        routing_examples=['Handle an internal knowledge request.'],\n"
        "        execution_mode='local_llm',\n"
        "    )\n"
        "]\n",
        encoding="utf-8",
    )

    # Act
    with caplog.at_level(logging.INFO, logger=AUDIT_LOGGER_NAME):
        with pytest.raises(RegistryValidationError):
            AgentRegistry.from_config_path(config_path)

    # Assert
    payloads = _audit_payloads_by_event(caplog)
    assert payloads["registry_reload_rejected"][-1]["source"] == str(config_path)
    rendered_payload = repr(payloads["registry_reload_rejected"][-1])
    assert SECRET_VALUE not in caplog.text
    assert SECRET_VALUE not in rendered_payload
    assert "apiToken" not in caplog.text
    assert "apiToken" not in rendered_payload
    assert "<redacted-secret>" in rendered_payload or "<redacted-key>" in rendered_payload
