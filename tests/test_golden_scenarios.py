from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from orchestrator_demo.orchestrator.service import OrchestratorService


GOLDEN_SCENARIOS_PATH = Path(__file__).with_name("golden_scenarios.json")


def _golden_cases() -> list[dict[str, Any]]:
    return json.loads(GOLDEN_SCENARIOS_PATH.read_text())


@pytest.mark.asyncio
@pytest.mark.parametrize("case", _golden_cases(), ids=lambda case: case["name"])
async def test_golden_scenarios_flow_through_orchestrator_service(
    case: dict[str, Any],
) -> None:
    # Arrange
    service = OrchestratorService()

    # Act
    result = await service.handle_user_request(case["input"])

    # Assert
    assert result.path == case["expected_path"]
    if case["expected_path"] == "direct":
        assert result.approval_plan is None
        assert result.a2ui_parts == ()
        assert [response.agent_id for response in result.specialist_responses] == [
            case["expected_agent"]
        ]
        assert service.specialist_call_counts() == {case["expected_agent"]: 1}
    else:
        assert result.approval_plan is not None
        assert result.approval_plan.selected_agents == case["expected_agents"]
        assert [step.agent_id for step in result.approval_plan.steps] == (
            case["expected_agents"]
        )
        assert result.specialist_responses == ()
        assert service.specialist_call_counts() == {}
        assert len(result.a2ui_parts) == 2
