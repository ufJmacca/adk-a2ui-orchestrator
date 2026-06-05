import json
from pathlib import Path

from a2a.types import AgentCard


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CARD_PATH = REPOSITORY_ROOT / "orchestrator_demo" / "orchestrator" / "agent.json"
A2UI_MIME_TYPE = "application/json+a2ui"


def test_orchestrator_agent_card_validates_with_required_discovery_contract() -> None:
    # Arrange
    raw_card = json.loads(CARD_PATH.read_text(encoding="utf-8"))

    # Act
    card = AgentCard(**raw_card)

    # Assert
    assert CARD_PATH.is_file()
    assert card.name == "orchestrator"
    assert card.description == (
        "Business banking relationship-manager orchestrator for request routing, "
        "draft plan review, plan approval, and approved graph execution."
    )
    assert card.version == "0.1.0"
    assert card.url == "http://127.0.0.1:8000/a2a/orchestrator"
    assert card.default_input_modes == ["text/plain"]
    assert card.default_output_modes == [
        "application/json",
        A2UI_MIME_TYPE,
        "text/plain",
    ]
    assert card.capabilities.streaming is True
    assert card.capabilities.state_transition_history is True


def test_orchestrator_agent_card_skills_cover_required_a2a_journeys() -> None:
    # Arrange
    card = AgentCard(**json.loads(CARD_PATH.read_text(encoding="utf-8")))

    # Act
    skills_by_id = {skill.id: skill for skill in card.skills}

    # Assert
    assert set(skills_by_id) == {
        "submit_orchestrator_request",
        "edit_orchestrator_plan",
        "approve_orchestrator_plan",
        "reject_orchestrator_plan",
        "review_orchestrator_a2ui_plan",
    }
    assert "submission" in skills_by_id["submit_orchestrator_request"].tags
    assert "draft-editing" in skills_by_id["edit_orchestrator_plan"].tags
    assert "approval" in skills_by_id["approve_orchestrator_plan"].tags
    assert "rejection" in skills_by_id["reject_orchestrator_plan"].tags
    assert {
        "a2ui",
        "plan-review",
    } <= set(skills_by_id["review_orchestrator_a2ui_plan"].tags)

    for skill in skills_by_id.values():
        assert skill.output_modes == ["application/json", A2UI_MIME_TYPE, "text/plain"]
