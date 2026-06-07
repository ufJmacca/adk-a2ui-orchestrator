"""Agent-facing facade and ADK Dev UI wrapper for the orchestrator service."""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import AsyncGenerator
from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import TYPE_CHECKING, Any

from google.adk.agents import Agent
from google.adk.apps.app import App
from google.adk.models import BaseLlm
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.tools import FunctionTool
from google.adk.tools.tool_context import ToolContext
from google.genai import types

from orchestrator_demo.a2a_support.transport import DataPart
from orchestrator_demo.a2ui_support.adk_a2a_plugin import A2uiA2AProtocolPlugin
from orchestrator_demo.a2ui_support.adk_ui_delivery import (
    deliver_a2ui_parts_to_adk_ui,
    install_a2ui_response_event_delivery,
)
from orchestrator_demo.a2ui_support.secret_safety import (
    safe_path_component,
)
from orchestrator_demo.app.bootstrap_llm import build_litellm_model
from orchestrator_demo.intent.classifier import LiteLlmIntentClassifier
from orchestrator_demo.orchestrator.approval_state import PlanAlreadyFinalError
from orchestrator_demo.orchestrator.artifacts import (
    LATEST_RESULT_ARTIFACT_NAME,
    ArtifactStorageError,
    plan_execution_artifact_name,
    save_response_artifact,
)
from orchestrator_demo.orchestrator.response_payloads import (
    build_error_response,
    build_request_response,
    build_user_action_response,
)
from orchestrator_demo.orchestrator.session_snapshot import SNAPSHOT_SCHEMA_VERSION
from orchestrator_demo.orchestrator.service import (
    OrchestratorRequestResult,
    OrchestratorService,
    OrchestratorUserActionResult,
)


ORCHESTRATOR_SESSION_STATE_KEY = "orchestrator_session"
DETERMINISTIC_MODEL_ENV = "ORCHESTRATOR_DEMO_DETERMINISTIC_MODEL"
ADK_EVAL_MODE_ENV = "ORCHESTRATOR_DEMO_ADK_EVAL_MODE"
_FINAL_APPROVAL_STATUSES = frozenset(
    {"approved", "approved_execution_failed", "rejected"}
)

install_a2ui_response_event_delivery()


class OrchestratorAgent:
    """Small facade matching the demo agent's request and userAction surfaces."""

    def __init__(self, service: OrchestratorService | None = None) -> None:
        self._service = service or OrchestratorService()

    async def handle_request(self, user_input: str) -> OrchestratorRequestResult:
        """Handle a natural-language request through the orchestrator service."""

        return await self._service.handle_user_request(user_input)

    async def handle_user_action(
        self,
        user_action: Any,
    ) -> OrchestratorUserActionResult:
        """Handle a structured A2UI userAction through deterministic routing."""

        return await self._service.handle_user_action(user_action)

    def export_session_snapshot(self) -> dict[str, Any]:
        """Return the JSON-safe orchestrator session snapshot."""

        return self._service.export_session_snapshot()

    def artifact_refs(self) -> dict[str, Any]:
        """Return saved artifact references tracked by the service."""

        return self._service.artifact_refs()

    def record_artifact_refs(self, artifact_refs: Mapping[str, Any]) -> None:
        """Record saved artifact references in service-backed session state."""

        self._service.record_artifact_refs(artifact_refs)

    def restore_session_snapshot(self, snapshot: Mapping[str, Any]) -> None:
        """Restore orchestrator session state from ADK session storage."""

        self._service.restore_session_snapshot(snapshot)

    def reset_session_snapshot(self) -> None:
        """Reset orchestrator session state for a fresh ADK session."""

        self.restore_session_snapshot(_empty_session_snapshot())


class AdkOrchestratorAdapter:
    """Expose orchestrator request, draft edit, and approval operations as ADK tools."""

    def __init__(self, agent: OrchestratorAgent | None = None) -> None:
        self._agent = agent or OrchestratorAgent()
        self._session_lock = asyncio.Lock()
        self._finalized_plan_snapshots_by_id: dict[str, dict[str, Any]] = {}
        self._latest_draft_plan_snapshots_by_id: dict[str, dict[str, Any]] = {}

    def tools(self) -> list[Any]:
        """Return ADK function tools backed by one stateful orchestrator agent."""

        return [
            FunctionTool(self.submit_orchestrator_request),
            FunctionTool(self.add_plan_instruction),
            FunctionTool(self.remove_plan_step),
            FunctionTool(self.replace_plan_agent),
            FunctionTool(self.reorder_plan_steps),
            FunctionTool(self.approve_orchestrator_plan),
            FunctionTool(self.reject_orchestrator_plan),
        ]

    async def submit_orchestrator_request(
        self,
        user_input: str,
        tool_context: ToolContext | None = None,
    ) -> dict[str, Any]:
        """Submit a natural-language RM request to the orchestrator."""

        session_ready = False
        async with self._session_lock:
            try:
                self._restore_session_from_context(tool_context)
                session_ready = True
                result = await self._agent.handle_request(user_input)
                response = build_request_response(result)
                try:
                    await self._save_request_artifacts(tool_context, result, response)
                except ArtifactStorageError as exc:
                    if result.path != "direct":
                        raise
                    response["artifactRefs"] = self._agent.artifact_refs()
                    response["artifactPersistence"] = {
                        "status": "failed",
                        "error": build_error_response(exc)["error"],
                    }
                self._deliver_a2ui_to_adk_ui(
                    tool_context,
                    response,
                    validated_a2ui_parts=result.a2ui_parts,
                )
                self._persist_session_to_context(tool_context)
            except Exception as exc:
                if session_ready:
                    self._persist_session_to_context(
                        tool_context,
                        suppress_errors=True,
                    )
                response = build_error_response(exc)
        self._skip_model_summarization(tool_context)
        return response

    async def add_plan_instruction(
        self,
        plan_id: str,
        approval_surface_id: str,
        step_id: str,
        instruction: str,
        edited_plan_version: int,
        tool_context: ToolContext | None = None,
    ) -> dict[str, Any]:
        """Add a supplemental instruction to one draft plan step."""

        return await self._handle_plan_action(
            "add_instruction",
            approval_surface_id,
            {
                "planId": plan_id,
                "editedPlanVersion": edited_plan_version,
                "stepId": step_id,
                "instruction": instruction,
            },
            tool_context=tool_context,
        )

    async def remove_plan_step(
        self,
        plan_id: str,
        approval_surface_id: str,
        step_id: str,
        edited_plan_version: int,
        tool_context: ToolContext | None = None,
    ) -> dict[str, Any]:
        """Remove one step from a pending draft plan."""

        return await self._handle_plan_action(
            "remove_step",
            approval_surface_id,
            {
                "planId": plan_id,
                "editedPlanVersion": edited_plan_version,
                "stepId": step_id,
            },
            tool_context=tool_context,
        )

    async def replace_plan_agent(
        self,
        plan_id: str,
        approval_surface_id: str,
        step_id: str,
        replacement_agent_id: str,
        edited_plan_version: int,
        tool_context: ToolContext | None = None,
    ) -> dict[str, Any]:
        """Replace the specialist agent assigned to one draft plan step."""

        return await self._handle_plan_action(
            "replace_agent",
            approval_surface_id,
            {
                "planId": plan_id,
                "editedPlanVersion": edited_plan_version,
                "stepId": step_id,
                "replacementAgentId": replacement_agent_id,
            },
            tool_context=tool_context,
        )

    async def reorder_plan_steps(
        self,
        plan_id: str,
        approval_surface_id: str,
        ordered_step_ids: list[str],
        edited_plan_version: int,
        tool_context: ToolContext | None = None,
    ) -> dict[str, Any]:
        """Reorder the current draft plan steps."""

        return await self._handle_plan_action(
            "reorder_steps",
            approval_surface_id,
            {
                "planId": plan_id,
                "editedPlanVersion": edited_plan_version,
                "orderedStepIds": ordered_step_ids,
            },
            tool_context=tool_context,
        )

    async def approve_orchestrator_plan(
        self,
        plan_id: str,
        approval_surface_id: str,
        approved_step_ids: list[str],
        edited_plan_version: int,
        tool_context: ToolContext,
    ) -> dict[str, Any]:
        """Approve a pending A2UI plan and execute the approved workflow."""

        return await self._handle_plan_action(
            "approve_plan",
            approval_surface_id,
            {
                "planId": plan_id,
                "editedPlanVersion": edited_plan_version,
                "approvedStepIds": approved_step_ids,
            },
            tool_context=tool_context,
        )

    async def reject_orchestrator_plan(
        self,
        plan_id: str,
        approval_surface_id: str,
        reason: str,
        *,
        tool_context: ToolContext,
        edited_plan_version: int | None = None,
    ) -> dict[str, Any]:
        """Reject a pending A2UI plan without executing specialists."""

        payload: dict[str, Any] = {
            "planId": plan_id,
            "reason": reason,
        }
        if edited_plan_version is not None:
            payload["editedPlanVersion"] = edited_plan_version

        return await self._handle_plan_action(
            "reject_plan",
            approval_surface_id,
            payload,
            tool_context=tool_context,
        )

    async def _handle_plan_action(
        self,
        action_type: str,
        approval_surface_id: str,
        payload: dict[str, Any],
        *,
        tool_context: ToolContext | None = None,
    ) -> dict[str, Any]:
        session_ready = False
        async with self._session_lock:
            try:
                restored_snapshot = self._restore_session_from_context(tool_context)
                session_ready = True
                if _snapshot_has_finalized_plan(
                    restored_snapshot,
                    payload.get("planId"),
                ):
                    raise PlanAlreadyFinalError("plan is already final")
                result = await self._agent.handle_user_action(
                    {
                        "userAction": {
                            "type": action_type,
                            "surfaceId": approval_surface_id,
                            "payload": payload,
                        }
                    }
                )
                response = build_user_action_response(result)
                if (
                    response.get("status") == "rejected"
                    and "approvalSurfaceId" not in response
                ):
                    response["approvalSurfaceId"] = approval_surface_id
                try:
                    await self._save_user_action_artifacts(
                        tool_context,
                        result,
                        response,
                    )
                except ArtifactStorageError as exc:
                    if result.status != "approved":
                        raise
                    response["artifactRefs"] = self._agent.artifact_refs()
                    response["artifactPersistence"] = {
                        "status": "failed",
                        "error": build_error_response(exc)["error"],
                    }
                self._deliver_a2ui_to_adk_ui(
                    tool_context,
                    response,
                    validated_a2ui_parts=result.a2ui_parts,
                )
                self._persist_session_to_context(tool_context)
            except Exception as exc:
                if session_ready:
                    self._persist_session_to_context(
                        tool_context,
                        suppress_errors=True,
                    )
                response = build_error_response(exc)
        self._skip_model_summarization(tool_context)
        return response

    async def _save_request_artifacts(
        self,
        tool_context: ToolContext | None,
        result: OrchestratorRequestResult,
        response: dict[str, Any],
    ) -> None:
        if result.path != "direct":
            return
        await self._save_json_artifact(
            tool_context,
            filename=LATEST_RESULT_ARTIFACT_NAME,
            response=response,
            document_type="direct_result",
        )

    async def _save_user_action_artifacts(
        self,
        tool_context: ToolContext | None,
        result: OrchestratorUserActionResult,
        response: dict[str, Any],
    ) -> None:
        approval_result = result.approval_result
        if (
            result.status != "approved"
            or approval_result is None
            or approval_result.plan_id is None
        ):
            return

        plan_id = approval_result.plan_id
        await self._save_json_artifact(
            tool_context,
            filename=LATEST_RESULT_ARTIFACT_NAME,
            response=response,
            document_type="approved_result",
            plan_id=plan_id,
        )
        await self._save_json_artifact(
            tool_context,
            filename=plan_execution_artifact_name(plan_id),
            response=response,
            document_type="approved_plan_execution",
            plan_id=plan_id,
        )

    async def _save_json_artifact(
        self,
        tool_context: ToolContext | None,
        *,
        filename: str,
        response: dict[str, Any],
        document_type: str,
        plan_id: str | None = None,
    ) -> None:
        artifact_ref = await save_response_artifact(
            tool_context,
            filename=filename,
            response=response,
            document_type=document_type,
            plan_id=plan_id,
        )
        if artifact_ref is None:
            return

        self._agent.record_artifact_refs({filename: artifact_ref})
        response["artifactRefs"] = self._agent.artifact_refs()

    def _deliver_a2ui_to_adk_ui(
        self,
        tool_context: ToolContext | None,
        response: Mapping[str, Any],
        *,
        validated_a2ui_parts: Sequence[DataPart] | None = None,
    ) -> None:
        deliver_a2ui_parts_to_adk_ui(
            response,
            tool_context,
            validated_a2ui_parts=validated_a2ui_parts,
        )

    def _restore_session_from_context(
        self,
        tool_context: ToolContext | None,
    ) -> Mapping[str, Any] | None:
        if tool_context is None:
            return None

        snapshot = tool_context.state.get(ORCHESTRATOR_SESSION_STATE_KEY)
        if snapshot is None:
            self._agent.reset_session_snapshot()
            return _empty_session_snapshot()
        if not isinstance(snapshot, Mapping):
            raise ValueError("orchestrator_session must be a JSON object")
        restored_snapshot = self._snapshot_preserving_finalized_plans(snapshot)
        restored_snapshot = self._snapshot_preserving_latest_drafts(restored_snapshot)
        self._agent.restore_session_snapshot(restored_snapshot)
        self._remember_finalized_plans(restored_snapshot)
        self._remember_latest_drafts(restored_snapshot)
        return restored_snapshot

    def _persist_session_to_context(
        self,
        tool_context: ToolContext | None,
        *,
        suppress_errors: bool = False,
    ) -> None:
        if tool_context is None:
            return

        try:
            snapshot = self._agent.export_session_snapshot()
            self._remember_finalized_plans(snapshot)
            self._remember_latest_drafts(snapshot)
            tool_context.state[ORCHESTRATOR_SESSION_STATE_KEY] = snapshot
        except Exception:
            if not suppress_errors:
                raise

    def _skip_model_summarization(self, tool_context: ToolContext | None) -> None:
        if tool_context is None:
            return

        actions = getattr(tool_context, "actions", None)
        if actions is None:
            return

        try:
            actions.skip_summarization = not _deterministic_adk_eval_mode_enabled()
        except AttributeError:
            return

    def _snapshot_preserving_finalized_plans(
        self,
        snapshot: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        approval_records = snapshot.get("approvalRecords")
        if not isinstance(approval_records, Mapping):
            return snapshot

        merged_snapshot: dict[str, Any] | None = None
        for plan_id, incoming_record in approval_records.items():
            if not isinstance(plan_id, str) or not isinstance(
                incoming_record,
                Mapping,
            ):
                continue
            if incoming_record.get("status") != "draft":
                continue
            finalized = self._finalized_plan_snapshots_by_id.get(plan_id)
            if finalized is None:
                continue
            finalized_record = finalized.get("approvalRecord")
            if not isinstance(finalized_record, Mapping):
                continue
            if not _approval_records_share_plan_identity(
                incoming_record,
                finalized_record,
                plan_id,
            ):
                continue

            if merged_snapshot is None:
                merged_snapshot = deepcopy(dict(snapshot))
            _merge_finalized_plan_snapshot(
                merged_snapshot,
                plan_id=plan_id,
                finalized=finalized,
            )

        return snapshot if merged_snapshot is None else merged_snapshot

    def _snapshot_preserving_latest_drafts(
        self,
        snapshot: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        approval_records = snapshot.get("approvalRecords")
        if not isinstance(approval_records, Mapping):
            return snapshot

        merged_snapshot: dict[str, Any] | None = None
        for plan_id, incoming_record in approval_records.items():
            if not isinstance(plan_id, str) or not isinstance(
                incoming_record,
                Mapping,
            ):
                continue
            if incoming_record.get("status") != "draft":
                continue
            latest = self._latest_draft_plan_snapshots_by_id.get(plan_id)
            if latest is None:
                continue
            latest_record = latest.get("approvalRecord")
            if not isinstance(latest_record, Mapping):
                continue
            if latest_record.get("status") != "draft":
                continue
            if not _approval_records_share_plan_identity(
                incoming_record,
                latest_record,
                plan_id,
            ):
                continue
            incoming_version = _draft_plan_version_from_record(incoming_record)
            latest_version = _draft_plan_version_from_record(latest_record)
            if (
                incoming_version is None
                or latest_version is None
                or latest_version <= incoming_version
            ):
                continue

            if merged_snapshot is None:
                merged_snapshot = deepcopy(dict(snapshot))
            _merge_draft_plan_snapshot(
                merged_snapshot,
                plan_id=plan_id,
                latest=latest,
            )

        return snapshot if merged_snapshot is None else merged_snapshot

    def _remember_finalized_plans(self, snapshot: Mapping[str, Any]) -> None:
        approval_records = snapshot.get("approvalRecords")
        if not isinstance(approval_records, Mapping):
            return
        request_contexts = snapshot.get("requestContextsByPlanId")
        surface_registry = snapshot.get("surfaceRegistry")
        artifact_refs = snapshot.get("artifactRefs")

        for plan_id, record in approval_records.items():
            if not isinstance(plan_id, str) or not _approval_record_is_final(record):
                continue
            approval_surface_id = _approval_surface_id_for_record(record, plan_id)
            self._latest_draft_plan_snapshots_by_id.pop(plan_id, None)
            self._finalized_plan_snapshots_by_id[plan_id] = {
                "approvalRecord": deepcopy(record),
                "requestContext": (
                    deepcopy(request_contexts.get(plan_id))
                    if isinstance(request_contexts, Mapping)
                    and isinstance(request_contexts.get(plan_id), Mapping)
                    else None
                ),
                "artifactRefs": _artifact_refs_for_plan(artifact_refs, plan_id),
                "approvalSurfaceId": approval_surface_id,
                "surfaceRegistry": (
                    deepcopy(surface_registry)
                    if isinstance(surface_registry, Mapping)
                    else None
                ),
            }

    def _remember_latest_drafts(self, snapshot: Mapping[str, Any]) -> None:
        approval_records = snapshot.get("approvalRecords")
        if not isinstance(approval_records, Mapping):
            return
        request_contexts = snapshot.get("requestContextsByPlanId")
        surface_registry = snapshot.get("surfaceRegistry")

        for plan_id, record in approval_records.items():
            if not isinstance(plan_id, str) or not isinstance(record, Mapping):
                continue
            if record.get("status") != "draft":
                if _approval_record_is_final(record):
                    self._latest_draft_plan_snapshots_by_id.pop(plan_id, None)
                continue

            latest = self._latest_draft_plan_snapshots_by_id.get(plan_id)
            latest_record = (
                latest.get("approvalRecord")
                if isinstance(latest, Mapping)
                else None
            )
            latest_version = (
                _draft_plan_version_from_record(latest_record)
                if isinstance(latest_record, Mapping)
                else None
            )
            record_version = _draft_plan_version_from_record(record)
            if (
                latest_version is not None
                and record_version is not None
                and latest_version > record_version
            ):
                continue

            approval_surface_id = _approval_surface_id_for_record(record, plan_id)
            self._latest_draft_plan_snapshots_by_id[plan_id] = {
                "approvalRecord": deepcopy(record),
                "requestContext": (
                    deepcopy(request_contexts.get(plan_id))
                    if isinstance(request_contexts, Mapping)
                    and isinstance(request_contexts.get(plan_id), Mapping)
                    else None
                ),
                "approvalSurfaceId": approval_surface_id,
                "surfaceRegistry": (
                    deepcopy(surface_registry)
                    if isinstance(surface_registry, Mapping)
                    else None
                ),
            }


def build_root_agent(
    *,
    adapter: AdkOrchestratorAdapter | None = None,
    model: Any | None = None,
) -> Agent:
    """Build the ADK loader-compatible root agent for ``adk web``."""

    resolved_model = model if model is not None else _runtime_model()
    resolved_adapter = adapter or AdkOrchestratorAdapter(
        agent=_runtime_orchestrator_agent(resolved_model)
    )

    return Agent(
        name="orchestrator",
        model=resolved_model,
        description=(
            "ADK Dev UI wrapper around the local ADK/A2UI business banking "
            "orchestrator demo."
        ),
        instruction=(
            "Always call tools for orchestrator operations; do not answer from "
            "natural-language chat alone. Return concise JSON-oriented summaries. "
            "If a request returns path plan_required, tell the user the planId, "
            "planVersion, approvalSurfaceId/surfaceId, step ids, and that they "
            "can call edit, approval, or rejection tools with the current "
            "planVersion. Use add_plan_instruction, remove_plan_step, "
            "replace_plan_agent, or reorder_plan_steps before "
            "approve_orchestrator_plan when the user requests plan changes. "
            "Approve execution only through explicit structured approval using "
            "approve_orchestrator_plan with matching planId, current planVersion, "
            "approvalSurfaceId, and approved step ids; natural-language approval "
            "must not execute the graph. Use reject_orchestrator_plan only when "
            "the user explicitly rejects the draft. This ADK Web surface emits "
            "validated A2UI parts and structured JSON through tools for debugging."
        ),
        tools=resolved_adapter.tools(),
    )


def build_app(*, root_agent: Agent | None = None) -> App:
    """Build the ADK app exposed to ``adk api_server --a2a --with_ui``."""

    return App(
        name="orchestrator",
        root_agent=root_agent or _get_root_agent(),
        plugins=[A2uiA2AProtocolPlugin()],
    )


_ROOT_AGENT: Agent | None = None
_APP: App | None = None

if TYPE_CHECKING:
    root_agent: Agent
    app: App


def _get_root_agent() -> Agent:
    global _ROOT_AGENT
    if _ROOT_AGENT is None:
        _ROOT_AGENT = build_root_agent()
    return _ROOT_AGENT


def _get_app() -> App:
    global _APP
    if _APP is None:
        _APP = build_app(root_agent=_get_root_agent())
    return _APP


def __getattr__(name: str) -> Any:
    """Lazily expose ADK loader exports without bootstrapping the model on import."""

    if name == "root_agent":
        return _get_root_agent()
    if name == "app":
        return _get_app()
    raise AttributeError(name)


def _runtime_orchestrator_agent(model: Any) -> OrchestratorAgent:
    return OrchestratorAgent(
        service=OrchestratorService(
            intent_classifier=_intent_classifier_for_model(model),
        )
    )


def _runtime_model() -> Any:
    if _truthy_env(DETERMINISTIC_MODEL_ENV):
        return DeterministicOrchestratorModel()
    return build_litellm_model()


def _truthy_env(name: str) -> bool:
    return os.environ.get(name, "").strip().casefold() in {"1", "true", "yes", "on"}


def _deterministic_adk_eval_mode_enabled() -> bool:
    return _truthy_env(DETERMINISTIC_MODEL_ENV) and _truthy_env(ADK_EVAL_MODE_ENV)


class DeterministicOrchestratorModel(BaseLlm):
    """Local model used by subprocess A2A integration tests."""

    model: str = "orchestrator-deterministic-test-model"

    async def generate_content_async(
        self,
        llm_request: LlmRequest,
        stream: bool = False,
    ) -> AsyncGenerator[LlmResponse, None]:
        if _current_turn_function_response(llm_request) is not None:
            yield _text_response(
                "Structured orchestrator tool response returned to the A2A client."
            )
            return

        if "submit_orchestrator_request" in llm_request.tools_dict:
            user_input = _latest_user_text(llm_request)
            function_call = types.Part.from_function_call(
                name="submit_orchestrator_request",
                args={"user_input": user_input},
            )
            if function_call.function_call is not None:
                function_call.function_call.id = "call_submit_orchestrator_request"
            yield LlmResponse(
                content=types.Content(role="model", parts=[function_call])
            )
            return

        yield _text_response(json.dumps(_deterministic_intent_assessment(llm_request)))


def _text_response(text: str) -> LlmResponse:
    return LlmResponse(
        content=types.Content(role="model", parts=[types.Part.from_text(text=text)])
    )


def _current_turn_function_response(llm_request: LlmRequest) -> Any:
    if not llm_request.contents:
        return None
    content = llm_request.contents[-1]
    for part in reversed(content.parts or []):
        if part.function_response is not None:
            return part.function_response.response
    return None


def _latest_user_text(llm_request: LlmRequest) -> str:
    for content in reversed(llm_request.contents):
        if content.role != "user":
            continue
        text = _content_text(content)
        if text:
            return text
    for content in reversed(llm_request.contents):
        text = _content_text(content)
        if text:
            return text
    return ""


def _content_text(content: types.Content) -> str:
    return "\n".join(
        part.text for part in content.parts or [] if isinstance(part.text, str)
    )


def _deterministic_intent_assessment(llm_request: LlmRequest) -> dict[str, Any]:
    text = _classifier_user_request_text(_latest_user_text(llm_request)).casefold()
    if "internal notes" in text or "crm" in text:
        return {
            "intents": ["internal_knowledge"],
            "confidence": 0.94,
            "complexity": "simple",
            "required_agents": ["internal_knowledge"],
            "rationale": "Deterministic test classifier selected internal knowledge.",
        }
    if "meeting" in text or "prepare" in text:
        return {
            "intents": [
                "meeting_prep",
                "relationship_summary",
                "internal_knowledge",
                "industry_research",
            ],
            "confidence": 0.93,
            "complexity": "complex",
            "required_agents": [
                "relationship_summary",
                "internal_knowledge",
                "industry_research",
                "synthesis",
            ],
            "rationale": "Deterministic test classifier selected meeting prep plan.",
        }
    if (
        "prospect" in text
        or "risks" in text
        or "opportunities" in text
        or "talking points" in text
    ):
        return {
            "intents": [
                "prospect_research",
                "industry_research",
                "product_opportunity",
                "credit_risk",
            ],
            "confidence": 0.92,
            "complexity": "complex",
            "required_agents": [
                "web_search",
                "industry_research",
                "product_opportunity",
                "credit_risk",
                "synthesis",
            ],
            "rationale": "Deterministic test classifier selected prospect research plan.",
        }
    return {
        "intents": ["unknown"],
        "confidence": 0.42,
        "complexity": "complex",
        "required_agents": ["data_quality"],
        "rationale": (
            "Deterministic test classifier found an ambiguous request with no "
            "safe single owner agent."
        ),
    }


def _classifier_user_request_text(text: str) -> str:
    if "User request:" not in text:
        return text
    user_request = text.split("User request:", maxsplit=1)[1]
    for delimiter in ("\nSLM suggestion:", "\nAvailable agents:"):
        user_request = user_request.split(delimiter, maxsplit=1)[0]
    return user_request.strip()


def _intent_classifier_for_model(model: Any) -> LiteLlmIntentClassifier:
    if isinstance(model, str):
        return LiteLlmIntentClassifier(model_name=model)
    return LiteLlmIntentClassifier(model=model)


def _empty_session_snapshot() -> dict[str, Any]:
    return {
        "schemaVersion": SNAPSHOT_SCHEMA_VERSION,
        "approvalRecords": {},
        "requestContextsByPlanId": {},
        "surfaceRegistry": {
            "ownersBySurfaceId": {},
            "componentsBySurfaceId": {},
        },
        "artifactRefs": {},
    }


def _snapshot_has_finalized_plan(
    snapshot: Mapping[str, Any] | None,
    plan_id: Any,
) -> bool:
    if not isinstance(plan_id, str) or not isinstance(snapshot, Mapping):
        return False
    approval_records = snapshot.get("approvalRecords")
    if not isinstance(approval_records, Mapping):
        return False
    return _approval_record_is_final(approval_records.get(plan_id))


def _approval_record_is_final(record: Any) -> bool:
    return (
        isinstance(record, Mapping)
        and record.get("status") in _FINAL_APPROVAL_STATUSES
    )


def _approval_records_share_plan_identity(
    incoming_record: Mapping[str, Any],
    finalized_record: Mapping[str, Any],
    plan_id: str,
) -> bool:
    incoming_draft = _draft_plan_from_record(incoming_record)
    finalized_draft = _draft_plan_from_record(finalized_record)
    if not isinstance(incoming_draft, Mapping) or not isinstance(
        finalized_draft,
        Mapping,
    ):
        return False

    incoming_plan_id = _plan_field(incoming_draft, "plan_id", "planId")
    finalized_plan_id = _plan_field(finalized_draft, "plan_id", "planId")
    if incoming_plan_id is not None and incoming_plan_id != plan_id:
        return False
    if finalized_plan_id is not None and finalized_plan_id != plan_id:
        return False

    incoming_surface_id = _approval_surface_id_for_record(incoming_record, plan_id)
    finalized_surface_id = _approval_surface_id_for_record(finalized_record, plan_id)
    if incoming_surface_id != finalized_surface_id:
        return False

    incoming_objective = _plan_field(incoming_draft, "objective")
    finalized_objective = _plan_field(finalized_draft, "objective")
    return (
        not isinstance(incoming_objective, str)
        or not isinstance(finalized_objective, str)
        or incoming_objective == finalized_objective
    )


def _merge_finalized_plan_snapshot(
    snapshot: dict[str, Any],
    *,
    plan_id: str,
    finalized: Mapping[str, Any],
) -> None:
    approval_records = snapshot.get("approvalRecords")
    finalized_record = finalized.get("approvalRecord")
    if isinstance(approval_records, dict) and isinstance(finalized_record, Mapping):
        approval_records[plan_id] = deepcopy(finalized_record)

    request_contexts = snapshot.get("requestContextsByPlanId")
    request_context = finalized.get("requestContext")
    if isinstance(request_contexts, dict) and isinstance(request_context, Mapping):
        request_contexts[plan_id] = deepcopy(request_context)

    artifact_refs = snapshot.get("artifactRefs")
    finalized_artifact_refs = finalized.get("artifactRefs")
    if isinstance(artifact_refs, dict) and isinstance(
        finalized_artifact_refs,
        Mapping,
    ):
        artifact_refs.update(deepcopy(finalized_artifact_refs))

    _merge_finalized_approval_surface(snapshot, finalized=finalized)


def _merge_draft_plan_snapshot(
    snapshot: dict[str, Any],
    *,
    plan_id: str,
    latest: Mapping[str, Any],
) -> None:
    approval_records = snapshot.get("approvalRecords")
    latest_record = latest.get("approvalRecord")
    if isinstance(approval_records, dict) and isinstance(latest_record, Mapping):
        approval_records[plan_id] = deepcopy(latest_record)

    request_contexts = snapshot.get("requestContextsByPlanId")
    request_context = latest.get("requestContext")
    if isinstance(request_contexts, dict) and isinstance(request_context, Mapping):
        request_contexts[plan_id] = deepcopy(request_context)

    _merge_draft_approval_surface(snapshot, latest=latest)


def _merge_finalized_approval_surface(
    snapshot: dict[str, Any],
    *,
    finalized: Mapping[str, Any],
) -> None:
    approval_surface_id = finalized.get("approvalSurfaceId")
    if not isinstance(approval_surface_id, str):
        return

    surface_registry = snapshot.get("surfaceRegistry")
    if not isinstance(surface_registry, dict):
        return
    finalized_registry = finalized.get("surfaceRegistry")

    for registry_key in ("ownersBySurfaceId", "componentsBySurfaceId"):
        target_entries = surface_registry.get(registry_key)
        if not isinstance(target_entries, dict):
            continue
        finalized_entries = (
            finalized_registry.get(registry_key)
            if isinstance(finalized_registry, Mapping)
            else None
        )
        if (
            isinstance(finalized_entries, Mapping)
            and approval_surface_id in finalized_entries
        ):
            target_entries[approval_surface_id] = deepcopy(
                finalized_entries[approval_surface_id]
            )
        else:
            target_entries.pop(approval_surface_id, None)


def _merge_draft_approval_surface(
    snapshot: dict[str, Any],
    *,
    latest: Mapping[str, Any],
) -> None:
    approval_surface_id = latest.get("approvalSurfaceId")
    if not isinstance(approval_surface_id, str):
        return

    surface_registry = snapshot.get("surfaceRegistry")
    if not isinstance(surface_registry, dict):
        return
    latest_registry = latest.get("surfaceRegistry")
    if not isinstance(latest_registry, Mapping):
        return

    for registry_key in ("ownersBySurfaceId", "componentsBySurfaceId"):
        target_entries = surface_registry.get(registry_key)
        latest_entries = latest_registry.get(registry_key)
        if (
            isinstance(target_entries, dict)
            and isinstance(latest_entries, Mapping)
            and approval_surface_id in latest_entries
        ):
            target_entries[approval_surface_id] = deepcopy(
                latest_entries[approval_surface_id]
            )


def _artifact_refs_for_plan(
    artifact_refs: Any,
    plan_id: str,
) -> dict[str, Any]:
    if not isinstance(artifact_refs, Mapping):
        return {}
    return {
        str(filename): deepcopy(artifact_ref)
        for filename, artifact_ref in artifact_refs.items()
        if _artifact_ref_belongs_to_plan(filename, artifact_ref, plan_id)
    }


def _artifact_ref_belongs_to_plan(
    filename: Any,
    artifact_ref: Any,
    plan_id: str,
) -> bool:
    if isinstance(artifact_ref, Mapping):
        ref_plan_id = artifact_ref.get("planId")
        if ref_plan_id == plan_id or ref_plan_id == safe_path_component(plan_id):
            return True

    return (
        isinstance(filename, str)
        and filename == plan_execution_artifact_name(plan_id)
    )


def _approval_surface_id_for_record(record: Any, plan_id: str) -> str:
    draft_plan = _draft_plan_from_record(record)
    if isinstance(draft_plan, Mapping):
        surface_id = _plan_field(
            draft_plan,
            "approval_surface_id",
            "approvalSurfaceId",
        )
        if isinstance(surface_id, str) and surface_id:
            return surface_id
    return f"surface_{plan_id}"


def _draft_plan_from_record(record: Any) -> Any:
    if not isinstance(record, Mapping):
        return None
    return record.get("draftPlan")


def _draft_plan_version_from_record(record: Mapping[str, Any]) -> int | None:
    draft_plan = _draft_plan_from_record(record)
    if not isinstance(draft_plan, Mapping):
        return None
    version = _plan_field(draft_plan, "plan_version", "planVersion")
    if isinstance(version, int) and not isinstance(version, bool):
        return version
    return None


def _plan_field(plan: Mapping[str, Any], *field_names: str) -> Any:
    for field_name in field_names:
        value = plan.get(field_name)
        if value is not None:
            return value
    return None


__all__ = [
    "ADK_EVAL_MODE_ENV",
    "AdkOrchestratorAdapter",
    "DETERMINISTIC_MODEL_ENV",
    "OrchestratorAgent",
    "app",
    "build_app",
    "build_root_agent",
    "root_agent",
]
