"""ADK Dev UI delivery helpers for validated A2UI parts."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Any

from a2a import types as a2a_types
from google.adk.a2a.converters import part_converter
from google.adk.events.event import Event
from google.adk.events.event_actions import EventActions
from google.adk.events.ui_widget import UiWidget
from google.genai import types as genai_types

from orchestrator_demo.a2a_support.transport import DataPart
from orchestrator_demo.a2ui_support.schema_manager import (
    CREATE_SURFACE_MESSAGE,
    DELETE_SURFACE_MESSAGE,
    UPDATE_COMPONENTS_MESSAGE,
    UPDATE_DATA_MODEL_MESSAGE,
)
from orchestrator_demo.a2ui_support.secret_safety import redact_secret_like_values
from orchestrator_demo.a2ui_support.validation import validate_outbound_a2ui
from orchestrator_demo.contracts import PLAN_USER_ACTION_TYPES


class A2UIWidgetDeliveryError(ValueError):
    """Raised when A2UI cannot be safely emitted through ADK UI delivery."""


def deliver_a2ui_parts_to_adk_ui(
    response: Mapping[str, Any],
    tool_context: Any,
    *,
    validated_a2ui_parts: Sequence[DataPart] | None = None,
) -> None:
    """Validate response ``a2uiParts`` and render an ADK UI widget when available."""

    if tool_context is None:
        return

    validated_parts = _validated_a2ui_part_payloads_for_response(
        response,
        validated_a2ui_parts=validated_a2ui_parts,
    )
    if not validated_parts:
        return

    _remember_validated_a2ui_part_payloads(tool_context, response, validated_parts)

    render_ui_widget = getattr(tool_context, "render_ui_widget", None)
    if not callable(render_ui_widget):
        return

    render_ui_widget(
        UiWidget(
            id=_widget_id(response, validated_parts),
            provider="a2ui",
            payload={"parts": validated_parts},
        )
    )


def adk_content_parts_for_a2ui_response(
    response: Mapping[str, Any],
    *,
    tool_context: Any = None,
    validated_a2ui_parts: Sequence[DataPart] | None = None,
) -> list[genai_types.Part]:
    """Return A2A-visible ADK parts for validated standard response A2UI."""

    return [
        _adk_content_part_for_a2ui(part)
        for part in _validated_a2ui_part_payloads_for_response(
            response,
            validated_a2ui_parts=validated_a2ui_parts,
            tool_context=tool_context,
        )
    ]


def adk_dev_ui_content_parts_for_a2ui_response(
    response: Mapping[str, Any],
    *,
    tool_context: Any = None,
    validated_a2ui_parts: Sequence[DataPart] | None = None,
) -> list[genai_types.Part]:
    """Return ADK Dev UI renderer parts for validated response A2UI."""

    return [
        _adk_content_part_for_a2ui(part)
        for part in _dev_ui_a2ui_part_payloads(
            _validated_a2ui_part_payloads_for_response(
                response,
                validated_a2ui_parts=validated_a2ui_parts,
                tool_context=tool_context,
            ),
            known_component_ids_by_surface_id=(
                _known_component_ids_by_surface_id(tool_context)
            ),
        )
    ]


_VALIDATED_A2UI_PARTS_BY_RESPONSE_ID_ATTR = (
    "_orchestrator_demo_validated_a2ui_parts_by_response_id"
)
_ORCHESTRATOR_SESSION_STATE_KEY = "orchestrator_session"


def _validated_a2ui_part_payloads_for_response(
    response: Mapping[str, Any],
    *,
    validated_a2ui_parts: Sequence[DataPart] | None = None,
    tool_context: Any = None,
) -> list[dict[str, Any]]:
    if validated_a2ui_parts is not None:
        return _trusted_part_payloads(validated_a2ui_parts)

    parts = response.get("a2uiParts")
    if parts is None or parts == []:
        _discard_cached_validated_a2ui_part_payloads(tool_context, response)
        return []
    if not isinstance(parts, Sequence) or isinstance(parts, str | bytes | bytearray):
        _discard_cached_validated_a2ui_part_payloads(tool_context, response)
        raise A2UIWidgetDeliveryError("a2uiParts must be a list of A2UI DataParts")

    cached_parts = _consume_cached_validated_a2ui_part_payloads(
        tool_context,
        response,
    )
    current_parts = _current_a2ui_part_payloads_for_cache_match(parts)
    if cached_parts is not None and cached_parts == current_parts:
        return cached_parts

    return _validated_part_payloads(parts)


def _trusted_part_payloads(parts: Sequence[DataPart]) -> list[dict[str, Any]]:
    return [part.model_dump(by_alias=True, mode="json") for part in parts]


def _remember_validated_a2ui_part_payloads(
    tool_context: Any,
    response: Mapping[str, Any],
    parts: Sequence[Mapping[str, Any]],
) -> None:
    try:
        cached_parts_by_response_id = getattr(
            tool_context,
            _VALIDATED_A2UI_PARTS_BY_RESPONSE_ID_ATTR,
            None,
        )
        if not isinstance(cached_parts_by_response_id, dict):
            cached_parts_by_response_id = {}
            setattr(
                tool_context,
                _VALIDATED_A2UI_PARTS_BY_RESPONSE_ID_ATTR,
                cached_parts_by_response_id,
            )
        cached_parts_by_response_id[id(response)] = [
            deepcopy(dict(part)) for part in parts
        ]
    except (AttributeError, TypeError):
        return


def _discard_cached_validated_a2ui_part_payloads(
    tool_context: Any,
    response: Mapping[str, Any],
) -> None:
    cached_parts_by_response_id = getattr(
        tool_context,
        _VALIDATED_A2UI_PARTS_BY_RESPONSE_ID_ATTR,
        None,
    )
    if isinstance(cached_parts_by_response_id, dict):
        cached_parts_by_response_id.pop(id(response), None)


def _consume_cached_validated_a2ui_part_payloads(
    tool_context: Any,
    response: Mapping[str, Any],
) -> list[dict[str, Any]] | None:
    cached_parts_by_response_id = getattr(
        tool_context,
        _VALIDATED_A2UI_PARTS_BY_RESPONSE_ID_ATTR,
        None,
    )
    if not isinstance(cached_parts_by_response_id, dict):
        return None

    cached_parts = cached_parts_by_response_id.pop(id(response), None)
    if not isinstance(cached_parts, Sequence) or isinstance(
        cached_parts,
        str | bytes | bytearray,
    ):
        return None

    part_payloads: list[dict[str, Any]] = []
    for part in cached_parts:
        if not isinstance(part, Mapping):
            return None
        part_payloads.append(deepcopy(dict(part)))
    return part_payloads


def _current_a2ui_part_payloads_for_cache_match(
    parts: Sequence[Any],
) -> list[dict[str, Any]] | None:
    part_payloads: list[dict[str, Any]] = []
    for part in parts:
        try:
            data_part = DataPart.model_validate(part)
        except ValueError:
            return None
        part_payloads.append(data_part.model_dump(by_alias=True, mode="json"))
    return part_payloads


def _known_component_ids_by_surface_id(tool_context: Any) -> dict[str, set[str]]:
    state = getattr(tool_context, "state", None)
    if not isinstance(state, Mapping):
        return {}

    snapshot = state.get(_ORCHESTRATOR_SESSION_STATE_KEY)
    if not isinstance(snapshot, Mapping):
        return {}

    surface_registry = snapshot.get("surfaceRegistry")
    if not isinstance(surface_registry, Mapping):
        return {}

    components_by_surface_id = surface_registry.get("componentsBySurfaceId")
    if not isinstance(components_by_surface_id, Mapping):
        return {}

    known_component_ids: dict[str, set[str]] = {}
    for surface_id, components in components_by_surface_id.items():
        if not isinstance(surface_id, str) or not isinstance(components, Mapping):
            continue
        component_ids = {
            component_id
            for component_id in components
            if isinstance(component_id, str) and component_id
        }
        if component_ids:
            known_component_ids[surface_id] = component_ids
    return known_component_ids


_ADK_A2UI_RESPONSE_EVENT_DELIVERY_ATTR = "_orchestrator_demo_a2ui_delivery_installed"
_ADK_A2UI_MODEL_HISTORY_FILTER_ATTR = (
    "_orchestrator_demo_a2ui_model_history_filter_installed"
)
_ADK_A2UI_PARALLEL_MERGE_ATTR = "_orchestrator_demo_a2ui_parallel_merge_installed"
_ADK_A2UI_RESPONSE_EVENT_SPLIT_ATTR = (
    "_orchestrator_demo_a2ui_response_event_split_installed"
)
_ADK_A2UI_LIVE_RESPONSE_EVENT_SPLIT_ATTR = (
    "_orchestrator_demo_a2ui_live_response_event_split_installed"
)
_ADK_A2UI_A2A_STANDARD_EXPORT_ATTR = (
    "_orchestrator_demo_a2ui_a2a_standard_export_installed"
)

_DEV_UI_COMPONENT_PRIMITIVE_FIELDS: dict[str, frozenset[str]] = {
    "AudioPlayer": frozenset({"description", "url"}),
    "CheckBox": frozenset({"label", "value"}),
    "ChoicePicker": frozenset({"filterable", "label", "value"}),
    "DateTimeInput": frozenset({"label", "max", "min", "value"}),
    "Icon": frozenset({"name"}),
    "Image": frozenset({"description", "url"}),
    "Slider": frozenset({"label", "value"}),
    "Text": frozenset({"text"}),
    "TextField": frozenset({"label", "value"}),
    "Video": frozenset({"url"}),
}
_DEV_UI_COMPONENT_NAME_ALIASES: dict[str, str] = {
    "ChoicePicker": "MultipleChoice",
}


def install_a2ui_response_event_delivery() -> None:
    """Wire validated A2UI parts into ADK Dev UI events and A2A export."""

    _install_a2ui_model_history_filter()
    _install_a2ui_parallel_response_merge()
    _install_a2ui_response_event_split()
    _install_a2ui_standard_a2a_export()

    from google.adk.flows.llm_flows import functions as adk_functions

    build_response_event = getattr(adk_functions, "__build_response_event")
    if getattr(build_response_event, _ADK_A2UI_RESPONSE_EVENT_DELIVERY_ATTR, False):
        return

    def _build_response_event_with_a2ui(
        tool: Any,
        function_result: dict[str, object],
        tool_context: Any,
        invocation_context: Any,
    ) -> Any:
        event = build_response_event(
            tool,
            function_result,
            tool_context,
            invocation_context,
        )
        if not isinstance(function_result, Mapping):
            return event

        a2ui_parts = adk_dev_ui_content_parts_for_a2ui_response(
            function_result,
            tool_context=tool_context,
        )
        if not a2ui_parts:
            return event

        if event.content is None:
            event.content = genai_types.Content(role="user", parts=a2ui_parts)
            _mark_event_as_a2a_response(event)
            return event

        event.content.parts = [
            *(event.content.parts or []),
            *a2ui_parts,
        ]
        _mark_event_as_a2a_response(event)
        return event

    setattr(
        _build_response_event_with_a2ui,
        _ADK_A2UI_RESPONSE_EVENT_DELIVERY_ATTR,
        True,
    )
    setattr(adk_functions, "__build_response_event", _build_response_event_with_a2ui)


def _install_a2ui_standard_a2a_export() -> None:
    from google.adk.a2a.converters import event_converter
    from google.adk.a2a.converters import from_adk_event

    _wrap_a2a_event_converter(
        from_adk_event,
        "convert_event_to_a2a_events",
    )
    _wrap_a2a_event_converter(
        event_converter,
        "convert_event_to_a2a_events",
    )

    try:
        from google.adk.a2a.executor.config import A2aAgentExecutorConfig

        A2aAgentExecutorConfig.model_fields[
            "adk_event_converter"
        ].default = from_adk_event.convert_event_to_a2a_events
        A2aAgentExecutorConfig.model_fields[
            "event_converter"
        ].default = event_converter.convert_event_to_a2a_events
        A2aAgentExecutorConfig.model_rebuild(force=True)
    except (AttributeError, KeyError, TypeError):
        return


def _wrap_a2a_event_converter(module: Any, function_name: str) -> None:
    convert_event_to_a2a_events = getattr(module, function_name)
    if getattr(
        convert_event_to_a2a_events,
        _ADK_A2UI_A2A_STANDARD_EXPORT_ATTR,
        False,
    ):
        return

    def _convert_event_to_a2a_events_with_standard_a2ui(
        event: Any,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        return convert_event_to_a2a_events(
            _event_with_standard_a2ui_parts_for_a2a_export(event),
            *args,
            **kwargs,
        )

    setattr(
        _convert_event_to_a2a_events_with_standard_a2ui,
        _ADK_A2UI_A2A_STANDARD_EXPORT_ATTR,
        True,
    )
    setattr(module, function_name, _convert_event_to_a2a_events_with_standard_a2ui)


def _install_a2ui_parallel_response_merge() -> None:
    from google.adk.flows.llm_flows import functions as adk_functions

    merge_response_events = getattr(
        adk_functions,
        "merge_parallel_function_response_events",
    )
    if getattr(merge_response_events, _ADK_A2UI_PARALLEL_MERGE_ATTR, False):
        return

    def _merge_parallel_function_response_events_with_a2ui_marker(
        function_response_events: list[Any],
    ) -> Any:
        merged_event = merge_response_events(function_response_events)
        if any(
            _is_a2ui_transport_response_event(event)
            or _has_a2ui_transport_inline_part(getattr(event, "content", None))
            for event in function_response_events
        ):
            _mark_event_as_a2a_response(merged_event)
        return merged_event

    setattr(
        _merge_parallel_function_response_events_with_a2ui_marker,
        _ADK_A2UI_PARALLEL_MERGE_ATTR,
        True,
    )
    setattr(
        adk_functions,
        "merge_parallel_function_response_events",
        _merge_parallel_function_response_events_with_a2ui_marker,
    )


def _install_a2ui_response_event_split() -> None:
    from google.adk.flows.llm_flows import base_llm_flow

    postprocess_function_calls = getattr(
        base_llm_flow.BaseLlmFlow,
        "_postprocess_handle_function_calls_async",
    )
    if not getattr(
        postprocess_function_calls,
        _ADK_A2UI_RESPONSE_EVENT_SPLIT_ATTR,
        False,
    ):

        async def _postprocess_handle_function_calls_with_a2ui_split(
            self: Any,
            *args: Any,
            **kwargs: Any,
        ) -> Any:
            async for event in postprocess_function_calls(self, *args, **kwargs):
                for delivery_event in _a2ui_transport_response_events_for_delivery(
                    event
                ):
                    yield delivery_event

        setattr(
            _postprocess_handle_function_calls_with_a2ui_split,
            _ADK_A2UI_RESPONSE_EVENT_SPLIT_ATTR,
            True,
        )
        setattr(
            base_llm_flow.BaseLlmFlow,
            "_postprocess_handle_function_calls_async",
            _postprocess_handle_function_calls_with_a2ui_split,
        )

    postprocess_live = getattr(base_llm_flow.BaseLlmFlow, "_postprocess_live")
    if getattr(postprocess_live, _ADK_A2UI_LIVE_RESPONSE_EVENT_SPLIT_ATTR, False):
        return

    async def _postprocess_live_with_a2ui_split(
        self: Any,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        async for event in postprocess_live(self, *args, **kwargs):
            for delivery_event in _a2ui_transport_response_events_for_delivery(event):
                yield delivery_event

    setattr(
        _postprocess_live_with_a2ui_split,
        _ADK_A2UI_LIVE_RESPONSE_EVENT_SPLIT_ATTR,
        True,
    )
    setattr(
        base_llm_flow.BaseLlmFlow,
        "_postprocess_live",
        _postprocess_live_with_a2ui_split,
    )


def _mark_event_as_a2a_response(event: Any) -> None:
    custom_metadata = event.custom_metadata
    if not isinstance(custom_metadata, Mapping):
        custom_metadata = {}
    else:
        custom_metadata = dict(custom_metadata)
    custom_metadata["a2a:response"] = True
    event.custom_metadata = custom_metadata


def _install_a2ui_model_history_filter() -> None:
    from google.adk.flows.llm_flows import contents as adk_contents

    get_contents = getattr(adk_contents, "_get_contents")
    if not getattr(get_contents, _ADK_A2UI_MODEL_HISTORY_FILTER_ATTR, False):

        def _get_contents_without_a2ui_model_history(
            *args: Any,
            **kwargs: Any,
        ) -> list[genai_types.Content]:
            filtered_args, filtered_kwargs = (
                _with_a2ui_transport_response_events_filtered(args, kwargs)
            )
            return get_contents(*filtered_args, **filtered_kwargs)

        setattr(
            _get_contents_without_a2ui_model_history,
            _ADK_A2UI_MODEL_HISTORY_FILTER_ATTR,
            True,
        )
        setattr(
            adk_contents,
            "_get_contents",
            _get_contents_without_a2ui_model_history,
        )

    get_current_turn_contents = getattr(adk_contents, "_get_current_turn_contents")
    if not getattr(
        get_current_turn_contents,
        _ADK_A2UI_MODEL_HISTORY_FILTER_ATTR,
        False,
    ):

        def _get_current_turn_contents_without_a2ui_model_history(
            *args: Any,
            **kwargs: Any,
        ) -> list[genai_types.Content]:
            filtered_args, filtered_kwargs = (
                _with_a2ui_transport_response_events_filtered(args, kwargs)
            )
            return get_current_turn_contents(*filtered_args, **filtered_kwargs)

        setattr(
            _get_current_turn_contents_without_a2ui_model_history,
            _ADK_A2UI_MODEL_HISTORY_FILTER_ATTR,
            True,
        )
        setattr(
            adk_contents,
            "_get_current_turn_contents",
            _get_current_turn_contents_without_a2ui_model_history,
        )


def _with_a2ui_transport_response_events_filtered(
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    if len(args) > 1:
        filtered_args = list(args)
        filtered_args[1] = _without_a2ui_transport_response_events(
            filtered_args[1],
        )
        return tuple(filtered_args), kwargs

    if "events" not in kwargs:
        return args, kwargs

    filtered_kwargs = dict(kwargs)
    filtered_kwargs["events"] = _without_a2ui_transport_response_events(
        filtered_kwargs["events"],
    )
    return args, filtered_kwargs


def _without_a2ui_transport_response_events(events: Any) -> Any:
    if not isinstance(events, Sequence) or isinstance(
        events,
        str | bytes | bytearray,
    ):
        return events

    filtered_events: list[Any] = []
    for event in events:
        if not _is_a2ui_transport_response_event(event):
            filtered_events.append(event)
            continue

        content = getattr(event, "content", None)
        if content is None:
            filtered_events.append(event)
            continue

        filtered_content = _without_a2ui_transport_parts(content)
        if filtered_content is content:
            filtered_events.append(event)
            continue

        filtered_event = event.model_copy(deep=True)
        filtered_event.content = filtered_content
        filtered_events.append(filtered_event)

    return filtered_events


def _is_a2ui_transport_response_event(event: Any) -> bool:
    custom_metadata = getattr(event, "custom_metadata", None)
    if (
        isinstance(custom_metadata, Mapping)
        and custom_metadata.get("a2a:response") is True
    ):
        return True

    content = getattr(event, "content", None)
    return _has_function_response_part(content) and _has_a2ui_transport_inline_part(
        content
    )


def _event_with_standard_a2ui_parts_for_a2a_export(event: Any) -> Any:
    custom_metadata = getattr(event, "custom_metadata", None)
    if not (
        isinstance(custom_metadata, Mapping)
        and custom_metadata.get("a2a:response") is True
    ):
        return event

    content = getattr(event, "content", None)
    if content is None or not content.parts:
        return event
    if not _has_a2ui_transport_inline_part(content):
        return event

    standard_a2ui_parts = _standard_a2ui_content_parts_from_function_responses(
        content,
    )
    retained_parts = [
        part for part in content.parts if not _is_a2ui_transport_inline_part(part)
    ]
    if not standard_a2ui_parts and len(retained_parts) == len(content.parts):
        return event

    export_event = event.model_copy(deep=True)
    export_content = content.model_copy(deep=True)
    export_content.parts = [*retained_parts, *standard_a2ui_parts]
    export_event.content = export_content
    return export_event


def _standard_a2ui_content_parts_from_function_responses(
    content: genai_types.Content,
) -> list[genai_types.Part]:
    standard_parts: list[genai_types.Part] = []
    for part in content.parts or []:
        function_response = part.function_response
        if function_response is None:
            continue
        response = function_response.response
        if not isinstance(response, Mapping):
            continue
        standard_parts.extend(
            adk_content_parts_for_a2ui_response(
                response,
                validated_a2ui_parts=_trusted_data_parts_from_response(response),
            )
        )
    return standard_parts


def _trusted_data_parts_from_response(
    response: Mapping[str, Any],
) -> list[DataPart]:
    parts = response.get("a2uiParts")
    if parts is None or parts == []:
        return []
    if not isinstance(parts, Sequence) or isinstance(parts, str | bytes | bytearray):
        raise A2UIWidgetDeliveryError("a2uiParts must be a list of A2UI DataParts")

    trusted_parts: list[DataPart] = []
    for part in parts:
        trusted_parts.append(DataPart.model_validate(part))
    return trusted_parts


def _a2ui_transport_response_events_for_delivery(event: Any) -> list[Any]:
    content = getattr(event, "content", None)
    if content is None:
        return [event]

    parts = content.parts
    if not parts:
        return [event]

    a2ui_groups = _a2ui_transport_inline_part_groups(parts)
    if not a2ui_groups:
        return [event]

    if len(a2ui_groups) == 1:
        _mark_event_as_a2a_response(event)
        return [event]

    base_parts = [
        part for part in parts if not _is_a2ui_transport_inline_part(part)
    ]
    delivery_events: list[Any] = []
    for index, a2ui_group in enumerate(a2ui_groups):
        delivery_event = event.model_copy(deep=True)
        delivery_content = content.model_copy(deep=True)
        if index == 0:
            delivery_content.parts = [*base_parts, *a2ui_group]
        else:
            delivery_content.parts = list(a2ui_group)
            delivery_event.actions = EventActions()
            delivery_event.id = Event.new_id()
        delivery_event.content = delivery_content
        _mark_event_as_a2a_response(delivery_event)
        delivery_events.append(delivery_event)

    return delivery_events


def _a2ui_transport_inline_part_groups(
    parts: Sequence[genai_types.Part],
) -> list[list[genai_types.Part]]:
    groups: list[list[genai_types.Part]] = []
    current_group: list[genai_types.Part] = []
    current_keys: set[str] = set()

    for part in parts:
        if not _is_a2ui_transport_inline_part(part):
            continue

        message_keys = _a2ui_transport_inline_part_message_keys(part)
        if current_group and current_keys.intersection(message_keys):
            groups.append(current_group)
            current_group = []
            current_keys = set()

        current_group.append(part)
        current_keys.update(message_keys)

    if current_group:
        groups.append(current_group)

    return groups


def _a2ui_transport_inline_part_message_keys(part: genai_types.Part) -> set[str]:
    data_part = _a2ui_transport_data_part(part)
    if data_part is None or not isinstance(data_part.data, Mapping):
        return set()
    return {str(key) for key in data_part.data if key != "version"}


def _has_function_response_part(content: genai_types.Content | None) -> bool:
    if content is None or not content.parts:
        return False
    return any(part.function_response is not None for part in content.parts)


def _has_a2ui_transport_inline_part(content: genai_types.Content | None) -> bool:
    if content is None or not content.parts:
        return False
    return any(_is_a2ui_transport_inline_part(part) for part in content.parts)


def _without_a2ui_transport_parts(
    content: genai_types.Content,
) -> genai_types.Content | None:
    parts = content.parts
    if not parts:
        return content

    filtered_parts = [
        part for part in parts if not _is_a2ui_transport_inline_part(part)
    ]
    if len(filtered_parts) == len(parts):
        return content
    if not filtered_parts:
        return None

    filtered_content = content.model_copy(deep=True)
    filtered_content.parts = filtered_parts
    return filtered_content


def _is_a2ui_transport_inline_part(part: genai_types.Part) -> bool:
    data_part = _a2ui_transport_data_part(part)
    if data_part is None:
        return False

    metadata = data_part.metadata
    return isinstance(metadata, Mapping) and (
        metadata.get("mimeType") == "application/json+a2ui"
    )


def _a2ui_transport_data_part(
    part: genai_types.Part,
) -> a2a_types.DataPart | None:
    inline_data = part.inline_data
    if (
        inline_data is None
        or inline_data.mime_type != part_converter.A2A_DATA_PART_TEXT_MIME_TYPE
        or not isinstance(inline_data.data, bytes)
        or not inline_data.data.startswith(part_converter.A2A_DATA_PART_START_TAG)
        or not inline_data.data.endswith(part_converter.A2A_DATA_PART_END_TAG)
    ):
        return None

    data_part_bytes = inline_data.data[
        len(part_converter.A2A_DATA_PART_START_TAG) : -len(
            part_converter.A2A_DATA_PART_END_TAG
        )
    ]
    try:
        return a2a_types.DataPart.model_validate_json(data_part_bytes)
    except ValueError:
        return None


def _adk_content_part_for_a2ui(part: Mapping[str, Any]) -> genai_types.Part:
    data = part.get("data")
    metadata = part.get("metadata")
    if not isinstance(data, Mapping) or not isinstance(metadata, Mapping):
        raise A2UIWidgetDeliveryError("A2UI DataPart payload is malformed")

    a2a_data_part = a2a_types.DataPart(
        data=dict(data),
        metadata=dict(metadata),
    )
    inline_blob = (
        part_converter.A2A_DATA_PART_START_TAG
        + json.dumps(
            a2a_data_part.model_dump(
                by_alias=True,
                exclude_none=True,
                mode="json",
            ),
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        + part_converter.A2A_DATA_PART_END_TAG
    )
    return genai_types.Part(
        inline_data=genai_types.Blob(
            data=inline_blob,
            mime_type=part_converter.A2A_DATA_PART_TEXT_MIME_TYPE,
        )
    )


def _dev_ui_a2ui_part_payloads(
    parts: Sequence[Mapping[str, Any]],
    *,
    known_component_ids_by_surface_id: dict[str, set[str]] | None = None,
) -> list[dict[str, Any]]:
    root_ids_by_surface_id = _root_component_ids_by_surface_id(parts)
    known_component_ids = (
        {}
        if known_component_ids_by_surface_id is None
        else {
            surface_id: set(component_ids)
            for surface_id, component_ids in known_component_ids_by_surface_id.items()
        }
    )
    translated_parts: list[dict[str, Any]] = []
    pending_begin_rendering_by_surface_id: dict[str, dict[str, Any]] = {}

    for part in parts:
        translated_part = _dev_ui_a2ui_part_payload(
            part,
            root_ids_by_surface_id=root_ids_by_surface_id,
            known_component_ids_by_surface_id=known_component_ids,
        )
        data = translated_part.get("data")
        if not isinstance(data, Mapping):
            raise A2UIWidgetDeliveryError("A2UI DataPart payload is malformed")

        begin_rendering = data.get("beginRendering")
        if isinstance(begin_rendering, Mapping):
            surface_id = begin_rendering.get("surfaceId")
            if isinstance(surface_id, str) and surface_id:
                pending_begin_rendering_by_surface_id[surface_id] = translated_part
                continue

        translated_parts.append(translated_part)
        surface_id = _surface_id_from_dev_ui_a2ui_payload(data)
        if (
            surface_id is not None
            and surface_id in pending_begin_rendering_by_surface_id
            and "surfaceUpdate" in data
        ):
            translated_parts.append(
                pending_begin_rendering_by_surface_id.pop(surface_id)
            )

    translated_parts.extend(pending_begin_rendering_by_surface_id.values())
    return translated_parts


def _dev_ui_a2ui_part_payload(
    part: Mapping[str, Any],
    *,
    root_ids_by_surface_id: Mapping[str, str],
    known_component_ids_by_surface_id: dict[str, set[str]],
) -> dict[str, Any]:
    data = part.get("data")
    metadata = part.get("metadata")
    if not isinstance(data, Mapping) or not isinstance(metadata, Mapping):
        raise A2UIWidgetDeliveryError("A2UI DataPart payload is malformed")

    translated_part = dict(part)
    translated_part["data"] = _dev_ui_a2ui_payload(
        data,
        root_ids_by_surface_id=root_ids_by_surface_id,
        known_component_ids_by_surface_id=known_component_ids_by_surface_id,
    )
    translated_part["metadata"] = dict(metadata)
    return translated_part


def _dev_ui_a2ui_payload(
    payload: Mapping[str, Any],
    *,
    root_ids_by_surface_id: Mapping[str, str],
    known_component_ids_by_surface_id: dict[str, set[str]],
) -> dict[str, Any]:
    create_surface = payload.get(CREATE_SURFACE_MESSAGE)
    if isinstance(create_surface, Mapping):
        return {
            "beginRendering": _begin_rendering_payload(
                create_surface,
                root_ids_by_surface_id=root_ids_by_surface_id,
            )
        }

    update_components = payload.get(UPDATE_COMPONENTS_MESSAGE)
    if isinstance(update_components, Mapping):
        return {
            "surfaceUpdate": _surface_update_payload(
                update_components,
                known_component_ids_by_surface_id=known_component_ids_by_surface_id,
            ),
        }

    update_data_model = payload.get(UPDATE_DATA_MODEL_MESSAGE)
    if isinstance(update_data_model, Mapping):
        return {
            "dataModelUpdate": _data_model_update_payload(update_data_model),
        }

    delete_surface = payload.get(DELETE_SURFACE_MESSAGE)
    if isinstance(delete_surface, Mapping):
        return {
            "surfaceUpdate": _delete_surface_payload(
                delete_surface,
                root_ids_by_surface_id=root_ids_by_surface_id,
            ),
        }

    return deepcopy(dict(payload))


def _begin_rendering_payload(
    create_surface: Mapping[str, Any],
    *,
    root_ids_by_surface_id: Mapping[str, str],
) -> dict[str, Any]:
    surface_id = create_surface.get("surfaceId")
    if not isinstance(surface_id, str) or not surface_id:
        raise A2UIWidgetDeliveryError("A2UI createSurface surfaceId is malformed")

    begin_rendering: dict[str, Any] = {
        "surfaceId": surface_id,
        "root": root_ids_by_surface_id.get(surface_id, "root"),
    }
    catalog_id = create_surface.get("catalogId")
    if isinstance(catalog_id, str) and catalog_id:
        begin_rendering["catalogId"] = catalog_id

    theme = create_surface.get("theme")
    if isinstance(theme, Mapping):
        begin_rendering["styles"] = dict(theme)

    return begin_rendering


def _surface_update_payload(
    update_components: Mapping[str, Any],
    *,
    known_component_ids_by_surface_id: dict[str, set[str]],
) -> dict[str, Any]:
    surface_id = update_components.get("surfaceId")
    surface_update = dict(update_components)
    components = surface_update.get("components")
    if isinstance(components, list):
        component_ids: set[str] = set()
        for component in components:
            if not isinstance(component, Mapping):
                continue
            component_id = component.get("id")
            if isinstance(component_id, str):
                component_ids.add(component_id)
        is_full_replacement = _is_full_component_replacement(update_components)
        if isinstance(surface_id, str) and not is_full_replacement:
            component_ids.update(
                known_component_ids_by_surface_id.get(surface_id, set())
            )
        converted_components: list[Any] = []
        for component in components:
            converted_components.extend(
                _dev_ui_component_payloads(component, component_ids=component_ids)
            )
        surface_update["components"] = converted_components
        if isinstance(surface_id, str):
            _remember_surface_component_ids(
                known_component_ids_by_surface_id,
                surface_id=surface_id,
                components=converted_components,
                replace=is_full_replacement,
            )
    return surface_update


def _is_full_component_replacement(message: Mapping[str, Any]) -> bool:
    return (
        message.get("replace") is True
        or message.get("fullReplacement") is True
        or message.get("mode") == "replace"
    )


def _remember_surface_component_ids(
    known_component_ids_by_surface_id: dict[str, set[str]],
    *,
    surface_id: str,
    components: Sequence[Any],
    replace: bool,
) -> None:
    surface_component_ids = known_component_ids_by_surface_id.setdefault(
        surface_id,
        set(),
    )
    if replace:
        surface_component_ids.clear()
    for component in components:
        if not isinstance(component, Mapping):
            continue
        component_id = component.get("id")
        if isinstance(component_id, str) and component_id:
            surface_component_ids.add(component_id)


def _delete_surface_payload(
    delete_surface: Mapping[str, Any],
    *,
    root_ids_by_surface_id: Mapping[str, str],
) -> dict[str, Any]:
    surface_id = delete_surface.get("surfaceId")
    if not isinstance(surface_id, str) or not surface_id:
        raise A2UIWidgetDeliveryError("A2UI deleteSurface surfaceId is malformed")

    root_id = root_ids_by_surface_id.get(surface_id, "root")
    closed_text_id = f"{root_id}_surface_closed"
    return {
        "surfaceId": surface_id,
        "components": [
            {
                "id": root_id,
                "component": {
                    "Column": {"children": {"explicitList": [closed_text_id]}}
                },
            },
            {
                "id": closed_text_id,
                "component": {
                    "Text": {
                        "text": {"literalString": "Plan review closed."},
                        "usageHint": "secondary",
                    }
                },
            },
        ],
    }


def _dev_ui_component_payloads(
    component: Any,
    *,
    component_ids: set[str],
) -> list[Any]:
    if not isinstance(component, Mapping):
        return [deepcopy(component)]

    component_id = component.get("id")
    component_name = component.get("component")
    if (
        component_name != "Button"
        or not isinstance(component_id, str)
        or not component_id
        or "child" in component
        or "label" not in component
    ):
        return [_dev_ui_component_payload(component)]

    label_child_id = _synthetic_button_label_child_id(component_id, component_ids)
    component_ids.add(label_child_id)
    return [
        _dev_ui_component_payload(
            component,
            synthetic_button_label_child_id=label_child_id,
        ),
        {
            "id": label_child_id,
            "component": {
                "Text": {"text": _dev_ui_text_value(component.get("label"))}
            },
        },
    ]


def _synthetic_button_label_child_id(
    component_id: str,
    component_ids: set[str],
) -> str:
    base_child_id = f"{component_id}_label"
    if base_child_id not in component_ids:
        return base_child_id

    suffix = 2
    while f"{base_child_id}_{suffix}" in component_ids:
        suffix += 1
    return f"{base_child_id}_{suffix}"


def _dev_ui_component_payload(
    component: Any,
    *,
    synthetic_button_label_child_id: str | None = None,
) -> Any:
    if not isinstance(component, Mapping):
        return deepcopy(component)

    component_id = component.get("id")
    component_name = component.get("component")
    if not isinstance(component_id, str) or not component_id:
        return deepcopy(dict(component))
    if not isinstance(component_name, str) or not component_name:
        return deepcopy(dict(component))

    dev_ui_component_name = _dev_ui_component_name(component_name)
    component_body = {}
    for key, value in component.items():
        if key in {"id", "component"}:
            continue
        if synthetic_button_label_child_id is not None and key == "label":
            continue
        component_body[
            _dev_ui_component_property_name(key, component_name=component_name)
        ] = _dev_ui_component_property(key, value, component_name=component_name)
    if synthetic_button_label_child_id is not None:
        component_body["child"] = synthetic_button_label_child_id
    return {
        "id": component_id,
        "component": {dev_ui_component_name: component_body},
    }


def _dev_ui_component_name(component_name: str) -> str:
    return _DEV_UI_COMPONENT_NAME_ALIASES.get(component_name, component_name)


def _dev_ui_component_property_name(key: str, *, component_name: str) -> str:
    if key == "variant" and component_name == "Text":
        return "usageHint"
    if component_name == "TextField":
        if key == "value":
            return "text"
        if key == "variant":
            return "type"
    if component_name == "Slider":
        if key == "min":
            return "minValue"
        if key == "max":
            return "maxValue"
    if component_name == "ChoicePicker" and key == "value":
        return "selections"
    if component_name == "Tabs" and key == "tabs":
        return "tabItems"
    if component_name == "Modal":
        if key == "trigger":
            return "entryPointChild"
        if key == "content":
            return "contentChild"
    return key


def _dev_ui_component_property(
    key: str,
    value: Any,
    *,
    component_name: str,
    top_level: bool = True,
) -> Any:
    if key == "children" and isinstance(value, list):
        return {"explicitList": list(value)}

    if key == "children" and isinstance(value, Mapping):
        template_children = _dev_ui_templated_children(value)
        if template_children is not None:
            return template_children

    if top_level and key == "text" and component_name == "Text":
        return _dev_ui_text_value(value)

    if top_level and key == "label" and component_name in {"Button", "TextField"}:
        return _dev_ui_text_value(value)

    if top_level and key == "value" and component_name == "TextField":
        return _dev_ui_text_value(value)

    if top_level and key == "action" and component_name == "Button":
        return _dev_ui_button_action(value)

    if (
        top_level
        and key in {"enableDate", "enableTime"}
        and component_name == "DateTimeInput"
    ):
        return deepcopy(value)

    if top_level and key in {"min", "max"} and component_name == "Slider":
        return deepcopy(value)

    if top_level and key in _DEV_UI_COMPONENT_PRIMITIVE_FIELDS.get(
        component_name,
        frozenset(),
    ):
        return _dev_ui_component_primitive_value(value)

    if top_level and key == "options" and component_name == "ChoicePicker":
        return _dev_ui_choice_picker_options(value)

    if top_level and key == "tabs" and component_name == "Tabs":
        return _dev_ui_tabs(value)

    if isinstance(value, Mapping):
        return {
            nested_key: _dev_ui_component_property(
                str(nested_key),
                nested_value,
                component_name=component_name,
                top_level=False,
            )
            for nested_key, nested_value in value.items()
        }

    if isinstance(value, list):
        return [
            _dev_ui_component_property(key, item, component_name=component_name)
            for item in value
        ]

    return deepcopy(value)


def _dev_ui_templated_children(value: Mapping[str, Any]) -> dict[str, Any] | None:
    component_id = value.get("componentId")
    path = value.get("path")
    if not isinstance(component_id, str) or not component_id:
        return None
    if not isinstance(path, str) or not path:
        return None
    return {
        "template": {
            "componentId": component_id,
            "dataBinding": {"path": path},
        }
    }


def _dev_ui_text_value(value: Any) -> Any:
    return _dev_ui_component_primitive_value(value)


def _dev_ui_component_primitive_value(value: Any) -> Any:
    encoded_value = _encoded_dev_ui_action_context_value(value)
    if encoded_value is not None:
        return encoded_value
    if isinstance(value, Mapping) and set(value) == {"path"}:
        path = value.get("path")
        if isinstance(path, str):
            return {"path": path}
    if isinstance(value, bool):
        return {"literalBoolean": value}
    if isinstance(value, int | float) and not isinstance(value, bool):
        return {"literalNumber": value}
    if isinstance(value, str):
        return {"literalString": value}
    return deepcopy(value)


def _dev_ui_choice_picker_options(value: Any) -> Any:
    if not isinstance(value, list):
        return deepcopy(value)

    options: list[Any] = []
    for option in value:
        if not isinstance(option, Mapping):
            options.append(deepcopy(option))
            continue
        converted_option = dict(option)
        label = converted_option.get("label")
        if label is not None:
            converted_option["label"] = _dev_ui_component_primitive_value(label)
        options.append(converted_option)
    return options


def _dev_ui_tabs(value: Any) -> Any:
    if not isinstance(value, list):
        return deepcopy(value)

    tabs: list[Any] = []
    for tab in value:
        if not isinstance(tab, Mapping):
            tabs.append(deepcopy(tab))
            continue
        converted_tab = dict(tab)
        title = converted_tab.get("title")
        if title is not None:
            converted_tab["title"] = _dev_ui_component_primitive_value(title)
        tabs.append(converted_tab)
    return tabs


def _dev_ui_button_action(value: Any) -> Any:
    if not isinstance(value, Mapping):
        return deepcopy(value)

    event = value.get("event")
    if not isinstance(event, Mapping):
        return deepcopy(dict(value))

    action = deepcopy(dict(event))
    context = action.get("context")
    if isinstance(context, Mapping):
        action["context"] = _dev_ui_action_context(context)
    return action


def _dev_ui_action_context(context: Mapping[str, Any]) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    namespace_payload = _dev_ui_context_uses_namespaced_payload(context)
    for key, value in context.items():
        if key == "payload" and not namespace_payload and isinstance(value, Mapping):
            entries.extend(_dev_ui_action_context(value))
            continue
        if key == "payload" and not namespace_payload and isinstance(value, list):
            payload_entries = _dev_ui_basic_catalog_context_entries(value)
            if payload_entries is not None:
                entries.extend(payload_entries)
                continue
        entries.append(
            {
                "key": str(key),
                "value": (
                    _dev_ui_action_payload_value(value)
                    if key == "payload"
                    else _dev_ui_action_context_value(value)
                ),
            }
        )
    return entries


def _dev_ui_context_uses_namespaced_payload(context: Mapping[str, Any]) -> bool:
    action_type = context.get("type")
    return not isinstance(action_type, str) or action_type not in PLAN_USER_ACTION_TYPES


def _dev_ui_action_payload_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return _dev_ui_action_context(value)
    if isinstance(value, list):
        payload_entries = _dev_ui_basic_catalog_context_entries(value)
        if payload_entries is not None:
            return payload_entries
    return _dev_ui_action_context_value(value)


def _dev_ui_basic_catalog_context_entries(
    value: list[Any],
) -> list[dict[str, Any]] | None:
    entries: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, Mapping):
            return None
        key = item.get("key")
        if not isinstance(key, str) or not key:
            return None
        entries.append(
            {
                "key": key,
                "value": _dev_ui_action_context_value(item.get("value")),
            }
        )
    return entries


def _dev_ui_action_context_value(value: Any) -> dict[str, Any]:
    encoded_value = _encoded_dev_ui_action_context_value(value)
    if encoded_value is not None:
        return encoded_value
    if isinstance(value, Mapping) and isinstance(value.get("path"), str):
        return {"path": value["path"]}
    if isinstance(value, bool):
        return {"literalBoolean": value}
    if isinstance(value, int | float) and not isinstance(value, bool):
        return {"literalNumber": value}
    if isinstance(value, str):
        return {"literalString": value}
    if isinstance(value, Mapping) or isinstance(value, list):
        return {
            "literalString": json.dumps(
                value,
                ensure_ascii=False,
                separators=(",", ":"),
            )
        }
    return {"literalString": "" if value is None else str(value)}


def _encoded_dev_ui_action_context_value(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, Mapping):
        return None

    if set(value) == {"literalString"} and isinstance(value.get("literalString"), str):
        return deepcopy(dict(value))
    if set(value) == {"literalBoolean"} and isinstance(value.get("literalBoolean"), bool):
        return deepcopy(dict(value))
    if (
        set(value) == {"literalNumber"}
        and isinstance(value.get("literalNumber"), int | float)
        and not isinstance(value.get("literalNumber"), bool)
    ):
        return deepcopy(dict(value))
    return None


def _data_model_update_payload(update_data_model: Mapping[str, Any]) -> dict[str, Any]:
    data_model_update = {
        key: deepcopy(value)
        for key, value in update_data_model.items()
        if key != "value"
    }
    if "value" in update_data_model:
        value = update_data_model["value"]
        scalar_target = _scalar_data_model_target(update_data_model.get("path"), value)
        if scalar_target is not None:
            data_model_update["path"], key = scalar_target
            data_model_update["contents"] = [_data_model_entry(key, value)]
        else:
            data_model_update["contents"] = _data_model_contents(value)
    return data_model_update


def _scalar_data_model_target(path: Any, value: Any) -> tuple[str, str] | None:
    if isinstance(value, Mapping) or not isinstance(path, str):
        return None
    if path in {"", "/"}:
        return None

    path_without_trailing_slash = path.rstrip("/")
    if not path_without_trailing_slash:
        return None

    parent_path, separator, key = path_without_trailing_slash.rpartition("/")
    if not separator or not key:
        return None

    if path_without_trailing_slash.startswith("/"):
        parent_path = parent_path or "/"
    return parent_path, _json_pointer_key(key)


def _data_model_contents(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, Mapping):
        return [
            _data_model_entry(str(key), nested_value)
            for key, nested_value in value.items()
        ]
    return [_data_model_entry("value", value)]


def _data_model_entry(key: str, value: Any) -> dict[str, Any]:
    entry: dict[str, Any] = {"key": key}
    if isinstance(value, bool):
        entry["valueBoolean"] = value
    elif isinstance(value, int | float) and not isinstance(value, bool):
        entry["valueNumber"] = value
    elif isinstance(value, Mapping):
        entry["valueMap"] = _data_model_contents(value)
    elif isinstance(value, list):
        entry["valueString"] = json.dumps(
            value,
            ensure_ascii=False,
            separators=(",", ":"),
        )
    else:
        entry["valueString"] = "" if value is None else str(value)
    return entry


def _json_pointer_key(value: str) -> str:
    return value.replace("~1", "/").replace("~0", "~")


def _root_component_ids_by_surface_id(
    parts: Sequence[Mapping[str, Any]],
) -> dict[str, str]:
    root_ids: dict[str, str] = {}
    for part in parts:
        data = part.get("data")
        if not isinstance(data, Mapping):
            continue

        update_components = data.get(UPDATE_COMPONENTS_MESSAGE)
        if not isinstance(update_components, Mapping):
            continue

        surface_id = update_components.get("surfaceId")
        components = update_components.get("components")
        if not isinstance(surface_id, str) or not isinstance(components, list):
            continue

        fallback_root_id: str | None = None
        for component in components:
            if not isinstance(component, Mapping):
                continue
            component_id = component.get("id")
            if not isinstance(component_id, str) or not component_id:
                continue
            if component_id == "root":
                root_ids[surface_id] = component_id
                break
            if fallback_root_id is None:
                fallback_root_id = component_id
        else:
            if fallback_root_id is not None:
                root_ids.setdefault(surface_id, fallback_root_id)
    return root_ids


def _surface_id_from_dev_ui_a2ui_payload(payload: Mapping[str, Any]) -> str | None:
    for message_type in (
        "beginRendering",
        "surfaceUpdate",
        "dataModelUpdate",
        DELETE_SURFACE_MESSAGE,
    ):
        message = payload.get(message_type)
        if not isinstance(message, Mapping):
            continue
        surface_id = message.get("surfaceId")
        if isinstance(surface_id, str) and surface_id:
            return surface_id
    return None


def _validated_part_payloads(parts: Sequence[Any]) -> list[dict[str, Any]]:
    result = validate_outbound_a2ui(list(parts))
    if not result.valid:
        errors = "; ".join(_safe_errors(result.validation_errors))
        raise A2UIWidgetDeliveryError(
            f"A2UI payload failed outbound validation: {errors}"
        )

    validated_parts: list[dict[str, Any]] = []
    for part in result.renderer_parts:
        if not isinstance(part, DataPart):
            errors = "; ".join(_safe_errors(result.validation_errors))
            raise A2UIWidgetDeliveryError(
                "A2UI payload validation did not produce DataParts"
                + (f": {errors}" if errors else "")
            )
        validated_parts.append(part.model_dump(by_alias=True, mode="json"))

    if not validated_parts:
        raise A2UIWidgetDeliveryError("A2UI payload produced no DataParts")
    return validated_parts


def _widget_id(
    response: Mapping[str, Any],
    parts: Sequence[Mapping[str, Any]],
) -> str:
    approval_surface_id = response.get("approvalSurfaceId")
    if isinstance(approval_surface_id, str) and approval_surface_id:
        return approval_surface_id

    for part in parts:
        data = part.get("data")
        if not isinstance(data, Mapping):
            continue
        surface_id = _surface_id_from_a2ui_payload(data)
        if surface_id is not None:
            return surface_id

    raise A2UIWidgetDeliveryError("A2UI widget id could not be derived")


def _surface_id_from_a2ui_payload(payload: Mapping[str, Any]) -> str | None:
    for message_type in (
        CREATE_SURFACE_MESSAGE,
        UPDATE_COMPONENTS_MESSAGE,
        DELETE_SURFACE_MESSAGE,
        UPDATE_DATA_MODEL_MESSAGE,
    ):
        message = payload.get(message_type)
        if not isinstance(message, Mapping):
            continue
        surface_id = message.get("surfaceId")
        if isinstance(surface_id, str) and surface_id:
            return surface_id
    return None


def _safe_errors(errors: Sequence[str]) -> list[str]:
    return [redact_secret_like_values(error) for error in errors]


__all__ = [
    "A2UIWidgetDeliveryError",
    "adk_content_parts_for_a2ui_response",
    "adk_dev_ui_content_parts_for_a2ui_response",
    "deliver_a2ui_parts_to_adk_ui",
    "install_a2ui_response_event_delivery",
]
