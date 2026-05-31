"""Dynamic specialist registry loaded from Python config."""

from __future__ import annotations

import logging
import runpy
import sys
from importlib import invalidate_caches
from importlib import util as importlib_util
from pathlib import Path
from types import ModuleType
from typing import Any

from orchestrator_demo.contracts import AgentDescriptor, ExecutionPlan
from orchestrator_demo.registry.descriptors import (
    DescriptorValidationError,
    validate_agent_descriptors,
)


LOGGER = logging.getLogger(__name__)
DEFAULT_CONFIG_MODULE = "orchestrator_demo.registry.agent_config"
MISSING = object()


class RegistryValidationError(DescriptorValidationError):
    """Raised when the registry config cannot be loaded or validated."""


class UnavailableAgentError(ValueError):
    """Raised when an approved plan references unavailable agent ids."""


class AgentRegistry:
    """Load, validate, and transactionally reload specialist descriptors."""

    def __init__(
        self,
        *,
        config_module: str = DEFAULT_CONFIG_MODULE,
        config_path: Path | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self._config_module = config_module
        self._config_path = Path(config_path) if config_path is not None else None
        self._logger = logger or LOGGER
        self._descriptors_by_id: dict[str, AgentDescriptor] = {}
        self.reload()

    @classmethod
    def from_config_path(cls, config_path: Path | str) -> "AgentRegistry":
        return cls(config_path=Path(config_path))

    @classmethod
    def from_default_config(cls) -> "AgentRegistry":
        return cls()

    def reload(self) -> None:
        """Reload config and keep the previous valid registry on failure."""

        previous_agent_ids = set(self._descriptors_by_id)
        try:
            raw_descriptors, module_to_publish = self._load_raw_descriptors()
            next_descriptors = validate_agent_descriptors(raw_descriptors)
        except RegistryValidationError as exc:
            self._logger.error(
                "agent registry reload rejected source=%s error=%s",
                self._source_label,
                exc,
            )
            raise
        except DescriptorValidationError as exc:
            registry_error = RegistryValidationError(str(exc))
            self._logger.error(
                "agent registry reload rejected source=%s error=%s",
                self._source_label,
                registry_error,
            )
            raise registry_error from exc

        next_agent_ids = set(next_descriptors)
        added_agent_ids = sorted(next_agent_ids - previous_agent_ids)
        removed_agent_ids = sorted(previous_agent_ids - next_agent_ids)
        if module_to_publish is not None:
            sys.modules[self._config_module] = module_to_publish
        self._descriptors_by_id = dict(sorted(next_descriptors.items()))
        self._logger.info(
            "agent registry reloaded source=%s added=%s removed=%s total=%d",
            self._source_label,
            _format_agent_ids(added_agent_ids),
            _format_agent_ids(removed_agent_ids),
            len(self._descriptors_by_id),
        )

    def agent_ids(self) -> list[str]:
        return list(self._descriptors_by_id)

    def descriptors(self) -> list[AgentDescriptor]:
        return list(self._descriptors_by_id.values())

    def get(self, agent_id: str) -> AgentDescriptor | None:
        return self._descriptors_by_id.get(agent_id)

    def is_available_for_new_plan(self, agent_id: str) -> bool:
        return agent_id in self._descriptors_by_id

    def find_unavailable_plan_agents(self, plan: ExecutionPlan) -> list[str]:
        unavailable: list[str] = []
        seen: set[str] = set()
        for agent_id in _plan_agent_ids(plan):
            if agent_id not in self._descriptors_by_id and agent_id not in seen:
                unavailable.append(agent_id)
                seen.add(agent_id)
        return unavailable

    def require_plan_agents_available(self, plan: ExecutionPlan) -> None:
        unavailable_agent_ids = self.find_unavailable_plan_agents(plan)
        if unavailable_agent_ids:
            unavailable = ", ".join(unavailable_agent_ids)
            raise UnavailableAgentError(
                f"approved plan {plan.plan_id} references unavailable agents: "
                f"{unavailable}"
            )

    @property
    def _source_label(self) -> str:
        if self._config_path is not None:
            return str(self._config_path)
        return self._config_module

    def _load_raw_descriptors(self) -> tuple[Any, ModuleType | None]:
        if self._config_path is not None:
            return self._load_raw_descriptors_from_path(), None

        try:
            module = _load_module_from_source(self._config_module)
        except Exception as exc:
            raise RegistryValidationError(
                f"failed to load registry config module: {type(exc).__name__}"
            ) from exc

        return _available_agents_from_namespace(module), module

    def _load_raw_descriptors_from_path(self) -> Any:
        if self._config_path is None:
            raise RegistryValidationError("config_path is not set")

        try:
            namespace = runpy.run_path(str(self._config_path))
        except Exception as exc:
            raise RegistryValidationError(
                f"failed to load registry config file: {type(exc).__name__}"
            ) from exc

        return _available_agents_from_namespace(namespace)


def _available_agents_from_namespace(namespace: ModuleType | dict[str, Any]) -> Any:
    if isinstance(namespace, ModuleType):
        raw_descriptors = getattr(namespace, "AVAILABLE_AGENTS", MISSING)
    else:
        raw_descriptors = namespace.get("AVAILABLE_AGENTS", MISSING)

    if raw_descriptors is MISSING:
        raise RegistryValidationError("registry config must define AVAILABLE_AGENTS")

    return raw_descriptors


def _load_module_from_source(module_name: str) -> ModuleType:
    invalidate_caches()
    spec = importlib_util.find_spec(module_name)
    if spec is None or spec.origin is None:
        raise RegistryValidationError(f"cannot find source for {module_name}")

    source_path = Path(spec.origin)
    source = source_path.read_text(encoding="utf-8")
    code = compile(source, str(source_path), "exec")
    module = importlib_util.module_from_spec(spec)
    exec(code, module.__dict__)
    return module


def _plan_agent_ids(plan: ExecutionPlan) -> list[str]:
    agent_ids: list[str] = []
    for agent_id in plan.selected_agents:
        if agent_id not in agent_ids:
            agent_ids.append(agent_id)
    for step in plan.steps:
        if step.agent_id not in agent_ids:
            agent_ids.append(step.agent_id)
    return agent_ids


def _format_agent_ids(agent_ids: list[str]) -> str:
    return ",".join(agent_ids) if agent_ids else "-"


__all__ = [
    "AgentRegistry",
    "RegistryValidationError",
    "UnavailableAgentError",
]
