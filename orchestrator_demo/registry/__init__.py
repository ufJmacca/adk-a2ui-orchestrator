"""Specialist agent registration support."""

from orchestrator_demo.registry.agent_registry import (
    AgentRegistry,
    RegistryValidationError,
    UnavailableAgentError,
)
from orchestrator_demo.registry.descriptors import REQUIRED_SPECIALIST_AGENT_IDS


__all__ = [
    "AgentRegistry",
    "RegistryValidationError",
    "REQUIRED_SPECIALIST_AGENT_IDS",
    "UnavailableAgentError",
]
