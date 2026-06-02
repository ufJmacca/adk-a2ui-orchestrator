"""A2A adapter and local remote-compatible wrapper support."""

from orchestrator_demo.a2a_support.local_remote_wrapper import (
    DEFAULT_LOCAL_REMOTE_AGENT_IDS,
    LocalRemoteAgentWrapper,
    build_default_local_remote_wrappers,
)
from orchestrator_demo.a2a_support.remote_agent_adapter import (
    RemoteA2AAgentAdapter,
    UserActionPayload,
)


__all__ = [
    "DEFAULT_LOCAL_REMOTE_AGENT_IDS",
    "LocalRemoteAgentWrapper",
    "RemoteA2AAgentAdapter",
    "UserActionPayload",
    "build_default_local_remote_wrappers",
]
