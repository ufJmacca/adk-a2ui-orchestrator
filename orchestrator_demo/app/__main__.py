"""Runtime entrypoint for ``python -m orchestrator_demo.app``."""

from __future__ import annotations

import sys

from orchestrator_demo.app.bootstrap_llm import build_litellm_model
from orchestrator_demo.app.settings import ConfigurationError


def main() -> int:
    """Validate runtime configuration and initialize the LiteLLM-backed model."""

    try:
        build_litellm_model()
    except ConfigurationError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    print("orchestrator_demo.app runtime configuration loaded.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
