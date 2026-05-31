"""Application entrypoints and runtime bootstrap."""

from __future__ import annotations

import os
import sys

from dotenv import find_dotenv, load_dotenv


REQUIRED_RUNTIME_ENV_VARS = ("OPENROUTER_API_KEY", "LLM_MODEL")


def main() -> int:
    """Validate runtime configuration for the documented module entrypoint."""

    load_dotenv(find_dotenv(usecwd=True))

    missing = [
        variable
        for variable in REQUIRED_RUNTIME_ENV_VARS
        if not os.environ.get(variable)
    ]
    if missing:
        sys.stderr.write(
            "Configuration error: missing required runtime environment variable(s): "
            f"{', '.join(missing)}\n"
        )
        return 2

    sys.stdout.write("orchestrator_demo.app runtime configuration validated\n")
    return 0


__all__ = ["REQUIRED_RUNTIME_ENV_VARS", "main"]
