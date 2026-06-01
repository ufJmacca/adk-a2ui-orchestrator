"""Runtime entrypoint for ``python -m orchestrator_demo.app``."""

from __future__ import annotations

import os
import sys

from orchestrator_demo.app.bootstrap_llm import build_litellm_model
from orchestrator_demo.app.server import create_server
from orchestrator_demo.app.settings import ConfigurationError


def main() -> int:
    """Validate runtime configuration and start the local HTTP app."""

    try:
        build_litellm_model()
    except ConfigurationError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2

    host = os.environ.get("ORCHESTRATOR_APP_HOST", "127.0.0.1")
    port = int(os.environ.get("ORCHESTRATOR_APP_PORT", "8000"))
    server = create_server(host=host, port=port)
    print(f"Local orchestrator app listening on {server.base_url}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
