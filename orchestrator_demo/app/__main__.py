"""Runtime entrypoint for ``python -m orchestrator_demo.app``."""

from __future__ import annotations

import sys

from orchestrator_demo.app.bootstrap_llm import build_litellm_model
from orchestrator_demo.app.server import LocalOrchestratorApp, create_server
from orchestrator_demo.app.settings import ConfigurationError, Settings, load_settings
from orchestrator_demo.intent.classifier import LiteLlmIntentClassifier
from orchestrator_demo.orchestrator.service import OrchestratorService


def build_runtime_app(settings: Settings | None = None) -> LocalOrchestratorApp:
    """Build the HTTP app with the configured LiteLLM/OpenRouter classifier."""

    model = build_litellm_model(settings)
    return LocalOrchestratorApp(
        service=OrchestratorService(
            intent_classifier=LiteLlmIntentClassifier(model=model),
        )
    )


def main() -> int:
    """Validate runtime configuration and start the local HTTP app."""

    try:
        settings = load_settings()
        app = build_runtime_app(settings)
    except ConfigurationError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2

    host = settings.orchestrator_app_host
    port = settings.orchestrator_app_port
    server = create_server(host=host, port=port, app=app)
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
