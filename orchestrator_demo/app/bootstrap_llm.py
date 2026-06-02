"""LiteLLM/OpenRouter bootstrap helpers."""

from collections.abc import Callable, MutableMapping
import os
from typing import Any

from orchestrator_demo.app.settings import Settings, load_settings


ModelFactory = Callable[[str], Any]


def configure_litellm_environment(
    settings: Settings,
    *,
    environ: MutableMapping[str, str] | None = None,
) -> MutableMapping[str, str]:
    """Populate the environment variables expected by LiteLLM/OpenRouter."""

    target_environ = os.environ if environ is None else environ
    target_environ["OPENROUTER_API_KEY"] = (
        settings.openrouter_api_key.get_secret_value()
    )
    target_environ["LLM_MODEL"] = settings.llm_model
    target_environ["OPENROUTER_API_BASE"] = settings.openrouter_api_base

    if settings.or_app_name is not None:
        target_environ["OR_APP_NAME"] = settings.or_app_name
    if settings.or_site_url is not None:
        target_environ["OR_SITE_URL"] = settings.or_site_url

    return target_environ


def build_litellm_model(
    settings: Settings | None = None,
    *,
    model_factory: ModelFactory | None = None,
    environ: MutableMapping[str, str] | None = None,
    configure_environment: bool = True,
) -> Any:
    """Build a LiteLLM model, with injectable factory for deterministic tests."""

    resolved_settings = settings if settings is not None else load_settings()
    if configure_environment:
        configure_litellm_environment(resolved_settings, environ=environ)

    factory = model_factory if model_factory is not None else _default_model_factory
    return factory(resolved_settings.llm_model)


def _default_model_factory(model: str) -> Any:
    from google.adk.models.lite_llm import LiteLlm

    return LiteLlm(model=model)
