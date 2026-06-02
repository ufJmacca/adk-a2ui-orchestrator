"""Runtime settings for model-provider configuration.

Settings are loaded explicitly by application entrypoints so importing the
package never requires runtime secrets.
"""

from pathlib import Path
from typing import Any

from pydantic import Field, SecretStr, ValidationError, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


REDACTED_SECRET = "**********"


class ConfigurationError(RuntimeError):
    """Raised when required runtime configuration is missing or invalid."""


class Settings(BaseSettings):
    """Environment-backed settings for OpenRouter-compatible LiteLLM access."""

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
        populate_by_name=True,
    )

    openrouter_api_key: SecretStr = Field(alias="OPENROUTER_API_KEY", repr=False)
    llm_model: str = Field(alias="LLM_MODEL")
    openrouter_api_base: str = Field(
        default="https://openrouter.ai/api/v1",
        alias="OPENROUTER_API_BASE",
    )
    or_app_name: str | None = Field(default=None, alias="OR_APP_NAME")
    or_site_url: str | None = Field(default=None, alias="OR_SITE_URL")
    orchestrator_app_host: str = Field(
        default="127.0.0.1",
        alias="ORCHESTRATOR_APP_HOST",
    )
    orchestrator_app_port: int = Field(
        default=8000,
        ge=0,
        le=65535,
        alias="ORCHESTRATOR_APP_PORT",
    )

    @field_validator("openrouter_api_key")
    @classmethod
    def require_non_empty_secret(cls, value: SecretStr) -> SecretStr:
        if not value.get_secret_value().strip():
            raise ValueError("OPENROUTER_API_KEY must be set")

        return value

    @field_validator("llm_model", "openrouter_api_base", "orchestrator_app_host")
    @classmethod
    def require_non_empty_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("value must not be empty")

        return value

    @field_validator("or_app_name", "or_site_url", mode="before")
    @classmethod
    def empty_optional_values_become_none(cls, value: Any) -> Any:
        if isinstance(value, str) and not value.strip():
            return None

        return value

    def redacted_diagnostics(self) -> dict[str, str | None]:
        """Return non-secret configuration values for diagnostics."""

        return {
            "OPENROUTER_API_KEY": REDACTED_SECRET,
            "LLM_MODEL": self.llm_model,
            "OPENROUTER_API_BASE": self.openrouter_api_base,
            "OR_APP_NAME": self.or_app_name,
            "OR_SITE_URL": self.or_site_url,
        }


_ENV_NAMES_BY_LOCATION = {
    "openrouter_api_key": "OPENROUTER_API_KEY",
    "OPENROUTER_API_KEY": "OPENROUTER_API_KEY",
    "llm_model": "LLM_MODEL",
    "LLM_MODEL": "LLM_MODEL",
}
_REQUIRED_ENV_ORDER = ["OPENROUTER_API_KEY", "LLM_MODEL"]
_REQUIRED_ENV_NAMES = set(_REQUIRED_ENV_ORDER)


def load_settings(env_file: str | Path | None = ".env") -> Settings:
    """Load settings and raise a redacted fail-fast error on failure."""

    redacted_error: ConfigurationError | None = None
    try:
        settings_kwargs: dict[str, Any] = {"_env_file": env_file}
        return Settings(**settings_kwargs)
    except ValidationError as exc:
        missing_or_invalid = _required_env_names_from_validation_error(exc)
        if missing_or_invalid:
            names = ", ".join(
                name for name in _REQUIRED_ENV_ORDER if name in missing_or_invalid
            )
            redacted_error = ConfigurationError(
                "Missing required runtime configuration: "
                f"{names}. Set these environment variables or provide a local "
                ".env file copied from .env.example."
            )
        else:
            redacted_error = ConfigurationError(
                "Invalid runtime configuration. Check environment variables and "
                ".env.example."
            )

    assert redacted_error is not None
    raise redacted_error


def _required_env_names_from_validation_error(exc: ValidationError) -> set[str]:
    required_names: set[str] = set()

    for error in exc.errors(include_input=False):
        location = ".".join(str(part) for part in error["loc"])
        env_name = _ENV_NAMES_BY_LOCATION.get(location)
        if env_name in _REQUIRED_ENV_NAMES:
            required_names.add(env_name)

    return required_names
