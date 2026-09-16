"""Provider configuration resolved from environment variables.

The point of this module: seven of the eight providers below speak the same
OpenAI-compatible HTTP API. Ollama on your laptop, llama.cpp, vLLM, Groq,
OpenRouter, Together and OpenAI itself differ only by base URL, model name and
whether a key is required. So "swap the model" is config, not code.

Anthropic is the one real exception -- different wire format, so it gets an
adapter in anthropic_client.py.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


@dataclass(frozen=True)
class ProviderPreset:
    base_url: str | None
    default_model: str
    api_key_env: str | None
    requires_key: bool
    openai_compatible: bool = True
    notes: str = ""


PRESETS: dict[str, ProviderPreset] = {
    # ---- local, no key -----------------------------------------------------
    "ollama": ProviderPreset(
        base_url="http://localhost:11434/v1",
        default_model="qwen2.5:7b-instruct",
        api_key_env=None,
        requires_key=False,
        notes="Easiest local option. Ollama exposes an OpenAI-compatible /v1 endpoint.",
    ),
    "llamacpp": ProviderPreset(
        base_url="http://localhost:8080/v1",
        default_model="local-model",
        api_key_env=None,
        requires_key=False,
        notes="`llama-server` from llama.cpp. One level below Ollama; you manage the GGUF yourself.",
    ),
    "vllm": ProviderPreset(
        base_url="http://localhost:8000/v1",
        default_model="Qwen/Qwen2.5-7B-Instruct",
        api_key_env=None,
        requires_key=False,
        notes="Production-grade local serving. Needs a real GPU -- not viable on this laptop.",
    ),
    "lmstudio": ProviderPreset(
        base_url="http://localhost:1234/v1",
        default_model="local-model",
        api_key_env=None,
        requires_key=False,
        notes="GUI alternative to Ollama.",
    ),
    # ---- hosted open-source models, free tier ------------------------------
    # Note on model names: hosted providers retire models on a regular cadence,
    # so any default here has a shelf life. `check_env.py` lists what the
    # provider currently serves -- trust that over this file.
    "groq": ProviderPreset(
        base_url="https://api.groq.com/openai/v1",
        default_model="openai/gpt-oss-120b",
        api_key_env="GROQ_API_KEY",
        requires_key=True,
        notes="Open weights, very fast, free tier. Good default for every lesson.",
    ),
    "openrouter": ProviderPreset(
        base_url="https://openrouter.ai/api/v1",
        default_model="qwen/qwen-2.5-72b-instruct",
        api_key_env="OPENROUTER_API_KEY",
        requires_key=True,
        notes="Many providers behind one API, including :free variants.",
    ),
    "together": ProviderPreset(
        base_url="https://api.together.xyz/v1",
        default_model="Qwen/Qwen2.5-7B-Instruct-Turbo",
        api_key_env="TOGETHER_API_KEY",
        requires_key=True,
        notes="Hosted open models, paid.",
    ),
    # ---- commercial --------------------------------------------------------
    "openai": ProviderPreset(
        base_url=None,  # the SDK default
        default_model="gpt-4o-mini",
        api_key_env="OPENAI_API_KEY",
        requires_key=True,
    ),
    "anthropic": ProviderPreset(
        base_url=None,
        default_model="claude-3-5-sonnet-20241022",
        api_key_env="ANTHROPIC_API_KEY",
        requires_key=True,
        openai_compatible=False,
        notes="Different wire format -- handled by a dedicated adapter.",
    ),
}

LOCAL_PROVIDERS = {"ollama", "llamacpp", "vllm", "lmstudio"}


class ConfigError(RuntimeError):
    """Raised for a misconfiguration a human can fix, with instructions."""


@dataclass(frozen=True)
class LLMConfig:
    provider: str
    model: str
    base_url: str | None
    api_key: str | None
    timeout: float
    openai_compatible: bool

    @property
    def is_local(self) -> bool:
        return self.provider in LOCAL_PROVIDERS

    @property
    def server_root(self) -> str | None:
        """Base URL without the trailing /v1, for native health endpoints."""
        if not self.base_url:
            return None
        return self.base_url.removesuffix("/").removesuffix("/v1")

    def describe(self) -> str:
        where = "local (CPU/GPU on this machine)" if self.is_local else "hosted API"
        return f"{self.provider}:{self.model}  [{where}]"


def load_config(
    provider: str | None = None,
    model: str | None = None,
    timeout: float | None = None,
) -> LLMConfig:
    """Build config from .env, with optional explicit overrides.

    Precedence: explicit argument > environment variable > preset default.
    Lessons call this with no arguments; tests and A/B comparisons pass
    overrides so one script can drive two models.
    """
    load_dotenv()  # no-op if .env is absent

    provider = (provider or os.getenv("LLM_PROVIDER") or "ollama").strip().lower()
    if provider not in PRESETS:
        raise ConfigError(
            f"Unknown LLM_PROVIDER={provider!r}.\n"
            f"Valid values: {', '.join(sorted(PRESETS))}"
        )

    preset = PRESETS[provider]
    model = (model or os.getenv("LLM_MODEL") or preset.default_model).strip()
    base_url = (os.getenv("LLM_BASE_URL") or preset.base_url) or None

    # Accept a generic LLM_API_KEY or the provider's conventional variable, so
    # people who already have OPENAI_API_KEY exported don't have to duplicate it.
    api_key = os.getenv("LLM_API_KEY")
    if not api_key and preset.api_key_env:
        api_key = os.getenv(preset.api_key_env)

    if preset.requires_key and not api_key:
        raise ConfigError(
            f"Provider {provider!r} needs an API key but none was found.\n"
            f"Set LLM_API_KEY (or {preset.api_key_env}) in your .env file.\n\n"
            f"No key? Stay on the local default instead:\n"
            f"    LLM_PROVIDER=ollama\n"
            f"    LLM_MODEL=qwen2.5:7b-instruct"
        )

    if timeout is None:
        timeout = float(os.getenv("LLM_TIMEOUT") or 180)

    return LLMConfig(
        provider=provider,
        model=model,
        base_url=base_url,
        # OpenAI-compatible servers still expect the header to exist, so send a
        # placeholder for keyless local servers.
        api_key=api_key or ("not-needed" if preset.openai_compatible else None),
        timeout=timeout,
        openai_compatible=preset.openai_compatible,
    )
