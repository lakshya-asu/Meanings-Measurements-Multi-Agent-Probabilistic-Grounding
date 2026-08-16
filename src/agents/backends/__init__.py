"""Thin LLM adapters for the unified agent stack (MAPG-09).

Three client shapes cover the four legacy families: alibaba is the
OpenAI client with a DashScope base_url, exactly as before.
"""

from typing import Optional

from src.agents.backends.claude import ClaudeBackend
from src.agents.backends.gemini import GeminiBackend
from src.agents.backends.openai_compat import OpenAICompatBackend

PROVIDER_ALIASES = {
    "claude": "claude",
    "anthropic": "claude",
    "openai": "openai",
    "alibaba": "alibaba",
    "qwen": "alibaba",
    "gemini": "gemini",
    "google": "gemini",
}


def create_backend(provider: str, model_name: Optional[str] = None):
    """Backend instance for a provider name.

    Unknown provider names fall back to gemini, matching the legacy
    AgentFactory's default branch.
    """
    key = PROVIDER_ALIASES.get(str(provider).lower().strip(), "gemini")
    if key == "claude":
        return ClaudeBackend(model_name) if model_name else ClaudeBackend()
    if key in ("openai", "alibaba"):
        return (
            OpenAICompatBackend(provider=key, model_name=model_name)
            if model_name
            else OpenAICompatBackend(provider=key)
        )
    return GeminiBackend(model_name) if model_name else GeminiBackend()
