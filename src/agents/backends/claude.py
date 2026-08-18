"""Anthropic adapter.

Request shape matches the legacy claude family: system string,
one user message whose content is text blocks plus optional base64
image source blocks, ``max_tokens=2048``. Temperature is pinned to the
stack-wide DEFAULT_TEMPERATURE (the legacy family left it at the
provider default, which differed from gemini's 0.1; see base.py).

Schema enforcement: the schema text already sits in the rendered
system prompt (identically for every backend); replies are fence-split
and json-parsed here, then validated in the role. The ``schema``
argument is accepted for interface parity and recorded in the request
for debugging, but the Messages API has no JSON mode to hand it to.
"""

import os
from typing import Any, Dict, List, Optional, Tuple

from src.agents.base import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_TEMPERATURE,
    Backend,
    BackendError,
    encode_image_b64,
    guess_mime,
    usage_dict,
)

DEFAULT_MODEL = "claude-opus-4-6"
API_KEY_ENVS = ("CLAUDE_API_KEY", "ANTHROPIC_API_KEY")


class ClaudeBackend(Backend):
    provider = "claude"

    def __init__(self, model_name: str = DEFAULT_MODEL):
        super().__init__(model_name)
        self._client = None

    # ------------------------------------------------------------------
    def build_request(
        self,
        system: str,
        user_parts: List[Dict[str, Any]],
        schema: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        content: List[Dict[str, Any]] = []
        for part in user_parts:
            if part.get("type") == "text":
                content.append({"type": "text", "text": part["text"]})
            elif part.get("type") == "image_path":
                path = part["path"]
                content.append(
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": guess_mime(path),
                            "data": encode_image_b64(path),
                        },
                    }
                )
            else:
                raise BackendError(f"unknown user part type: {part.get('type')!r}")
        return {
            "model": self.model_name,
            "max_tokens": DEFAULT_MAX_TOKENS,
            "temperature": DEFAULT_TEMPERATURE,
            "system": system,
            "messages": [{"role": "user", "content": content}],
        }

    # ------------------------------------------------------------------
    def _get_client(self):
        if self._client is None:
            import anthropic

            api_key = next(
                (os.environ[k] for k in API_KEY_ENVS if os.environ.get(k)), None
            )
            if not api_key:
                raise BackendError(
                    "CLAUDE_API_KEY (or ANTHROPIC_API_KEY) must be set in the environment."
                )
            self._client = anthropic.Anthropic(api_key=api_key)
        return self._client

    def _transport(self, request: Dict[str, Any]) -> Tuple[str, Dict[str, Optional[int]]]:
        response = self._get_client().messages.create(**request)
        return response.content[0].text, usage_dict(response)
