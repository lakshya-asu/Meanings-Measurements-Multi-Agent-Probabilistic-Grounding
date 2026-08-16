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

Prompt caching (MAPG-10): the system text is sent as a content block
with ``cache_control: {"type": "ephemeral"}``, and any user text part
marked ``cache: True`` (the stable scene-graph prefix emitted by
render_parts) gets the same annotation. Anthropic caches the byte
prefix up to each breakpoint at 0.1x read pricing (write 1.25x, 5 min
TTL); prefixes below the model's minimum cacheable length are simply
not cached, so the annotation is safe unconditionally. cache_control
is request structure, not text: the prompt bytes every provider sees
stay identical (golden-tested).
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

# MAPG-25: Anthropic models from Opus 4.7 onward reject a non-default
# temperature (and top_p / top_k) with a 400. Older models, including
# the pinned claude-opus-4-6 and the claude-haiku-4-5 cheap tier,
# accept them.
#
# This is deliberately an ALLOW-list of families known to accept
# sampling params, not a deny-list of families known to reject them. A
# deny-list silently breaks on the next model release, and the failure
# mode is a hard 400 mid-run. An allow-list fails the safe way: an
# unrecognized model simply omits temperature and still runs, at the
# provider default.
#
# Anything matched here keeps sending temperature exactly as before,
# so the pinned arm's request bytes are unchanged and MAPG-09's
# cross-backend confound fix and any collected results still hold.
SAMPLING_PARAM_FAMILIES = (
    "claude-opus-4-6",
    "claude-sonnet-4-6",
    "claude-haiku-4-5",
    "claude-3",
)


def accepts_sampling_params(model_name: Any) -> bool:
    """True if this model accepts an explicit temperature.

    Matching is by family prefix, so a dated snapshot id such as
    claude-haiku-4-5-20251001 resolves to its family.
    """
    name = str(model_name or "").strip().lower()
    if not name:
        return False
    return any(name.startswith(f) for f in SAMPLING_PARAM_FAMILIES)


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
                block: Dict[str, Any] = {"type": "text", "text": part["text"]}
                if part.get("cache"):
                    block["cache_control"] = {"type": "ephemeral"}
                content.append(block)
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
        req: Dict[str, Any] = {
            "model": self.model_name,
            "max_tokens": DEFAULT_MAX_TOKENS,
            # System as a block list so the stable system text carries a
            # cache breakpoint; the text bytes are unchanged.
            "system": [
                {
                    "type": "text",
                    "text": system,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            "messages": [{"role": "user", "content": content}],
        }
        # MAPG-25: omitted rather than defaulted for models that reject
        # it, since there is no value (0.1 or otherwise) those models
        # accept. Key order keeps temperature next to max_tokens for
        # readability in logged requests.
        if accepts_sampling_params(self.model_name):
            req["temperature"] = DEFAULT_TEMPERATURE
        return req

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
