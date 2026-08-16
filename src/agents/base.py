"""Backend protocol and shared plumbing for the unified agent stack.

A backend is a thin adapter around one provider API. It receives the
already-rendered prompt text (system + user parts) and the role's JSON
schema, and returns ``(parsed_dict, usage_dict, latency_ms)``.

Invariants (these are what make the backend factorial meaningful):

- The adapter NEVER edits prompt text. Task wording lives only in
  src/agents/prompts/, byte-identical across backends; the adapter
  only packages it into the provider's request shape.
- Images ride as separate request parts (base64 block on Anthropic,
  data-URI image_url on OpenAI-compatible, inline_data on Gemini);
  the text parts are unaffected by whether an image is attached.
- Retries live OUTSIDE, in the planner's existing capped
  verifier-rejection loop. ``send`` performs exactly one logical call.
- Usage comes from the provider response via
  src.results.calls.extract_usage and is returned to the caller so
  CallLog token counts are real (MAPG-02 follow-through), never
  estimated.
- Provider SDKs import lazily at transport time; this module and every
  ``build_request`` stay importable and testable with no SDK
  installed (host test suite).

Sampling parameters are uniform across backends on purpose:
``temperature=0.1`` (the legacy gemini setting; legacy claude/openai
used provider defaults, which differ per provider and would confound
the factorial) and ``max_tokens=2048`` (the legacy claude/openai
setting; Anthropic requires an explicit value).
"""

import base64
import json
import mimetypes
import time
from typing import Any, Dict, List, Optional, Tuple

from src.results.calls import extract_cache_usage, extract_usage

DEFAULT_MAX_TOKENS = 2048
DEFAULT_TEMPERATURE = 0.1


class BackendError(RuntimeError):
    """Raised by adapters for transport or configuration failures."""


def text_part(text: str, cache: bool = False) -> Dict[str, Any]:
    """A text user-part. ``cache=True`` marks it as the end of a
    stable prefix (MAPG-10): the claude adapter turns the mark into a
    ``cache_control: ephemeral`` breakpoint; openai/gemini adapters
    ignore it because their prefix caching is provider-automatic. The
    mark is request structure, never text: prompt bytes are identical
    with or without it (golden-tested)."""
    part: Dict[str, Any] = {"type": "text", "text": str(text)}
    if cache:
        part["cache"] = True
    return part


def image_part(path: str) -> Dict[str, Any]:
    """An image user-part by file path; encoded at build_request time."""
    return {"type": "image_path", "path": str(path)}


def encode_image_b64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def guess_mime(path: str) -> str:
    return mimetypes.guess_type(path)[0] or "image/png"


def strip_json_fences(text: str) -> str:
    """The legacy claude-family fence splitter, kept byte-compatible."""
    text = text.strip()
    if "```json" in text:
        text = text.split("```json")[-1].split("```")[0].strip()
    elif "```" in text:
        text = text.split("```")[-1].split("```")[0].strip()
    return text


def parse_json_reply(raw_text: str) -> Dict[str, Any]:
    parsed = json.loads(strip_json_fences(raw_text))
    if not isinstance(parsed, dict):
        raise ValueError(
            f"model reply is JSON but not an object: {type(parsed).__name__}"
        )
    return parsed


def usage_dict(response: Any) -> Dict[str, Optional[int]]:
    """Provider usage as the dict shape CallLog's extract_usage reads.

    Cache token counts (MAPG-10) are included only when the provider
    reported them, so the dict shape is unchanged for providers and
    tests that do not surface caching."""
    prompt_tokens, completion_tokens = extract_usage(response)
    out: Dict[str, Optional[int]] = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
    }
    cache_read, cache_write = extract_cache_usage(response)
    if cache_read is not None:
        out["cache_read_tokens"] = cache_read
    if cache_write is not None:
        out["cache_write_tokens"] = cache_write
    return out


class Backend:
    """One provider adapter. Subclasses set ``provider`` and implement
    ``build_request`` (pure, SDK-free) and ``_transport`` (does the
    lazy SDK import and the network call, returns (raw_text, usage))."""

    provider: str = "base"

    def __init__(self, model_name: str):
        self.model_name = str(model_name)

    # ------------------------------------------------------------------
    # Pure request shaping (unit-testable without any SDK)
    # ------------------------------------------------------------------
    def build_request(
        self,
        system: str,
        user_parts: List[Dict[str, Any]],
        schema: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Network
    # ------------------------------------------------------------------
    def _transport(self, request: Dict[str, Any]) -> Tuple[str, Dict[str, Optional[int]]]:
        raise NotImplementedError

    def send(
        self,
        system: str,
        user_parts: List[Dict[str, Any]],
        schema: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Optional[int]], float]:
        """One call: (parsed dict, usage dict, latency_ms).

        Raises on transport errors and unparseable replies; schema
        validation happens in the role (so the failure is counted with
        the role's error shape).
        """
        request = self.build_request(system, user_parts, schema)
        t0 = time.perf_counter()
        raw_text, usage = self._transport(request)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        parsed = parse_json_reply(raw_text)
        return parsed, usage, latency_ms
