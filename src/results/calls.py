"""Per-call LLM accounting (MAPG-02).

The runners used to hardcode ``vlm_calls += 4`` per step while the
planner loop actually makes 1, 2, or 4 calls depending on path, and
retries did not exist as a concept. CallLog records one row per actual
LLM invocation at the planner call sites; the episode rollup is the
row count, retries included.

Token counts come from provider usage when the invocation result
exposes it, else None; they are never estimated. Today the
src/multi_agent agent classes drop the provider response object and
return a parsed dict, so prompt/completion tokens are None at those
call sites until MAPG-09 unifies the agents behind adapters that
return usage. The Gemini planner (vlm_planner_msp) builds its
responses in-file and records real usage_metadata token counts.

Recognized usage shapes (duck-typed, no SDK imports):

- Anthropic:  response.usage.input_tokens / output_tokens
- OpenAI:     response.usage.prompt_tokens / completion_tokens
- Gemini:     response.usage_metadata.prompt_token_count /
              candidates_token_count
- dicts carrying a "usage" dict with any of those key pairs

Stdlib only.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

_PROMPT_KEYS = ("input_tokens", "prompt_tokens", "prompt_token_count")
_COMPLETION_KEYS = (
    "output_tokens",
    "completion_tokens",
    "candidates_token_count",
)

# MAPG-10: cached-token fields, where providers report them.
# - Anthropic: usage.cache_read_input_tokens / cache_creation_input_tokens
# - OpenAI:    usage.prompt_tokens_details.cached_tokens (read only)
# - Gemini:    usage_metadata.cached_content_token_count (read only)
# - normalized dicts from src.agents.base.usage_dict carry
#   cache_read_tokens / cache_write_tokens
_CACHE_READ_KEYS = (
    "cache_read_input_tokens",
    "cached_content_token_count",
    "cache_read_tokens",
    "cached_tokens",
)
_CACHE_WRITE_KEYS = ("cache_creation_input_tokens", "cache_write_tokens")


def _as_opt_int(value: Any) -> Optional[int]:
    """Coerce to int or return None. Never raises."""
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _first_key(container: Any, keys: Tuple[str, ...]) -> Optional[int]:
    for key in keys:
        if isinstance(container, dict):
            value = container.get(key)
        else:
            value = getattr(container, key, None)
        value = _as_opt_int(value)
        if value is not None:
            return value
    return None


def extract_usage(result: Any) -> Tuple[Optional[int], Optional[int]]:
    """(prompt_tokens, completion_tokens) from a response-ish object.

    Accepts provider response objects (Anthropic / OpenAI ``.usage``,
    Gemini ``.usage_metadata``) and plain dicts carrying a "usage"
    dict. Returns (None, None) when nothing usable is exposed.
    """
    if result is None:
        return None, None
    candidates = []
    if isinstance(result, dict):
        usage = result.get("usage")
        if usage is not None:
            candidates.append(usage)
    else:
        for attr in ("usage", "usage_metadata"):
            usage = getattr(result, attr, None)
            if usage is not None:
                candidates.append(usage)
        # A raw usage object itself (e.g. resp.usage passed directly).
        candidates.append(result)
    for usage in candidates:
        prompt = _first_key(usage, _PROMPT_KEYS)
        completion = _first_key(usage, _COMPLETION_KEYS)
        if prompt is not None or completion is not None:
            return prompt, completion
    return None, None


def _usage_candidates(result: Any) -> List[Any]:
    """The same usage-container search order extract_usage walks."""
    if result is None:
        return []
    candidates: List[Any] = []
    if isinstance(result, dict):
        usage = result.get("usage")
        if usage is not None:
            candidates.append(usage)
        # A bare usage dict passed directly.
        candidates.append(result)
    else:
        for attr in ("usage", "usage_metadata"):
            usage = getattr(result, attr, None)
            if usage is not None:
                candidates.append(usage)
        candidates.append(result)
    return candidates


def extract_cache_usage(result: Any) -> Tuple[Optional[int], Optional[int]]:
    """(cache_read_tokens, cache_write_tokens) where providers report
    them (MAPG-10), else (None, None). Never estimated.

    Handles the Anthropic flat fields, the Gemini
    ``cached_content_token_count``, the OpenAI nested
    ``prompt_tokens_details.cached_tokens``, and dicts already
    normalized by src.agents.base.usage_dict.
    """
    for usage in _usage_candidates(result):
        read = _first_key(usage, _CACHE_READ_KEYS)
        write = _first_key(usage, _CACHE_WRITE_KEYS)
        if read is None:
            # OpenAI nests the cached count one level down.
            details = (
                usage.get("prompt_tokens_details")
                if isinstance(usage, dict)
                else getattr(usage, "prompt_tokens_details", None)
            )
            if details is not None:
                read = _first_key(details, ("cached_tokens",))
        if read is not None or write is not None:
            return read, write
    return None, None


def model_name_of(agent: Any) -> Optional[str]:
    """Best-effort model name from an agent instance.

    The claude/openai/alibaba agent families store ``model_name``; the
    gemini family stores a ``GenerativeModel`` on ``.model`` which
    itself exposes ``model_name``. None when neither is present.
    """
    name = getattr(agent, "model_name", None)
    if name:
        return str(name)
    inner = getattr(agent, "model", None)
    name = getattr(inner, "model_name", None) if inner is not None else None
    return str(name) if name else None


@dataclass
class CallRecord:
    """One LLM invocation. Token fields are None when the provider
    usage was not exposed to the call site (never estimated)."""

    role: str
    model_name: Optional[str] = None
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    is_retry: bool = False
    latency_ms: Optional[float] = None
    step_idx: Optional[int] = None
    # MAPG-10: provider-reported cached token counts, None when the
    # provider did not report them (never estimated).
    cache_read_tokens: Optional[int] = None
    cache_write_tokens: Optional[int] = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "role": self.role,
            "model_name": self.model_name,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "is_retry": bool(self.is_retry),
            "latency_ms": self.latency_ms,
            "step_idx": self.step_idx,
            "cache_read_tokens": self.cache_read_tokens,
            "cache_write_tokens": self.cache_write_tokens,
        }


class CallLog:
    """Append-only per-episode log of actual LLM invocations.

    One CallLog per planner instance (one episode). The episode
    ``vlm_calls`` rollup is ``total()``, retries included.
    """

    def __init__(self) -> None:
        self._records: List[CallRecord] = []

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------
    def record(
        self,
        role: str,
        *,
        model_name: Optional[str] = None,
        prompt_tokens: Optional[int] = None,
        completion_tokens: Optional[int] = None,
        is_retry: bool = False,
        latency_ms: Optional[float] = None,
        step_idx: Optional[int] = None,
        cache_read_tokens: Optional[int] = None,
        cache_write_tokens: Optional[int] = None,
    ) -> CallRecord:
        rec = CallRecord(
            role=str(role),
            model_name=model_name,
            prompt_tokens=_as_opt_int(prompt_tokens),
            completion_tokens=_as_opt_int(completion_tokens),
            is_retry=bool(is_retry),
            latency_ms=(float(latency_ms) if latency_ms is not None else None),
            step_idx=(int(step_idx) if step_idx is not None else None),
            cache_read_tokens=_as_opt_int(cache_read_tokens),
            cache_write_tokens=_as_opt_int(cache_write_tokens),
        )
        self._records.append(rec)
        return rec

    def call(
        self,
        role: str,
        fn: Callable[..., Any],
        *args: Any,
        model_name: Optional[str] = None,
        is_retry: bool = False,
        step_idx: Optional[int] = None,
        record_if: Optional[Callable[[Any], bool]] = None,
        **kwargs: Any,
    ) -> Any:
        """Invoke ``fn`` and record it as one LLM call.

        Latency is measured around the invocation; token counts are
        pulled from the result when it exposes provider usage (see
        extract_usage). ``record_if(result)`` can veto the row for
        invocations that turn out not to hit the LLM at all (e.g. the
        programmatic-only verifier path, which sets llm_used=False).
        An exception is still recorded as a call (the API was
        invoked; tokens unknown) and then re-raised.
        """
        t0 = time.perf_counter()
        try:
            result = fn(*args, **kwargs)
        except Exception as exc:
            # A transport may have completed and exposed provider usage
            # before response parsing failed. BackendReplyError keeps that
            # usage on the exception so the paid call remains exact rather
            # than falling back to an estimate in the cost governor.
            prompt_tokens, completion_tokens = extract_usage(exc)
            cache_read_tokens, cache_write_tokens = extract_cache_usage(exc)
            self.record(
                role,
                model_name=model_name,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                is_retry=is_retry,
                latency_ms=(time.perf_counter() - t0) * 1000.0,
                step_idx=step_idx,
                cache_read_tokens=cache_read_tokens,
                cache_write_tokens=cache_write_tokens,
            )
            raise
        latency_ms = (time.perf_counter() - t0) * 1000.0
        if record_if is not None and not record_if(result):
            return result
        prompt_tokens, completion_tokens = extract_usage(result)
        cache_read_tokens, cache_write_tokens = extract_cache_usage(result)
        self.record(
            role,
            model_name=model_name,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            is_retry=is_retry,
            latency_ms=latency_ms,
            step_idx=step_idx,
            cache_read_tokens=cache_read_tokens,
            cache_write_tokens=cache_write_tokens,
        )
        return result

    # ------------------------------------------------------------------
    # Rollups
    # ------------------------------------------------------------------
    def total(self) -> int:
        """Episode vlm_calls: every recorded call, retries included."""
        return len(self._records)

    def retries(self) -> int:
        return sum(1 for r in self._records if r.is_retry)

    def records(self) -> List[CallRecord]:
        return list(self._records)

    def rows(self) -> List[Dict[str, Any]]:
        """Dicts in call order, ready for ResultsStore.record_calls."""
        return [r.as_dict() for r in self._records]
