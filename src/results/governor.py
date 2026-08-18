"""Cost governor: hard per-provider USD caps (MAPG-11).

EXPERIMENT_PLAN section 2 hard requirement: no LLM run without a cost
governor. Spend accrues from CallLog rows (MAPG-02); every cap in cfg
``cost_caps`` (per provider, plus ``total``) is a hard limit. On breach
the governor raises CostCapExceeded; the runners catch it, mark the run
aborted in the store with the breach detail in the manifest, flush what
was recorded so far, and exit nonzero. Nothing silent.

Pricing is a pinned $/Mtok table in cfg ``model_prices`` and is
REQUIRED for any provider that has a cap: unpriceable spend cannot be
governed, so an unpriced model under a capped provider fails loudly
(GovernorConfigError) instead of accruing $0. Calls whose provider did
not expose token counts accrue a conservative estimate from cfg
``fallback_tokens_per_call`` ({prompt: 6000, completion: 400} by
default) and are additionally tracked as estimated spend.

Stdlib only, same contract as the rest of src/results.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Tuple

#: Conservative per-call token estimate used when a CallLog row has no
#: provider-reported token counts. Overridable via cfg
#: fallback_tokens_per_call: {prompt: ..., completion: ...}.
DEFAULT_FALLBACK_TOKENS = {"prompt": 6000, "completion": 400}

#: Model-name prefix -> provider (the cap keys in cfg cost_caps).
_PROVIDER_PREFIXES: Tuple[Tuple[str, str], ...] = (
    ("claude", "claude"),
    ("anthropic", "claude"),
    ("gpt", "openai"),
    ("openai", "openai"),
    ("o1", "openai"),
    ("o3", "openai"),
    ("o4", "openai"),
    ("gemini", "gemini"),
    ("google", "gemini"),
    ("qwen", "qwen"),
)

_PRICE_IN_KEYS = ("input", "in", "prompt")
_PRICE_OUT_KEYS = ("output", "out", "completion")


class GovernorConfigError(RuntimeError):
    """Cost governor cannot govern: bad caps, bad prices, or an
    unpriceable model. Raised at run start or at first accrual, never
    swallowed."""


class CostCapExceeded(RuntimeError):
    """A hard cost cap was breached. ``scope`` is a provider name or
    'total'. ``detail()`` is manifest-ready."""

    def __init__(self, scope: str, cap_usd: float, spend_usd: float,
                 summary: Dict[str, Any]):
        self.scope = str(scope)
        self.cap_usd = float(cap_usd)
        self.spend_usd = float(spend_usd)
        self.summary = summary
        super().__init__(
            "cost cap breached for '{}': spend ${:.4f} exceeds cap "
            "${:.2f}".format(self.scope, self.spend_usd, self.cap_usd)
        )

    def detail(self) -> Dict[str, Any]:
        """Breach record for the run manifest."""
        return {
            "scope": self.scope,
            "cap_usd": self.cap_usd,
            "spend_usd": self.spend_usd,
            "summary": self.summary,
        }


# ---------------------------------------------------------------------------
# Pure helpers (also used by preflight)
# ---------------------------------------------------------------------------

def _strip_models_prefix(name: str) -> str:
    return name[len("models/"):] if name.startswith("models/") else name


def provider_of(model_name: Any) -> Optional[str]:
    """Map a model name to its provider cap key, or None.

    'models/gemini-2.5-pro' -> 'gemini', 'claude-opus-4-6' -> 'claude',
    'gpt-5.2-chat-latest' -> 'openai', 'qwen3-vl-plus' -> 'qwen'.
    A prefix only matches at a word boundary (end of name, or a digit /
    separator right after), so 'googleplex' is not 'google'.
    """
    if not model_name:
        return None
    name = _strip_models_prefix(str(model_name).strip().lower())
    for prefix, provider in _PROVIDER_PREFIXES:
        if name == prefix:
            return provider
        if name.startswith(prefix):
            nxt = name[len(prefix)]
            if nxt in "-_./0123456789":
                return provider
    return None


def normalize_price_row(row: Any) -> Tuple[float, float]:
    """($/Mtok input, $/Mtok output) from a cfg model_prices row.

    Accepts {input, output} (also in/out, prompt/completion) or a
    two-element list. Raises ValueError on anything unusable, empty
    placeholders included: a price that does not parse is not pinned.
    """
    if isinstance(row, dict):
        pin = next((row[k] for k in _PRICE_IN_KEYS if k in row), None)
        pout = next((row[k] for k in _PRICE_OUT_KEYS if k in row), None)
    elif isinstance(row, (list, tuple)) and len(row) == 2:
        pin, pout = row
    else:
        raise ValueError(f"unusable price row {row!r}")
    try:
        pin_f = float(pin)
        pout_f = float(pout)
    except (TypeError, ValueError):
        raise ValueError(
            f"non-numeric price row {row!r}; fill pinned $/Mtok values"
        )
    if pin_f < 0 or pout_f < 0:
        raise ValueError(f"negative price in row {row!r}")
    return pin_f, pout_f


def resolve_price_key(prices: Dict[str, Any], model_name: Any) -> Optional[str]:
    """The price-table key covering ``model_name``, or None.

    Exact match first (with and without a 'models/' prefix), then the
    longest table key that is a boundary prefix of the model name, so a
    snapshot id like 'gemini-2.5-pro-001' is covered by a pinned row
    'gemini-2.5-pro'.
    """
    if not model_name:
        return None
    raw = str(model_name).strip()
    name = _strip_models_prefix(raw)
    keyed = {_strip_models_prefix(str(k).strip()): str(k) for k in prices}
    if name in keyed:
        return keyed[name]
    best = None
    for stripped, original in keyed.items():
        if not stripped or not name.startswith(stripped):
            continue
        if len(name) > len(stripped) and name[len(stripped)] not in "-_./":
            continue
        if best is None or len(stripped) > len(_strip_models_prefix(best)):
            best = original
    return best


def _plain(obj: Any) -> Any:
    """Duck-typed cfg node -> plain dict/list/scalar (no omegaconf import)."""
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if hasattr(obj, "items"):
        return {str(k): _plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_plain(v) for v in obj]
    return obj


def _get(rec: Any, key: str) -> Any:
    if isinstance(rec, dict):
        return rec.get(key)
    return getattr(rec, key, None)


# ---------------------------------------------------------------------------
# Governor
# ---------------------------------------------------------------------------

class CostGovernor:
    """Accrues USD spend from CallLog rows and enforces hard caps.

    caps: {provider: usd, ..., 'total': usd} from cfg cost_caps.
    prices: {model_name: {input: $/Mtok, output: $/Mtok}} from cfg
    model_prices. fallback_tokens_per_call fills in for rows without
    provider token counts; that spend is charged AND tracked separately
    as estimated.
    """

    def __init__(self, caps: Dict[str, Any], prices: Dict[str, Any],
                 fallback_tokens_per_call: Optional[Dict[str, Any]] = None):
        caps = _plain(caps)
        prices = _plain(prices) or {}
        if not isinstance(caps, dict) or not caps:
            raise GovernorConfigError(
                "cost_caps must be a non-empty mapping of provider -> USD cap"
            )
        self.caps: Dict[str, float] = {}
        for key, value in caps.items():
            try:
                cap = float(value)
            except (TypeError, ValueError):
                raise GovernorConfigError(
                    f"cost cap for '{key}' is not a number: {value!r}"
                )
            if cap < 0:
                raise GovernorConfigError(f"cost cap for '{key}' is negative")
            self.caps[str(key).strip().lower()] = cap

        self.prices: Dict[str, Tuple[float, float]] = {}
        for model, row in (prices or {}).items():
            try:
                self.prices[str(model)] = normalize_price_row(row)
            except ValueError as e:
                raise GovernorConfigError(
                    f"model_prices['{model}']: {e}"
                )

        fallback = _plain(fallback_tokens_per_call) or dict(DEFAULT_FALLBACK_TOKENS)
        try:
            self.fallback_prompt = int(fallback["prompt"])
            self.fallback_completion = int(fallback["completion"])
        except (KeyError, TypeError, ValueError):
            raise GovernorConfigError(
                "fallback_tokens_per_call needs integer 'prompt' and "
                f"'completion' entries, got {fallback!r}"
            )

        self._spend: Dict[str, float] = {}
        self._estimated_spend: Dict[str, float] = {}
        self._calls = 0
        self._estimated_calls = 0
        self._tracked: Any = None
        self._tracked_count = 0

    # ------------------------------------------------------------------
    # Construction from cfg
    # ------------------------------------------------------------------
    @classmethod
    def from_cfg(cls, cfg: Any) -> Optional["CostGovernor"]:
        """Governor from a run cfg, or None when cfg has no cost_caps.

        Raises GovernorConfigError (loudly, at run start) when caps are
        present but the price table or fallback is unusable.
        """
        get = cfg.get if hasattr(cfg, "get") else lambda k, d=None: getattr(cfg, k, d)
        caps = get("cost_caps", None)
        if not caps:
            return None
        return cls(
            caps,
            get("model_prices", None) or {},
            get("fallback_tokens_per_call", None),
        )

    # ------------------------------------------------------------------
    # Pricing
    # ------------------------------------------------------------------
    def _price_for(self, model_name: str, provider: str) -> Tuple[float, float]:
        key = resolve_price_key(self.prices, model_name)
        if key is not None:
            return self.prices[key]
        raise GovernorConfigError(
            f"no model_prices row covers '{model_name}' (provider "
            f"'{provider}' is governed): unpriceable spend cannot be "
            "governed. Pin its $/Mtok prices in cfg model_prices."
        )

    def _governed(self, provider: Optional[str]) -> bool:
        if "total" in self.caps:
            return True
        return provider is not None and provider in self.caps

    def validate_models(self, model_names: Iterable[Any]) -> None:
        """Fail loudly at run start if any model cannot be priced."""
        problems: List[str] = []
        for name in model_names:
            provider = provider_of(name)
            if provider is None:
                problems.append(f"'{name}' maps to no known provider")
                continue
            if not self._governed(provider):
                continue
            if resolve_price_key(self.prices, name) is None:
                problems.append(f"'{name}' has no model_prices row")
        if problems:
            raise GovernorConfigError(
                "cost governor cannot price these models: "
                + "; ".join(problems)
            )

    # ------------------------------------------------------------------
    # Accrual
    # ------------------------------------------------------------------
    def charge_rows(self, rows: Iterable[Any]) -> None:
        """Accrue spend from CallLog rows (dicts or CallRecords), then
        enforce every cap. Raises CostCapExceeded on breach and
        GovernorConfigError on an unpriceable row."""
        for row in rows:
            model_name = _get(row, "model_name")
            provider = provider_of(model_name)
            if provider is None:
                raise GovernorConfigError(
                    f"call row has model_name {model_name!r}, which maps "
                    "to no known provider; spend cannot be attributed or "
                    "governed"
                )
            if not self._governed(provider):
                continue
            pin, pout = self._price_for(str(model_name), provider)
            prompt_tokens = _get(row, "prompt_tokens")
            completion_tokens = _get(row, "completion_tokens")
            estimated = prompt_tokens is None or completion_tokens is None
            if prompt_tokens is None:
                prompt_tokens = self.fallback_prompt
            if completion_tokens is None:
                completion_tokens = self.fallback_completion
            cost = (int(prompt_tokens) * pin + int(completion_tokens) * pout) / 1e6
            self._spend[provider] = self._spend.get(provider, 0.0) + cost
            self._calls += 1
            if estimated:
                self._estimated_spend[provider] = (
                    self._estimated_spend.get(provider, 0.0) + cost
                )
                self._estimated_calls += 1
        self.check_caps()

    def track(self, call_log: Any) -> None:
        """Watch one episode's CallLog; charge_tracked() accrues only
        the rows added since the last charge."""
        self._tracked = call_log
        self._tracked_count = 0

    def charge_tracked(self) -> None:
        """Charge rows the tracked CallLog gained since the last call,
        then enforce caps. With no tracked log this is just a check."""
        if self._tracked is None:
            self.check_caps()
            return
        rows = self._tracked.rows()
        new_rows = rows[self._tracked_count:]
        self._tracked_count = len(rows)
        self.charge_rows(new_rows)

    # ------------------------------------------------------------------
    # Enforcement and reporting
    # ------------------------------------------------------------------
    def total_spend(self) -> float:
        return sum(self._spend.values())

    def spend(self, provider: Optional[str] = None) -> float:
        if provider is None:
            return self.total_spend()
        return self._spend.get(provider, 0.0)

    def check_caps(self) -> None:
        """Raise CostCapExceeded if any per-provider or total cap is
        strictly exceeded."""
        for scope, cap in self.caps.items():
            spent = self.total_spend() if scope == "total" else self._spend.get(scope, 0.0)
            if spent > cap:
                raise CostCapExceeded(scope, cap, spent, self.summary())

    def summary(self) -> Dict[str, Any]:
        """Manifest-ready spend snapshot. Estimated spend (fallback
        token counts) is a subset of spend_usd, reported separately."""
        return {
            "caps_usd": dict(self.caps),
            "spend_usd": {k: round(v, 6) for k, v in sorted(self._spend.items())},
            "estimated_spend_usd": {
                k: round(v, 6) for k, v in sorted(self._estimated_spend.items())
            },
            "total_spend_usd": round(self.total_spend(), 6),
            "calls_charged": self._calls,
            "calls_estimated": self._estimated_calls,
        }
