"""Tests for the MAPG-11 cost governor.

Covers spend arithmetic (known and estimated token paths), provider
mapping, price resolution, hard-cap breach semantics, the kill path
(simulated breach aborts a fake run and the store shows status
aborted with the breach detail in the manifest), and the preflight
cost-governor check. Stdlib + pytest only.
"""

import pytest

from src.results.calls import CallLog
from src.results.governor import (
    CostCapExceeded,
    CostGovernor,
    GovernorConfigError,
    normalize_price_row,
    provider_of,
    resolve_price_key,
)
from src.results.store import ResultsStore
from src.scripts.preflight import cost_governor_problems

PRICES = {
    "gemini-2.5-pro": {"input": 2.0, "output": 10.0},
    "claude-opus-4-6": {"input": 15.0, "output": 75.0},
    "gpt-5.2-chat-latest": {"input": 1.25, "output": 10.0},
    "qwen3-vl-plus": {"input": 0.2, "output": 0.6},
}

BIG_CAPS = {"claude": 100.0, "openai": 100.0, "gemini": 100.0,
            "qwen": 100.0, "total": 400.0}


def gov(caps=None, prices=None, fallback=None):
    return CostGovernor(caps or dict(BIG_CAPS), prices or dict(PRICES),
                        fallback)


def row(model="models/gemini-2.5-pro", prompt=None, completion=None):
    return {"model_name": model, "prompt_tokens": prompt,
            "completion_tokens": completion}


# ---------------------------------------------------------------------------
# Provider mapping and price resolution
# ---------------------------------------------------------------------------

class TestProviderOf:
    def test_known_providers(self):
        assert provider_of("claude-opus-4-6") == "claude"
        assert provider_of("gpt-5.2-chat-latest") == "openai"
        assert provider_of("o3-mini") == "openai"
        assert provider_of("models/gemini-2.5-pro") == "gemini"
        assert provider_of("gemini-3-pro-preview") == "gemini"
        assert provider_of("qwen3-vl-plus") == "qwen"

    def test_boundary_and_unknowns(self):
        assert provider_of("googleplex") is None
        assert provider_of("o1x") is None
        assert provider_of("llama-3-70b") is None
        assert provider_of(None) is None
        assert provider_of("") is None


class TestPriceResolution:
    def test_exact_and_models_prefix(self):
        assert resolve_price_key(PRICES, "gemini-2.5-pro") == "gemini-2.5-pro"
        assert resolve_price_key(PRICES, "models/gemini-2.5-pro") == "gemini-2.5-pro"

    def test_snapshot_resolves_to_base_row(self):
        assert resolve_price_key(PRICES, "gemini-2.5-pro-001") == "gemini-2.5-pro"
        assert resolve_price_key(PRICES, "claude-opus-4-6-20260115") == "claude-opus-4-6"

    def test_no_boundary_no_match(self):
        prices = {"gemini-2.5": {"input": 1.0, "output": 1.0}}
        assert resolve_price_key(prices, "gemini-2.55") is None

    def test_missing(self):
        assert resolve_price_key(PRICES, "gemini-9-ultra") is None
        assert resolve_price_key(PRICES, None) is None

    def test_normalize_rejects_placeholders(self):
        with pytest.raises(ValueError):
            normalize_price_row({"input": "", "output": ""})
        with pytest.raises(ValueError):
            normalize_price_row("cheap")
        with pytest.raises(ValueError):
            normalize_price_row({"input": -1.0, "output": 2.0})
        assert normalize_price_row({"input": "1.5", "output": 3}) == (1.5, 3.0)
        assert normalize_price_row([2, 4]) == (2.0, 4.0)


# ---------------------------------------------------------------------------
# Spend arithmetic
# ---------------------------------------------------------------------------

class TestSpendArithmetic:
    def test_known_tokens_exact_cost(self):
        g = gov()
        g.charge_rows([row(prompt=1_000_000, completion=100_000)])
        # 1M * $2/Mtok + 0.1M * $10/Mtok = 2.0 + 1.0
        assert g.spend("gemini") == pytest.approx(3.0)
        s = g.summary()
        assert s["calls_charged"] == 1
        assert s["calls_estimated"] == 0
        assert s["estimated_spend_usd"] == {}

    def test_unknown_tokens_use_fallback_and_count_as_estimated(self):
        g = gov()
        g.charge_rows([row()])  # no token counts at all
        # 6000 * 2/1e6 + 400 * 10/1e6 = 0.012 + 0.004
        assert g.spend("gemini") == pytest.approx(0.016)
        s = g.summary()
        assert s["calls_estimated"] == 1
        assert s["estimated_spend_usd"]["gemini"] == pytest.approx(0.016)

    def test_partially_known_tokens_still_estimated(self):
        g = gov()
        g.charge_rows([row(prompt=1000)])  # completion unknown
        # 1000 * 2/1e6 + 400 * 10/1e6 = 0.002 + 0.004
        assert g.spend("gemini") == pytest.approx(0.006)
        assert g.summary()["calls_estimated"] == 1

    def test_custom_fallback(self):
        g = gov(fallback={"prompt": 1000, "completion": 100})
        g.charge_rows([row()])
        # 1000 * 2/1e6 + 100 * 10/1e6 = 0.002 + 0.001
        assert g.spend("gemini") == pytest.approx(0.003)

    def test_spend_accumulates_across_providers(self):
        g = gov()
        g.charge_rows([
            row(prompt=100_000, completion=10_000),
            row(model="claude-opus-4-6", prompt=100_000, completion=10_000),
        ])
        assert g.spend("gemini") == pytest.approx(0.3)   # 0.2 + 0.1
        assert g.spend("claude") == pytest.approx(2.25)  # 1.5 + 0.75
        assert g.total_spend() == pytest.approx(2.55)

    def test_uncapped_provider_without_total_is_not_charged(self):
        g = CostGovernor({"gemini": 5.0}, PRICES)
        g.charge_rows([row(model="claude-opus-4-6", prompt=10, completion=10)])
        assert g.total_spend() == 0.0


# ---------------------------------------------------------------------------
# Fail-loud config paths
# ---------------------------------------------------------------------------

class TestFailLoud:
    def test_unpriced_model_raises(self):
        g = gov()
        with pytest.raises(GovernorConfigError):
            g.charge_rows([row(model="gemini-9-ultra", prompt=1, completion=1)])

    def test_unknown_provider_raises(self):
        g = gov()
        with pytest.raises(GovernorConfigError):
            g.charge_rows([row(model="llama-3-70b", prompt=1, completion=1)])
        with pytest.raises(GovernorConfigError):
            g.charge_rows([row(model=None, prompt=1, completion=1)])

    def test_total_cap_governs_every_provider(self):
        # claude has no per-provider cap but total is capped, so an
        # unpriced claude model must still fail loudly.
        g = CostGovernor({"gemini": 5.0, "total": 5.0},
                         {"gemini-2.5-pro": {"input": 1.0, "output": 1.0}})
        with pytest.raises(GovernorConfigError):
            g.charge_rows([row(model="claude-opus-4-6", prompt=1, completion=1)])

    def test_validate_models(self):
        g = gov()
        g.validate_models(["models/gemini-2.5-pro", "claude-opus-4-6-20260115"])
        with pytest.raises(GovernorConfigError) as ei:
            g.validate_models(["gemini-9-ultra", "llama-3-70b"])
        assert "gemini-9-ultra" in str(ei.value)
        assert "llama-3-70b" in str(ei.value)

    def test_bad_caps_and_prices_rejected_at_construction(self):
        with pytest.raises(GovernorConfigError):
            CostGovernor({}, PRICES)
        with pytest.raises(GovernorConfigError):
            CostGovernor({"gemini": "lots"}, PRICES)
        with pytest.raises(GovernorConfigError):
            CostGovernor({"gemini": -1}, PRICES)
        with pytest.raises(GovernorConfigError):
            CostGovernor({"gemini": 1.0},
                         {"gemini-2.5-pro": {"input": "", "output": ""}})
        with pytest.raises(GovernorConfigError):
            CostGovernor({"gemini": 1.0}, {}, {"prompt": "many"})

    def test_from_cfg(self):
        assert CostGovernor.from_cfg({"seed": 1}) is None
        g = CostGovernor.from_cfg({
            "cost_caps": {"gemini": 1.0, "total": 2.0},
            "model_prices": {"gemini-2.5-pro": {"input": 1.0, "output": 2.0}},
            "fallback_tokens_per_call": {"prompt": 10, "completion": 5},
        })
        assert g is not None
        assert g.caps == {"gemini": 1.0, "total": 2.0}
        assert g.fallback_prompt == 10
        with pytest.raises(GovernorConfigError):
            CostGovernor.from_cfg({
                "cost_caps": {"gemini": 1.0},
                "model_prices": {"gemini-2.5-pro": {"input": "", "output": ""}},
            })


# ---------------------------------------------------------------------------
# Breach semantics
# ---------------------------------------------------------------------------

class TestBreach:
    def test_provider_cap_breach(self):
        g = gov(caps={"gemini": 0.01, "total": 100.0})
        with pytest.raises(CostCapExceeded) as ei:
            g.charge_rows([row()])  # estimated 0.016 > 0.01
        exc = ei.value
        assert exc.scope == "gemini"
        assert exc.cap_usd == pytest.approx(0.01)
        assert exc.spend_usd == pytest.approx(0.016)
        detail = exc.detail()
        assert detail["scope"] == "gemini"
        assert detail["summary"]["total_spend_usd"] == pytest.approx(0.016)

    def test_total_cap_breach(self):
        g = gov(caps={"gemini": 10.0, "claude": 10.0, "total": 0.02})
        with pytest.raises(CostCapExceeded) as ei:
            g.charge_rows([
                row(prompt=5000, completion=100),                       # 0.011
                row(model="claude-opus-4-6", prompt=1000, completion=0),  # 0.015
            ])
        assert ei.value.scope == "total"
        assert ei.value.spend_usd == pytest.approx(0.026)

    def test_spend_at_cap_is_not_a_breach(self):
        g = gov(caps={"gemini": 0.016, "total": 1.0})
        g.charge_rows([row()])  # exactly 0.016
        g.check_caps()  # no raise: caps are exceeded strictly

    def test_check_caps_keeps_raising_once_breached(self):
        g = gov(caps={"gemini": 0.01, "total": 1.0})
        with pytest.raises(CostCapExceeded):
            g.charge_rows([row()])
        with pytest.raises(CostCapExceeded):
            g.check_caps()

    def test_charge_tracked_accrues_only_new_rows(self):
        g = gov(caps={"gemini": 0.02, "total": 1.0})
        log = CallLog()
        g.track(log)
        log.record("kernel", model_name="models/gemini-2.5-pro")  # 0.016 est
        g.charge_tracked()
        assert g.spend("gemini") == pytest.approx(0.016)
        g.charge_tracked()  # no new rows, no double charge
        assert g.spend("gemini") == pytest.approx(0.016)
        log.record("selector", model_name="models/gemini-2.5-pro")
        with pytest.raises(CostCapExceeded):
            g.charge_tracked()  # 0.032 > 0.02


# ---------------------------------------------------------------------------
# Kill test: breach aborts a fake run, store shows aborted + reason
# ---------------------------------------------------------------------------

class TestKillPath:
    def test_breach_aborts_fake_run_in_store(self, tmp_path):
        with ResultsStore(tmp_path / "results.sqlite") as store:
            manifest = {"run_id": "govtest_run", "seed": 0}
            run_id = store.start_run(manifest)

            governor = CostGovernor(
                {"gemini": 0.02, "total": 1.0},
                {"gemini-2.5-pro": {"input": 2.0, "output": 10.0}},
            )

            # Episode 1 stays under the cap and is recorded normally.
            log1 = CallLog()
            governor.track(log1)
            log1.record("kernel", model_name="models/gemini-2.5-pro",
                        prompt_tokens=5000, completion_tokens=100)  # $0.011
            governor.charge_tracked()
            store.record_episode(run_id, {"qid": "ep1", "success": True,
                                          "vlm_calls": log1.total()})
            store.record_calls(run_id, "ep1", log1.rows())

            # Episode 2 breaches mid-episode; simulate the runners'
            # catch block: breach detail into the manifest, partial
            # calls flushed, run marked aborted.
            log2 = CallLog()
            governor.track(log2)
            log2.record("kernel", model_name="models/gemini-2.5-pro")
            with pytest.raises(CostCapExceeded) as ei:
                governor.charge_tracked()  # 0.011 + 0.016 est > 0.02
            exc = ei.value

            manifest["aborted_reason"] = "cost_cap_exceeded"
            manifest["cost_cap_breach"] = exc.detail()
            store.record_calls(run_id, "ep2_partial", log2.rows())
            store.start_run(manifest)  # refresh stored manifest copy
            store.finish_run(run_id, "aborted")

            # The store shows the aborted status and the breach reason.
            assert store.run_status(run_id) == "aborted"
            stored = store.runs()[0]["manifest"]
            assert stored["aborted_reason"] == "cost_cap_exceeded"
            breach = stored["cost_cap_breach"]
            assert breach["scope"] == "gemini"
            assert breach["cap_usd"] == pytest.approx(0.02)
            # Spend arithmetic incl. the estimated-tokens path:
            # ep1 real: 5000*2/1e6 + 100*10/1e6 = 0.011
            # ep2 estimated: 6000*2/1e6 + 400*10/1e6 = 0.016
            assert breach["spend_usd"] == pytest.approx(0.027)
            summary = breach["summary"]
            assert summary["spend_usd"]["gemini"] == pytest.approx(0.027)
            assert summary["estimated_spend_usd"]["gemini"] == pytest.approx(0.016)
            assert summary["calls_charged"] == 2
            assert summary["calls_estimated"] == 1

            # Everything recorded before the breach was flushed.
            assert [e["qid"] for e in store.episodes(run_id)] == ["ep1"]
            assert len(store.calls(run_id, "ep1")) == 1
            assert len(store.calls(run_id, "ep2_partial")) == 1

    def test_pre_episode_check_blocks_next_episode(self):
        governor = CostGovernor({"gemini": 0.01, "total": 1.0},
                                {"gemini-2.5-pro": {"input": 2.0, "output": 10.0}})
        log = CallLog()
        governor.track(log)
        log.record("kernel", model_name="models/gemini-2.5-pro")
        with pytest.raises(CostCapExceeded):
            governor.charge_tracked()
        # The runners call check_caps() before each episode; a breached
        # governor must keep refusing.
        with pytest.raises(CostCapExceeded):
            governor.check_caps()


# ---------------------------------------------------------------------------
# Preflight check (h)
# ---------------------------------------------------------------------------

GOV_CFG = {
    "vlm": {"name": "gemini"},
    "model_pins": {"claude-opus-4-6": "claude-opus-4-6-20260115"},
    "cost_caps": {"claude": 25.0, "gemini": 10.0, "total": 50.0},
    "model_prices": {"claude-opus-4-6": {"input": 15.0, "output": 75.0}},
}


def cfg_with(**overrides):
    cfg = {k: (dict(v) if isinstance(v, dict) else v)
           for k, v in GOV_CFG.items()}
    cfg.update(overrides)
    return cfg


class TestPreflightCostGovernor:
    def test_complete_cfg_has_no_problems(self):
        assert cost_governor_problems(GOV_CFG) == []

    def test_missing_cost_caps(self):
        problems = cost_governor_problems(cfg_with(cost_caps=None))
        assert any("cost_caps missing" in p for p in problems)

    def test_missing_provider_cap(self):
        problems = cost_governor_problems(
            cfg_with(cost_caps={"claude": 25.0, "total": 50.0}))
        assert any("provider 'gemini'" in p for p in problems)

    def test_missing_total_cap(self):
        problems = cost_governor_problems(
            cfg_with(cost_caps={"claude": 25.0, "gemini": 10.0}))
        assert any("'total'" in p for p in problems)

    def test_placeholder_price_is_a_problem(self):
        problems = cost_governor_problems(
            cfg_with(model_prices={"claude-opus-4-6": {"input": "", "output": ""}}))
        assert any("claude-opus-4-6" in p for p in problems)

    def test_non_numeric_cap_is_a_problem(self):
        problems = cost_governor_problems(
            cfg_with(cost_caps={"claude": "lots", "gemini": 10.0, "total": 50.0}))
        assert any("claude" in p and "USD" in p for p in problems)

    def test_missing_price_row_for_pinned_alias(self):
        problems = cost_governor_problems(cfg_with(model_prices={}))
        assert any("no usable row for 'claude-opus-4-6'" in p for p in problems)

    def test_pin_snapshot_row_covers_alias(self):
        # No row for the alias itself, but the pinned snapshot id has
        # an exact row: that is coverage.
        cfg = cfg_with(
            model_pins={"gpt-5.2-chat-latest": "gpt-5.2-2026-01-01"},
            cost_caps={"openai": 5.0, "gemini": 5.0, "total": 10.0},
            model_prices={"gpt-5.2-2026-01-01": {"input": 1.25, "output": 10.0}},
        )
        assert cost_governor_problems(cfg) == []

    def test_no_backends_no_requirements(self):
        assert cost_governor_problems({"habitat": {"scene_type": "hm3d"}}) == []
