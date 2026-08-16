"""Projected $/episode tool (MAPG-10). No live calls anywhere."""

import subprocess
import sys

from src.agents.cost_estimate import (
    DEFAULT_PRICES,
    _price_row,
    estimate,
    estimate_episode,
    measure_serialization,
)


def test_estimate_shape_and_direction():
    result = estimate({})
    assert result["before"]["total_usd"] > result["after"]["total_usd"] > 0
    # This ticket's levers alone (serialization + parse-once + caching
    # + tiering) project >= 2x; the residual is image tokens, which
    # MAPG-10 does not touch.
    assert result["savings_factor"] >= 2.0
    assert result["after_haiku_tiered"]["total_usd"] <= result["after"]["total_usd"]
    # Parse-once: one orchestrator call after, one per step before.
    assert result["before"]["roles"]["orchestrator"]["calls"] == 12
    assert result["after"]["roles"]["orchestrator"]["calls"] == 1
    # Assumptions are part of the output contract (they feed the PRD).
    for key in ("stable_prefix_hit_rate", "chars_per_token", "prices_source"):
        assert key in result["assumptions"]


def test_tiered_roles_use_the_tier_model():
    result = estimate({})
    tiered = result["after_haiku_tiered"]["roles"]
    assert tiered["orchestrator"]["model"] == "claude-haiku-4-5-20251001"
    assert tiered["grounding"]["model"] == "claude-opus-4-6"
    assert tiered["spatial"]["model"] == "claude-opus-4-6"


def test_price_row_prefers_pinned_cfg_over_defaults():
    cfg_prices = {"claude-opus-4-6": {"input": 4.0, "output": 20.0}}
    row = _price_row(cfg_prices, "claude-opus-4-6")
    assert row["input"] == 4.0
    assert row["output"] == 20.0
    # Cache prices derive from input when not pinned explicitly.
    assert row["cache_read"] == 0.4
    assert row["cache_write"] == 5.0


def test_price_row_ignores_empty_placeholders():
    # The shipped yaml carries '' placeholders until pins land.
    cfg_prices = {"claude-opus-4-6": {"input": "", "output": ""}}
    row = _price_row(cfg_prices, "claude-opus-4-6")
    assert row["input"] == DEFAULT_PRICES["claude-opus-4-6"]["input"]


def test_caching_never_increases_cost_beyond_uncached():
    sizes = measure_serialization()
    graph_tokens = sizes["clean"]["compact_tokens"]
    uncached = estimate_episode("after", graph_tokens, None,
                                "claude-opus-4-6", caching=False)
    cached = estimate_episode("after", graph_tokens, None,
                              "claude-opus-4-6", caching=True)
    # The single-call orchestrator pays the write premium; the multi
    # call roles more than recoup it, so the total must go down.
    assert cached["total_usd"] < uncached["total_usd"]


def test_cli_runs_against_the_benchmark_cfg():
    proc = subprocess.run(
        [sys.executable, "-m", "src.agents.cost_estimate",
         "--cfg", "mapg_benchmark"],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    out = proc.stdout
    assert "savings factor" in out
    assert "before (base)" in out
    assert "after (MAPG-10)" in out
    assert "stable_prefix_hit_rate" in out
