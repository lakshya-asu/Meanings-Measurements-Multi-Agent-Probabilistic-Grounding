"""Projected $/episode before/after the MAPG-10 rework (no live calls).

Usage, from the repo root:

    python3 -m src.agents.cost_estimate --cfg mapg_benchmark

This is a PROJECTION for the PRD success criterion (B6): it computes
per-episode cost from (a) MEASURED serialized scene-graph text sizes
(legacy vs compact, on a deterministic synthetic graph), (b) the
planner's actual role call pattern, (c) documented cache-hit
assumptions, and (d) $/MTok prices (cfg ``model_prices`` rows when
pinned, else the documented defaults below). Real measured numbers
come from the MAPG-08 smoke run; until then every assumption is
printed with the table.

ASSUMPTIONS (all constants below, all printed by the CLI):

- Tokens ~= chars / 4 (English-ish prompt text).
- Typical "where" episode: 12 steps; 2 pre-lock steps (orchestrator +
  grounding) and 10 locked steps (orchestrator + grounding + spatial,
  the verifier being programmatic-only by default). BEFORE = the base
  commit's behavior (orchestrator re-called every step, legacy JSON
  serialization, indent=2 candidate dumps, no caching, no tiering).
  AFTER = MAPG-10 (parse-once orchestrator: 1 call/episode; compact
  serialization; cache breakpoints; cfg model_tiers).
- Scene graph mid-episode: 60 real nodes (45 objects, 3 rooms, 12
  frontiers) plus 150 accumulated historical agent-pose nodes in the
  BEFORE text (research/method-scenegraph-grounding.md section 1.2);
  the compact format drops them by contract.
- One image per vision call (grounding, spatial, qa): ~1600 tokens.
- Output ~250 tokens per call.
- Caching (AFTER only): per role, the system text is written to cache
  on the role's first call (cache-write price) and read on every later
  call (cache-read price). The stable scene-graph prefix additionally
  hits cache for STABLE_PREFIX_HIT_RATE of its tokens on steps 2+
  (the graph grows append-mostly under the stable-prefix ordering
  contract). Providers without explicit breakpoints cache shared
  prefixes automatically; the same rates are assumed.

Default prices are August 2026 list prices (USD per MTok), receipts in
research/method-architecture-cost.md section 2.3; cache_read defaults
to 0.1x input and cache_write to 1.25x input when a row has no
explicit value.
"""

from __future__ import annotations

import argparse
import json
import random
from typing import Any, Dict, Iterable, List, Optional, Tuple

from src.agents.serialization import serialize_scene_graph

CHARS_PER_TOKEN = 4.0
IMAGE_TOKENS = 1600
OUTPUT_TOKENS = 250
USER_OVERHEAD_TOKENS = 150  # pose, history, anchors, headers per call
CANDIDATE_TOKENS_LEGACY = 50  # per object, json indent=2 dump
CANDIDATE_TOKENS_COMPACT = 12  # per object, one line, 2 dp
STABLE_PREFIX_HIT_RATE = 0.7  # fraction of graph tokens read from cache, steps 2+

TYPICAL_STEPS = 12
PRELOCK_STEPS = 2

#: August 2026 list prices, USD per MTok. Used only when the cfg
#: model_prices row is absent or unpinned (empty strings).
DEFAULT_PRICES: Dict[str, Dict[str, float]] = {
    "claude-opus-4-6": {"input": 5.0, "output": 25.0},
    "claude-haiku-4-5": {"input": 1.0, "output": 5.0},
    "claude-haiku-4-5-20251001": {"input": 1.0, "output": 5.0},
    "gpt-5.2-chat-latest": {"input": 1.75, "output": 14.0},
    "gemini-2.5-pro": {"input": 1.25, "output": 10.0},
    "gemini-3-pro-preview": {"input": 2.0, "output": 12.0},
    "qwen3-vl-plus": {"input": 0.4, "output": 1.2},
}

DEFAULT_MAIN_MODEL = "claude-opus-4-6"
# The cheap-tier pin the MAPG-08 smoke will use (dated haiku snapshot,
# matching cfg model_pins).
HAIKU_TIER_MODEL = "claude-haiku-4-5-20251001"


# ---------------------------------------------------------------------------
# Minimal graph stand-in (networkx-protocol subset the serializer uses)
# ---------------------------------------------------------------------------

class _NodeView:
    def __init__(self, data: Dict[str, Dict[str, Any]]):
        self._data = data

    def __iter__(self):
        return iter(self._data)

    def __getitem__(self, key):
        return self._data[key]


class SimpleGraph:
    """Directed graph implementing exactly the protocol
    serialize_scene_graph reads: ``.nodes`` iteration + indexing,
    ``.predecessors(id)``, ``.successors(id)``. Lets the serializer be
    measured and tested on hosts without networkx installed."""

    def __init__(self):
        self._nodes: Dict[str, Dict[str, Any]] = {}
        # source -> [(target, attrs), ...] in insertion order, matching
        # networkx DiGraph adjacency storage.
        self._succ: Dict[str, List[Tuple[str, Dict[str, Any]]]] = {}
        self._pred: Dict[str, List[str]] = {}

    @property
    def nodes(self):
        return _NodeView(self._nodes)

    def add_node(self, node_id: str, **attrs: Any) -> None:
        self._nodes[str(node_id)] = dict(attrs)

    def add_edge(self, source: str, target: str, **attrs: Any) -> None:
        source, target = str(source), str(target)
        self._succ.setdefault(source, []).append((target, dict(attrs)))
        self._pred.setdefault(target, []).append(source)

    def successors(self, node_id: str) -> Iterable[str]:
        return iter(t for t, _ in self._succ.get(str(node_id), []))

    def predecessors(self, node_id: str) -> Iterable[str]:
        return iter(self._pred.get(str(node_id), []))

    def edges(self) -> List[Tuple[str, str, Dict[str, Any]]]:
        """(source, target, attrs) in networkx DiGraph.edges order:
        grouped by source node insertion order, then target insertion
        order within each source."""
        return [
            (s, t, dict(attrs))
            for s in self._nodes
            for t, attrs in self._succ.get(s, [])
        ]


def legacy_node_link_json(graph: SimpleGraph) -> str:
    """The legacy serialization's byte format for a SimpleGraph:
    matches ``json.dumps(nx.node_link_data(G))`` for a DiGraph with the
    same insertion order (asserted against real networkx in the test
    suite where networkx is installed)."""
    payload = {
        "directed": True,
        "multigraph": False,
        "graph": {},
        "nodes": [{**attrs, "id": nid} for nid, attrs in graph._nodes.items()],
        "links": [
            {**attrs, "source": s, "target": t} for s, t, attrs in graph.edges()
        ],
    }
    return json.dumps(payload)


# ---------------------------------------------------------------------------
# Synthetic scene graph (deterministic)
# ---------------------------------------------------------------------------

_NAMES = [
    "sofa", "table", "chair", "bed", "tv", "lamp", "shelf", "cabinet",
    "plant", "sink", "toilet", "mirror", "counter", "stool", "desk",
    "wardrobe", "fridge", "oven", "couch", "picture",
]


def build_synthetic_graph(
    n_objects: int = 45,
    n_rooms: int = 3,
    n_frontiers: int = 12,
    historical_agent_nodes: int = 0,
    seed: int = 7,
) -> Tuple[SimpleGraph, str]:
    """(graph, current_agent_id). Default 45+3+12 = 60 real nodes (the
    MAPG-10 unit-measurement graph); ``historical_agent_nodes`` adds
    stale agent-pose nodes to model mid-episode Hydra accumulation."""
    rng = random.Random(seed)
    g = SimpleGraph()

    for r in range(n_rooms):
        g.add_node(f"room_{r}", name=rng.choice(
            ["living room", "kitchen", "bedroom", "bathroom", "hallway"]
        ), layer=4, position=[rng.uniform(-8, 8) for _ in range(3)])

    for i in range(n_objects):
        pos = [rng.uniform(-8, 8), rng.uniform(-8, 8), rng.uniform(0, 2)]
        g.add_node(
            f"object_{i}",
            name=rng.choice(_NAMES),
            layer=2,
            label=rng.randrange(40),
            position=pos,
            bbox_extents=[round(rng.uniform(0.2, 2.2), 2) for _ in range(3)],
        )
        g.add_edge(f"room_{i % n_rooms}", f"object_{i}",
                   source_name="room", target_name="object",
                   type="room-to-object")

    for f in range(n_frontiers):
        g.add_node(f"frontier_{f}", name="frontier", layer=2,
                   position=[rng.uniform(-8, 8) for _ in range(3)])
        for obj in rng.sample(range(n_objects), k=min(2, n_objects)):
            g.add_edge(f"frontier_{f}", f"object_{obj}",
                       source_name="frontier", target_name="object",
                       type="frontier-to-object")

    total_agents = historical_agent_nodes + 1
    for a in range(total_agents):
        g.add_node(f"agent_{a}", name="agent", layer=2,
                   position=[rng.uniform(-8, 8) for _ in range(3)],
                   timestamp=float(a))
    current_agent_id = f"agent_{total_agents - 1}"
    return g, current_agent_id


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------

def measure_serialization() -> Dict[str, Any]:
    """Measured text sizes, legacy vs compact, on the synthetic graphs.

    ``clean``: the ticket's 60-node graph (no agent accumulation).
    ``accumulated``: the same graph plus 150 historical agent-pose
    nodes, the realistic mid-episode shape the BEFORE pipeline
    serialized every step.
    """
    out: Dict[str, Any] = {}
    for label, hist in (("clean", 0), ("accumulated", 150)):
        graph, agent_id = build_synthetic_graph(historical_agent_nodes=hist)
        legacy = legacy_node_link_json(graph)
        compact = serialize_scene_graph(graph, mode="compact",
                                        current_agent_id=agent_id)
        out[label] = {
            "legacy_chars": len(legacy),
            "compact_chars": len(compact),
            "ratio": len(legacy) / max(1, len(compact)),
            "legacy_tokens": int(len(legacy) / CHARS_PER_TOKEN),
            "compact_tokens": int(len(compact) / CHARS_PER_TOKEN),
        }
    return out


def _system_tokens() -> Dict[str, int]:
    """Rendered system-prompt sizes, measured from the real templates."""
    from src.agents.prompts import grounding, orchestrator, qa, spatial, verifier

    return {
        "orchestrator": int(len(orchestrator.SYSTEM) / CHARS_PER_TOKEN),
        "grounding": int(len(grounding.SYSTEM) / CHARS_PER_TOKEN),
        "spatial": int(len(spatial.SYSTEM) / CHARS_PER_TOKEN),
        "verifier": int(len(verifier.SYSTEM) / CHARS_PER_TOKEN),
        "qa": int(len(qa.SYSTEM) / CHARS_PER_TOKEN),
    }


# ---------------------------------------------------------------------------
# Pricing
# ---------------------------------------------------------------------------

def _price_row(prices_cfg: Optional[Dict[str, Any]], model: str) -> Dict[str, float]:
    """Resolved {input, output, cache_read, cache_write} $/MTok for a
    model: pinned cfg row when usable, else the documented default."""
    row: Dict[str, Any] = {}
    if prices_cfg:
        raw = prices_cfg.get(model)
        if isinstance(raw, dict):
            row = {
                k: float(v)
                for k, v in raw.items()
                if v not in (None, "") and str(v).strip() != ""
            }
    if "input" not in row or "output" not in row:
        default = DEFAULT_PRICES.get(model)
        if default is None:
            raise ValueError(
                f"no usable price for model {model!r}: not pinned in cfg "
                "model_prices and no documented default in "
                "src/agents/cost_estimate.DEFAULT_PRICES"
            )
        row = {**default, **row}
    row.setdefault("cache_read", 0.1 * row["input"])
    row.setdefault("cache_write", 1.25 * row["input"])
    return row


# ---------------------------------------------------------------------------
# Episode cost model
# ---------------------------------------------------------------------------

def _episode_calls(scenario: str) -> Dict[str, int]:
    """LLM calls per role for a typical 12-step "where" episode."""
    if scenario == "before":
        # Base-commit behavior: orchestrator re-called every step.
        return {
            "orchestrator": TYPICAL_STEPS,
            "grounding": TYPICAL_STEPS,
            "spatial": TYPICAL_STEPS - PRELOCK_STEPS,
        }
    # after: parse-once orchestrator.
    return {
        "orchestrator": 1,
        "grounding": TYPICAL_STEPS,
        "spatial": TYPICAL_STEPS - PRELOCK_STEPS,
    }


def _role_input_tokens(role: str, scenario: str, graph_tokens: int,
                       n_candidates: int = 45) -> Dict[str, int]:
    """{stable, volatile, image} input tokens for one call of a role.

    ``stable`` is the cacheable part (system text, plus the scene-graph
    stable prefix for graph-consuming roles); ``volatile`` bills full
    price every call.
    """
    sys_tokens = _system_tokens()[role]
    per_candidate = (
        CANDIDATE_TOKENS_LEGACY if scenario == "before" else CANDIDATE_TOKENS_COMPACT
    )
    if role == "orchestrator":
        return {"stable": sys_tokens, "volatile": USER_OVERHEAD_TOKENS, "image": 0}
    if role == "grounding":
        return {
            "stable": sys_tokens,
            "volatile": USER_OVERHEAD_TOKENS + n_candidates * per_candidate,
            "image": IMAGE_TOKENS,
        }
    if role == "spatial":
        return {
            "stable": sys_tokens + graph_tokens,
            "volatile": USER_OVERHEAD_TOKENS + n_candidates * 0,  # frontiers in overhead
            "image": IMAGE_TOKENS,
        }
    raise ValueError(role)


def estimate_episode(
    scenario: str,
    graph_tokens: int,
    prices_cfg: Optional[Dict[str, Any]],
    main_model: str,
    tiers: Optional[Dict[str, Optional[str]]] = None,
    caching: bool = False,
) -> Dict[str, Any]:
    """Projected cost of one typical episode under one scenario.

    ``caching=True`` applies the documented cache assumptions: the
    stable part is cache-written on a role's first call and cache-read
    afterward (scene-graph prefix at STABLE_PREFIX_HIT_RATE).
    """
    tiers = tiers or {}
    calls = _episode_calls(scenario)
    per_role: Dict[str, Dict[str, Any]] = {}
    total = 0.0
    for role, n_calls in calls.items():
        model = tiers.get(role) or main_model
        price = _price_row(prices_cfg, model)
        toks = _role_input_tokens(role, scenario, graph_tokens)
        cost = 0.0
        for call_idx in range(n_calls):
            stable, volatile, image = toks["stable"], toks["volatile"], toks["image"]
            if not caching:
                cost += (stable + volatile + image) / 1e6 * price["input"]
            elif call_idx == 0:
                cost += stable / 1e6 * price["cache_write"]
                cost += (volatile + image) / 1e6 * price["input"]
            else:
                hit = int(stable * STABLE_PREFIX_HIT_RATE)
                cost += hit / 1e6 * price["cache_read"]
                cost += (stable - hit) / 1e6 * price["input"]
                cost += (volatile + image) / 1e6 * price["input"]
            cost += OUTPUT_TOKENS / 1e6 * price["output"]
        per_role[role] = {
            "model": model,
            "calls": n_calls,
            "input_tokens_per_call": toks,
            "cost_usd": round(cost, 4),
        }
        total += cost
    return {"scenario": scenario, "total_usd": round(total, 4), "roles": per_role}


def estimate(cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """The full before/after projection (pure; used by the CLI and
    tests). ``cfg`` is the loaded benchmark yaml dict or None."""
    cfg = cfg or {}
    vlm = cfg.get("vlm") or {}
    prices_cfg = cfg.get("model_prices") or None
    tiers_cfg = vlm.get("model_tiers") or {}
    tiers = {str(k): (str(v) if v not in (None, "", "null") else None)
             for k, v in dict(tiers_cfg).items()}

    sizes = measure_serialization()
    before_graph_tokens = sizes["accumulated"]["legacy_tokens"]
    after_graph_tokens = sizes["clean"]["compact_tokens"]

    before = estimate_episode(
        "before", before_graph_tokens, prices_cfg, DEFAULT_MAIN_MODEL,
        tiers=None, caching=False,
    )
    after = estimate_episode(
        "after", after_graph_tokens, prices_cfg, DEFAULT_MAIN_MODEL,
        tiers=tiers, caching=True,
    )
    haiku_tiers = dict(tiers)
    haiku_tiers["orchestrator"] = haiku_tiers.get("orchestrator") or HAIKU_TIER_MODEL
    haiku_tiers["verifier"] = haiku_tiers.get("verifier") or HAIKU_TIER_MODEL
    after_tiered = estimate_episode(
        "after", after_graph_tokens, prices_cfg, DEFAULT_MAIN_MODEL,
        tiers=haiku_tiers, caching=True,
    )
    return {
        "serialization": sizes,
        "before": before,
        "after": after,
        "after_haiku_tiered": after_tiered,
        "savings_factor": round(
            before["total_usd"] / max(1e-9, after["total_usd"]), 2
        ),
        "assumptions": {
            "chars_per_token": CHARS_PER_TOKEN,
            "image_tokens": IMAGE_TOKENS,
            "output_tokens": OUTPUT_TOKENS,
            "typical_steps": TYPICAL_STEPS,
            "prelock_steps": PRELOCK_STEPS,
            "stable_prefix_hit_rate": STABLE_PREFIX_HIT_RATE,
            "candidate_tokens": {
                "before_indent2": CANDIDATE_TOKENS_LEGACY,
                "after_compact": CANDIDATE_TOKENS_COMPACT,
            },
            "historical_agent_nodes_before": 150,
            "main_model": DEFAULT_MAIN_MODEL,
            "prices_source": (
                "cfg model_prices where pinned, else "
                "src/agents/cost_estimate.DEFAULT_PRICES (Aug 2026 list)"
            ),
        },
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _load_cfg(name: str) -> Dict[str, Any]:
    import os

    import yaml

    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(os.path.dirname(here), "cfg", f"{name}.yaml")
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--cfg", default="mapg_benchmark",
                        help="cfg yaml name under src/cfg/ (no extension)")
    parser.add_argument("--json", action="store_true",
                        help="emit the raw result dict as JSON")
    args = parser.parse_args()

    result = estimate(_load_cfg(args.cfg))
    if args.json:
        print(json.dumps(result, indent=2))
        return

    sizes = result["serialization"]
    print("MAPG-10 projected $/episode (no live calls; assumptions below)")
    print()
    print("Serialized scene-graph text (synthetic 60-node graph):")
    print("  {:<28} {:>10} {:>10} {:>8}".format(
        "graph", "legacy", "compact", "ratio"))
    for label, row in sizes.items():
        print("  {:<28} {:>9}c {:>9}c {:>7.1f}x".format(
            label + (" (+150 stale agents)" if label == "accumulated" else ""),
            row["legacy_chars"], row["compact_chars"], row["ratio"]))
    print()
    print("Projected cost, typical 12-step 'where' episode, main model "
          + result["assumptions"]["main_model"] + ":")
    print("  {:<22} {:>12}".format("scenario", "$/episode"))
    for key, label in (
        ("before", "before (base)"),
        ("after", "after (MAPG-10)"),
        ("after_haiku_tiered", "after + haiku tiers"),
    ):
        print("  {:<22} {:>12.4f}".format(label, result[key]["total_usd"]))
    print()
    print("  savings factor (before/after): {:.1f}x".format(
        result["savings_factor"]))
    print()
    print("Per-role (after):")
    for role, row in result["after"]["roles"].items():
        print("  {:<14} {:>2} calls  model={:<20} ${:.4f}".format(
            role, row["calls"], row["model"], row["cost_usd"]))
    print()
    print("Assumptions:")
    for k, v in result["assumptions"].items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
