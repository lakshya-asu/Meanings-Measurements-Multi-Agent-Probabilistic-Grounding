"""Preregistered statistical report over a gate 4 results database.

Usage:
    python3 -m src.stats.report --db results.sqlite --split bench_v1_98

Produces a plain-text summary (stdout, or --text-out) and a JSON
summary (--json-out, or appended to stdout) containing:

- per-cell (per-method) point estimates for the two headline
  endpoints, SR@1.0m and median censored d_h, each with a 95 percent
  hierarchical BCa bootstrap CI over the scene clusters,
- the K = 5 preregistered comparisons (endpoints.
  PREREGISTERED_COMPARISONS) with the cluster-permutation raw p-values
  and Holm-adjusted p-values (family size held at K = 5),
- row-count integrity results per method: the analysis refuses to run
  when any (qid, seed) row is missing, listing the missing pairs,
  unless --allow-incomplete explicitly downgrades missing rows to
  censored failures,
- per-seed aggregates and the across-seed SD (df = n_seeds - 1, a
  crude estimate, reported separately from the query bootstrap per
  metrics.md step 7).

Determinism: the same database, split, and analysis seed produce a
byte-identical JSON file. Everything is ordered (sorted methods,
fixed metric and comparison order, sorted JSON keys); a single numpy
Generator seeded with --analysis-seed (default 20260815) is consumed
in that fixed order; no timestamps or absolute paths enter the JSON.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from src.splits import SplitIntegrityError, load_split, split_sha
from src.stats import endpoints as ep
from src.stats.bootstrap import BootstrapResult, bca_interval
from src.stats.permutation import cluster_permutation_test, holm_bonferroni

# Fixed order in which the RNG is consumed per cell: one bootstrap for
# each of these metrics, methods in sorted order, then comparisons in
# preregistered order. Changing this order changes the (still valid)
# random streams, so it is part of the determinism contract.
_CELL_METRICS = ("sr_1m", "med_dh")


def _r(x: Optional[float]) -> Optional[float]:
    """Round for the JSON summary; deterministic and diff-friendly."""
    return None if x is None else round(float(x), 10)


def _ci_dict(res: BootstrapResult) -> Dict[str, Any]:
    return {
        "point": _r(res.point),
        "lo": _r(res.lo),
        "hi": _r(res.hi),
        "method": res.method,
        "fallback_reason": res.fallback_reason,
        "z0": _r(res.z0),
        "accel": _r(res.accel),
        "n_boot": res.n_boot,
        "n_scenes": res.n_scenes,
        "n_queries": res.n_queries,
    }


def build_summary(
    db_path: str,
    split: str,
    analysis_seed: int = ep.DEFAULT_ANALYSIS_SEED,
    n_boot: int = 10000,
    n_perm: int = 10000,
    censor_cap: float = ep.CENSOR_CAP_M,
    alpha: float = 0.05,
    seeds: Optional[Sequence[int]] = None,
    aliases: Optional[Dict[str, str]] = None,
    split_manifest: Optional[str] = None,
    allow_incomplete: bool = False,
) -> Dict[str, Any]:
    """Compute the full preregistered summary as a JSON-ready dict.

    Raises ep.MissingQueriesError when rows are missing and
    allow_incomplete is False. aliases maps the canonical system names
    used in PREREGISTERED_COMPARISONS (mapg, grapheqa_adapted,
    anchor_centroid) to the method names actually stored in the
    database, when they differ.
    """
    alias_map = dict(aliases or {})

    split_rows = load_split(split, split_manifest)
    qids, qid_to_scene = ep.split_qids_and_scenes(split_rows)
    try:
        pinned_sha: Optional[str] = split_sha(split, split_manifest)
    except SplitIntegrityError:
        pinned_sha = None

    method_rows = ep.load_method_rows(db_path, split)
    if not method_rows:
        raise ValueError(
            f"no episode rows found for split {split!r} in {db_path}; "
            "nothing to analyze"
        )
    methods = sorted(method_rows)

    if seeds is not None:
        seed_list = sorted(int(s) for s in seeds)
    else:
        observed = {
            s
            for rows in method_rows.values()
            for (_, s) in rows.keys()
            if s is not None
        }
        seed_list = sorted(int(s) for s in observed)
    if not seed_list:
        raise ValueError("no seeds found in the database and none given")

    warnings: List[str] = []
    if len(seed_list) != ep.PREREGISTERED_SEEDS_PER_CELL:
        warnings.append(
            f"seed count is {len(seed_list)}, preregistered design is "
            f"{ep.PREREGISTERED_SEEDS_PER_CELL} seeds per cell"
        )

    # Row-count integrity: no dropped queries, ever. Fail loudly unless
    # the caller explicitly downgraded missing rows.
    integrity: Dict[str, Any] = {}
    shortfalls: List[str] = []
    for method in methods:
        check = ep.check_row_counts(list(method_rows[method].keys()), qids, seed_list)
        integrity[method] = check
        if not check["ok"]:
            shortfalls.append(method)
    if shortfalls and not allow_incomplete:
        # Re-raise through the loud single-method assertion for the
        # first offender so the error lists the missing pairs.
        first = shortfalls[0]
        ep.assert_no_dropped_queries(
            first, list(method_rows[first].keys()), qids, seed_list
        )
    if shortfalls:
        warnings.append(
            "missing rows scored as censored failures (allow_incomplete) "
            "for methods: " + ", ".join(shortfalls)
        )

    rng = np.random.default_rng(int(analysis_seed))

    # Per-cell estimates, methods in sorted order, metrics in fixed
    # order. Each bootstrap consumes the shared RNG in this exact
    # sequence; that ordering is the determinism contract.
    cells: Dict[str, Any] = {}
    per_query_cache: Dict[str, Dict[str, np.ndarray]] = {}
    for method in methods:
        rows = method_rows[method]
        succ = ep.value_matrix(
            rows, qids, seed_list, "success_1m", censor_cap,
            allow_missing_rows=True,
        )
        err = ep.value_matrix(
            rows, qids, seed_list, "censored_error", censor_cap,
            allow_missing_rows=True,
        )
        s_i = ep.per_query_means(succ)
        x_i = ep.per_query_means(err)
        per_query_cache[method] = {"success_1m": s_i, "censored_error": x_i}

        cell: Dict[str, Any] = {
            "n_rows_scored_success": ep.scored_row_count(rows, qids, seed_list, "success_1m"),
            "n_rows_scored_error": ep.scored_row_count(rows, qids, seed_list, "censored_error"),
        }
        for metric in _CELL_METRICS:
            if metric == "sr_1m":
                values, stat = s_i, lambda m: float(np.mean(m[:, 0]))
                per_seed = [float(np.mean(succ[:, k])) for k in range(len(seed_list))]
            else:
                values, stat = x_i, lambda m: float(np.median(m[:, 0]))
                per_seed = [float(np.median(err[:, k])) for k in range(len(seed_list))]
            grouped = ep.group_by_scene(qids, values, qid_to_scene)
            res = bca_interval(grouped, stat, n_boot=n_boot, rng=rng, alpha=alpha)
            sd = float(np.std(per_seed, ddof=1)) if len(per_seed) >= 2 else None
            cell[metric] = {
                "estimate": _ci_dict(res),
                "per_seed": [_r(v) for v in per_seed],
                "across_seed_sd": _r(sd),
            }
        cells[method] = cell

    # Preregistered comparisons in fixed (rank) order.
    scene_labels = [qid_to_scene[q] for q in qids]
    comparisons: List[Dict[str, Any]] = []
    for comp in ep.PREREGISTERED_COMPARISONS:
        entry: Dict[str, Any] = {
            "id": comp["id"],
            "rank": comp["rank"],
            "endpoint": comp["endpoint"],
            "description": comp["description"],
            "system_a": comp["system_a"],
            "system_b": comp["system_b"],
            "value_kind": comp["value_kind"],
            "statistic": comp["statistic"],
        }
        name_a = alias_map.get(comp["system_a"], comp["system_a"])
        name_b = alias_map.get(comp["system_b"], comp["system_b"])
        entry["method_a"] = name_a
        entry["method_b"] = name_b

        missing = [n for n in (name_a, name_b) if n not in method_rows]
        if missing:
            entry["status"] = "missing_system"
            entry["reason"] = (
                "method(s) not present in the database: " + ", ".join(sorted(missing))
            )
            comparisons.append(entry)
            continue

        kind = comp["value_kind"]
        rows_a, rows_b = method_rows[name_a], method_rows[name_b]
        if ep.optional_kind(kind):
            scored_a = ep.scored_row_count(rows_a, qids, seed_list, kind)
            scored_b = ep.scored_row_count(rows_b, qids, seed_list, kind)
            entry["scored_rows_a"] = scored_a
            entry["scored_rows_b"] = scored_b
            if scored_a == 0 or scored_b == 0:
                entry["status"] = "not_evaluable"
                entry["reason"] = (
                    f"no scored rows for value kind '{kind}' "
                    "(field absent from every episode of at least one system)"
                )
                comparisons.append(entry)
                continue

        if kind in per_query_cache.get(name_a, {}):
            a_q = per_query_cache[name_a][kind]
        else:
            a_q = ep.per_query_means(
                ep.value_matrix(rows_a, qids, seed_list, kind, censor_cap,
                                allow_missing_rows=True)
            )
        if kind in per_query_cache.get(name_b, {}):
            b_q = per_query_cache[name_b][kind]
        else:
            b_q = ep.per_query_means(
                ep.value_matrix(rows_b, qids, seed_list, kind, censor_cap,
                                allow_missing_rows=True)
            )

        perm = cluster_permutation_test(
            a_q, b_q, scene_labels, statistic=comp["statistic"],
            n_perm=n_perm, rng=rng,
        )
        paired = np.column_stack([a_q, b_q])
        if comp["statistic"] == "median_diff":
            delta_stat = lambda m: float(np.median(m[:, 0]) - np.median(m[:, 1]))
        else:
            delta_stat = lambda m: float(np.mean(m[:, 0]) - np.mean(m[:, 1]))
        grouped_pairs = ep.group_by_scene(qids, paired, qid_to_scene)
        delta_ci = bca_interval(
            grouped_pairs, delta_stat, n_boot=n_boot, rng=rng, alpha=alpha
        )
        entry["status"] = "evaluated"
        entry["t_obs"] = _r(perm.t_obs)
        entry["p_raw"] = _r(perm.p_value)
        entry["n_perm"] = perm.n_perm
        entry["n_scenes"] = perm.n_scenes
        entry["n_queries"] = perm.n_queries
        entry["delta"] = _ci_dict(delta_ci)
        comparisons.append(entry)

    # Holm-Bonferroni over the evaluated comparisons, family size held
    # at the preregistered K = 5 (conservative when some comparisons
    # were not evaluable).
    evaluated = [c for c in comparisons if c["status"] == "evaluated"]
    if evaluated:
        holm = holm_bonferroni(
            [c["p_raw"] for c in evaluated],
            alpha=alpha,
            family_size=ep.K_COMPARISONS,
        )
        for c, adj, rej in zip(evaluated, holm["adjusted"], holm["reject"]):
            c["p_holm"] = _r(adj)
            c["reject_at_alpha"] = bool(rej)

    return {
        "analysis": {
            "analysis_seed": int(analysis_seed),
            "n_boot": int(n_boot),
            "n_perm": int(n_perm),
            "censor_cap_m": _r(censor_cap),
            "alpha": _r(alpha),
            "primary_tau_m": 1.0,
            "holm_family_size": ep.K_COMPARISONS,
            "seeds": seed_list,
            "allow_incomplete": bool(allow_incomplete),
            "aliases": {k: alias_map[k] for k in sorted(alias_map)},
        },
        "split": {
            "name": split,
            "sha256": pinned_sha,
            "n_queries": len(qids),
            "n_scenes": len(set(qid_to_scene.values())),
        },
        "integrity": integrity,
        "cells": cells,
        "comparisons": comparisons,
        "warnings": warnings,
    }


# ----------------------------------------------------------------------
# Rendering
# ----------------------------------------------------------------------
def render_text(summary: Dict[str, Any]) -> str:
    """Fixed-format plain-text rendering of the summary."""
    a = summary["analysis"]
    sp = summary["split"]
    lines: List[str] = []
    lines.append("MAPG preregistered statistical report")
    lines.append(
        f"split={sp['name']} n_queries={sp['n_queries']} "
        f"n_scenes={sp['n_scenes']} sha256={sp['sha256']}"
    )
    lines.append(
        f"analysis_seed={a['analysis_seed']} n_boot={a['n_boot']} "
        f"n_perm={a['n_perm']} censor_cap={a['censor_cap_m']} m "
        f"alpha={a['alpha']} seeds={a['seeds']}"
    )
    lines.append("")

    lines.append("Row-count integrity (expected = n_queries x n_seeds):")
    for method in sorted(summary["integrity"]):
        chk = summary["integrity"][method]
        status = "OK" if chk["ok"] else "SHORTFALL"
        lines.append(
            f"  {method}: {status} present={chk['present_rows']}/"
            f"{chk['expected_rows']}"
        )
        if chk["missing_qids"]:
            lines.append("    missing qids: " + ", ".join(chk["missing_qids"]))
    lines.append("")

    lines.append("Per-cell estimates (95% CI, hierarchical bootstrap over scenes):")
    header = (
        f"  {'method':<28} {'SR@1.0m':>8} {'[lo, hi]':>18} "
        f"{'med d_h (m)':>12} {'[lo, hi]':>18}"
    )
    lines.append(header)
    for method in sorted(summary["cells"]):
        cell = summary["cells"][method]
        sr = cell["sr_1m"]["estimate"]
        md = cell["med_dh"]["estimate"]
        tag = ""
        if sr["method"] != "bca" or md["method"] != "bca":
            tag = "  (percentile fallback)"
        lines.append(
            f"  {method:<28} {sr['point']:>8.4f} "
            f"[{sr['lo']:.4f}, {sr['hi']:.4f}]".ljust(60)
            + f" {md['point']:>10.4f} [{md['lo']:.4f}, {md['hi']:.4f}]" + tag
        )
    lines.append("")

    lines.append(
        f"Preregistered comparisons (K={a['holm_family_size']}, "
        "cluster permutation, Holm-adjusted):"
    )
    for comp in summary["comparisons"]:
        if comp["status"] != "evaluated":
            lines.append(
                f"  {comp['id']} {comp['description']}: "
                f"{comp['status']} ({comp.get('reason', '')})"
            )
            continue
        rej = "REJECT" if comp.get("reject_at_alpha") else "no"
        lines.append(
            f"  {comp['id']} {comp['description']}: "
            f"delta={comp['t_obs']:.4f} "
            f"[{comp['delta']['lo']:.4f}, {comp['delta']['hi']:.4f}] "
            f"p_raw={comp['p_raw']:.6f} p_holm={comp['p_holm']:.6f} "
            f"reject_at_alpha={rej}"
        )
    if summary["warnings"]:
        lines.append("")
        lines.append("Warnings:")
        for w in summary["warnings"]:
            lines.append(f"  - {w}")
    lines.append("")
    return "\n".join(lines)


def summary_json_bytes(summary: Dict[str, Any]) -> bytes:
    """Canonical JSON encoding: sorted keys, fixed separators, LF."""
    return (
        json.dumps(summary, sort_keys=True, indent=2, ensure_ascii=True) + "\n"
    ).encode("utf-8")


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def _parse_aliases(pairs: Optional[Sequence[str]]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for pair in pairs or ():
        if "=" not in pair:
            raise ValueError(
                f"bad --alias {pair!r}: expected canonical=db_method_name"
            )
        canonical, db_name = pair.split("=", 1)
        out[canonical.strip()] = db_name.strip()
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Preregistered statistical report over a results database"
    )
    parser.add_argument("--db", required=True, help="path to the sqlite results db")
    parser.add_argument("--split", required=True, help="frozen split name, e.g. bench_v1_98")
    parser.add_argument("--analysis-seed", type=int, default=ep.DEFAULT_ANALYSIS_SEED,
                        help="RNG seed for bootstrap and permutations (default 20260815)")
    parser.add_argument("--n-boot", type=int, default=10000)
    parser.add_argument("--n-perm", type=int, default=10000)
    parser.add_argument("--censor-cap", type=float, default=ep.CENSOR_CAP_M,
                        help="censoring cap C in meters (preregistered 10.0)")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--seeds", default=None,
                        help="comma-separated run seeds; default: all seeds in the db")
    parser.add_argument("--alias", action="append", default=None, metavar="CANON=DBNAME",
                        help="map a canonical system name to its db method name")
    parser.add_argument("--split-manifest", default=None,
                        help="alternate splits/MANIFEST.json (tests only)")
    parser.add_argument("--json-out", default=None, help="write the JSON summary here")
    parser.add_argument("--text-out", default=None, help="write the text summary here")
    parser.add_argument("--allow-incomplete", action="store_true",
                        help="score missing rows as censored failures instead of aborting")
    args = parser.parse_args(argv)

    seeds = None
    if args.seeds:
        seeds = [int(s) for s in args.seeds.split(",") if s.strip() != ""]

    try:
        summary = build_summary(
            db_path=args.db,
            split=args.split,
            analysis_seed=args.analysis_seed,
            n_boot=args.n_boot,
            n_perm=args.n_perm,
            censor_cap=args.censor_cap,
            alpha=args.alpha,
            seeds=seeds,
            aliases=_parse_aliases(args.alias),
            split_manifest=args.split_manifest,
            allow_incomplete=args.allow_incomplete,
        )
    except ep.MissingQueriesError as e:
        print(f"INTEGRITY FAILURE: {e}", file=sys.stderr)
        print(
            "Refusing to produce statistics from an incomplete design. "
            "Fix the run (record every query, failures included) or rerun "
            "with --allow-incomplete to score missing rows as censored "
            "failures.",
            file=sys.stderr,
        )
        return 2

    text = render_text(summary)
    if args.text_out:
        with open(args.text_out, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
    print(text, end="")

    payload = summary_json_bytes(summary)
    if args.json_out:
        with open(args.json_out, "wb") as f:
            f.write(payload)
    else:
        print("=== JSON ===")
        sys.stdout.write(payload.decode("utf-8"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
