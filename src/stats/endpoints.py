"""Preregistered endpoints and per-query aggregation (item 9).

Implements the metric side of the statistical protocol in
research/metrics.md (mirrored from flux-work/mapg-paper/research/
metrics.md), sections 1, 3 and 4:

- Primary endpoint: SR@1.0m, the mean over queries of the per-query
  success fraction s_i = (1/n_seeds) sum_k 1[success at seed k].
- Co-primary continuous endpoint: median horizontal error med(d_h),
  computed as the median over queries of the per-query mean of the
  CENSORED d_h values (median of per-query means, the bootstrap unit,
  as recommended in metrics.md section 3 step 2).
- Censoring (section 4): continuous errors are censored at the
  preregistered cap C = 10.0 m. Errors above C are set to C. A missing
  or unparseable error (terminal failure, missing prediction, or a
  missing row when the caller explicitly allows incomplete data) is
  also set to C. Successes are NOT affected by censoring: the binary
  success flags are computed upstream from the raw d_h against
  tau = 1.0 m (src/evals/success.py) and censoring touches only the
  continuous statistics.
- No dropped queries, ever (section 4 rule 0): a missing PREDICTION is
  a failure with censored error, but a missing ROW in the results
  store is a shortfall in the run itself. assert_no_dropped_queries
  fails loudly, listing every missing (qid, seed) pair, when the row
  count is not exactly split_size * n_seeds.

The preregistered confirmatory family (K = 5 comparisons, metrics.md
section 3 step 6) is encoded as data in PREREGISTERED_COMPARISONS so
the report cannot drift from the prose.

Episode rows come from the gate 4 SQLite store (src/results/store.py,
episodes table). The per-episode scored fields (d_h, success_gt_1m,
success_gt_node, spl) live in the row's "final" JSON column, merged in
by the runners via src/evals/success.py (item 5).
"""

from __future__ import annotations

import json
import math
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

# Preregistered censoring cap for continuous error statistics, meters.
# metrics.md section 4: C = 10.0 m, declared a priori.
CENSOR_CAP_M = 10.0

# Default analysis RNG seed, metrics.md section 3 step 3:
# numpy.random.default_rng(20260815), fixed and stated.
DEFAULT_ANALYSIS_SEED = 20260815

# Preregistered design: 3 seeds per (system, query) cell.
PREREGISTERED_SEEDS_PER_CELL = 3

# The preregistered confirmatory family, metrics.md section 3 step 6.
# Holm-Bonferroni is applied over exactly these K = 5 comparisons;
# everything else is exploratory. Encoded as data, not prose, so
# report.py and the paper table are generated from one source.
#
#   1. SR@1.0m:            MAPG vs GraphEQA (adapted)
#   2. med(d_h):           MAPG vs GraphEQA
#   3. O-O strict accuracy: MAPG vs GraphEQA
#   4. SR@1.0m:            MAPG vs anchor-centroid baseline
#   5. SPL (embodied):     MAPG vs GraphEQA
#
# value_kind selects the field read from the episode's final JSON and
# its missing-value rule (see extract_value). statistic selects the
# permutation-test statistic (see src/stats/permutation.py).
PREREGISTERED_COMPARISONS: Tuple[Dict[str, Any], ...] = (
    {
        "id": "C1",
        "rank": 1,
        "endpoint": "SR@1.0m",
        "description": "SR@1.0m: MAPG vs GraphEQA (adapted)",
        "system_a": "mapg",
        "system_b": "grapheqa_adapted",
        "value_kind": "success_1m",
        "statistic": "mean_diff",
    },
    {
        "id": "C2",
        "rank": 2,
        "endpoint": "med(d_h)",
        "description": "median d_h (censored at C): MAPG vs GraphEQA (adapted)",
        "system_a": "mapg",
        "system_b": "grapheqa_adapted",
        "value_kind": "censored_error",
        "statistic": "median_diff",
    },
    {
        "id": "C3",
        "rank": 3,
        "endpoint": "O-O strict accuracy",
        "description": "O-O strict node-ID accuracy: MAPG vs GraphEQA (adapted)",
        "system_a": "mapg",
        "system_b": "grapheqa_adapted",
        "value_kind": "oo_strict",
        "statistic": "mean_diff",
    },
    {
        "id": "C4",
        "rank": 4,
        "endpoint": "SR@1.0m",
        "description": "SR@1.0m: MAPG vs anchor-centroid baseline",
        "system_a": "mapg",
        "system_b": "anchor_centroid",
        "value_kind": "success_1m",
        "statistic": "mean_diff",
    },
    {
        "id": "C5",
        "rank": 5,
        "endpoint": "SPL",
        "description": "SPL (embodied): MAPG vs GraphEQA (adapted)",
        "system_a": "mapg",
        "system_b": "grapheqa_adapted",
        "value_kind": "spl",
        "statistic": "mean_diff",
    },
)

# Family size for Holm-Bonferroni. Fixed at 5 by preregistration even
# if some comparison cannot be evaluated on a given database (using
# the full family size for the multipliers is conservative).
K_COMPARISONS = 5

# value kinds whose per-row value can be genuinely absent (unscored)
# rather than "absent means failure". For these, a comparison is only
# evaluable if both systems have at least one scored row.
_OPTIONAL_KINDS = ("oo_strict", "spl")


class MissingQueriesError(RuntimeError):
    """Raised when episode rows are missing for expected (qid, seed) pairs."""


# ----------------------------------------------------------------------
# Split conventions
# ----------------------------------------------------------------------
def qid_for_split_row(i: int, row: Dict[str, Any]) -> str:
    """qid for split row index i: '{i}_{scene}_{floor}'.

    Replicates the convention shared by the runners (experiment_id in
    run_msp_benchmark.py / run_multi_agent_benchmark.py) and the
    offline evaluator (_qid_from_row in eval_offset_distances.py):
    floor is coerced through int(float(floor)) when possible.
    """
    scene = row.get("scene", "")
    floor = row.get("floor", "")
    try:
        f: Any = int(float(floor))
    except (TypeError, ValueError):
        f = str(floor)
    return f"{i}_{scene}_{f}"


def split_qids_and_scenes(split_rows: Sequence[Dict[str, Any]]) -> Tuple[List[str], Dict[str, str]]:
    """Expected qids (in split order) and the qid -> scene map."""
    qids: List[str] = []
    qid_to_scene: Dict[str, str] = {}
    for i, row in enumerate(split_rows):
        qid = qid_for_split_row(i, row)
        if qid in qid_to_scene:
            raise ValueError(f"duplicate qid {qid!r} in split rows")
        qids.append(qid)
        qid_to_scene[qid] = str(row.get("scene", ""))
    return qids, qid_to_scene


# ----------------------------------------------------------------------
# Pulling episode rows out of the ResultsStore database
# ----------------------------------------------------------------------
def load_method_rows(db_path, split: str) -> Dict[str, Dict[Tuple[str, Optional[int]], Dict[str, Any]]]:
    """Load episode rows for one split, grouped by method.

    Returns {method: {(qid, seed): final_dict}}. The database is opened
    read-only. When the same (method, seed, qid) exists in several runs
    the row from the latest run wins (ordered by created_at, then
    run_id, so the choice is deterministic).
    """
    path = Path(db_path)
    if not path.is_file():
        raise FileNotFoundError(f"results database not found: {path}")
    conn = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
    try:
        conn.row_factory = sqlite3.Row
        cur = conn.execute(
            "SELECT method, seed, qid, final FROM episodes WHERE split=? "
            "ORDER BY method, seed, qid, created_at, run_id",
            (str(split),),
        )
        out: Dict[str, Dict[Tuple[str, Optional[int]], Dict[str, Any]]] = {}
        for r in cur.fetchall():
            method = str(r["method"])
            seed = r["seed"] if r["seed"] is None else int(r["seed"])
            try:
                final = json.loads(r["final"])
            except (TypeError, ValueError):
                final = {}
            if not isinstance(final, dict):
                final = {}
            # Later rows (latest created_at / run_id) overwrite earlier.
            out.setdefault(method, {})[(str(r["qid"]), seed)] = final
        return out
    finally:
        conn.close()


# ----------------------------------------------------------------------
# Censoring and value extraction
# ----------------------------------------------------------------------
def censor_error(d: Any, cap: float = CENSOR_CAP_M) -> float:
    """Censor a continuous error at the preregistered cap C.

    metrics.md section 4: d_h_cens = min(d_h, C), C = 10.0 m declared
    a priori. Errors above C are set to C. A missing, non-numeric,
    non-finite, or negative value (terminal failure, missing
    prediction, unparseable output) is also set to C: failures carry
    the censored error, they are never dropped.

    Successes are unaffected: this function is used only for the
    continuous error statistics. The binary success flags are computed
    upstream from the raw (uncensored) d_h against tau = 1.0 m.
    """
    cap_f = float(cap)
    try:
        f = float(d)
    except (TypeError, ValueError):
        return cap_f
    if math.isnan(f) or math.isinf(f) or f < 0.0:
        return cap_f
    return min(f, cap_f)


def extract_value(final: Optional[Dict[str, Any]], kind: str, cap: float = CENSOR_CAP_M) -> float:
    """One scalar per episode row for a given endpoint kind.

    Missing-value rules (metrics.md section 4, no dropped queries):
    - success_1m:     final['success_gt_1m']; None or absent counts as
                      failure (0.0). Failure to answer is failure.
    - censored_error: final['d_h'] censored at C; None or absent gets C.
    - oo_strict:      final['success_gt_node']; None means "unscored",
                      which extract_value still maps to 0.0 (incorrect,
                      never dropped). Use has_value to decide whether a
                      comparison is evaluable at all.
    - spl:            final['spl']; None or absent is 0.0 (a failed
                      episode has SPL 0 by definition). Use has_value
                      for evaluability.

    final may be None (a missing row, when the caller explicitly allows
    incomplete data): it is scored as a total failure.
    """
    row = final if isinstance(final, dict) else {}
    if kind == "success_1m":
        v = row.get("success_gt_1m")
        return 1.0 if bool(v) else 0.0
    if kind == "censored_error":
        return censor_error(row.get("d_h"), cap)
    if kind == "oo_strict":
        v = row.get("success_gt_node")
        return 1.0 if bool(v) else 0.0
    if kind == "spl":
        v = row.get("spl")
        try:
            f = float(v)
        except (TypeError, ValueError):
            return 0.0
        if math.isnan(f) or math.isinf(f) or f < 0.0:
            return 0.0
        return f
    raise ValueError(f"unknown value kind {kind!r}")


def has_value(final: Optional[Dict[str, Any]], kind: str) -> bool:
    """Whether the row carries a real (scored) value for this kind."""
    row = final if isinstance(final, dict) else {}
    key = {
        "success_1m": "success_gt_1m",
        "censored_error": "d_h",
        "oo_strict": "success_gt_node",
        "spl": "spl",
    }.get(kind)
    if key is None:
        raise ValueError(f"unknown value kind {kind!r}")
    return row.get(key) is not None


# ----------------------------------------------------------------------
# Row-count integrity: no dropped queries, ever
# ----------------------------------------------------------------------
def check_row_counts(
    present_pairs: Sequence[Tuple[str, Optional[int]]],
    expected_qids: Sequence[str],
    seeds: Sequence[int],
) -> Dict[str, Any]:
    """Compare present (qid, seed) pairs against the full design.

    The design is the cross product expected_qids x seeds, so the
    expected row count is split_size * n_seeds. Returns a dict with
    ok, expected_rows, present_rows, missing (sorted list of
    [qid, seed] pairs), missing_qids (sorted unique qids), and
    unexpected (rows outside the design, sorted).
    """
    expected = {(q, int(s)) for q in expected_qids for s in seeds}
    present = set(present_pairs)
    missing = sorted(expected - present, key=lambda p: (p[0], p[1]))
    unexpected = sorted(
        present - expected,
        key=lambda p: (p[0], -1 if p[1] is None else int(p[1])),
    )
    return {
        "ok": not missing,
        "expected_rows": len(expected),
        "present_rows": len(present & expected),
        "missing": [[q, s] for q, s in missing],
        "missing_qids": sorted({q for q, _ in missing}),
        "unexpected": [[q, s] for q, s in unexpected],
    }


def assert_no_dropped_queries(
    method: str,
    present_pairs: Sequence[Tuple[str, Optional[int]]],
    expected_qids: Sequence[str],
    seeds: Sequence[int],
) -> Dict[str, Any]:
    """Fail loudly if any (qid, seed) row is missing for a method.

    metrics.md section 4 rule 0: no query is ever dropped. A missing
    PREDICTION inside a row is a failure with censored error and is
    handled by extract_value. A missing ROW means the run itself is
    incomplete, so the analysis must not proceed silently. Raises
    MissingQueriesError naming the method and listing every missing
    (qid, seed) pair. Returns the check dict when everything is there.
    """
    check = check_row_counts(present_pairs, expected_qids, seeds)
    if not check["ok"]:
        pairs = ", ".join(f"({q}, seed {s})" for q, s in check["missing"])
        raise MissingQueriesError(
            f"ROW COUNT SHORTFALL for method '{method}': expected "
            f"{check['expected_rows']} rows ({len(expected_qids)} queries x "
            f"{len(seeds)} seeds) but only {check['present_rows']} are in the "
            f"design. No query may ever be dropped: a missing prediction must "
            f"still be recorded as a failed row. Missing (qid, seed) pairs: "
            f"{pairs}. Missing qids: {', '.join(check['missing_qids'])}."
        )
    return check


# ----------------------------------------------------------------------
# Per-query aggregation (metrics.md section 3 steps 1 and 2)
# ----------------------------------------------------------------------
def value_matrix(
    rows: Dict[Tuple[str, Optional[int]], Dict[str, Any]],
    expected_qids: Sequence[str],
    seeds: Sequence[int],
    kind: str,
    cap: float = CENSOR_CAP_M,
    allow_missing_rows: bool = False,
) -> np.ndarray:
    """(n_queries, n_seeds) matrix of scalar values, design order.

    Row order follows expected_qids, column order follows seeds. By
    default a missing (qid, seed) row raises MissingQueriesError via
    assert_no_dropped_queries. With allow_missing_rows=True (an
    explicit, caller-visible choice) missing rows are scored as total
    failures (success 0, error censored at C), consistent with the
    missing-prediction rule; the caller is expected to have surfaced
    the shortfall to the user already.
    """
    if not allow_missing_rows:
        assert_no_dropped_queries("(matrix)", list(rows.keys()), expected_qids, seeds)
    seeds_i = [int(s) for s in seeds]
    out = np.empty((len(expected_qids), len(seeds_i)), dtype=float)
    for qi, qid in enumerate(expected_qids):
        for si, seed in enumerate(seeds_i):
            out[qi, si] = extract_value(rows.get((qid, seed)), kind, cap)
    return out


def per_query_means(matrix: np.ndarray) -> np.ndarray:
    """s_i / x_i: mean over seeds for each query (metrics.md step 1)."""
    return np.asarray(matrix, dtype=float).mean(axis=1)


def scored_row_count(
    rows: Dict[Tuple[str, Optional[int]], Dict[str, Any]],
    expected_qids: Sequence[str],
    seeds: Sequence[int],
    kind: str,
) -> int:
    """Number of design rows carrying a real value for this kind."""
    seeds_i = [int(s) for s in seeds]
    count = 0
    for qid in expected_qids:
        for seed in seeds_i:
            if has_value(rows.get((qid, seed)), kind):
                count += 1
    return count


def optional_kind(kind: str) -> bool:
    """Kinds that require at least one scored row to be evaluable."""
    return kind in _OPTIONAL_KINDS


# ----------------------------------------------------------------------
# Point estimates (metrics.md section 3 step 2)
# ----------------------------------------------------------------------
def sr_at_1m(per_query_s: np.ndarray) -> float:
    """SR@1.0m: mean over queries of the per-query success fraction."""
    return float(np.mean(np.asarray(per_query_s, dtype=float)))


def median_dh(per_query_x: np.ndarray) -> float:
    """med(d_h): median over queries of the per-query mean censored d_h.

    Median of per-query means, matching the bootstrap resampling unit
    (metrics.md section 3 step 2). Numpy's default linear interpolation
    is used for even counts; it is deterministic.
    """
    return float(np.median(np.asarray(per_query_x, dtype=float)))


def group_by_scene(
    qids: Sequence[str],
    values: np.ndarray,
    qid_to_scene: Dict[str, str],
) -> Dict[str, np.ndarray]:
    """Group per-query value rows into {scene: array} for the bootstrap.

    values may be 1D (one value per query) or 2D (n_queries, k), rows
    aligned with qids. Scene keys are sorted; row order within a scene
    follows the qid order given.
    """
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.shape[0] != len(qids):
        raise ValueError("values rows must align with qids")
    buckets: Dict[str, List[np.ndarray]] = {}
    for i, qid in enumerate(qids):
        scene = qid_to_scene[qid]
        buckets.setdefault(scene, []).append(arr[i])
    return {scene: np.vstack(rows) for scene, rows in sorted(buckets.items())}
