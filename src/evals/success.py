"""GT-checked success metrics (item 5 of the research harness).

One implementation per metric, shared by the online runners
(run_multi_agent_benchmark.py, run_msp_benchmark.py) and the offline
evaluator (src/evals/eval_offset_distances.py).

Preregistered metric spec (research/metrics.md):

- Primary endpoint is SR@1.0m on O-W: an episode is a success iff
  d_h(p_hat, p_gt) <= 1.0 m, where d_h is the HORIZONTAL distance.
  Habitat convention: y is up (gravity-aligned), so d_h projects out y
  and is computed on x and z only.
- 3D Euclidean distance d_3 is reported as a secondary column.
- tau = 1.0 m is preregistered as primary; 0.5 and 2.0 m are
  descriptive only.
- The legacy "success" flag stored by the runners was self-reported VLM
  confidence (is_conf or conf > 0.9), never checked against GT. It is
  renamed to self_reported_conf_success; the fields computed here
  (success_gt_1m and friends) are the real, GT-checked metrics.

GT column convention (reused, not invented):

- The offline evaluator eval_offset_distances.py reads the GT point
  from columns metric_corrected_x, metric_corrected_y,
  metric_corrected_z.
- Those columns are produced by src/tools/offset_metric.py as
  metric_corrected = anchor_center + distance_m * unit(ann_pos - anchor_center)
  from columns anchor_center_x/y/z, ann_pos_x/y/z, distance_m.
- The frozen bench split splits/bench_v1_98.csv carries the ingredient
  columns (ann_pos_*, anchor_center_*, distance_m) but not the derived
  metric_corrected_* columns, so gt_xyz_from_row uses
  metric_corrected_* when present and otherwise recomputes them with
  the exact offset_metric.py formula (same EPS guard: if
  ann_pos == anchor_center within EPS the direction is undefined and,
  exactly like offset_metric.py with metric_corrected_ok=False, the
  point degenerates to the anchor center).
- Zero-distance rows (distance_m == 0) are the between-style queries
  whose answer IS the annotated point, not an offset from an anchor.
  For those the scoring GT is ann_pos. This is Lakshya decision D5
  (2026-08-16, option b) and it takes priority over both the derived
  metric_corrected_* columns and the offset formula, because the
  offline offset tool degenerates a zero-distance row to the anchor
  center, which is the wrong target for a between query.

Stdlib + math only: no numpy, no pandas, importable anywhere including
inside the Habitat docker image. Inputs may be lists, tuples, or numpy
arrays (indexed and coerced with float(), never via numpy APIs).
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence

from src.schema.prediction import normalize_prediction

# Same epsilon as src/tools/offset_metric.py: below this, the direction
# from anchor to annotated point is undefined.
EPS = 1e-9

# Preregistered primary threshold (meters). 0.5 and 2.0 are descriptive.
PRIMARY_TAU = 1.0
DESCRIPTIVE_TAUS = (0.5, 2.0)


def _as_xyz(p: Any) -> Optional[List[float]]:
    """Coerce p to [x, y, z] floats or return None. Never raises."""
    if p is None:
        return None
    try:
        out = [float(p[0]), float(p[1]), float(p[2])]
    except (TypeError, ValueError, IndexError, KeyError):
        return None
    for v in out:
        if math.isnan(v) or math.isinf(v):
            return None
    return out


def horizontal_error(pred_xyz: Sequence[float], gt_xyz: Sequence[float]) -> float:
    """d_h: horizontal (gravity-projected) distance in meters.

    Habitat convention: y is up, so the gravity-aligned y axis is
    projected out and the distance uses x and z only. This is the
    distance the primary endpoint SR@1.0m thresholds.
    """
    dx = float(pred_xyz[0]) - float(gt_xyz[0])
    dz = float(pred_xyz[2]) - float(gt_xyz[2])
    return math.sqrt(dx * dx + dz * dz)


def euclidean_error(pred_xyz: Sequence[float], gt_xyz: Sequence[float]) -> float:
    """d_3: full 3D Euclidean distance in meters (secondary column)."""
    dx = float(pred_xyz[0]) - float(gt_xyz[0])
    dy = float(pred_xyz[1]) - float(gt_xyz[1])
    dz = float(pred_xyz[2]) - float(gt_xyz[2])
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def gt_success(
    pred_xyz: Optional[Sequence[float]],
    gt_xyz: Sequence[float],
    tau: float = PRIMARY_TAU,
) -> Optional[bool]:
    """GT-checked success: d_h(pred, gt) <= tau. Boundary counts.

    Returns None when pred_xyz is None. A missing prediction IS a
    failure for SR purposes (metrics.md section 4: failure to answer is
    failure, binary metrics score it 0), but its distances are
    undefined, so this pure function returns None and lets the caller
    store the failure explicitly (score_episode stores False).
    """
    p = _as_xyz(pred_xyz)
    if p is None:
        return None
    g = _as_xyz(gt_xyz)
    if g is None:
        return None
    return horizontal_error(p, g) <= tau


def _row_float(row: Dict[str, Any], key: str) -> Optional[float]:
    """Fetch row[key] as a finite float, else None (CSV rows are strings)."""
    v = row.get(key)
    if v is None or v == "":
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if math.isnan(f) or math.isinf(f):
        return None
    return f


def _row_vec3(row: Dict[str, Any], prefix: str) -> Optional[List[float]]:
    """Fetch [prefix_x, prefix_y, prefix_z] from a row dict, else None."""
    x = _row_float(row, prefix + "_x")
    y = _row_float(row, prefix + "_y")
    z = _row_float(row, prefix + "_z")
    if x is None or y is None or z is None:
        return None
    return [x, y, z]


def gt_xyz_from_row(gt_row: Dict[str, Any]) -> Optional[List[float]]:
    """Extract the GT target point from a bench CSV row dict.

    Column convention, in priority order (see module docstring):
    0. distance_m == 0 with ann_pos_x/y/z present: the GT is ann_pos
       itself (decision D5, between-style queries). Checked first, so
       a stale metric_corrected_* column derived by the offset tool
       (which degenerates such a row to the anchor center) cannot
       override the annotated point.
    1. metric_corrected_x/y/z: the exact columns
       eval_offset_distances.py reads today (derived offline by
       src/tools/offset_metric.py).
    2. anchor_center_x/y/z + ann_pos_x/y/z + distance_m, as in
       splits/bench_v1_98.csv: recompute metric_corrected with the
       offset_metric.py formula
       anchor + distance_m * unit(ann_pos - anchor).
       If ann_pos == anchor within EPS the direction is undefined and
       the result degenerates to the anchor center, exactly as
       offset_metric.py does (it flags such rows metric_corrected_ok
       False; none exist in the frozen bench split).
    3. Neither available: return None (episode cannot be GT-scored).
    """
    d = _row_float(gt_row, "distance_m")
    ann = _row_vec3(gt_row, "ann_pos")
    if d is not None and abs(d) <= EPS and ann is not None:
        # D5: zero-distance (between-style) rows score against the
        # annotated point, never against the anchor center.
        return [ann[0], ann[1], ann[2]]

    mc = _row_vec3(gt_row, "metric_corrected")
    if mc is not None:
        return mc

    anchor = _row_vec3(gt_row, "anchor_center")
    if anchor is None or ann is None or d is None:
        return None

    vx = ann[0] - anchor[0]
    vy = ann[1] - anchor[1]
    vz = ann[2] - anchor[2]
    norm = math.sqrt(vx * vx + vy * vy + vz * vz)
    if norm <= EPS:
        # Direction undefined: unit vector treated as zero, same as
        # offset_metric.py (metric_corrected_ok = False there).
        return [anchor[0], anchor[1], anchor[2]]
    return [
        anchor[0] + d * vx / norm,
        anchor[1] + d * vy / norm,
        anchor[2] + d * vz / norm,
    ]


def object_answer_success(final_pred: Dict[str, Any], gt_row: Dict[str, Any]) -> Optional[bool]:
    """O-O success stub: correct scene-graph node ID.

    Item 6 of the harness build order (the object matcher in
    src/evals/object_selection.py) will replace this hook with real
    strict node-ID matching. Until then it always returns None
    (unscored), never False, so O-O episodes are not miscounted as
    failures by a matcher that does not exist yet.
    """
    return None


def score_episode(final_pred: Optional[Dict[str, Any]], gt_row: Dict[str, Any]) -> Dict[str, Any]:
    """Score one episode's final prediction against its GT row.

    final_pred: the planner's final prediction dict (any legacy key
    shape; it is passed through normalize_prediction, so the point is
    read from the canonical target_point_xyz key). None is treated as
    a missing prediction.
    gt_row: the episode's row from the bench CSV (see gt_xyz_from_row
    for the column convention).

    Returns a dict of the new GT-checked fields, ready to merge into
    the episode record before it is persisted:

    - d_h: horizontal error in meters, None if pred or GT missing
    - d_3: 3D Euclidean error in meters, None if pred or GT missing
    - success_gt_1m: primary endpoint, d_h <= 1.0 m. False when the
      prediction is missing (failure to answer is failure); None only
      when the GT itself cannot be parsed.
    - success_gt_0p5m, success_gt_2m: descriptive thresholds, same
      missing-value rules
    - success_gt_node: O-O node-ID success, always None until item 6
      fills the object_answer_success hook
    - gt_xyz: the GT point used, for auditability
    """
    norm_pred = normalize_prediction(final_pred if final_pred is not None else {})
    pred = _as_xyz(norm_pred.get("target_point_xyz"))
    gt = gt_xyz_from_row(gt_row if isinstance(gt_row, dict) else {})

    out: Dict[str, Any] = {
        "d_h": None,
        "d_3": None,
        "success_gt_1m": None,
        "success_gt_0p5m": None,
        "success_gt_2m": None,
        "success_gt_node": object_answer_success(norm_pred, gt_row or {}),
        "gt_xyz": gt,
    }

    if gt is None:
        # No GT: nothing can be scored either way.
        return out

    if pred is None:
        # Missing prediction: distances undefined, but SR counts it as
        # a failure (metrics.md section 4).
        out["success_gt_1m"] = False
        out["success_gt_0p5m"] = False
        out["success_gt_2m"] = False
        return out

    d_h = horizontal_error(pred, gt)
    out["d_h"] = d_h
    out["d_3"] = euclidean_error(pred, gt)
    out["success_gt_1m"] = d_h <= PRIMARY_TAU
    out["success_gt_0p5m"] = d_h <= DESCRIPTIVE_TAUS[0]
    out["success_gt_2m"] = d_h <= DESCRIPTIVE_TAUS[1]
    return out
