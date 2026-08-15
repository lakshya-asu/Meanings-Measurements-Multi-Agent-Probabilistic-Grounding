"""O-W error decomposition (item 8 of the research harness).

Single implementation of the preregistered error decomposition from
flux-work/mapg-paper/research/metrics.md, section 2.1, shared by the
online runners (run_multi_agent_benchmark.py, run_msp_benchmark.py) and
the offline evaluator (src/evals/eval_offset_distances.py).

With GT anchor centroid a (horizontally projected) and commanded
distance d_cmd:

- Radial error   e_r = | dist(p_hat, a) - d_cmd | in meters. Did the
  metric kernel honor the stated distance.
- Angular error  e_theta = angle(p_hat - a, p_gt - a) in degrees, in
  [0, 180]. Directional kernel quality. Computed by
  src.evals.angular.angular_errors_from_points (one implementation per
  metric; this module does not reimplement it).
- Anchor error   e_a = dist(a_hat, a) in meters, where a_hat is the
  anchor position the system actually used. Semantic kernel quality.
- Frame-flip     e_theta >= 150 degrees. Isolates the egocentric versus
  object-centric "in front of" ambiguity.
- Best-of-frames oracle: SR@1m scored against the better of the GT
  frame and the mirrored frame. The frame ambiguity negates the
  direction vector (p_gt - a), so the mirrored-frame GT point is the GT
  point reflected through the anchor in the horizontal plane (a 180
  degree rotation about the vertical axis through a):
      p_gt_mirror = [2*a_x - gt_x, gt_y, 2*a_z - gt_z]
  The y component is kept from the GT point; it does not enter the
  horizontal distance anyway. The gap (SR_best_of_frames - SR)
  upper-bounds how much residual error is frame ambiguity rather than
  grounding failure (metrics.md section 2.1).
- Ratio band (SpatialVLM precedent): 0.5 <= dist(p_hat, a)/d_cmd <= 2.0
  inclusive at both edges. Scale-free.

All distances are HORIZONTAL (Habitat y-up convention, y projected
out), consistent with src/evals/success.py: every distance here goes
through success.horizontal_error, never a private reimplementation.

Where a_hat comes from: the multi-agent planner
(src/planners/multi_agent_msp_planner.py) records pdf_params in its
final prediction, and those params carry the anchor position the MSP
kernel was centered on as x0/y0/z0 (see
vlm_planner_msp._get_metric_semantic_params and src/msp/pdf.py, which
read params['x0'] etc.). mu_x/mu_y/mu_z are accepted as a fallback
(they equal the anchor position in the same params dict). Predictions
without pdf_params (for example the smart MSP planner, whose plan
carries a selector trace but no pdf_params) get e_a = None with a
counted reason, never a fabricated value.

Missing-value policy (metrics.md section 4): no query is dropped and
nothing is imputed. Every component that cannot be computed is None
with a short reason string in its matching *_reason field, and
summarize() aggregates each component only over the rows where it
exists, reporting how many were missing. The one binary exception is
the best-of-frames success: a missing prediction is a failure (False),
same as success.py scores success_gt_1m, because failure to answer is
failure for binary metrics.

Stdlib math only: no numpy, no pandas, importable anywhere including
inside the Habitat docker image.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

from src.evals.angular import angular_errors_from_points
from src.evals.success import (
    PRIMARY_TAU,
    _as_xyz,
    _row_float,
    _row_vec3,
    gt_xyz_from_row,
    horizontal_error,
)
from src.schema.prediction import normalize_prediction

# Preregistered frame-flip threshold in degrees (metrics.md 2.1),
# boundary included: e_theta >= 150 is a flip.
FRAME_FLIP_DEG = 150.0

# Preregistered ratio band, both edges inclusive (metrics.md 2.1,
# SpatialVLM precedent).
RATIO_BAND = (0.5, 2.0)

# The decomposition fields decompose_episode emits, in output order.
FIELDS = (
    "e_r",
    "e_theta_deg",
    "e_a",
    "ratio",
    "ratio_in_band",
    "frame_flip",
    "success_best_of_frames_1m",
)


def anchor_hat_from_pred(final_pred: Dict[str, Any]) -> Tuple[Optional[List[float]], Optional[str]]:
    """Extract a_hat, the anchor position the system actually used.

    Reads pdf_params x0/y0/z0 (canonical, what src/msp/pdf.py consumes)
    with mu_x/mu_y/mu_z as fallback. Returns ([x, y, z], None) or
    (None, reason). Never raises.
    """
    if not isinstance(final_pred, dict):
        return None, "prediction is not a dict"
    params = final_pred.get("pdf_params")
    if not isinstance(params, dict):
        return None, "no pdf_params recorded in prediction"
    for kx, ky, kz in (("x0", "y0", "z0"), ("mu_x", "mu_y", "mu_z")):
        a_hat = _as_xyz([params.get(kx), params.get(ky), params.get(kz)])
        if a_hat is not None:
            return a_hat, None
    return None, "pdf_params carries no usable anchor position (x0/y0/z0 or mu_x/mu_y/mu_z)"


def mirrored_gt(gt_xyz: Sequence[float], anchor_xyz: Sequence[float]) -> List[float]:
    """Mirrored-frame GT point: reflect gt through the anchor horizontally.

    180 degree rotation about the vertical axis through the anchor
    (metrics.md 2.1: the egocentric versus object-centric frame flip
    negates the horizontal direction vector p_gt - a). The y component
    is kept from the GT point; the horizontal distance ignores it.
    """
    return [
        2.0 * float(anchor_xyz[0]) - float(gt_xyz[0]),
        float(gt_xyz[1]),
        2.0 * float(anchor_xyz[2]) - float(gt_xyz[2]),
    ]


def decompose_episode(final_pred: Optional[Dict[str, Any]], gt_row: Dict[str, Any]) -> Dict[str, Any]:
    """Decompose one episode's error into radial, angular, anchor parts.

    final_pred: the planner's final prediction dict (any legacy key
    shape; passed through normalize_prediction). None means no
    prediction.
    gt_row: the episode's bench CSV row dict. Uses anchor_center_x/y/z
    (GT anchor centroid a), distance_m (commanded distance d_cmd), and
    the GT point per success.gt_xyz_from_row (metric_corrected_* or the
    offset_metric.py recomputation).

    Returns a dict ready to merge into the episode record, every value
    in FIELDS plus a *_reason string for each component that is
    unavailable (see module docstring for the policy):

    - e_r: |dist_h(p_hat, a) - d_cmd| in meters, or None
    - e_theta_deg: angle(p_hat - a, p_gt - a) in degrees [0, 180], or None
    - e_a: dist_h(a_hat, a) in meters, or None
    - ratio: dist_h(p_hat, a) / d_cmd, or None
    - ratio_in_band: 0.5 <= ratio <= 2.0 inclusive, or None
    - frame_flip: e_theta_deg >= 150.0, or None with e_theta
    - success_best_of_frames_1m: d_h to the better of GT and mirrored
      GT <= 1.0 m. False when the prediction is missing (failure to
      answer is failure); None only when the GT itself is missing.
      Without an anchor the mirrored frame is undefined and the value
      falls back to the plain GT-frame success, with a reason saying so.
    """
    norm_pred = normalize_prediction(final_pred if final_pred is not None else {})
    pred = _as_xyz(norm_pred.get("target_point_xyz"))
    row = gt_row if isinstance(gt_row, dict) else {}
    gt = gt_xyz_from_row(row)
    anchor = _row_vec3(row, "anchor_center")
    d_cmd = _row_float(row, "distance_m")

    out: Dict[str, Any] = {name: None for name in FIELDS}
    out.update({name + "_reason": None for name in FIELDS})

    pred_reason = None if pred is not None else "no predicted point"
    anchor_reason = None if anchor is not None else "no anchor_center in GT row"
    d_cmd_reason = None if d_cmd is not None else "no commanded distance (distance_m) in GT row"

    # Horizontal distance from the prediction to the GT anchor, shared
    # by e_r and the ratio metrics.
    dist_pred_anchor = None
    if pred is not None and anchor is not None:
        dist_pred_anchor = horizontal_error(pred, anchor)

    # Radial error e_r.
    if dist_pred_anchor is None:
        out["e_r_reason"] = pred_reason or anchor_reason
    elif d_cmd is None:
        out["e_r_reason"] = d_cmd_reason
    else:
        out["e_r"] = abs(dist_pred_anchor - d_cmd)

    # Angular error e_theta: the one canonical implementation in
    # src/evals/angular.py (metrics.md 2.1, full 3D angle between the
    # anchor-to-point direction vectors).
    ang = angular_errors_from_points(pred, gt, anchor)
    out["e_theta_deg"] = ang["angular_error_deg"]
    out["e_theta_deg_reason"] = ang["reason"]

    # Frame flip: defined exactly when e_theta is.
    if out["e_theta_deg"] is None:
        out["frame_flip_reason"] = out["e_theta_deg_reason"]
    else:
        out["frame_flip"] = out["e_theta_deg"] >= FRAME_FLIP_DEG

    # Anchor error e_a: system anchor versus GT anchor.
    a_hat, a_hat_reason = anchor_hat_from_pred(norm_pred)
    if anchor is None:
        out["e_a_reason"] = anchor_reason
    elif a_hat is None:
        out["e_a_reason"] = a_hat_reason
    else:
        out["e_a"] = horizontal_error(a_hat, anchor)

    # Ratio and ratio band.
    if dist_pred_anchor is None:
        out["ratio_reason"] = pred_reason or anchor_reason
    elif d_cmd is None:
        out["ratio_reason"] = d_cmd_reason
    elif d_cmd <= 0.0:
        out["ratio_reason"] = "commanded distance is not positive, ratio undefined"
    else:
        out["ratio"] = dist_pred_anchor / d_cmd
        out["ratio_in_band"] = RATIO_BAND[0] <= out["ratio"] <= RATIO_BAND[1]
    if out["ratio_reason"] is not None:
        out["ratio_in_band_reason"] = out["ratio_reason"]

    # Best-of-frames oracle SR@1m.
    if gt is None:
        out["success_best_of_frames_1m_reason"] = "no GT point in row"
    elif pred is None:
        # Failure to answer is failure (metrics.md section 4), in both
        # frames; same convention as success.score_episode.
        out["success_best_of_frames_1m"] = False
        out["success_best_of_frames_1m_reason"] = "no predicted point; scored as failure"
    else:
        d_best = horizontal_error(pred, gt)
        if anchor is None:
            out["success_best_of_frames_1m_reason"] = (
                "no anchor_center: mirrored frame undefined, scored against GT frame only"
            )
        else:
            d_best = min(d_best, horizontal_error(pred, mirrored_gt(gt, anchor)))
        out["success_best_of_frames_1m"] = d_best <= PRIMARY_TAU

    return out


def _median(vals: List[float]) -> float:
    """Median of a non-empty list (average of middle two when even)."""
    s = sorted(vals)
    n = len(s)
    mid = n // 2
    if n % 2 == 1:
        return s[mid]
    return 0.5 * (s[mid - 1] + s[mid])


def summarize(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate decomposition fields over episode rows.

    Each component is aggregated only over the rows where it exists
    (metrics.md section 4: conditioned on valid output, with the n
    stated); the *_missing counts make the conditioning visible. Rates
    and means are None when no row carries the component.

    Returns a flat dict:
    - n: total rows
    - e_r_mean/median/n/missing, and likewise for e_theta_deg, e_a,
      ratio
    - frame_flip_rate, frame_flip_n, frame_flip_missing
    - ratio_band_rate, ratio_band_n, ratio_band_missing
    - sr_best_of_frames_1m, best_of_frames_n, best_of_frames_missing
    """
    n = len(rows)
    out: Dict[str, Any] = {"n": n}

    for name in ("e_r", "e_theta_deg", "e_a", "ratio"):
        vals = [r[name] for r in rows if r.get(name) is not None]
        out[name + "_mean"] = (sum(vals) / len(vals)) if vals else None
        out[name + "_median"] = _median(vals) if vals else None
        out[name + "_n"] = len(vals)
        out[name + "_missing"] = n - len(vals)

    rate_specs = (
        ("frame_flip", "frame_flip_rate", "frame_flip_n", "frame_flip_missing"),
        ("ratio_in_band", "ratio_band_rate", "ratio_band_n", "ratio_band_missing"),
        (
            "success_best_of_frames_1m",
            "sr_best_of_frames_1m",
            "best_of_frames_n",
            "best_of_frames_missing",
        ),
    )
    for field, rate_key, n_key, missing_key in rate_specs:
        flags = [bool(r[field]) for r in rows if r.get(field) is not None]
        out[rate_key] = (sum(flags) / len(flags)) if flags else None
        out[n_key] = len(flags)
        out[missing_key] = n - len(flags)

    return out
