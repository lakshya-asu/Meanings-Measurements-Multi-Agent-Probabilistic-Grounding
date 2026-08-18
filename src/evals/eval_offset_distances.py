#!/usr/bin/env python3
import argparse
import json
import os
import sys
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from src.schema.prediction import normalize_prediction
    from src.evals.angular import angular_errors_from_points
    from src.evals.decomposition import decompose_episode, summarize
    from src.evals.success import euclidean_error, horizontal_error
except ImportError:
    # Allow running this script from anywhere, not just the repo root.
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from src.schema.prediction import normalize_prediction
    from src.evals.angular import angular_errors_from_points
    from src.evals.decomposition import decompose_episode, summarize
    from src.evals.success import euclidean_error, horizontal_error


def _euclid(a: np.ndarray, b: np.ndarray) -> float:
    # Shared implementation with the online path (item 5): one
    # implementation per metric lives in src/evals/success.py.
    return euclidean_error(a, b)


def _safe_float(x) -> Optional[float]:
    try:
        if x is None:
            return None
        v = float(x)
        if np.isnan(v) or np.isinf(v):
            return None
        return v
    except Exception:
        return None


def _get_vec3(d: Dict[str, Any], keys: Tuple[str, str, str]) -> Optional[np.ndarray]:
    x = _safe_float(d.get(keys[0]))
    y = _safe_float(d.get(keys[1]))
    z = _safe_float(d.get(keys[2]))
    if x is None or y is None or z is None:
        return None
    return np.array([x, y, z], dtype=np.float32)


def _qid_from_row(i: int, scene: str, floor: Any) -> str:
    # Matches your JSON keys: "{i}_{scene}_{floor}"
    try:
        f = int(float(floor))
    except Exception:
        f = str(floor)
    return f"{i}_{scene}_{f}"


def _extract_pred_xyz(jentry: Dict[str, Any]) -> Tuple[Optional[np.ndarray], str]:
    """
    Return (pred_xyz, pred_kind)
    pred_kind in {"point", "object", "missing"}
    """
    metrics = (jentry or {}).get("metrics", {}) if isinstance(jentry, dict) else {}
    final_pred = metrics.get("final_pred", None)

    if not isinstance(final_pred, dict):
        return None, "missing"

    # 0) Normalize first: maps every legacy key (target_location,
    # target_xyz_hab, selected_object_xyz, ...) onto target_point_xyz.
    final_pred = normalize_prediction(final_pred)

    # 1) Point mode: target_point_xyz (canonical key)
    tp = final_pred.get("target_point_xyz", None)
    if isinstance(tp, (list, tuple)) and len(tp) == 3:
        try:
            return np.array([float(tp[0]), float(tp[1]), float(tp[2])], dtype=np.float32), "point"
        except Exception:
            pass

    # 2) Object mode (optional): if your JSON ever includes xyz for the selected object
    # (Some pipelines store this as selected_object_xyz/selected_object_center_xyz.)
    for k in ("selected_object_xyz", "selected_object_center_xyz", "selected_center_xyz"):
        v = final_pred.get(k, None)
        if isinstance(v, (list, tuple)) and len(v) == 3:
            try:
                return np.array([float(v[0]), float(v[1]), float(v[2])], dtype=np.float32), "object"
            except Exception:
                pass

    return None, "missing"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="questions_msp_sample_1_metric_corrected.csv")
    ap.add_argument("--json", required=True, help="results JSON (e.g., results_where.json or gemini_images_True.json)")
    ap.add_argument("--out", required=True, help="output eval CSV path")
    ap.add_argument("--method", default=None, help="method name label (e.g., msp_point, msp_object, baseline_point)")
    args = ap.parse_args()

    if not os.path.exists(args.csv):
        raise FileNotFoundError(f"CSV not found: {args.csv}")
    if not os.path.exists(args.json):
        raise FileNotFoundError(f"JSON not found: {args.json}")

    df = pd.read_csv(args.csv)

    # required GT point columns
    required = ["scene", "floor", "metric_corrected_x", "metric_corrected_y", "metric_corrected_z"]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"CSV missing required column: {c}")

    # optional anchor center columns
    anchor_cols = ["anchor_center_x", "anchor_center_y", "anchor_center_z"]
    has_anchor = all(c in df.columns for c in anchor_cols)

    # Load JSON
    with open(args.json, "r") as f:
        J = json.load(f)

    rows = []
    decomp_rows = []
    rows_missing_orientation = 0
    for i, r in df.reset_index(drop=True).iterrows():
        scene = str(r["scene"])
        floor = r["floor"]
        qid = _qid_from_row(i, scene, floor)

        gt = np.array(
            [float(r["metric_corrected_x"]), float(r["metric_corrected_y"]), float(r["metric_corrected_z"])],
            dtype=np.float32,
        )

        anchor = None
        if has_anchor:
            try:
                anchor = np.array([float(r["anchor_center_x"]), float(r["anchor_center_y"]), float(r["anchor_center_z"])], dtype=np.float32)
            except Exception:
                anchor = None

        jentry = J.get(qid, None)
        success = None
        mode = None
        answer_mode = None
        is_conf = None
        conf_lvl = None
        vlm_steps = None
        overall_steps = None
        traj_len = None

        pred, pred_kind = None, "missing"
        if isinstance(jentry, dict):
            success = jentry.get("Success", None)
            metrics = jentry.get("metrics", {})
            if isinstance(metrics, dict):
                mode = metrics.get("mode", None)
                answer_mode = metrics.get("answer_mode", None)
                is_conf = metrics.get("is_confident", None)
                conf_lvl = metrics.get("confidence_level", None)
                vlm_steps = metrics.get("vlm_steps", None)
                overall_steps = metrics.get("overall_steps", None)
                traj_len = metrics.get("traj_length", None)
            pred, pred_kind = _extract_pred_xyz(jentry)

        error_m = None
        d_h_m = None
        anchor_to_pred_m = None
        anchor_to_gt_m = None

        if pred is not None:
            # error_m is the 3D Euclidean distance d_3 (secondary column);
            # d_h_m is the horizontal distance d_h that the preregistered
            # primary endpoint SR@1.0m thresholds (y projected out,
            # Habitat y-up convention). Both come from src/evals/success.py
            # so the offline and online paths share one implementation.
            error_m = _euclid(pred, gt)
            d_h_m = horizontal_error(pred, gt)

        if anchor is not None:
            anchor_to_gt_m = _euclid(anchor, gt)
            if pred is not None:
                anchor_to_pred_m = _euclid(anchor, pred)

        # Angular errors (item 7). Both orientations are derived from
        # point geometry per metrics.md section 2.1:
        # e_theta = angle(pred - anchor, gt - anchor); yaw and pitch
        # errors are its horizontal and vertical decomposition. The one
        # canonical implementation lives in src/evals/angular.py.
        ang = angular_errors_from_points(
            pred.tolist() if pred is not None else None,
            gt.tolist(),
            anchor.tolist() if anchor is not None else None,
        )
        if ang["yaw_error_deg"] is None or ang["pitch_error_deg"] is None:
            rows_missing_orientation += 1

        # Error decomposition (item 8): e_r, e_a, ratio band, frame flip
        # and the best-of-frames oracle from src/evals/decomposition.py.
        # Its e_theta_deg equals the angular_error_deg column above (both
        # come from src/evals/angular.py), so no duplicate column is
        # written for it. decompose_episode re-reads the anchor and GT
        # from the raw CSV row, which carries the same columns used here.
        final_pred_raw = None
        if isinstance(jentry, dict):
            jmetrics = jentry.get("metrics", {})
            if isinstance(jmetrics, dict):
                final_pred_raw = jmetrics.get("final_pred")
        decomp = decompose_episode(
            final_pred_raw if isinstance(final_pred_raw, dict) else None,
            r.to_dict(),
        )
        decomp_rows.append(decomp)

        rows.append(
            {
                "method": args.method or "unknown",
                "qid": qid,
                "i": i,
                "scene": scene,
                "floor": floor,
                "predicate": r.get("predicate", None),
                "distance_m": r.get("distance_m", None),
                "Success": success,
                "mode": mode,
                "answer_mode": answer_mode,
                "is_confident": is_conf,
                "confidence_level": conf_lvl,
                "vlm_steps": vlm_steps,
                "overall_steps": overall_steps,
                "traj_length": traj_len,
                "pred_kind": pred_kind,
                "gt_x": float(gt[0]),
                "gt_y": float(gt[1]),
                "gt_z": float(gt[2]),
                "pred_x": float(pred[0]) if pred is not None else np.nan,
                "pred_y": float(pred[1]) if pred is not None else np.nan,
                "pred_z": float(pred[2]) if pred is not None else np.nan,
                "error_m": error_m if error_m is not None else np.nan,
                "d_h_m": d_h_m if d_h_m is not None else np.nan,
                "anchor_x": float(anchor[0]) if anchor is not None else np.nan,
                "anchor_y": float(anchor[1]) if anchor is not None else np.nan,
                "anchor_z": float(anchor[2]) if anchor is not None else np.nan,
                "anchor_to_gt_m": anchor_to_gt_m if anchor_to_gt_m is not None else np.nan,
                "anchor_to_pred_m": anchor_to_pred_m if anchor_to_pred_m is not None else np.nan,
                "yaw_error_deg": ang["yaw_error_deg"] if ang["yaw_error_deg"] is not None else np.nan,
                "pitch_error_deg": ang["pitch_error_deg"] if ang["pitch_error_deg"] is not None else np.nan,
                "angular_error_deg": ang["angular_error_deg"] if ang["angular_error_deg"] is not None else np.nan,
                # Item 8 decomposition columns; blank when unavailable.
                # angular_error_deg above already carries e_theta_deg.
                "e_r_m": decomp["e_r"] if decomp["e_r"] is not None else np.nan,
                "e_a_m": decomp["e_a"] if decomp["e_a"] is not None else np.nan,
                "ratio_pred_over_cmd": decomp["ratio"] if decomp["ratio"] is not None else np.nan,
                "ratio_in_band": decomp["ratio_in_band"],
                "frame_flip": decomp["frame_flip"],
                "success_best_of_frames_1m": decomp["success_best_of_frames_1m"],
            }
        )

    out_df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    out_df.to_csv(args.out, index=False)
    print(f"[OK] Wrote: {args.out}  (rows={len(out_df)})")
    n_scored = len(out_df) - rows_missing_orientation
    print(
        f"[OK] Angular errors: {n_scored}/{len(out_df)} rows scored; "
        f"{rows_missing_orientation} rows lacked orientation "
        f"(missing prediction or anchor, or point coincides with anchor)."
    )

    # Item 8 decomposition summary: every component is aggregated only
    # over the rows where it exists; the missing counts state the
    # conditioning explicitly (metrics.md section 4).
    s = summarize(decomp_rows)
    n = s["n"]

    def _fmt(v, spec):
        return format(v, spec) if v is not None else "n/a"

    print(
        f"[OK] Decomposition (item 8) over {n} rows: "
        f"e_r n={s['e_r_n']} (missing {s['e_r_missing']}) "
        f"median={_fmt(s['e_r_median'], '.3f')} m; "
        f"e_theta n={s['e_theta_deg_n']} (missing {s['e_theta_deg_missing']}) "
        f"median={_fmt(s['e_theta_deg_median'], '.1f')} deg; "
        f"e_a n={s['e_a_n']} (missing {s['e_a_missing']}) "
        f"median={_fmt(s['e_a_median'], '.3f')} m; "
        f"ratio-band rate={_fmt(s['ratio_band_rate'], '.3f')} "
        f"(n={s['ratio_band_n']}, missing {s['ratio_band_missing']}); "
        f"frame-flip rate={_fmt(s['frame_flip_rate'], '.3f')} "
        f"(n={s['frame_flip_n']}, missing {s['frame_flip_missing']}); "
        f"SR@1m best-of-frames={_fmt(s['sr_best_of_frames_1m'], '.3f')} "
        f"(n={s['best_of_frames_n']}, missing {s['best_of_frames_missing']})."
    )


if __name__ == "__main__":
    main()