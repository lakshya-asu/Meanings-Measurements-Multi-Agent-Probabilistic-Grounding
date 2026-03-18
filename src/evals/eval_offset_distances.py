#!/usr/bin/env python3
import argparse
import json
import os
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd


def _euclid(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


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

    # 1) Point mode: target_point_xyz
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
        anchor_to_pred_m = None
        anchor_to_gt_m = None

        if pred is not None:
            error_m = _euclid(pred, gt)

        if anchor is not None:
            anchor_to_gt_m = _euclid(anchor, gt)
            if pred is not None:
                anchor_to_pred_m = _euclid(anchor, pred)

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
                "anchor_x": float(anchor[0]) if anchor is not None else np.nan,
                "anchor_y": float(anchor[1]) if anchor is not None else np.nan,
                "anchor_z": float(anchor[2]) if anchor is not None else np.nan,
                "anchor_to_gt_m": anchor_to_gt_m if anchor_to_gt_m is not None else np.nan,
                "anchor_to_pred_m": anchor_to_pred_m if anchor_to_pred_m is not None else np.nan,
            }
        )

    out_df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    out_df.to_csv(args.out, index=False)
    print(f"[OK] Wrote: {args.out}  (rows={len(out_df)})")


if __name__ == "__main__":
    main()