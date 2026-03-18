#!/usr/bin/env python3
"""
offset_metric.py

Goal:
Given a CSV dataset containing:
  - anchor_center_x/y/z  (anchor object center)
  - ann_pos_x/y/z        (human-annotated target position / superquadric center)
  - distance_m           (desired metric offset in meters)

Create new columns:
  - metric_corrected_x
  - metric_corrected_y
  - metric_corrected_z

Such that:
  metric_corrected = anchor_center + distance_m * unit(ann_pos - anchor_center)

Also adds:
  - metric_error_m        (actual_ann_dist - distance_m)
  - metric_corrected_ok   (False if ann == anchor, direction undefined)
  - ann_dist_m            (distance from anchor to ann_pos)

Default input (per your setup):
  /datasets/explore-eqa/questions_msp_sample_1.csv

Typical usage:
  python3 offset_metric.py \
    --in_csv  /datasets/explore-eqa/questions_msp_sample_1.csv \
    --out_csv /datasets/explore-eqa/questions_msp_sample_1_metric_corrected.csv
"""

import argparse
import os
import sys
from typing import List

import numpy as np
import pandas as pd


EPS = 1e-9

REQUIRED_COLS: List[str] = [
    "anchor_center_x", "anchor_center_y", "anchor_center_z",
    "ann_pos_x", "ann_pos_y", "ann_pos_z",
    "distance_m",
]


def _check_required_columns(df: pd.DataFrame, required: List[str]) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns: {missing}\n"
            f"Available columns: {list(df.columns)}"
        )


def add_metric_corrected_xyz(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds:
      metric_corrected_x/y/z
      ann_dist_m
      metric_error_m
      metric_corrected_ok
    """
    _check_required_columns(df, REQUIRED_COLS)

    # Anchor center A
    Ax = df["anchor_center_x"].to_numpy(dtype=float)
    Ay = df["anchor_center_y"].to_numpy(dtype=float)
    Az = df["anchor_center_z"].to_numpy(dtype=float)

    # Annotated point P
    Px = df["ann_pos_x"].to_numpy(dtype=float)
    Py = df["ann_pos_y"].to_numpy(dtype=float)
    Pz = df["ann_pos_z"].to_numpy(dtype=float)

    # Desired distance
    d = df["distance_m"].to_numpy(dtype=float)

    # v = P - A
    vx = Px - Ax
    vy = Py - Ay
    vz = Pz - Az

    # ||v||
    norm = np.sqrt(vx * vx + vy * vy + vz * vz)

    # unit vector u = v / ||v|| (safe)
    safe = norm > EPS
    ux = np.zeros_like(vx)
    uy = np.zeros_like(vy)
    uz = np.zeros_like(vz)
    ux[safe] = vx[safe] / norm[safe]
    uy[safe] = vy[safe] / norm[safe]
    uz[safe] = vz[safe] / norm[safe]

    # corrected point: A + d * u
    df["metric_corrected_x"] = Ax + d * ux
    df["metric_corrected_y"] = Ay + d * uy
    df["metric_corrected_z"] = Az + d * uz

    # diagnostics
    df["ann_dist_m"] = norm
    df["metric_error_m"] = norm - d
    df["metric_corrected_ok"] = safe

    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--in_csv",
        default="/datasets/explore-eqa/questions_msp_sample_1.csv",
        help="Input CSV path",
    )
    ap.add_argument(
        "--out_csv",
        default="",
        help="Output CSV path. If empty, writes next to input with suffix _metric_corrected.csv",
    )
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting out_csv if it already exists",
    )
    ap.add_argument(
        "--print_stats",
        action="store_true",
        help="Print a quick summary of correction errors",
    )
    args = ap.parse_args()

    in_csv = args.in_csv
    if not os.path.exists(in_csv):
        print(f"[ERROR] Input CSV not found: {in_csv}", file=sys.stderr)
        sys.exit(1)

    if args.out_csv.strip():
        out_csv = args.out_csv
    else:
        base, ext = os.path.splitext(in_csv)
        out_csv = f"{base}_metric_corrected.csv"

    if os.path.exists(out_csv) and not args.overwrite:
        print(
            f"[ERROR] Output CSV already exists: {out_csv}\n"
            f"Use --overwrite or choose a different --out_csv.",
            file=sys.stderr,
        )
        sys.exit(1)

    df = pd.read_csv(in_csv)

    # Add corrected columns
    df = add_metric_corrected_xyz(df)

    # Save
    df.to_csv(out_csv, index=False)

    print(f"[OK] Wrote: {out_csv}")
    print("[OK] Added columns: metric_corrected_x, metric_corrected_y, metric_corrected_z, ann_dist_m, metric_error_m, metric_corrected_ok")

    if args.print_stats:
        ok = df["metric_corrected_ok"].astype(bool)
        if ok.any():
            err = df.loc[ok, "metric_error_m"].to_numpy(dtype=float)
            print("\n--- Stats (only rows with metric_corrected_ok=True) ---")
            print(f"count: {err.size}")
            print(f"mean error (m): {err.mean():.6f}")
            print(f"median error (m): {np.median(err):.6f}")
            print(f"abs mean error (m): {np.abs(err).mean():.6f}")
            print(f"max abs error (m): {np.abs(err).max():.6f}")
        else:
            print("\n[WARN] No rows had metric_corrected_ok=True (ann_pos == anchor_center everywhere?)")


if __name__ == "__main__":
    main()