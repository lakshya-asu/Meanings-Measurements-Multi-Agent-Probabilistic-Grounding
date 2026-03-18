#!/usr/bin/env python3
import argparse
import os
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _parse_inputs(items: List[str]) -> List[Tuple[str, str]]:
    """
    Expect repeated args like:
      --input msp_point=/path/to/eval_msp_point.csv
      --input baseline_point=/path/to/eval_baseline_point.csv
    """
    out = []
    for s in items:
        if "=" not in s:
            raise ValueError(f"--input must be NAME=PATH, got: {s}")
        name, path = s.split("=", 1)
        name = name.strip()
        path = path.strip()
        if not name:
            raise ValueError(f"Empty method name in --input: {s}")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Input CSV not found: {path}")
        out.append((name, path))
    return out


def _method_summary(df: pd.DataFrame, method: str) -> dict:
    d = df[df["method"] == method].copy()
    total = len(d)

    # "has prediction" = error_m not NaN
    has_pred = d["error_m"].notna()
    n_pred = int(has_pred.sum())

    # success can be True/False/None
    success = d["Success"]
    n_success = int((success == True).sum())  # noqa: E712
    success_rate = (n_success / total) if total > 0 else np.nan

    # error stats on rows that have pred
    derr = d.loc[has_pred, "error_m"].astype(float)
    mean_err = float(derr.mean()) if len(derr) else np.nan
    med_err = float(derr.median()) if len(derr) else np.nan
    p90_err = float(np.percentile(derr, 90)) if len(derr) else np.nan

    return {
        "method": method,
        "n_rows": total,
        "n_pred": n_pred,
        "n_success": n_success,
        "success_rate": success_rate,
        "mean_error_m": mean_err,
        "median_error_m": med_err,
        "p90_error_m": p90_err,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", action="append", required=True, help="Repeatable: NAME=/path/to/eval.csv")
    ap.add_argument("--out_dir", required=True, help="Output directory for merged csv + plots")
    args = ap.parse_args()

    inputs = _parse_inputs(args.input)
    os.makedirs(args.out_dir, exist_ok=True)

    # Load and normalize
    dfs = []
    for name, path in inputs:
        df = pd.read_csv(path)
        if "qid" not in df.columns:
            raise ValueError(f"{path} missing 'qid' column")
        if "method" not in df.columns:
            df["method"] = name
        else:
            # overwrite with the provided name so merges are consistent
            df["method"] = name
        dfs.append(df)

    # Merge: outer join on qid/scene/floor (scene/floor might be missing in some baselines; handle gracefully)
    # Strategy:
    # 1) build a master index of qids
    all_qids = sorted(set().union(*[set(d["qid"].astype(str).tolist()) for d in dfs]))
    master = pd.DataFrame({"qid": all_qids})

    # helper: choose join cols if present
    def merge_one(master_df: pd.DataFrame, d: pd.DataFrame, method: str) -> pd.DataFrame:
        d = d.copy()
        d["qid"] = d["qid"].astype(str)

        keep_cols = [
            "qid",
            "Success",
            "error_m",
            "vlm_steps",
            "overall_steps",
            "traj_length",
            "confidence_level",
            "is_confident",
            "mode",
            "answer_mode",
            "pred_kind",
        ]
        # also keep scene/floor if present (useful for filtering later)
        for c in ["scene", "floor", "predicate", "distance_m"]:
            if c in d.columns and c not in keep_cols:
                keep_cols.append(c)

        d = d[[c for c in keep_cols if c in d.columns]]

        # prefix method
        rename = {c: f"{method}__{c}" for c in d.columns if c != "qid"}
        d = d.rename(columns=rename)

        return master_df.merge(d, on="qid", how="left")

    merged = master
    for (method, _path), d in zip(inputs, dfs):
        merged = merge_one(merged, d, method)

    merged_path = os.path.join(args.out_dir, "merged_eval.csv")
    merged.to_csv(merged_path, index=False)

    # Build per-method summaries from original dfs (cleaner)
    summaries = []
    for method, _path in inputs:
        dfm = next(dd for (nm, _), dd in zip(inputs, dfs) if nm == method)
        summaries.append(_method_summary(dfm, method))

    summary_df = pd.DataFrame(summaries)
    summary_path = os.path.join(args.out_dir, "summary_by_method.csv")
    summary_df.to_csv(summary_path, index=False)

    # -------- plots --------
    # Plot 1: error hist per method
    for method, _path in inputs:
        dfm = next(dd for (nm, _), dd in zip(inputs, dfs) if nm == method)
        derr = dfm["error_m"].dropna().astype(float)
        if len(derr) == 0:
            continue
        plt.figure()
        plt.hist(derr.values, bins=30)
        plt.title(f"Error histogram: {method}")
        plt.xlabel("Euclidean error (m)")
        plt.ylabel("count")
        plt.tight_layout()
        plt.savefig(os.path.join(args.out_dir, f"error_hist_{method}.png"))
        plt.close()

    # Plot 2: boxplot across methods
    box_data = []
    labels = []
    for method, _path in inputs:
        dfm = next(dd for (nm, _), dd in zip(inputs, dfs) if nm == method)
        derr = dfm["error_m"].dropna().astype(float)
        if len(derr) == 0:
            continue
        box_data.append(derr.values)
        labels.append(method)

    if box_data:
        plt.figure()
        plt.boxplot(box_data, labels=labels, showfliers=False)
        plt.title("Error boxplot by method")
        plt.ylabel("Euclidean error (m)")
        plt.tight_layout()
        plt.savefig(os.path.join(args.out_dir, "error_boxplot.png"))
        plt.close()

    # Plot 3: success rate bar chart
    if "success_rate" in summary_df.columns and len(summary_df):
        plt.figure()
        plt.bar(summary_df["method"].values, summary_df["success_rate"].values)
        plt.title("Success rate by method")
        plt.ylabel("success_rate")
        plt.ylim(0, 1.0)
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        plt.savefig(os.path.join(args.out_dir, "success_rate.png"))
        plt.close()

    print(f"[OK] merged:  {merged_path}")
    print(f"[OK] summary: {summary_path}")
    print(f"[OK] plots in: {args.out_dir}")


if __name__ == "__main__":
    main()