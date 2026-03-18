#!/usr/bin/env python3
"""
List semantic objects by region for HM3D scenes.

Two modes:
1) Single region:
   python src/tools/list_region_objects.py \
     --scene-dir 00006-HkseAnWCgqk --scene-id HkseAnWCgqk --region-id _9

2) All regions that contain a category substring (e.g., "couch"/"sofa"):
   python src/tools/list_region_objects.py \
     --scene-dir 00006-HkseAnWCgqk --scene-id HkseAnWCgqk --regions-with couch

Options:
  --object-filter: substring applied when printing objects (does not affect region selection in --regions-with mode)
  --max-list:      max objects to print per region (default 500)
  --out-json:      optional JSON dump of results
"""

import argparse
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Optional, Dict, Any, List, Tuple

import numpy as np
from src.envs.habitat_interface import HabitatInterface

DATASET_BASE_PATH = "/datasets/hm3d/train"

# ---------- helpers ----------
def call_maybe(x):
    return x() if callable(x) else x

def safe_center(aabb) -> np.ndarray:
    try:
        c = call_maybe(aabb.center)
    except Exception:
        c = (np.asarray(aabb.max, dtype=np.float32) + np.asarray(aabb.min, dtype=np.float32)) / 2.0
    return np.asarray(c, dtype=np.float32)

def safe_cat_name(cat) -> Optional[str]:
    if cat is None:
        return None
    try:
        n = call_maybe(cat.name)
    except Exception:
        try:
            n = cat.name()
        except Exception:
            n = None
    return None if n is None else str(n).lower()

def valid_xyz(v) -> bool:
    v = np.asarray(v, dtype=np.float32)
    return v.shape[-1] == 3 and np.all(np.isfinite(v))

def aabb_is_degenerate(region) -> bool:
    try:
        aabb = region.aabb
        mn = np.asarray(aabb.min, dtype=np.float32)
        mx = np.asarray(aabb.max, dtype=np.float32)
        if not (np.isfinite(mn).all() and np.isfinite(mx).all()):
            return True
        if np.allclose(mn, 0.0) and np.allclose(mx, 0.0):
            return True
        if np.linalg.norm(mx - mn) <= 1e-3:
            return True
        return False
    except Exception:
        return True

def get_region_center(region) -> Optional[np.ndarray]:
    if region is None:
        return None
    if not aabb_is_degenerate(region):
        try:
            ctr = safe_center(region.aabb)
            if valid_xyz(ctr):
                return ctr
        except Exception:
            pass
    # fallback: mean of object centers
    objs = getattr(region, "objects", []) or []
    cents = []
    for o in objs:
        try:
            c = safe_center(o.aabb)
            if valid_xyz(c):
                cents.append(c)
        except Exception:
            continue
    if cents:
        return np.mean(np.stack(cents, axis=0), axis=0)
    return None

def make_hab_cfg():
    return SimpleNamespace(
        scene_type="hm3d",
        dataset_type="train",
        sim_gpu=0,
        inflation_radius=0.25,
        img_width=640,
        img_height=480,
        camera_height=1.5,
        camera_tilt_deg=-30,
        agent_z_offset=0.0,
        hfov=90.0,
        z_offset=0,
        use_semantic_data=True,
    )

# simple synonym shim for common words
SYNONYMS = {
    "couch": ["sofa", "couch", "loveseat"],
    "sofa":  ["sofa", "couch", "loveseat"],
}

def _norm_tokens(s: str) -> List[str]:
    s = (s or "").strip().lower()
    if s in SYNONYMS:
        return SYNONYMS[s]
    return [s] if s else []

def _region_to_dict(region, obj_rows, hist) -> Dict[str, Any]:
    return {
        "region_id": getattr(region, "id", None),
        "region_name": safe_cat_name(getattr(region, "category", None)),
        "region_center": (get_region_center(region).tolist()
                          if get_region_center(region) is not None else None),
        "aabb_degenerate": bool(aabb_is_degenerate(region)),
        "objects": obj_rows,
        "class_histogram": hist,
    }

def list_objects_in_region(region, object_filter: str, max_list: int) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    objs = getattr(region, "objects", []) or []
    rows, hist = [], {}
    flt = (object_filter or "").strip().lower()
    for o in objs:
        nm = safe_cat_name(o.category) or "unknown"
        c = safe_center(o.aabb)
        if not valid_xyz(c):
            continue
        if flt and flt not in nm:
            continue
        rows.append({
            "id": o.id,
            "name": nm,
            "position": [float(c[0]), float(c[1]), float(c[2])],
        })
        hist[nm] = hist.get(nm, 0) + 1
    rows.sort(key=lambda d: (d["name"], d["id"]))
    return rows[:max_list], hist

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene-dir", required=True)
    ap.add_argument("--scene-id", required=True)

    sel = ap.add_mutually_exclusive_group(required=True)
    sel.add_argument("--region-id", help="Region id string (e.g., _9)")
    sel.add_argument("--regions-with", help="Substring category to select regions by contained object (e.g., 'couch'/'sofa')")

    ap.add_argument("--object-filter", type=str, default="", help="Substring filter for printed objects")
    ap.add_argument("--max-list", type=int, default=500, help="Max objects printed per region")
    ap.add_argument("--out-json", type=str, default="", help="Optional: write results to JSON")
    args = ap.parse_args()

    glb = os.path.join(DATASET_BASE_PATH, args.scene_dir, f"{args.scene_id}.basis.glb")
    assert os.path.exists(glb), f"Missing: {glb}"

    habitat_data = HabitatInterface(glb, cfg=make_hab_cfg(), device="cpu")
    sim = habitat_data._sim
    scene = sim.semantic_scene

    regions = list(getattr(scene, "regions", []) or [])
    assert regions, "[ERR] No regions present in this scene."

    results_json: List[Dict[str, Any]] = []

    # --- Mode 1: specific region id ---
    if args.region_id:
        rid = str(args.region_id)
        region = next((r for r in regions if str(getattr(r, "id", "")) == rid), None)
        if region is None:
            print(f"[ERR] Region {rid} not found. Available ids:")
            for r in regions:
                print(" ", getattr(r, "id", None))
            return

        rname = safe_cat_name(getattr(region, "category", None))
        rcenter = get_region_center(region)
        deg = aabb_is_degenerate(region)

        print(f"\n=== Region {rid} ===")
        print(f"name: {rname}   aabb_degenerate: {deg}")
        if rcenter is not None:
            print(f"center_est: {np.round(rcenter, 4).tolist()}")

        rows, hist = list_objects_in_region(region, args.object_filter, args.max_list)

        print(f"\nobjects_in_region (after filter='{(args.object_filter or '').strip().lower()}') : {len(rows)}")
        for r in rows:
            p = np.round(np.array(r["position"]), 3).tolist()
            print(f"  - {r['id']:>24} | {r['name']:<18} | {p}")

        print("\nclass_histogram (top 40):")
        for k, v in sorted(hist.items(), key=lambda kv: (-kv[1], kv[0]))[:40]:
            print(f"  {k:>18}: {v}")

        results_json.append(_region_to_dict(region, rows, hist))

    # --- Mode 2: all regions that *contain* an object category substring ---
    else:
        tokens = _norm_tokens(args.regions_with)
        assert tokens, "[ERR] --regions-with must be a non-empty string"
        print(f"\n=== Regions containing any of: {tokens} ===")

        matched_regions = []
        for r in regions:
            objs = getattr(r, "objects", []) or []
            names = [(safe_cat_name(o.category) or "") for o in objs]
            if any(any(tok in nm for tok in tokens) for nm in names):
                matched_regions.append(r)

        if not matched_regions:
            print("No regions matched.")
        else:
            for r in matched_regions:
                rid = getattr(r, "id", None)
                rname = safe_cat_name(getattr(r, "category", None))
                rcenter = get_region_center(r)
                deg = aabb_is_degenerate(r)
                # count matches of the selector tokens
                objs = getattr(r, "objects", []) or []
                match_count = sum(1 for o in objs if any(tok in (safe_cat_name(o.category) or "") for tok in tokens))

                print(f"\n--- Region {rid} (name={rname})  aabb_degenerate={deg}  matches={match_count}")
                if rcenter is not None:
                    print(f"center_est: {np.round(rcenter, 4).tolist()}")

                # list ALL (optionally filtered for printing by --object-filter)
                rows, hist = list_objects_in_region(r, args.object_filter, args.max_list)
                print(f"objects_in_region (after filter='{(args.object_filter or '').strip().lower()}') : {len(rows)}")
                for rr in rows:
                    p = np.round(np.array(rr["position"]), 3).tolist()
                    print(f"  - {rr['id']:>24} | {rr['name']:<18} | {p}")

                print("class_histogram (top 40):")
                for k, v in sorted(hist.items(), key=lambda kv: (-kv[1], kv[0]))[:40]:
                    print(f"  {k:>18}: {v}")

                results_json.append(_region_to_dict(r, rows, hist))

    # optional JSON
    if args.out_json:
        out = {
            "scene_dir": args.scene_dir,
            "scene_id": args.scene_id,
            "mode": ("region_id" if args.region_id else "regions_with"),
            "query": (args.region_id or args.regions_with),
            "object_filter": (args.object_filter or "").strip().lower(),
            "regions": results_json,
        }
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        import json
        with open(args.out_json, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\n[saved] {args.out_json}")

    # cleanup
    try:
        sim.close(destroy=True)
    except Exception:
        pass

if __name__ == "__main__":
    main()
