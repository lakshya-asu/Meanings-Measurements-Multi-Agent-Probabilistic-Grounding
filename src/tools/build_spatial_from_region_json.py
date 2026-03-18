#!/usr/bin/env python3
"""
Build a tiny spatial_dataset_v2 from a region dump JSON and also emit
distance/direction tables wrt a chosen reference object.

Inputs:
- A JSON produced by list_region_objects.py (use --regions-with ... --out-json ...),
  which contains one or multiple regions. We’ll pick one by id.

Example:
  # 1) Dump regions that contain a couch
  python src/tools/list_region_objects.py \
    --scene-dir 00006-HkseAnWCgqk --scene-id HkseAnWCgqk \
    --regions-with couch \
    --out-json src/outputs_v2/couch_regions.json

  # 2) Build 10 questions for region _8 using the 'couch' as reference,
  #    and also compute distances wrt 'dining table'
  python src/tools/build_spatial_from_region_json.py \
    --scene-dir 00006-HkseAnWCgqk --scene-id HkseAnWCgqk \
    --region-json src/outputs_v2/couch_regions.json \
    --region-id _8 \
    --ref-substr "couch" \
    --also-ref2-substr "dining table" \
    --num-questions 10 \
    --min-dist 0.2 --max-dist 2.0 \
    --dataset-out src/outputs_v2/spatial_dataset_v2_single.json \
    --table-out   src/outputs_v2/rel_table_00006.csv
"""

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import csv

SPATIAL_REL = {
    "in front of": np.array([ 0.0, 0.0, -1.0], dtype=np.float32),  # -Z
    "behind":      np.array([ 0.0, 0.0,  1.0], dtype=np.float32),  # +Z
    "to the right of": np.array([ 1.0, 0.0,  0.0], dtype=np.float32),  # +X
    "to the left of":  np.array([-1.0, 0.0,  0.0], dtype=np.float32),  # -X
}

def _to_xyz(p) -> np.ndarray:
    return np.array([float(p[0]), float(p[1]), float(p[2])], dtype=np.float32)

def _horizontal_dist(a: np.ndarray, b: np.ndarray) -> float:
    d = a - b
    return float(np.sqrt(d[0]**2 + d[2]**2))

def _classify_lr_fb(delta_x: float, delta_z: float) -> str:
    # Use the dominant horizontal axis. (-Z) is "front", (+Z) is "behind".
    if abs(delta_x) >= abs(delta_z):
        return "to the right of" if delta_x > 0 else "to the left of"
    else:
        return "in front of" if delta_z < 0 else "behind"

def _gather_candidates(objects: List[Dict[str, Any]],
                       ref_pos: np.ndarray,
                       direction_vec: np.ndarray,
                       desired_dist: float,
                       k: int = 5) -> List[Dict[str, Any]]:
    """Ray-based selection (same idea as your generator, but offline on the JSON)."""
    ray_target = ref_pos + direction_vec * desired_dist
    max_angle_deg = 35.0
    cos_th = math.cos(math.radians(max_angle_deg))
    min_t = 0.4 * desired_dist
    max_t = 1.8 * desired_dist
    dir_u = direction_vec / (np.linalg.norm(direction_vec) + 1e-8)

    cand: List[Dict[str, Any]] = []
    for o in objects:
        c = _to_xyz(o["position"])
        v = c - ref_pos
        # Ignore Y; project in 3D but the dot is okay since dir is horizontal.
        t = float(np.dot(v, dir_u))
        if t <= 0 or not (min_t <= t <= max_t):
            continue
        v_u = v / (np.linalg.norm(v) + 1e-8)
        if float(np.dot(v_u, dir_u)) < cos_th:
            continue
        d_err = float(np.linalg.norm(c - ray_target))
        cand.append({
            "id": o["id"],
            "name": o["name"],
            "position": [float(c[0]), float(c[1]), float(c[2])],
            "distance_to_target": d_err,
        })
    cand.sort(key=lambda x: x["distance_to_target"])
    # Dedup by (id, name)
    seen, uniq = set(), []
    for it in cand:
        key = (it["id"], it["name"])
        if key in seen:
            continue
        seen.add(key)
        uniq.append(it)
        if len(uniq) >= k:
            break
    return uniq

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene-dir", required=True)
    ap.add_argument("--scene-id", required=True)
    ap.add_argument("--region-json", required=True, help="JSON file saved by list_region_objects.py")
    ap.add_argument("--region-id", required=True, help="Which region id from that JSON to use (e.g., _8)")
    ap.add_argument("--ref-substr", required=True, help='Reference object substring (e.g., "couch")')
    ap.add_argument("--also-ref2-substr", default="", help='Optional 2nd ref to compute table against (e.g., "dining table")')
    ap.add_argument("--num-questions", type=int, default=10)
    ap.add_argument("--min-dist", type=float, default=0.2)
    ap.add_argument("--max-dist", type=float, default=2.0)
    ap.add_argument("--dataset-out", required=True)
    ap.add_argument("--table-out", required=True)
    args = ap.parse_args()

    with open(args.region_json, "r") as f:
        data = json.load(f)

    # Find the region dict
    regions = data.get("regions", [])
    region = None
    for r in regions:
        if str(r.get("region_id")) == str(args.region_id):
            region = r
            break
    assert region is not None, f"Region {args.region_id} not found in {args.region_json}"

    objects: List[Dict[str, Any]] = region.get("objects", [])
    assert objects, "No objects in region JSON."

    # Find reference object (first match by substring)
    ref_lc = args.ref_substr.strip().lower()
    ref_obj = next((o for o in objects if ref_lc in (o["name"] or "").lower()), None)
    assert ref_obj is not None, f"Reference substring '{args.ref_substr}' not found in region {args.region_id}"

    ref2_lc = args.also_ref2_substr.strip().lower()
    ref2_obj = None
    if ref2_lc:
        ref2_obj = next((o for o in objects if ref2_lc in (o["name"] or "").lower()), None)

    ref_pos = _to_xyz(ref_obj["position"])
    room_center = np.array(region.get("region_center") or ref_pos.tolist(), dtype=np.float32)

    # ---- 1) Write distance/direction table(s) ----
    def build_rel_table(anchor: Dict[str, Any]) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        anc_pos = _to_xyz(anchor["position"])
        for o in objects:
            if o["id"] == anchor["id"]:
                continue
            p = _to_xyz(o["position"])
            dx, dz = float(p[0] - anc_pos[0]), float(p[2] - anc_pos[2])
            horiz = math.sqrt(dx*dx + dz*dz)
            rel = _classify_lr_fb(dx, dz)
            rows.append({
                "anchor_id": anchor["id"],
                "anchor_name": anchor["name"],
                "object_id": o["id"],
                "object_name": o["name"],
                "object_x": float(p[0]),
                "object_y": float(p[1]),
                "object_z": float(p[2]),
                "dx": dx,
                "dz": dz,
                "horiz_dist": horiz,
                "relation_primary": rel,
            })
        rows.sort(key=lambda r: r["horiz_dist"])
        return rows

    table_rows = build_rel_table(ref_obj)
    if ref2_obj is not None:
        table_rows += build_rel_table(ref2_obj)

    # Save CSV table
    Path(args.table_out).parent.mkdir(parents=True, exist_ok=True)
    if table_rows:
        with open(args.table_out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(table_rows[0].keys()))
            w.writeheader()
            w.writerows(table_rows)
    print(f"[table] wrote {len(table_rows)} rows → {args.table_out}")

    # ---- 2) Build 10-item spatial_dataset_v2 for this (scene, region, ref) ----
    # Pick distances and relations (ensure we actually have a candidate; retry if needed)
    rng = random.Random(123)
    all_rel = list(SPATIAL_REL.keys())
    tasks: List[Dict[str, Any]] = []

    for _ in range(args.num_questions):
        for _retry in range(32):
            rel = rng.choice(all_rel)
            dist = rng.uniform(args.min_dist, args.max_dist)
            cand = _gather_candidates(objects, ref_pos, SPATIAL_REL[rel], dist, k=5)
            if cand:
                gt = dict(cand[0])
                gt.pop("distance_to_target", None)
                # initial pose = room center; yaw to face the reference
                face_vec = ref_pos - room_center
                yaw = float(np.arctan2(face_vec[0], -face_vec[2]))
                # quaternion (x,y,z,w) for yaw about +Y
                half = yaw * 0.5
                q = [0.0, math.sin(half), 0.0, math.cos(half)]

                tasks.append({
                    "scene_dir": args.scene_dir,
                    "scene_id": args.scene_id,
                    "scene": args.scene_dir,
                    "question": f"What object is approximately {round(dist,1)} meters {rel} the {ref_obj['name']}?",
                    "initial_pose": {
                        "position": [float(room_center[0]), float(room_center[1]), float(room_center[2])],
                        "rotation": q,
                    },
                    "reference_object": {
                        "id": ref_obj["id"],
                        "name": ref_obj["name"],
                        "position": [float(ref_pos[0]), float(ref_pos[1]), float(ref_pos[2])],
                    },
                    "reference_room": {
                        "id": args.region_id,
                        "name": region.get("region_name"),
                        "center": [float(room_center[0]), float(room_center[1]), float(room_center[2])],
                    },
                    "candidate_targets": cand,
                    "ground_truth_target": gt,
                })
                break

    Path(args.dataset_out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.dataset_out, "w") as f:
        json.dump(tasks, f, indent=2)
    print(f"[dataset] wrote {len(tasks)} items → {args.dataset_out}")

if __name__ == "__main__":
    main()
