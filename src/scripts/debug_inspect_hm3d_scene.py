#!/usr/bin/env python3
"""
Minimal one-scene debug (robust):
- Lists regions with degenerate-AABB flags and object-centroid estimates
- Picks a reference object from REFERENCE_CATEGORIES
- Finds a 'containing' room robustly (containment if possible, else nearest by object-centroid)
- Computes a robust room_center: region AABB center (if valid) -> region object-centroid -> local object cluster -> ref_pos
- Always snaps the chosen center to the navmesh
- Builds and prints a "would-be" dataset item (NOT saved)

Run:
  python src/scripts/debug_inspect_hm3d_scene.py \
    --scene-dir 00149-UuwwmrTsfBN --scene-id UuwwmrTsfBN --num-candidates 5
"""

import argparse
import os
import json
from types import SimpleNamespace
from typing import Optional, Dict, Any, List

import numpy as np
import habitat_sim
from habitat_sim.utils.common import quat_from_angle_axis

from src.envs.habitat_interface import HabitatInterface

# ---- HM3D root ----
DATASET_BASE_PATH = "/datasets/hm3d/train"

REFERENCE_CATEGORIES = {c.lower() for c in [
    "sofa", "bed", "oven", "refrigerator", "dining table", "tv stand",
    "armchair", "desk", "toilet", "kitchen island",
]}

SPATIAL_REL = {
    "in front of": np.array([0.0, 0.0, -1.0], dtype=np.float32),
    "behind":      np.array([0.0, 0.0,  1.0], dtype=np.float32),
    "to the right of": np.array([1.0, 0.0, 0.0], dtype=np.float32),
    "to the left of":  np.array([-1.0,0.0, 0.0], dtype=np.float32),
}

# ---------- helpers ----------
def safe_call(x):
    return x() if callable(x) else x

def safe_center(aabb) -> np.ndarray:
    try:
        c = safe_call(aabb.center)
    except Exception:
        c = (np.asarray(aabb.max, dtype=np.float32) + np.asarray(aabb.min, dtype=np.float32)) / 2.0
    return np.asarray(c, dtype=np.float32)

def safe_cat_name(cat) -> Optional[str]:
    if cat is None:
        return None
    try:
        n = safe_call(cat.name)
    except Exception:
        try:
            n = cat.name()
        except Exception:
            n = None
    return None if n is None else str(n).lower()

def valid_xyz(v) -> bool:
    v = np.asarray(v, dtype=np.float32)
    return v.shape[-1] == 3 and np.all(np.isfinite(v))

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
        hfov=120,
        z_offset=0,
        use_semantic_data=True,
    )

def open_sim(scene_dir: str, scene_id: str):
    basis_glb = os.path.join(DATASET_BASE_PATH, scene_dir, f"{scene_id}.basis.glb")
    if not os.path.exists(basis_glb):
        print(f"[FATAL] Missing GLB: {basis_glb}")
        return None, None
    habitat_data = None
    try:
        hab_cfg = make_hab_cfg()
        habitat_data = HabitatInterface(basis_glb, cfg=hab_cfg, device="cpu")
        sim = habitat_data._sim
        if not sim.semantic_scene or not getattr(sim.semantic_scene, "objects", None):
            print(f"[FATAL] No semantics for {scene_dir}/{scene_id}")
            habitat_data._sim.close(destroy=True)
            return None, None
        return habitat_data, sim
    except Exception as e:
        print(f"[FATAL] HabitatInterface failed: {e}")
        try:
            if habitat_data is not None:
                habitat_data._sim.close(destroy=True)
        except Exception:
            pass
        return None, None

def aabb_is_degenerate(region) -> bool:
    """True if AABB is zero-volume or all zeros."""
    try:
        aabb = region.aabb
        mn = np.asarray(aabb.min, dtype=np.float32)
        mx = np.asarray(aabb.max, dtype=np.float32)
        if not (np.all(np.isfinite(mn)) and np.all(np.isfinite(mx))):
            return True
        if np.linalg.norm(mx - mn) <= 1e-3:
            return True
        if np.allclose(mn, 0.0) and np.allclose(mx, 0.0):
            return True
        return False
    except Exception:
        return True

def region_contains_point(region, p: np.ndarray) -> bool:
    """Containment is meaningful only if AABB is non-degenerate."""
    try:
        if aabb_is_degenerate(region):
            return False
        aabb = region.aabb
        p = np.asarray(p, dtype=np.float32)
        mn = np.asarray(aabb.min, dtype=np.float32)
        mx = np.asarray(aabb.max, dtype=np.float32)
        return np.all(p >= mn) and np.all(p <= mx)
    except Exception:
        return False

def region_object_centroid(region) -> Optional[np.ndarray]:
    objs = getattr(region, "objects", []) or []
    cents = []
    for o in objs:
        try:
            c = safe_center(o.aabb)
            if valid_xyz(c) and not np.allclose(c, 0.0):
                cents.append(c)
        except Exception:
            continue
    if cents:
        return np.mean(np.stack(cents, axis=0), axis=0)
    return None

def get_region_center(region) -> Optional[np.ndarray]:
    """Prefer valid AABB center; else object-centroid; else None."""
    if region is None:
        return None
    if not aabb_is_degenerate(region):
        try:
            ctr = safe_center(region.aabb)
            if valid_xyz(ctr):
                return ctr
        except Exception:
            pass
    return region_object_centroid(region)

def estimate_local_center(scene, ref_pos: np.ndarray, radius: float = 3.5) -> Optional[np.ndarray]:
    """Mean of object centers within `radius` of ref_pos."""
    pts = []
    for obj in scene.objects:
        try:
            c = safe_center(obj.aabb)
        except Exception:
            continue
        if not valid_xyz(c):
            continue
        if np.linalg.norm(c - ref_pos) <= radius:
            pts.append(c)
    if pts:
        return np.mean(np.stack(pts, axis=0), axis=0)
    return None

def robust_find_room_for_point(scene, ref_pos: np.ndarray):
    """Containment if possible; else nearest region by object-centroid (ignoring degenerate centers)."""
    regions = getattr(scene, "regions", []) or []
    # 1) containment
    for r in regions:
        if region_contains_point(r, ref_pos):
            return r, "containment"
    if not regions:
        return None, "no_regions"
    # 2) nearest by centroid
    scored = []
    for r in regions:
        cen = region_object_centroid(r)
        if cen is None:
            continue
        scored.append((r, float(np.linalg.norm(cen - ref_pos))))
    if scored:
        r, _ = min(scored, key=lambda t: t[1])
        return r, "nearest_centroid"
    # 3) give up
    return None, "no_valid_centroid"

def robust_room_center(scene, sim, ref_pos: np.ndarray, region) -> np.ndarray:
    """Valid region center -> local cluster mean -> ref_pos; then navmesh-snap."""
    rc = get_region_center(region)
    if rc is None or np.allclose(rc, 0.0):
        rc = estimate_local_center(scene, ref_pos, radius=3.5)
    if rc is None:
        rc = np.array(ref_pos, dtype=np.float32)
    # snap to navmesh
    try:
        rc = np.asarray(sim.pathfinder.snap_point(np.asarray(rc, dtype=np.float32)), dtype=np.float32)
    except Exception:
        rc = np.asarray(rc, dtype=np.float32)
    return rc

def gather_candidates(scene, ref_pos: np.ndarray, direction_vec: np.ndarray,
                      desired_dist: float, k: int = 5) -> List[Dict[str, Any]]:
    ray_target = ref_pos + direction_vec * desired_dist
    cand = []
    max_angle_deg = 35.0
    cos_th = np.cos(np.deg2rad(max_angle_deg))
    min_t = 0.4 * desired_dist
    max_t = 1.8 * desired_dist
    dir_u = direction_vec / (np.linalg.norm(direction_vec) + 1e-8)

    for obj in scene.objects:
        if not obj or obj.category is None:
            continue
        c = safe_center(obj.aabb)
        if not valid_xyz(c):
            continue
        v = c - ref_pos
        t = float(np.dot(v, dir_u))
        if t <= 0:
            continue
        if not (min_t <= t <= max_t):
            continue
        v_u = v / (np.linalg.norm(v) + 1e-8)
        if float(np.dot(v_u, dir_u)) < cos_th:
            continue
        d_to_ray_target = float(np.linalg.norm(c - ray_target))
        cand.append({
            "id": obj.id,
            "name": safe_cat_name(obj.category) or "object",
            "position": c.astype(float).tolist(),
            "distance_to_target": d_to_ray_target
        })
    cand.sort(key=lambda x: x["distance_to_target"])
    # dedup
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

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene-dir", required=True)
    ap.add_argument("--scene-id", required=True)
    ap.add_argument("--num-candidates", type=int, default=5)
    ap.add_argument("--relation", default="in front of", choices=list(SPATIAL_REL.keys()))
    ap.add_argument("--distance", type=float, default=2.0)
    args = ap.parse_args()

    hd, sim = open_sim(args.scene_dir, args.scene_id)
    assert sim is not None, "Failed to open sim."

    scene = sim.semantic_scene
    regions = getattr(scene, "regions", []) or []
    print("\n=== Scene Summary ===")
    print(f"scene_dir: {args.scene_dir}  scene_id: {args.scene_id}")
    print(f"num_objects: {len(scene.objects)}")
    print(f"num_regions (rooms): {len(regions)}")

    # Sample objects
    print("\nSample objects [id, category, center]:")
    for obj in (scene.objects[:10] if len(scene.objects) > 10 else scene.objects):
        nm = safe_cat_name(obj.category)
        c = safe_center(obj.aabb)
        print(f"  - {obj.id:>20} | {nm:<20} | {np.round(c,3).tolist()}")

    # Rooms info
    print("\nRooms/Regions (first up to 12):")
    for i, r in enumerate(regions[:12]):
        rid = getattr(r, "id", None)
        rname = safe_cat_name(getattr(r, "category", None))
        deg = aabb_is_degenerate(r)
        try:
            aabb = r.aabb
            mn = np.round(np.asarray(aabb.min, dtype=np.float32), 3).tolist()
            mx = np.round(np.asarray(aabb.max, dtype=np.float32), 3).tolist()
            ctr = None if deg else np.round(safe_center(aabb), 3).tolist()
        except Exception:
            mn, mx, ctr, deg = None, None, None, True
        n_objs = len(getattr(r, "objects", []) or [])
        obj_cent = region_object_centroid(r)
        obj_cent_str = None if obj_cent is None else np.round(obj_cent, 3).tolist()
        print(f"  [{i}] id={rid} name={rname} degenerate_aabb={deg} "
              f"aabb_min={mn} aabb_max={mx} aabb_center={ctr} "
              f"objects_in_region={n_objs} objects_centroid={obj_cent_str}")

    # Pick a reference object
    ref_obj = None
    for obj in scene.objects:
        nm = safe_cat_name(obj.category)
        if nm in REFERENCE_CATEGORIES:
            ref_obj = obj
            break
    if ref_obj is None:
        print("\n[WARN] No reference-category object found. Exiting.")
        try:
            hd._sim.close(destroy=True)
        except Exception:
            pass
        return

    ref_name = safe_cat_name(ref_obj.category)
    ref_pos = safe_center(ref_obj.aabb)
    print("\nChosen reference object:")
    print(f"  id={ref_obj.id} name={ref_name} center={np.round(ref_pos,3).tolist()}")

    # Robust 'containing' room + center
    region, how = robust_find_room_for_point(scene, ref_pos)
    print(f"\nRoom selection method: {how}")
    if region is None:
        print("[WARN] No usable region found; will rely on local object cluster / ref_pos.")
    else:
        rid = getattr(region, "id", None)
        rname = safe_cat_name(getattr(region, "category", None))
        print(f"  chosen_region id={rid} name={rname} "
              f"degenerate_aabb={aabb_is_degenerate(region)}")
    room_center = robust_room_center(scene, sim, ref_pos, region)

    # Containment check (only meaningful if AABB valid)
    inside = (region is not None) and region_contains_point(region, ref_pos)
    print(f"  ref_pos inside region aabb? {inside}")

    # Show center variants
    aabb_ctr = get_region_center(region) if region is not None else None
    obj_ctr = region_object_centroid(region) if region is not None else None
    loc_ctr = estimate_local_center(scene, ref_pos)  # always try
    print("\nCenter candidates:")
    print(f"  region_aabb/centroid = {None if aabb_ctr is None else np.round(aabb_ctr,3).tolist()}")
    print(f"  region_object_centroid = {None if obj_ctr is None else np.round(obj_ctr,3).tolist()}")
    print(f"  local_cluster_centroid = {None if loc_ctr is None else np.round(loc_ctr,3).tolist()}")
    print(f"  ROBUST room_center (snapped) = {np.round(room_center,3).tolist()}")

    # Initial pose facing the ref
    face_vec = ref_pos - room_center
    yaw = float(np.arctan2(face_vec[0], -face_vec[2]))  # -Z forward convention
    q = quat_from_angle_axis(yaw, np.array([0.0, 1.0, 0.0], dtype=np.float32))

    print("\nInitial pose (would-be):")
    print(f"  position (room_center)      = {np.round(room_center,3).tolist()}")
    print(f"  rotation (yaw quaternion)   = {[q.x, q.y, q.z, q.w]}")

    # Candidates (optional)
    rel = args.relation
    dist = float(args.distance)
    dir_vec = SPATIAL_REL[rel]
    cands = gather_candidates(scene, ref_pos, dir_vec, dist, k=max(1, args.num_candidates))
    print(f"\nCandidate targets toward '{rel}' at ~{dist}m (up to {args.num_candidates}):")
    if not cands:
        print("  (none)")
    for i, c in enumerate(cands):
        print(f"  [{i}] id={c['id']} name={c['name']} pos={np.round(np.asarray(c['position']),3).tolist()} d_err={c['distance_to_target']:.3f}")

    # Would-be dataset item (NOT saved)
    gt = dict(cands[0]) if cands else {"id":"", "name":"", "position":[0,0,0]}
    gt.pop("distance_to_target", None)
    item = {
        "scene_dir": args.scene_dir,
        "scene_id": args.scene_id,
        "scene": args.scene_dir,
        "question": f"What object is approximately {dist} meters {rel} the {ref_name}?",
        "initial_pose": {
            "position": room_center.astype(float).tolist(),
            "rotation": [q.x, q.y, q.z, q.w],
        },
        "reference_object": {
            "id": ref_obj.id,
            "name": ref_name,
            "position": ref_pos.astype(float).tolist(),
        },
        "reference_room": {
            "id": getattr(region, "id", None) if region is not None else None,
            "name": safe_cat_name(getattr(region, "category", None)) if region is not None else None,
            "center": room_center.astype(float).tolist(),
        },
        "candidate_targets": cands,
        "ground_truth_target": gt,
    }
    print("\nWould-be dataset JSON entry:")
    print(json.dumps(item, indent=2))

    # Cleanup
    try:
        hd._sim.close(destroy=True)
    except Exception:
        pass

if __name__ == "__main__":
    main()
