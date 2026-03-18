#!/usr/bin/env python3
"""
Dump the region (room) and its objects for a given scene + spawn pose (no exploration),
optionally capture a 360° sweep at the spawn, list/capture neighboring rooms, and
NAME IMAGES WITH REPRO COORDINATES/ANGLES.

Filenames include:
- 360°: spawn_x{X}_y{Y}_z{Z}_view_{i}_yaw_{YAW}.png
- neighbor: neighbor_{REGIONID}_x{X}_y{Y}_z{Z}_yaw_{YAW}.png

JSON also records exact floats and camera params for reproducibility.

Example:
  python src/tools/region_dump_from_spawn.py \
    --scene-dir 00006-HkseAnWCgqk --scene-id HkseAnWCgqk \
    --spawn-x 1.1563 --spawn-y 2.2306 --spawn-z -0.5356 --spawn-yaw 5.3783249199074 \
    --out-json src/outputs_v2/region_dump_00006.json \
    --save-360-dir src/outputs_v2/region_dump_00006_360 --views 8 \
    --neighbors-k 6 --neighbor-dist 5.0 --door-region-dist 3.0 \
    --save-neighbor-dir src/outputs_v2/region_dump_00006_neighbors \
    --name-precision 4
"""

import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Optional, Dict, Any, List, Tuple, Set

import numpy as np
import cv2

import habitat_sim
from habitat_sim.utils.common import quat_from_angle_axis

from src.envs.habitat_interface import HabitatInterface

DATASET_BASE_PATH = "/datasets/hm3d/train"
DOOR_CATS = {"door", "door frame", "doorway", "archway", "entrance"}

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

def region_contains_point(region, p: np.ndarray) -> bool:
    try:
        if aabb_is_degenerate(region):
            return False
        aabb = region.aabb
        mn = np.asarray(aabb.min, dtype=np.float32)
        mx = np.asarray(aabb.max, dtype=np.float32)
        p = np.asarray(p, dtype=np.float32)
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

def robust_find_room_for_point(scene, p: np.ndarray):
    regions = getattr(scene, "regions", []) or []
    for r in regions:
        if region_contains_point(r, p):
            return r, "containment"
    scored = []
    for r in regions:
        cen = region_object_centroid(r)
        if cen is not None:
            scored.append((r, float(np.linalg.norm(cen - p))))
    if scored:
        r, _ = min(scored, key=lambda t: t[1])
        return r, "nearest_centroid"
    return None, "no_regions_or_centroids"

def _make_hab_cfg(w=640, h=480, eye_h=1.5, hfov=90.0):
    return SimpleNamespace(
        scene_type="hm3d",
        dataset_type="train",
        sim_gpu=0,
        inflation_radius=0.25,
        img_width=w,
        img_height=h,
        camera_height=eye_h,
        camera_tilt_deg=-30,
        agent_z_offset=0.0,
        hfov=hfov,
        z_offset=0,
        use_semantic_data=True,
    )

def _set_agent_pose(sim: habitat_sim.Simulator, pos_xyz: np.ndarray, yaw_rad: float):
    p = np.asarray(pos_xyz, dtype=np.float32)
    try:
        snapped = sim.pathfinder.snap_point(p)
        if np.isfinite(snapped).all():
            p = np.asarray(snapped, dtype=np.float32)
    except Exception:
        pass
    q = quat_from_angle_axis(float(yaw_rad), np.array([0.0, 1.0, 0.0], dtype=np.float32))
    st = habitat_sim.AgentState()
    st.position = p
    st.rotation = q
    sim.get_agent(0).set_state(st)
    return p, q

def _find_rgb_sensor_uuid(sim: habitat_sim.Simulator) -> Optional[str]:
    try:
        agent = sim.get_agent(0)
        for uuid, s in getattr(agent, "_sensors", {}).items():
            spec = s.specification()
            if str(getattr(spec, "sensor_type", "")).lower().endswith("color"):
                return uuid
            if "rgb" in uuid.lower() or "color" in uuid.lower():
                return uuid
    except Exception:
        pass
    try:
        return next(iter(sim._sensors.keys()))
    except Exception:
        return None

def _grab_rgb(sim: habitat_sim.Simulator, rgb_uuid: str) -> Optional[np.ndarray]:
    try:
        obs = sim.get_sensor_observations()
        frame = obs.get(rgb_uuid, None)
        if frame is None:
            return None
        arr = np.asarray(frame)
        if arr.ndim == 3 and arr.shape[2] == 4:
            arr = arr[:, :, :3]
        return arr
    except Exception:
        return None

def _save_image_rgb(path: Path, rgb: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), bgr)

def _fmt(x: float, nd: int) -> str:
    """format float for filenames; keep minus/dot (safe on Linux)."""
    return f"{float(x):.{nd}f}"

def _capture_360(sim: habitat_sim.Simulator, base_dir: Path, base_yaw: float,
                 spawn_pos: np.ndarray, views: int, name_prec: int) -> List[Dict[str, Any]]:
    base_dir.mkdir(parents=True, exist_ok=True)
    rgb_uuid = _find_rgb_sensor_uuid(sim)
    if rgb_uuid is None:
        print("[warn] No RGB sensor found; skipping 360 capture.")
        return []

    agent = sim.get_agent(0)
    state0 = agent.get_state()

    sx, sy, sz = [ _fmt(v, name_prec) for v in spawn_pos ]
    metas, frames = [], []
    for i in range(views):
        yaw = base_yaw + (2.0 * np.pi * i / max(1, views))
        _set_agent_pose(sim, np.array(state0.position), yaw)
        rgb = _grab_rgb(sim, rgb_uuid)
        if rgb is None:
            print(f"[warn] No RGB on view {i}; skipping.")
            continue
        fname = base_dir / f"spawn_x{sx}_y{sy}_z{sz}_view_{i:02d}_yaw_{_fmt(yaw, name_prec)}.png"
        _save_image_rgb(fname, rgb)
        metas.append({
            "index": i,
            "yaw_rad": float(yaw),
            "file": str(fname),
            "spawn": {"x": float(spawn_pos[0]), "y": float(spawn_pos[1]), "z": float(spawn_pos[2])},
        })
        frames.append(rgb)

    if frames:
        h = min(f.shape[0] for f in frames)
        w = min(f.shape[1] for f in frames)
        resized = [cv2.resize(f, (w, h)) for f in frames]
        pano = np.concatenate(resized, axis=1)
        pano_name = base_dir / f"spawn_x{sx}_y{sy}_z{sz}_panorama.png"
        _save_image_rgb(pano_name, pano)
        metas.append({"panorama_file": str(pano_name)})

    agent.set_state(state0)
    return metas

# ---- adjacency via doors + kNN ----
def _region_centers(scene) -> Dict[str, np.ndarray]:
    centers = {}
    for r in getattr(scene, "regions", []) or []:
        rid = getattr(r, "id", None)
        if rid is None:
            continue
        rc = get_region_center(r)
        if rc is not None and valid_xyz(rc):
            centers[str(rid)] = rc.astype(np.float32)
    return centers

def _door_edges(scene, centers: Dict[str, np.ndarray], max_door_region_dist: float = 3.0) -> Set[Tuple[str, str]]:
    edges: Set[Tuple[str, str]] = set()
    if not centers:
        return edges
    items = list(centers.items())
    for obj in getattr(scene, "objects", []) or []:
        nm = safe_cat_name(obj.category)
        if nm not in DOOR_CATS:
            continue
        c = safe_center(obj.aabb)
        if not valid_xyz(c):
            continue
        # find two nearest region centers to the door
        items_sorted = sorted(items, key=lambda kv: float(np.linalg.norm(kv[1] - c)))
        near = [kv for kv in items_sorted[:3] if float(np.linalg.norm(kv[1] - c)) <= max_door_region_dist]
        if len(near) >= 2:
            a, b = near[0][0], near[1][0]
            if a != b:
                edges.add(tuple(sorted((a, b))))
    return edges

def _knn_edges(centers: Dict[str, np.ndarray], k: int = 4, max_dist: float = 5.0) -> Set[Tuple[str, str]]:
    edges: Set[Tuple[str, str]] = set()
    keys = list(centers.keys())
    for i, a in enumerate(keys):
        ca = centers[a]
        dists = [(b, float(np.linalg.norm(ca - centers[b]))) for b in keys if b != a]
        dists.sort(key=lambda t: t[1])
        for b, d in dists[:k]:
            if d <= max_dist:
                edges.add(tuple(sorted((a, b))))
    return edges

def _neighbors_for(region_id: str, edges: Set[Tuple[str, str]]) -> List[str]:
    n = []
    for a, b in edges:
        if a == region_id:
            n.append(b)
        elif b == region_id:
            n.append(a)
    # dedup preserve order
    seen, out = set(), []
    for x in n:
        if x not in seen:
            out.append(x); seen.add(x)
    return out

def _capture_region_center(sim: habitat_sim.Simulator, center: np.ndarray, face_yaw: float,
                           out_path: Path) -> Optional[str]:
    rgb_uuid = _find_rgb_sensor_uuid(sim)
    if rgb_uuid is None:
        return None
    _set_agent_pose(sim, center, face_yaw)
    rgb = _grab_rgb(sim, rgb_uuid)
    if rgb is None:
        return None
    _save_image_rgb(out_path, rgb)
    return str(out_path)

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene-dir", required=True)
    ap.add_argument("--scene-id", required=True)
    ap.add_argument("--spawn-x", type=float, required=True)
    ap.add_argument("--spawn-y", type=float, required=True)
    ap.add_argument("--spawn-z", type=float, required=True)
    ap.add_argument("--spawn-yaw", type=float, required=True, help="Yaw in radians (Habitat convention)")
    ap.add_argument("--out-json", type=str, default="", help="Optional path to save JSON dump")
    ap.add_argument("--max-list", type=int, default=120, help="Max objects to print from the room")

    # 360° capture options
    ap.add_argument("--save-360-dir", type=str, default="", help="If set, save N views + panorama here")
    ap.add_argument("--views", type=int, default=8, help="Number of views for 360° sweep")

    # neighbor discovery / capture
    ap.add_argument("--neighbors-k", type=int, default=4, help="kNN neighbors per region")
    ap.add_argument("--neighbor-dist", type=float, default=5.0, help="Max distance (m) for kNN edges")
    ap.add_argument("--door-region-dist", type=float, default=3.0, help="Max region-center distance to a door to form an edge")
    ap.add_argument("--save-neighbor-dir", type=str, default="", help="If set, saves one RGB per neighbor region center")

    # filename precision
    ap.add_argument("--name-precision", type=int, default=4, help="Digits after decimal for numbers in filenames")

    args = ap.parse_args()

    glb = os.path.join(DATASET_BASE_PATH, args.scene_dir, f"{args.scene_id}.basis.glb")
    if not os.path.exists(glb):
        raise FileNotFoundError(glb)

    hab_cfg = _make_hab_cfg()
    habitat_data = HabitatInterface(glb, cfg=hab_cfg, device="cpu")
    sim = habitat_data._sim
    scene = sim.semantic_scene
    if not scene or not scene.objects:
        raise RuntimeError("[FATAL] No semantic objects in scene.")

    # Set spawn
    spawn = np.array([args.spawn_x, args.spawn_y, args.spawn_z], dtype=np.float32)
    pose_p, pose_q = _set_agent_pose(sim, spawn, args.spawn_yaw)

    # Region at spawn
    region, method = robust_find_room_for_point(scene, pose_p)
    rid = getattr(region, "id", None) if region else None
    rname = safe_cat_name(getattr(region, "category", None)) if region else None
    rcenter = get_region_center(region) if region else None
    contain_ok = (region_contains_point(region, pose_p) if region else False)
    deg = (aabb_is_degenerate(region) if region else True)

    # Objects in region
    reg_objs = list(getattr(region, "objects", []) or []) if region else []
    objs_dump = []
    class_hist: Dict[str, int] = {}
    for o in reg_objs:
        nm = safe_cat_name(o.category) or "unknown"
        c = safe_center(o.aabb)
        if not valid_xyz(c):
            continue
        objs_dump.append({
            "id": o.id,
            "name": nm,
            "position": [float(c[0]), float(c[1]), float(c[2])],
        })
        class_hist[nm] = class_hist.get(nm, 0) + 1
    objs_dump.sort(key=lambda d: (d["name"], d["id"]))
    class_hist_sorted = sorted(class_hist.items(), key=lambda kv: (-kv[1], kv[0]))

    # ---- PRINT spawn room ----
    print("\n=== Scene ===")
    print(f"scene_dir: {args.scene_dir}  scene_id: {args.scene_id}")
    print(f"num_objects_total: {len(scene.objects)}  num_regions: {len(getattr(scene, 'regions', []) or [])}")

    print("\n=== Spawn Pose (snapped) ===")
    print(f"position: {np.round(pose_p, 6).tolist()}")
    print(f"yaw(rad): {float(args.spawn_yaw)}")

    print("\n=== Region at Spawn ===")
    print(f"select_method: {method}")
    print(f"region_id: {rid}  name: {rname}  aabb_degenerate: {deg}")
    if rcenter is not None:
        print(f"region_center_est: {np.round(rcenter, 6).tolist()}")
    print(f"spawn_inside_region_aabb: {contain_ok}")
    print(f"objects_in_region: {len(objs_dump)}")

    print("\n--- Objects in Region (up to max-list) ---")
    for o in objs_dump[:args.max_list]:
        print(f"  - {o['id']:>24} | {o['name']:<18} | {np.round(np.array(o['position']), 3).tolist()}")
    if len(objs_dump) > args.max_list:
        print(f"  ... ({len(objs_dump)-args.max_list} more)")

    print("\n--- Class Histogram (room) ---")
    for cls, cnt in class_hist_sorted[:40]:
        print(f"  {cls:>18}: {cnt}")
    if len(class_hist_sorted) > 40:
        print(f"  ... ({len(class_hist_sorted)-40} more classes)")

    # ---- 360° capture at spawn (names encode spawn & yaw) ----
    pano_meta = []
    if args.save_360_dir:
        out_dir = Path(args.save_360_dir)
        print(f"\n[capture] capturing {args.views} views at spawn into: {out_dir}")
        pano_meta = _capture_360(sim, out_dir, float(args.spawn_yaw), pose_p, views=max(1, args.views), name_prec=args.name_precision)
        if pano_meta:
            print(f"[capture] saved {len([m for m in pano_meta if 'index' in m])} frames + panorama.png")

    # ---- Neighbor discovery ----
    centers = _region_centers(scene)
    edges = set()
    edges |= _door_edges(scene, centers, max_door_region_dist=float(args.door_region_dist))
    edges |= _knn_edges(centers, k=int(args.neighbors_k), max_dist=float(args.neighbor_dist))

    spawn_rid = str(rid) if rid is not None else None
    neighbor_ids = _neighbors_for(spawn_rid, edges) if spawn_rid is not None else []

    print("\n=== Neighbor Regions (heuristic adjacency) ===")
    print(f"neighbors_of({spawn_rid}) -> {neighbor_ids}")
    neighbor_dump = []
    neighbor_imgs = []

    if args.save_neighbor_dir:
        Path(args.save_neighbor_dir).mkdir(parents=True, exist_ok=True)

    for nid in neighbor_ids:
        # fetch region obj
        nr = None
        for r in getattr(scene, "regions", []) or []:
            if str(getattr(r, "id", "")) == nid:
                nr = r
                break
        if nr is None:
            continue
        ncenter = get_region_center(nr)
        nobjs = list(getattr(nr, "objects", []) or [])
        n_hist: Dict[str, int] = {}
        for o in nobjs:
            nm = safe_cat_name(o.category) or "unknown"
            n_hist[nm] = n_hist.get(nm, 0) + 1
        n_hist_sorted = sorted(n_hist.items(), key=lambda kv: (-kv[1], kv[0]))

        print(f"\n  - neighbor id={nid}, num_objs={len(nobjs)}")
        if ncenter is not None:
            print(f"    center_est: {np.round(ncenter, 6).tolist()}")
        if n_hist_sorted:
            topk = ", ".join([f"{k}:{v}" for k, v in n_hist_sorted[:6]])
            print(f"    class_hist_top: {topk}")

        # optional: save one RGB at neighbor center, facing spawn
        img_path = None
        if args.save_neighbor_dir and (ncenter is not None):
            vec = np.asarray(pose_p) - np.asarray(ncenter)
            yaw = float(np.arctan2(vec[0], -vec[2]))  # face toward spawn
            nx, ny, nz = [ _fmt(v, args.name_precision) for v in ncenter ]
            img_path = Path(args.save_neighbor_dir) / f"neighbor_{nid}_x{nx}_y{ny}_z{nz}_yaw_{_fmt(yaw, args.name_precision)}.png"
            ok = _capture_region_center(sim, ncenter, yaw, img_path)
            if ok:
                neighbor_imgs.append({
                    "region_id": nid, "file": str(img_path),
                    "center": [float(ncenter[0]), float(ncenter[1]), float(ncenter[2])],
                    "yaw_rad": yaw,
                })

        neighbor_dump.append({
            "id": nid,
            "center": (ncenter.tolist() if ncenter is not None else None),
            "num_objects": len(nobjs),
            "class_histogram": dict(n_hist),
        })

    # ---- Optional JSON (with repro params) ----
    if args.out_json:
        out = {
            "scene_dir": args.scene_dir,
            "scene_id": args.scene_id,
            "camera": {
                "img_width": int(habitat_data.cfg.img_width),
                "img_height": int(habitat_data.cfg.img_height),
                "hfov": float(habitat_data.cfg.hfov),
                "camera_tilt_deg": float(habitat_data.cfg.camera_tilt_deg),
            },
            "spawn": {"position": pose_p.tolist(), "yaw": float(args.spawn_yaw)},
            "region": {
                "id": rid, "name": rname,
                "center": (rcenter.tolist() if rcenter is not None else None),
                "aabb_degenerate": bool(deg),
                "spawn_inside_region_aabb": bool(contain_ok),
            },
            "objects_in_region": objs_dump,
            "class_histogram": dict(class_hist),
            "panorama": pano_meta,
            "neighbors": {
                "edges": sorted([list(e) for e in edges]),
                "of_spawn": neighbor_ids,
                "regions": neighbor_dump,
                "neighbor_images": neighbor_imgs,
            },
            "repro_notes": {
                "360_filenames_encode": "spawn_x{X}_y{Y}_z{Z}_view_{i}_yaw_{YAW}.png",
                "neighbor_filenames_encode": "neighbor_{REGIONID}_x{X}_y{Y}_z{Z}_yaw_{YAW}.png",
                "floats_precision_in_filenames": int(args.name_precision),
            }
        }
        op = Path(args.out_json)
        op.parent.mkdir(parents=True, exist_ok=True)
        with open(op, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\n[saved] {op}")

    # Cleanup
    try:
        sim.close(destroy=True)
    except Exception:
        pass

if __name__ == "__main__":
    main()
