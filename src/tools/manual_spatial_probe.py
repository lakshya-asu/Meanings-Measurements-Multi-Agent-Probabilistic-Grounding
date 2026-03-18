#!/usr/bin/env python3
# Minimal manual probe for HM3D scenes:
# - Load scene
# - Pick reference object by substring (e.g., "toilet")
# - Teleport to canonical viewpoints around it (front/behind/left/right at a few radii), snapped to navmesh
# - (Optional) pick a target object by substring (e.g., "tray") and save a JSON with the exact poses
#
# Keys:
#   h            : help
#   [ / ]        : cycle reference matches (prev/next)
#   , / .        : cycle target matches (prev/next)
#   1..8         : jump to canonical viewpoints around reference (see on-screen)
#   0            : jump to reference center (snapped)
#   space        : capture frame to disk
#   s            : save JSON record with current spawn pose + reference + (optional) target
#   q / ESC      : quit

import argparse, json, os, time
from pathlib import Path
from types import SimpleNamespace
from typing import List, Tuple, Optional

import numpy as np
import cv2
import habitat_sim
from habitat_sim.utils.common import quat_from_angle_axis

from src.envs.habitat_interface import HabitatInterface

DATASET_BASE_PATH = "/datasets/hm3d/train"

def safe_call(x): return x() if callable(x) else x

def safe_center(aabb) -> np.ndarray:
    try:
        c = safe_call(aabb.center)
    except Exception:
        c = (np.asarray(aabb.max, dtype=np.float32) + np.asarray(aabb.min, dtype=np.float32))/2.0
    return np.asarray(c, dtype=np.float32)

def safe_cat_name(cat) -> Optional[str]:
    if cat is None: return None
    try: n = safe_call(cat.name)
    except Exception:
        try: n = cat.name()
        except Exception: n = None
    return None if n is None else str(n).lower()

def make_hab_cfg():
    return SimpleNamespace(
        scene_type="hm3d", dataset_type="train", sim_gpu=0,
        inflation_radius=0.25, img_width=640, img_height=480,
        camera_height=1.5, camera_tilt_deg=-30, agent_z_offset=0.0, hfov=120,
        z_offset=0, use_semantic_data=True,
    )

def open_sim(scene_dir: str, scene_id: str):
    basis_glb = os.path.join(DATASET_BASE_PATH, scene_dir, f"{scene_id}.basis.glb")
    if not os.path.exists(basis_glb):
        raise FileNotFoundError(f"Missing GLB: {basis_glb}")
    hd = HabitatInterface(basis_glb, cfg=make_hab_cfg(), device="cpu")
    sim = hd._sim
    if not sim.semantic_scene or not getattr(sim.semantic_scene, "objects", None):
        try: hd._sim.close(destroy=True)
        except Exception: pass
        raise RuntimeError("No semantics in this scene")
    return hd, sim

def grab_rgb(sim: "habitat_sim.Simulator") -> Optional[np.ndarray]:
    obs = sim.get_sensor_observations()
    for k, v in obs.items():
        if isinstance(v, np.ndarray) and v.ndim==3 and v.shape[2] in (3,4):
            img = v[..., :3]  # RGB
            return img[:, :, ::-1].copy()  # to BGR for cv2
    return None

def yaw_face(src_xyz: np.ndarray, dst_xyz: np.ndarray) -> float:
    v = np.asarray(dst_xyz, np.float32) - np.asarray(src_xyz, np.float32)
    return float(np.arctan2(v[0], -v[2]))  # -Z forward

def snap(sim, p: np.ndarray) -> np.ndarray:
    try: return np.asarray(sim.pathfinder.snap_point(np.asarray(p, np.float32)), np.float32)
    except Exception: return np.asarray(p, np.float32)

def canonical_viewpoints(ref_pos: np.ndarray, y_level: float, radii=(1.2, 1.8, 2.4)) -> List[Tuple[str, np.ndarray]]:
    dirs = [
        ("front",  np.array([ 0,0,-1], np.float32)),
        ("behind", np.array([ 0,0, 1], np.float32)),
        ("right",  np.array([ 1,0, 0], np.float32)),
        ("left",   np.array([-1,0, 0], np.float32)),
    ]
    out = []
    rpos = np.asarray(ref_pos, np.float32)
    for r in radii:
        for name, d in dirs:
            p = rpos + d * float(r)
            p[1] = y_level
            out.append((f"{name}_{r:.1f}m", p))
    return out  # 12 viewpoints

def find_matches(scene, substr: Optional[str]) -> List[Tuple[object, str, np.ndarray]]:
    if not substr: return []
    s = substr.lower()
    matches = []
    for obj in scene.objects:
        nm = safe_cat_name(obj.category)
        if nm and s in nm:
            c = safe_center(obj.aabb)
            if np.all(np.isfinite(c)):
                matches.append((obj, nm, c))
    return matches

def draw_overlay(img, lines: List[str]):
    y = 18
    for ln in lines:
        cv2.putText(img, ln, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,0,0), 3, cv2.LINE_AA)
        cv2.putText(img, ln, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1, cv2.LINE_AA)
        y += 20

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene-dir", required=True)
    ap.add_argument("--scene-id", required=True)
    ap.add_argument("--ref-filter", default="toilet", help="substring for reference object (e.g., toilet/bed/sofa)")
    ap.add_argument("--tgt-filter", default="", help="substring for optional target object (e.g., tray)")
    ap.add_argument("--out", default="src/outputs_v2/manual_probe", help="output dir")
    args = ap.parse_args()

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)

    hd, sim = open_sim(args.scene_dir, args.scene_id)
    scene = sim.semantic_scene

    ref_list = find_matches(scene, args.ref_filter)
    tgt_list = find_matches(scene, args.tgt_filter) if args.tgt_filter else []
    if not ref_list:
        print(f"[ERR] No reference matches for '{args.ref_filter}'"); return

    ref_idx = 0
    tgt_idx = 0 if tgt_list else -1

    def set_pose(pos_xyz, look_at_xyz):
        pos_xyz = snap(sim, np.asarray(pos_xyz, np.float32))
        yaw = yaw_face(pos_xyz, np.asarray(look_at_xyz, np.float32))
        st = habitat_sim.AgentState()
        st.position = pos_xyz
        st.rotation = quat_from_angle_axis(yaw, np.array([0,1,0], np.float32))
        sim.get_agent(0).set_state(st)
        return pos_xyz, yaw

    # initial ref + viewpoints
    ref_obj, ref_name, ref_pos = ref_list[ref_idx]
    # set eye-height from current agent (keep whatever sim uses)
    y_level = float(sim.get_agent(0).get_state().position[1])
    vps = canonical_viewpoints(ref_pos, y_level)
    # jump to viewpoint 0 by default
    cur_pos, cur_yaw = set_pose(vps[0][1], ref_pos)

    print("[INFO] Controls: h=help, [/]=prev/next ref, ,/.=prev/next target, 1-8=go to view, 0=ref center, space=save frame, s=save JSON, q=quit")

    kmap = {ord(str(i)): i for i in range(10)}  # 0..9

    shot_id = 0
    while True:
        frame = grab_rgb(sim)
        if frame is None:
            # force a sensor update
            sim.step(None)
            frame = grab_rgb(sim)

        # overlay
        lines = [
            f"scene: {args.scene_dir}/{args.scene_id}",
            f"ref[{ref_idx+1}/{len(ref_list)}]: {ref_name} @ {np.round(ref_pos,3).tolist()}",
            f"pos: {np.round(cur_pos,3).tolist()}  yaw(deg): {np.degrees(cur_yaw):.1f}",
            "views: " + "  ".join([f"{i+1}:{lab}" for i,(lab,_) in enumerate(vps[:8])]),
            "keys: h,[,],,,.,1-8,0,space,s,q",
        ]
        if tgt_list:
            tgt_obj, tgt_name, tgt_pos = tgt_list[tgt_idx]
            lines.insert(2, f"tgt[{tgt_idx+1}/{len(tgt_list)}]: {tgt_name} @ {np.round(tgt_pos,3).tolist()}")

        vis = frame.copy()
        draw_overlay(vis, lines)
        cv2.imshow("manual_spatial_probe", vis)
        key = cv2.waitKey(30) & 0xFF

        if key in (ord('q'), 27):
            break
        elif key == ord('h'):
            print("[help] [/]=cycle ref  ,/.=cycle target  1..8=goto view  0=ref center  space=snap  s=save JSON  q=quit")
        elif key == ord('['):
            ref_idx = (ref_idx - 1) % len(ref_list)
            ref_obj, ref_name, ref_pos = ref_list[ref_idx]
            vps = canonical_viewpoints(ref_pos, y_level)
            cur_pos, cur_yaw = set_pose(vps[0][1], ref_pos)
        elif key == ord(']'):
            ref_idx = (ref_idx + 1) % len(ref_list)
            ref_obj, ref_name, ref_pos = ref_list[ref_idx]
            vps = canonical_viewpoints(ref_pos, y_level)
            cur_pos, cur_yaw = set_pose(vps[0][1], ref_pos)
        elif key == ord(',') and tgt_list:
            tgt_idx = (tgt_idx - 1) % len(tgt_list)
        elif key == ord('.') and tgt_list:
            tgt_idx = (tgt_idx + 1) % len(tgt_list)
        elif key in kmap and 1 <= kmap[key] <= min(8, len(vps)):
            i = kmap[key] - 1
            cur_pos, cur_yaw = set_pose(vps[i][1], ref_pos)
        elif key == ord('0'):
            cur_pos, cur_yaw = set_pose(ref_pos, ref_pos + np.array([0,0,-1], np.float32))
        elif key == ord(' '):
            shot_path = out_dir / f"shot_{shot_id:04d}.png"
            cv2.imwrite(str(shot_path), vis)
            print(f"[snap] {shot_path}")
            shot_id += 1
        elif key == ord('s'):
            # Save a JSON snippet you can copy into your dataset later
            state = sim.get_agent(0).get_state()
            payload = {
                "scene_dir": args.scene_dir,
                "scene_id": args.scene_id,
                "initial_pose": {
                    "position": np.asarray(state.position, np.float32).astype(float).tolist(),
                    "rotation": [float(state.rotation.x), float(state.rotation.y),
                                 float(state.rotation.z), float(state.rotation.w)],
                },
                "reference_object": {
                    "id": ref_obj.id, "name": ref_name,
                    "position": np.asarray(ref_pos, np.float32).astype(float).tolist(),
                },
            }
            if tgt_list:
                tgt_obj, tgt_name, tgt_pos = tgt_list[tgt_idx]
                payload["manual_target"] = {
                    "id": tgt_obj.id, "name": tgt_name,
                    "position": np.asarray(tgt_pos, np.float32).astype(float).tolist(),
                }
            out_json = out_dir / f"manual_pick_{int(time.time())}.json"
            with open(out_json, "w") as f:
                json.dump(payload, f, indent=2)
            print(f"[save] wrote {out_json}")

    try:
        hd._sim.close(destroy=True)
    except Exception:
        pass
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
