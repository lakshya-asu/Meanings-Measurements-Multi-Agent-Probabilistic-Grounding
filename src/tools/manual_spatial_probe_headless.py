#!/usr/bin/env python3
"""
Headless manual spatial probe for HM3D (no GUI).

- Loads an HM3D scene through HabitatInterface (semantics enabled).
- Filters reference/target objects by substring on their semantic category.
- Builds canonical viewpoints (front/back/left/right at given radii) around the chosen reference.
- Teleports to viewpoints or ref/target centers, grabs RGB, and saves frames + a JSON log.

Usage
-----
python src/tools/manual_spatial_probe_headless.py \
  --scene-dir 00149-UuwwmrTsfBN --scene-id UuwwmrTsfBN \
  --ref-filter toilet --tgt-filter tray \
  --out src/outputs_v2/manual_probe

Interactive commands
--------------------
 help           : show this help
 sensors        : list available sensors (uuids + shapes)
 refs           : list filtered reference candidates
 targs          : list filtered target candidates
 pick_ref i     : choose reference object index i from 'refs'
 pick_tgt j     : choose target object index j from 'targs'
 vps            : list canonical viewpoints around current reference
 goto n         : teleport to viewpoint index n (faces ref)
 goto_ref       : teleport to reference center (faces world -Z)
 goto_tgt       : teleport to target center (faces ref)
 snap           : capture RGB and add to session frames
 save           : write JSON with ref/tgt, viewpoints, and captured frames
 quit / q       : exit
"""

import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
import habitat_sim
from habitat_sim.utils.common import quat_from_angle_axis

from src.envs.habitat_interface import HabitatInterface

DATASET_BASE_PATH = "/datasets/hm3d/train"

def _make_hab_cfg(img_w=640, img_h=480, eye_h=1.5, hfov=90.0):
    return SimpleNamespace(
        scene_type="hm3d",
        dataset_type="train",
        sim_gpu=0,
        inflation_radius=0.25,
        img_width=img_w,
        img_height=img_h,
        camera_height=eye_h,
        camera_tilt_deg=-30,
        agent_z_offset=0.0,
        hfov=hfov,
        z_offset=0,
        use_semantic_data=True,
    )

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

def canonical_viewpoints(ref_pos: np.ndarray, radii=(1.6, 2.2)) -> List[Tuple[str, np.ndarray]]:
    ref = np.asarray(ref_pos, dtype=np.float32)
    dirs = [
        ("front", np.array([ 0.0, 0.0, -1.0], dtype=np.float32)),
        ("behind",np.array([ 0.0, 0.0,  1.0], dtype=np.float32)),
        ("right", np.array([ 1.0, 0.0,  0.0], dtype=np.float32)),
        ("left",  np.array([-1.0, 0.0,  0.0], dtype=np.float32)),
    ]
    vps = []
    for r in radii:
        for name, d in dirs:
            vps.append((f"{name}_{r:.1f}m", ref + d * float(r)))
    return vps

def _rgb_from_observations(obs: Dict[str, Any]) -> Optional[np.ndarray]:
    # Find any color-like array in obs dict
    for k, v in obs.items():
        if v is None:
            continue
        arr = np.asarray(v)
        if arr.ndim == 3 and arr.shape[2] >= 3 and arr.dtype == np.uint8:
            return arr[...,:3].copy()
    return None

def list_sensors(sim: habitat_sim.Simulator) -> List[Tuple[str, Tuple[int,int,int]]]:
    out = []
    # get all sensors visible to agent 0
    sens = sim.get_agent(0).get_subtree_sensors()
    for k, s in sens.items():
        spec = s.specification()
        # try to pull current obs shape
        obs = sim.get_sensor_observations()
        sh = None
        if k in obs and obs[k] is not None:
            a = np.asarray(obs[k])
            sh = tuple(a.shape)
        out.append((k, sh))
    return out

def set_pose(sim: habitat_sim.Simulator, pos_xyz: np.ndarray, yaw_rad: float):
    # snap to navmesh, keep height (y) as provided (assumes dataset center already snapped)
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
    # refresh sensors for the new pose
    sim.get_sensor_observations()
    return p, yaw_rad

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene-dir", required=True)
    ap.add_argument("--scene-id", required=True)
    ap.add_argument("--ref-filter", default="toilet")
    ap.add_argument("--tgt-filter", default="tray")
    ap.add_argument("--out", default="src/outputs_v2/manual_probe")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    args = ap.parse_args()

    scene_glb = os.path.join(DATASET_BASE_PATH, args.scene_dir, f"{args.scene_id}.basis.glb")
    if not os.path.exists(scene_glb):
        raise FileNotFoundError(scene_glb)

    # Create sim through HabitatInterface (semantics + sensors)
    hab_cfg = _make_hab_cfg(img_w=args.width, img_h=args.height)
    habitat_data = HabitatInterface(scene_glb, cfg=hab_cfg, device="cpu")
    sim = habitat_data._sim

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Renderer: {sim.gfx_replay_manager.get_renderer_info().gpu_renderer if hasattr(sim, 'gfx_replay_manager') else 'N/A'}")
    print(f"OpenGL version: {sim.gfx_replay_manager.get_renderer_info().gpu_version if hasattr(sim, 'gfx_replay_manager') else 'N/A'}")
    print("\n[INFO] Controls: h=help, 'sensors' to list sensors, [/]=prev/next ref, ,/.=prev/next target, "
          "1-8=go to view, 0=ref center, 'snap'=save frame, 'save'=write JSON, 'q'=quit")

    scene = sim.semantic_scene
    if not scene or not scene.objects:
        raise RuntimeError("[FATAL] No semantic objects found in scene; exiting.")

    # Collect ref/target lists
    refs = []
    targs = []
    for o in scene.objects:
        nm = safe_cat_name(o.category)
        if nm is None:
            continue
        c = safe_center(o.aabb)
        if not valid_xyz(c):
            continue
        if args.ref_filter.lower() in nm:
            refs.append((o, c, nm))
        if args.tgt_filter.lower() in nm:
            targs.append((o, c, nm))

    if not refs:
        print("[WARN] No reference matches found.")
    if not targs:
        print("[WARN] No target matches found.")

    # pick first by default
    ref_idx = 0 if refs else -1
    tgt_idx = 0 if targs else -1

    def _print_refs():
        print("\n[REFS]")
        for i, (_, c, nm) in enumerate(refs):
            print(f"  [{i}] {nm} @ {np.round(c,3).tolist()}")
        if not refs:
            print("  (none)")

    def _print_targs():
        print("\n[TARGETS]")
        for i, (_, c, nm) in enumerate(targs):
            print(f"  [{i}] {nm} @ {np.round(c,3).tolist()}")
        if not targs:
            print("  (none)")

    _print_refs()
    _print_targs()

    frames = []  # dicts with path + pose
    vp_list: List[Tuple[str, np.ndarray]] = []
    cur_pos = None
    cur_yaw = 0.0

    def _rebuild_vps():
        nonlocal vp_list
        if ref_idx < 0:
            vp_list = []
            return
        ref_obj, ref_pos, ref_nm = refs[ref_idx]
        vp_list = canonical_viewpoints(ref_pos, radii=(1.6, 2.2))

    def _face_towards(src: np.ndarray, dst: np.ndarray) -> float:
        v = np.asarray(dst, dtype=np.float32) - np.asarray(src, dtype=np.float32)
        return float(np.arctan2(v[0], -v[2]))  # -Z forward convention

    _rebuild_vps()

    def _snap_rgb(tag: str):
        obs = sim.get_sensor_observations()
        rgb = _rgb_from_observations(obs)
        if rgb is None:
            # Help the user debug sensors
            sens = list_sensors(sim)
            print("[ERR] no image from sensors")
            if sens:
                print("Available sensors and shapes:")
                for k, sh in sens:
                    print(f"  - {k}: {sh}")
            return
        fname = out_dir / f"{tag}.png"
        try:
            # use cv2 if present, else imageio
            import cv2
            cv2.imwrite(str(fname), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        except Exception:
            import imageio
            imageio.imwrite(str(fname), rgb)
        frames.append({
            "file": str(fname),
            "position": np.asarray(sim.get_agent(0).get_state().position, dtype=float).tolist(),
            "yaw": cur_yaw,
        })
        print(f"[saved] {fname.name}")

    # initial goto: ref center
    if ref_idx >= 0:
        ref_obj, ref_pos, ref_nm = refs[ref_idx]
        cur_yaw = 0.0
        cur_pos, cur_yaw = set_pose(sim, ref_pos, cur_yaw)

    # REPL loop
    import shlex
    while True:
        try:
            cmd = input("\ncmd> ").strip()
        except EOFError:
            cmd = "quit"
        if not cmd:
            continue

        if cmd in ("help", "h"):
            print(__doc__)
            continue

        if cmd == "sensors":
            sens = list_sensors(sim)
            if not sens:
                print("(no sensors found)")
            else:
                print("\n[SENSORS]")
                for k, sh in sens:
                    print(f"  - {k}: shape={sh}")
            continue

        if cmd == "refs":
            _print_refs()
            continue

        if cmd == "targs":
            _print_targs()
            continue

        if cmd.startswith("pick_ref"):
            parts = shlex.split(cmd)
            if len(parts) != 2:
                print("usage: pick_ref <index>")
                continue
            try:
                i = int(parts[1])
            except Exception:
                print("index must be int")
                continue
            if i < 0 or i >= len(refs):
                print("out of range")
                continue
            ref_idx = i
            _rebuild_vps()
            ref_obj, ref_pos, ref_nm = refs[ref_idx]
            cur_yaw = 0.0
            cur_pos, cur_yaw = set_pose(sim, ref_pos, cur_yaw)
            print(f"[ok] picked ref {i} ({ref_nm})")
            continue

        if cmd.startswith("pick_tgt"):
            parts = shlex.split(cmd)
            if len(parts) != 2:
                print("usage: pick_tgt <index>")
                continue
            try:
                j = int(parts[1])
            except Exception:
                print("index must be int")
                continue
            if j < 0 or j >= len(targs):
                print("out of range")
                continue
            tgt_idx = j
            print(f"[ok] picked target {j} ({targs[j][2]})")
            continue

        if cmd == "vps":
            if not vp_list:
                print("(no viewpoints; pick a ref first)")
            else:
                print("\n[VIEWPOINTS]")
                for i, (lbl, p) in enumerate(vp_list):
                    print(f"  [{i}] {lbl} @ {np.round(p,3).tolist()}")
            continue

        if cmd.startswith("goto "):
            parts = shlex.split(cmd)
            if len(parts) != 2:
                print("usage: goto <index>")
                continue
            if not vp_list:
                print("(no viewpoints; pick a ref first)")
                continue
            try:
                i = int(parts[1])
            except Exception:
                print("index must be int")
                continue
            if i < 0 or i >= len(vp_list):
                print("out of range")
                continue
            ref_obj, ref_pos, ref_nm = refs[ref_idx]
            lbl, p = vp_list[i]
            yaw = _face_towards(p, ref_pos)
            cur_pos, cur_yaw = set_pose(sim, p, yaw)
            print(f"[ok] teleported to VP[{i}] {lbl}")
            continue

        if cmd == "goto_ref":
            if ref_idx < 0:
                print("(no reference selected)")
                continue
            ref_obj, ref_pos, ref_nm = refs[ref_idx]
            cur_pos, cur_yaw = set_pose(sim, ref_pos, 0.0)
            print("[ok] teleported to ref center")
            continue

        if cmd == "goto_tgt":
            if tgt_idx < 0:
                print("(no target selected)")
                continue
            tgt_obj, tgt_pos, _ = targs[tgt_idx]
            if ref_idx >= 0:
                ref_obj, ref_pos, _ = refs[ref_idx]
                yaw = _face_towards(tgt_pos, ref_pos)
            else:
                yaw = 0.0
            cur_pos, cur_yaw = set_pose(sim, tgt_pos, yaw)
            print("[ok] teleported to target center")
            continue

        if cmd == "snap":
            tag = f"frame_{len(frames):04d}"
            _snap_rgb(tag)
            continue

        if cmd == "save":
            ref_json = None if ref_idx < 0 else {
                "id": refs[ref_idx][0].id,
                "name": refs[ref_idx][2],
                "position": np.asarray(refs[ref_idx][1], dtype=float).tolist(),
            }
            tgt_json = None if tgt_idx < 0 else {
                "id": targs[tgt_idx][0].id,
                "name": targs[tgt_idx][2],
                "position": np.asarray(targs[tgt_idx][1], dtype=float).tolist(),
            }
            vps_json = [
                {"label": lbl, "pose": np.asarray(p, dtype=float).tolist()} for lbl, p in vp_list
            ]
            out = {
                "scene_dir": args.scene_dir,
                "scene_id": args.scene_id,
                "reference": ref_json,
                "target": tgt_json,
                "viewpoints": vps_json,
                "frames": frames,
            }
            of = out_dir / "manual_probe.json"
            with open(of, "w") as f:
                json.dump(out, f, indent=2)
            print(f"[saved] {of}")
            continue

        if cmd in ("quit", "q", "exit"):
            break

        print("Unknown command. Type 'help' for usage.")

    # Cleanup
    try:
        sim.close(destroy=True)
    except Exception:
        pass

if __name__ == "__main__":
    main()
