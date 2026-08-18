#!/usr/bin/env python3
"""Per-query visual context for bench-v2-150 authoring and review (MAPG-07).

A drafter must have the scene in front of them before writing a query
(protocol 4.1: no blind authoring from label lists), a reviewer needs to see
where the scoring GT actually lands, and a test-retest annotator must see the
scene while seeing nothing about any annotation. Those are three different
amounts of information, so this script has three modes and the mode decides
what the rendering code is even given:

  author  scene context around the anchor: the frozen start pose view, four
          anchor-facing orbit views, and a top-down view. No GT marker.
  review  everything author shows, plus a marked view: the annotated point
          and the recomputed scoring GT drawn as crosshairs, and a
          context.json carrying both. This is the spot-check mode.
  blind   scene context only, built from a row that has already been through
          ``redact_for_blind``. The annotation columns are removed from the
          dict before any rendering function receives it, so there is no
          code path that could draw a marker even by mistake. This is the
          mode the D6 test-retest passes use, and --noise-floor-pass forces
          it.

The scoring GT drawn in review mode comes from
``src.evals.success.gt_xyz_from_row``, the same function the runners score
with, including the D5 zero-distance rule. Nothing here recomputes it.

Camera convention matches collect_init_poses.py and the benchmark config:
Habitat y up, yaw about +y with forward = -z, pinhole camera, 640x480, hfov
120. Rendering falls back to software mesa under WSL, so gpu_device_id is -1
by default (override with MAPG_POSE_GPU).

Run in the container:

    docker exec -w /workspace mapg_dev python3 -m src.scripts.render_query_context \\
        --csv splits/bench_v1_98.csv --rows 1,2,3 --mode author --out-dir /tmp/ctx
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import traceback
from typing import Any, Dict, List, Optional, Sequence, Tuple

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.scripts.author_bench_v2 import redact_for_blind  # noqa: E402
from src.scripts.bench_v2_common import (  # noqa: E402
    POSE_CSV_V2,
    V1_CSV_REL,
    parse_csv_text,
    read_bytes,
    repo_path,
    scene_floor_key,
)

# Camera, matching collect_init_poses.py.
IMG_WIDTH = 640
IMG_HEIGHT = 480
HFOV_DEG = 120.0
CAMERA_HEIGHT = 1.5

# Anchor orbit: the manual_spatial_probe viewpoints, at two radii.
ORBIT_RADII_M = (1.6, 2.2)
ORBIT_DIRECTIONS: Tuple[Tuple[str, Tuple[float, float, float]], ...] = (
    ("front", (0.0, 0.0, -1.0)),
    ("behind", (0.0, 0.0, 1.0)),
    ("right", (1.0, 0.0, 0.0)),
    ("left", (-1.0, 0.0, 0.0)),
)

# Top-down view: camera above the anchor looking straight down.
TOPDOWN_HEIGHT_M = 6.0
TOPDOWN_PITCH_DEG = -89.0

MODES = ("author", "review", "blind")


# ---------------------------------------------------------------------------
# Pure logic (unit tested in tests/test_bench_authoring.py)
# ---------------------------------------------------------------------------

def yaw_towards(src: Sequence[float], dst: Sequence[float]) -> float:
    """Habitat yaw that points a camera at src towards dst (forward = -z)."""
    dx = float(dst[0]) - float(src[0])
    dz = float(dst[2]) - float(src[2])
    return math.atan2(-dx, -dz)


def orbit_viewpoints(anchor: Sequence[float],
                     radii: Sequence[float] = ORBIT_RADII_M
                     ) -> List[Tuple[str, List[float], float]]:
    """Anchor-facing viewpoints as (label, position, yaw).

    Deterministic order: radius outer loop, then front, behind, right, left.
    Eye height is the camera height above the anchor's own y, so a low
    anchor is still framed sensibly.
    """
    a = [float(anchor[0]), float(anchor[1]), float(anchor[2])]
    out: List[Tuple[str, List[float], float]] = []
    for r in radii:
        for name, d in ORBIT_DIRECTIONS:
            pos = [a[0] + d[0] * float(r), a[1], a[2] + d[2] * float(r)]
            out.append((f"{name}_{float(r):.1f}m", pos, yaw_towards(pos, a)))
    return out


def project_to_pixel(point: Sequence[float], cam_pos: Sequence[float],
                     yaw: float, pitch: float, width: int = IMG_WIDTH,
                     height: int = IMG_HEIGHT, hfov_deg: float = HFOV_DEG
                     ) -> Optional[Tuple[float, float]]:
    """Pixel coordinates of a world point, or None when it is not in view.

    Pinhole camera in the Habitat convention: y up, the camera looks along
    its local -z, yaw rotates about +y and pitch about the camera's local
    +x (negative pitch looks down). Returns (u, v) with the origin at the
    top left. Points behind the camera or outside the frame return None.
    """
    px = float(point[0]) - float(cam_pos[0])
    py = float(point[1]) - float(cam_pos[1])
    pz = float(point[2]) - float(cam_pos[2])

    # World to camera: undo yaw about y, then pitch about x.
    cy, sy = math.cos(-yaw), math.sin(-yaw)
    x1 = cy * px + sy * pz
    z1 = -sy * px + cy * pz
    y1 = py
    cp, sp = math.cos(-pitch), math.sin(-pitch)
    y2 = cp * y1 - sp * z1
    z2 = sp * y1 + cp * z1

    if z2 >= -1e-6:
        return None  # behind the camera or exactly in its plane
    depth = -z2
    tan_h = math.tan(math.radians(hfov_deg) / 2.0)
    tan_v = tan_h * (float(height) / float(width))
    u = (width / 2.0) * (1.0 + x1 / (depth * tan_h))
    v = (height / 2.0) * (1.0 - y2 / (depth * tan_v))
    if not (0.0 <= u <= width - 1 and 0.0 <= v <= height - 1):
        return None
    return (u, v)


def crosshair_pixels(u: float, v: float, width: int, height: int,
                     arm: int = 12) -> List[Tuple[int, int]]:
    """Pixels of a crosshair centred on (u, v), clipped to the frame."""
    cu, cv = int(round(u)), int(round(v))
    out = []
    for d in range(-arm, arm + 1):
        for (x, y) in ((cu + d, cv), (cu, cv + d)):
            if 0 <= x < width and 0 <= y < height:
                out.append((x, y))
    return out


def plan_views(row: Dict[str, str], mode: str,
               start_pose: Optional[Dict[str, float]]) -> List[Dict[str, Any]]:
    """The render plan for one row: which cameras, in which order.

    Pure: takes the row dict (already redacted in blind mode) and the frozen
    start pose, returns camera descriptions. Never touches habitat.
    """
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}, expected one of {MODES}")
    views: List[Dict[str, Any]] = []
    if start_pose is not None:
        views.append({
            "name": "start_pose",
            "position": [start_pose["init_x"], start_pose["init_y"],
                         start_pose["init_z"]],
            "yaw": start_pose["init_angle"],
            "pitch": 0.0,
            "eye_height": CAMERA_HEIGHT,
        })
    if mode == "blind":
        # Blind rows carry no anchor centre, by construction. The start pose
        # view is all the geometry the annotator gets from this tool.
        return views

    anchor = _anchor_center(row)
    if anchor is None:
        return views
    for label, pos, yaw in orbit_viewpoints(anchor):
        views.append({"name": f"orbit_{label}", "position": pos, "yaw": yaw,
                      "pitch": 0.0, "eye_height": CAMERA_HEIGHT})
    views.append({
        "name": "topdown",
        "position": [anchor[0], anchor[1] + TOPDOWN_HEIGHT_M, anchor[2]],
        "yaw": 0.0,
        "pitch": math.radians(TOPDOWN_PITCH_DEG),
        "eye_height": 0.0,
    })
    return views


def _anchor_center(row: Dict[str, str]) -> Optional[List[float]]:
    try:
        return [float(row["anchor_center_x"]), float(row["anchor_center_y"]),
                float(row["anchor_center_z"])]
    except (KeyError, TypeError, ValueError):
        return None


def markers_for_row(row: Dict[str, str], mode: str) -> Dict[str, List[float]]:
    """Points to draw. Empty for every mode except review.

    The blind and author paths never reach this function with annotation
    columns present, but it refuses anyway: two independent guards, because
    a single guard is one refactor away from being removed.
    """
    if mode != "review":
        return {}
    from src.evals.success import gt_xyz_from_row

    out: Dict[str, List[float]] = {}
    try:
        ann = [float(row["ann_pos_x"]), float(row["ann_pos_y"]),
               float(row["ann_pos_z"])]
        out["ann_pos"] = ann
    except (KeyError, TypeError, ValueError):
        pass
    gt = gt_xyz_from_row(row)
    if gt is not None:
        out["scoring_gt"] = list(gt)
    anchor = _anchor_center(row)
    if anchor is not None:
        out["anchor_center"] = anchor
    return out


def context_payload(row_idx: int, row: Dict[str, str], mode: str,
                    views: Sequence[Dict[str, Any]],
                    markers: Dict[str, List[float]]) -> Dict[str, Any]:
    """The context.json written next to the renders."""
    return {
        "row_idx": row_idx,
        "mode": mode,
        "row": dict(row),
        "views": [v["name"] for v in views],
        "markers": markers,
    }


# ---------------------------------------------------------------------------
# Habitat-dependent rendering (container only)
# ---------------------------------------------------------------------------

def _make_sim(scene_dir_name: str):
    from src.scripts.collect_init_poses import _make_sim as make
    return make(scene_dir_name)


def _render_view(sim, view: Dict[str, Any], markers: Dict[str, List[float]],
                 png_path: str) -> None:
    import numpy as np
    import habitat_sim
    from habitat_sim.utils.common import quat_from_angle_axis

    agent = sim.get_agent(0)
    state = habitat_sim.AgentState()
    pos = view["position"]
    state.position = np.array(
        [float(pos[0]), float(pos[1]) - float(view.get("eye_height", 0.0)),
         float(pos[2])], dtype=np.float32)
    rot = quat_from_angle_axis(float(view["yaw"]), np.array([0.0, 1.0, 0.0]))
    if abs(float(view.get("pitch", 0.0))) > 1e-9:
        rot = rot * quat_from_angle_axis(float(view["pitch"]),
                                         np.array([1.0, 0.0, 0.0]))
    state.rotation = rot
    agent.set_state(state)
    rgb = np.asarray(sim.get_sensor_observations()["color"])[:, :, :3].copy()

    cam = [float(pos[0]), float(pos[1]), float(pos[2])]
    colors = {"ann_pos": (255, 40, 40), "scoring_gt": (40, 255, 40),
              "anchor_center": (60, 120, 255)}
    for name, point in sorted(markers.items()):
        uv = project_to_pixel(point, cam, float(view["yaw"]),
                              float(view.get("pitch", 0.0)),
                              width=rgb.shape[1], height=rgb.shape[0])
        if uv is None:
            continue
        for (x, y) in crosshair_pixels(uv[0], uv[1], rgb.shape[1], rgb.shape[0]):
            rgb[y, x] = colors.get(name, (255, 255, 0))

    os.makedirs(os.path.dirname(png_path), exist_ok=True)
    try:
        from PIL import Image
        Image.fromarray(rgb).save(png_path)
    except ImportError:
        import imageio
        imageio.imwrite(png_path, rgb)


def _load_start_poses(path: str) -> Dict[str, Dict[str, float]]:
    import csv as _csv
    out: Dict[str, Dict[str, float]] = {}
    if not os.path.exists(path):
        return out
    with open(path, newline="", encoding="utf-8") as f:
        for r in _csv.DictReader(f):
            try:
                out[r["scene_floor"]] = {
                    "init_x": float(r["init_x"]), "init_y": float(r["init_y"]),
                    "init_z": float(r["init_z"]),
                    "init_angle": float(r["init_angle"]),
                }
            except (KeyError, TypeError, ValueError):
                continue
    return out


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--csv", default=V1_CSV_REL)
    ap.add_argument("--rows", required=True,
                    help="comma-separated 1-based data-row indices")
    ap.add_argument("--mode", choices=MODES, default="author")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--pose-csv", default=POSE_CSV_V2)
    ap.add_argument("--noise-floor-pass", type=int, default=0, choices=(0, 1, 2),
                    help="test-retest pass; forces --mode blind")
    ap.add_argument("--plan-only", action="store_true",
                    help="write the render plan and context, render nothing "
                         "(host-safe, no habitat needed)")
    args = ap.parse_args(argv)

    mode = args.mode
    if args.noise_floor_pass:
        if mode != "blind":
            print(f"[render] --noise-floor-pass {args.noise_floor_pass} forces "
                  f"--mode blind (was {mode})")
        mode = "blind"

    csv_path = args.csv if os.path.isabs(args.csv) else repo_path(args.csv)
    _header, rows = parse_csv_text(read_bytes(csv_path).decode("utf-8"))
    indices = [int(x) for x in args.rows.split(",") if x.strip()]
    for i in indices:
        if not (1 <= i <= len(rows)):
            raise SystemExit(f"row {i} is outside 1..{len(rows)}")

    from src.paths import resolve_data_path
    pose_path = resolve_data_path(args.pose_csv)
    poses = _load_start_poses(pose_path)
    if not poses:
        print(f"[render] no start poses loaded from {pose_path}; the start-pose "
              "view is skipped. That file is the MAPG-05 deliverable.")

    os.makedirs(args.out_dir, exist_ok=True)
    sims: Dict[str, Any] = {}
    written = 0
    try:
        for row_idx in indices:
            raw = rows[row_idx - 1]
            # Redaction happens HERE, before anything else sees the row.
            row = redact_for_blind(raw) if mode == "blind" else raw
            scene = (row.get("scene") or "").strip()
            key = scene_floor_key(scene, row.get("floor") or 0)
            views = plan_views(row, mode, poses.get(key))
            markers = markers_for_row(row, mode)
            row_dir = os.path.join(args.out_dir, f"row_{row_idx:03d}")
            os.makedirs(row_dir, exist_ok=True)
            with open(os.path.join(row_dir, "context.json"), "w",
                      encoding="utf-8") as f:
                json.dump(context_payload(row_idx, row, mode, views, markers),
                          f, indent=2)
                f.write("\n")

            if args.plan_only:
                written += 1
                continue
            if scene not in sims:
                sims[scene] = _make_sim(scene)
            for view in views:
                png = os.path.join(row_dir, f"{view['name']}.png")
                try:
                    _render_view(sims[scene], view, markers, png)
                except Exception:
                    traceback.print_exc()
                    print(f"[render] row {row_idx} view {view['name']} failed")
            written += 1
            print(f"[render] row {row_idx} {scene}: {len(views)} views -> {row_dir}")
    finally:
        for sim in sims.values():
            try:
                sim.close(destroy=True)
            except Exception:
                pass

    print(f"[render] mode {mode}, {written} rows into {args.out_dir}")
    if mode == "blind":
        print("[render] blind mode: rows were redacted before planning, so no "
              "annotation value reached any rendering call.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
