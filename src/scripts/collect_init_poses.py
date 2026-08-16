#!/usr/bin/env python3
"""Init-pose collection tooling for the 15 bench pairs with no start pose.

Board item 6 (COLLECT). The benchmark needs one start pose per scene_floor
pair. ``datasets/explore-eqa/scene_init_poses_semantic_only_new.csv`` covers
49 of the 64 pairs in ``splits/bench_v1_98.csv``. This script:

1. ``measure``  : loads every scene referenced by the existing 49 poses and
   measures their quality metrics (clearance from walls and obstacles,
   island radius and area, openness, distance to object bboxes). The
   aggregate envelope is written to ``pose_candidates/envelope.json`` and
   is what the ``collect`` thresholds are calibrated against.
2. ``collect``  : for each missing pair, loads the scene in habitat_sim,
   rebuilds the navmesh exactly like ``HabitatInterface._make_navgraph``
   (agent_radius = 0.25), determines the target floor height band from the
   pair's annotation heights in bench_v1_98.csv, samples K candidate poses
   on the largest navmesh island of that band subject to quality checks,
   picks the yaw with the maximum mean free range over a 90 degree cone,
   renders an RGB frame per candidate and writes
   ``pose_candidates/<scene_floor>/cand_<k>.png``, a ``candidates.csv``
   with all measurements, and a human review sheet ``pose_review.md``.
3. ``finalize`` : after human approval, takes the approved candidate per
   pair and writes a NEW file
   ``datasets/explore-eqa/scene_init_poses_semantic_only_v2.csv``
   (verbatim copy of the original 49 rows plus the 15 new rows; the
   original file is never edited) and prints its sha256 for
   splits/MANIFEST.json.

Selection is NOT automated: ``collect`` marks a RECOMMENDED row per pair
(best composite score) but a human checks ACCEPT boxes in pose_review.md
(or writes approved.csv) before ``finalize`` will do anything.

Coordinate conventions (match run_multi_agent_benchmark.py):
- init_x/init_y/init_z are the agent position in the HABITAT frame
  (y up). They are passed unchanged to ``get_init_poses_eqa``.
- init_angle is the habitat yaw in radians about +y with forward = -z,
  i.e. forward = (-sin(a), 0, -cos(a)) and a = atan2(-dx, -dz).
- The floor height used by the runner (tsdf bounds) is simply the init
  pose's y, so the pose itself defines the floor. The ``_N`` suffix of a
  scene_floor key is an episode index, not a storey index (e.g. all four
  00057-1UnKg1rAb8A pairs sit at y = -2.8), so the physical floor band
  for a missing pair is inferred from that pair's annotation heights
  (ann_pos_y / anchor_center_y) in bench_v1_98.csv.

Determinism: every sampling step is seeded from ``--seed`` (default
20260815) plus a stable per-pair hash, so re-running reproduces identical
candidates.

Run inside the mapg_dev container:
  docker exec -w /workspace mapg_dev python3 -m src.scripts.collect_init_poses measure
  docker exec -w /workspace mapg_dev python3 -m src.scripts.collect_init_poses collect
  docker exec -w /workspace mapg_dev python3 -m src.scripts.collect_init_poses \
      finalize --from-review          # or: --approved <csv with scene_floor,cand>
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import traceback
import zlib

# Pure stdlib at module import time: the pure-logic functions below must be
# importable on any host (tests run without habitat_sim or numpy).

# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------

POSES_CSV = "/datasets/explore-eqa/scene_init_poses_semantic_only_new.csv"
V2_CSV = "/datasets/explore-eqa/scene_init_poses_semantic_only_v2.csv"
OUT_DIR = "/datasets/explore-eqa/pose_candidates"
SCENES_ROOT = "/datasets/hm3d/train"
SPLIT_NAME = "bench_v1_98"

DEFAULT_SEED = 20260815
DEFAULT_K = 5

# Navmesh must match HabitatInterface._make_navgraph (mapg_benchmark.yaml
# habitat.inflation_radius = 0.25).
AGENT_RADIUS = 0.25
# Camera must match mapg_benchmark.yaml habitat block (tilt 0 at init, see
# get_init_poses_eqa call sites).
IMG_WIDTH = 640
IMG_HEIGHT = 480
HFOV = 120.0
CAMERA_HEIGHT = 1.5

# Floor band clustering: navigable heights closer than this belong to the
# same floor.
BAND_GAP = 0.5
# An annotation at height h votes for a band [lo, hi] if
# lo - ANCHOR_BELOW <= h <= hi + ANCHOR_ABOVE (objects sit above the floor).
ANCHOR_BELOW = 0.3
ANCHOR_ABOVE = 2.3

# Openness probing.
N_YAW_BINS = 72
FREE_RANGE_MAX = 5.0
FREE_RANGE_STEP = 0.25
CONE_DEG = 90.0

# Object bbox check: skip structural / walk-on categories whose aabbs cover
# whole rooms, plus anything with a huge footprint.
BBOX_SKIP_SUBSTRINGS = (
    "floor", "ceiling", "wall", "rug", "carpet", "mat", "stair", "door",
    "window", "beam", "column", "railing", "banister", "curtain", "unknown",
)
BBOX_MAX_FOOTPRINT_M2 = 20.0
# Vertical slab of the agent body used for bbox overlap tests.
BODY_Z_LO = 0.2
BODY_Z_HI = 1.6
# HM3D aabbs are loose: measured on the existing 49 poses, 32 sit inside
# some raw aabb, and about half of ALL navigable points do. The hard gate
# therefore only rejects candidates that penetrate a bbox deeper than this
# (metres); shallow overlap only lowers the score via bbox_dist_m.
BBOX_PENETRATION_TOL = 0.10

# Composite score weights and normalizers.
SCORE_WEIGHTS = {"clearance": 0.4, "openness": 0.3, "island": 0.2, "bbox": 0.1}
CLEARANCE_NORM = 1.0
OPENNESS_NORM = 3.0
BBOX_NORM = 1.0


def repo_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def resolve(p: str) -> str:
    sys.path.insert(0, repo_root())
    from src.paths import resolve_data_path
    return resolve_data_path(p)


# ---------------------------------------------------------------------------
# Pure logic (unit-tested in tests/test_collect_poses.py, no habitat/numpy)
# ---------------------------------------------------------------------------

def git_head(root=None) -> str:
    """Current commit hash, read from .git directly (stdlib only)."""
    root = root or repo_root()
    try:
        with open(os.path.join(root, ".git", "HEAD")) as f:
            head = f.read().strip()
        if head.startswith("ref: "):
            ref = head[5:].strip()
            ref_path = os.path.join(root, ".git", *ref.split("/"))
            if os.path.exists(ref_path):
                with open(ref_path) as f:
                    return f.read().strip()
            packed = os.path.join(root, ".git", "packed-refs")
            if os.path.exists(packed):
                with open(packed) as f:
                    for line in f:
                        parts = line.strip().split(" ")
                        if len(parts) == 2 and parts[1] == ref:
                            return parts[0]
            return "unknown"
        return head
    except OSError:
        return "unknown"


def pair_seed(base_seed: int, scene_floor: str) -> int:
    """Stable per-pair seed (crc32 is stable across runs and platforms)."""
    return (int(base_seed) + zlib.crc32(scene_floor.encode("utf-8"))) % (2 ** 31)


def cluster_heights(ys, gap: float = BAND_GAP):
    """Cluster navigable heights into floor bands.

    Returns a list of (lo, hi, count) tuples sorted by ascending height.
    """
    ys = sorted(float(y) for y in ys)
    if not ys:
        return []
    bands = []
    lo = hi = ys[0]
    count = 1
    for y in ys[1:]:
        if y - hi <= gap:
            hi = y
            count += 1
        else:
            bands.append((lo, hi, count))
            lo = hi = y
            count = 1
    bands.append((lo, hi, count))
    return bands


def pick_floor_band(bands, anchor_heights,
                    below: float = ANCHOR_BELOW, above: float = ANCHOR_ABOVE):
    """Pick the floor band the pair's annotations live on.

    Each annotation height votes for every band whose ground interval
    extended by (below, above) contains it. The band with the most votes
    wins; ties break toward the band nearest (from below) to the median
    anchor height, then toward the band with the most navigable samples.

    Returns (band_index, votes_list).
    """
    if not bands:
        raise ValueError("no navigable height bands")
    votes = [0] * len(bands)
    for h in anchor_heights:
        for i, (lo, hi, _n) in enumerate(bands):
            if lo - below <= float(h) <= hi + above:
                votes[i] += 1
    best = max(votes) if votes else 0
    tied = [i for i, v in enumerate(votes) if v == best]
    if len(tied) == 1:
        return tied[0], votes
    if anchor_heights:
        med = sorted(float(h) for h in anchor_heights)[len(anchor_heights) // 2]
        # Nearest band, preferring bands at or below the median anchor.
        def key(i):
            lo, hi, n = bands[i]
            d = 0.0 if lo - below <= med <= hi + above else min(
                abs(med - lo), abs(med - hi))
            penalty = 0 if lo <= med + below else 1
            return (penalty, d, -n)
        tied.sort(key=key)
        return tied[0], votes
    tied.sort(key=lambda i: -bands[i][2])
    return tied[0], votes


def angle_from_direction(dx: float, dz: float) -> float:
    """Habitat yaw for a horizontal forward direction (forward = -z)."""
    return math.atan2(-dx, -dz)


def yaw_bin_angle(b: int, n_bins: int = N_YAW_BINS) -> float:
    """Yaw angle of bin b, wrapped to (-pi, pi]."""
    a = 2.0 * math.pi * b / n_bins
    if a > math.pi:
        a -= 2.0 * math.pi
    return a


def choose_yaw(free_ranges, cone_deg: float = CONE_DEG):
    """Pick the yaw whose 90 degree cone has the max mean free range.

    ``free_ranges[b]`` is the free range along yaw_bin_angle(b). Returns
    (best_yaw_rad, best_cone_mean). Deterministic: first best bin wins.
    """
    n = len(free_ranges)
    if n == 0:
        raise ValueError("empty free range profile")
    half = int(round((cone_deg / 2.0) / (360.0 / n)))
    best_b, best_mean = 0, -1.0
    for b in range(n):
        vals = [free_ranges[(b + o) % n] for o in range(-half, half + 1)]
        m = sum(vals) / len(vals)
        if m > best_mean + 1e-12:
            best_b, best_mean = b, m
    return yaw_bin_angle(best_b, n), best_mean


def select_diverse(points, k: int):
    """Deterministic farthest-point selection of k indices from xyz points."""
    n = len(points)
    if n <= k:
        return list(range(n))

    def d2(a, b):
        return sum((float(a[i]) - float(b[i])) ** 2 for i in range(3))

    # Start from the point closest to the centroid (stable, seed-free).
    cx = [sum(float(p[i]) for p in points) / n for i in range(3)]
    chosen = [min(range(n), key=lambda i: (d2(points[i], cx), i))]
    while len(chosen) < k:
        best_i, best_d = -1, -1.0
        for i in range(n):
            if i in chosen:
                continue
            dmin = min(d2(points[i], points[j]) for j in chosen)
            if dmin > best_d + 1e-12:
                best_i, best_d = i, dmin
        chosen.append(best_i)
    return chosen


def composite_score(clearance_m, openness_m, island_frac, bbox_dist_m,
                    weights=None):
    """Weighted composite in [0, 1]. Higher is better."""
    w = dict(SCORE_WEIGHTS if weights is None else weights)
    c = min(max(float(clearance_m) / CLEARANCE_NORM, 0.0), 1.0)
    o = min(max(float(openness_m) / OPENNESS_NORM, 0.0), 1.0)
    i = min(max(float(island_frac), 0.0), 1.0)
    b = min(max(float(bbox_dist_m) / BBOX_NORM, 0.0), 1.0)
    return (w["clearance"] * c + w["openness"] * o
            + w["island"] * i + w["bbox"] * b)


def fmt_num(v: float, ndigits: int) -> str:
    """Format like the existing CSV rows (round, trim trailing zeros)."""
    r = round(float(v), ndigits)
    if r == int(r):
        return str(int(r))
    return repr(r)


def format_pose_row(scene_floor: str, x: float, y: float, z: float,
                    angle: float) -> str:
    return ",".join([
        scene_floor, fmt_num(x, 2), fmt_num(y, 2), fmt_num(z, 2),
        fmt_num(angle, 3),
    ])


def build_v2_text(original_text: str, new_rows) -> str:
    """Verbatim copy of the original CSV plus the new rows appended.

    ``new_rows`` is a list of (scene_floor, x, y, z, angle). Existing pairs
    must not be duplicated.
    """
    have = set()
    for line in original_text.strip().splitlines()[1:]:
        if line.strip():
            have.add(line.split(",")[0])
    out = original_text
    if not out.endswith("\n"):
        out += "\n"
    for row in new_rows:
        sf = row[0]
        if sf in have:
            raise ValueError(f"pair {sf} already present in the original CSV")
        have.add(sf)
        out += format_pose_row(*row) + "\n"
    return out


def parse_review_selections(review_text: str):
    """Parse ACCEPT checkboxes out of pose_review.md.

    A candidate row looks like
      | cand_2 | ... | [x] | [ ] |
    where the last two cells are ACCEPT and REJECT. Returns
    {scene_floor: cand_id} for rows whose ACCEPT box is checked.
    Raises ValueError if a pair has more than one accepted candidate.
    """
    selections = {}
    current_pair = None
    for raw in review_text.splitlines():
        line = raw.strip()
        if line.startswith("## "):
            current_pair = line[3:].strip().split()[0]
            continue
        if not (line.startswith("|") and current_pair):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) < 3 or not cells[0].startswith("cand_"):
            continue
        accept = cells[-2].lower()
        if accept in ("[x]", "x"):
            if current_pair in selections:
                raise ValueError(
                    f"pair {current_pair} has more than one ACCEPT checked")
            selections[current_pair] = cells[0]
    return selections


def missing_pairs(bench_rows, pose_pairs):
    """Bench scene_floor pairs that have no init pose, sorted."""
    bench = set()
    for r in bench_rows:
        bench.add(f"{r['scene']}_{int(float(r['floor']))}")
    return sorted(bench - set(pose_pairs))


def anchor_heights_for_pair(bench_rows, scene_floor: str):
    """All annotation heights (habitat y) recorded for a pair."""
    hs = []
    for r in bench_rows:
        key = f"{r['scene']}_{int(float(r['floor']))}"
        if key != scene_floor:
            continue
        for col in ("ann_pos_y", "anchor_center_y"):
            v = (r.get(col) or "").strip()
            if v:
                try:
                    hs.append(float(v))
                except ValueError:
                    pass
    return hs


# ---------------------------------------------------------------------------
# Habitat-dependent helpers (only imported/executed inside the container)
# ---------------------------------------------------------------------------

def _load_bench_rows():
    sys.path.insert(0, repo_root())
    from src.splits import load_split
    return load_split(SPLIT_NAME)


def _load_pose_rows(path=None):
    p = resolve(path or POSES_CSV)
    with open(p, newline="") as f:
        return list(csv.DictReader(f))


def _make_sim(scene_dir_name: str):
    """Plain habitat_sim Simulator with the benchmark's camera settings."""
    import habitat_sim

    scene_id = scene_dir_name.split("-")[-1]
    root = resolve(SCENES_ROOT)
    glb = os.path.join(root, scene_dir_name, f"{scene_id}.basis.glb")
    if not os.path.exists(glb):
        raise FileNotFoundError(glb)

    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_dataset_config_file = os.path.join(
        root, "hm3d_annotated_train_basis.scene_dataset_config.json")
    # -1 skips the CUDA-to-EGL device match and takes the first EGL device.
    # Under WSL there is no NVIDIA EGL ICD, only mesa, so the CUDA match can
    # never succeed; mesa renders the review frames fine. Override with
    # MAPG_POSE_GPU=0 on hosts with a proper NVIDIA EGL stack.
    sim_cfg.gpu_device_id = int(os.environ.get("MAPG_POSE_GPU", "-1"))
    sim_cfg.scene_id = glb
    sim_cfg.enable_physics = False

    spec = habitat_sim.CameraSensorSpec()
    spec.uuid = "color"
    spec.sensor_type = habitat_sim.SensorType.COLOR
    spec.resolution = [IMG_HEIGHT, IMG_WIDTH]
    spec.position = [0.0, CAMERA_HEIGHT, 0.0]
    spec.hfov = HFOV
    spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE

    agent_cfg = habitat_sim.agent.AgentConfiguration(
        radius=0.1, sensor_specifications=[spec], action_space={},
        body_type="cylinder")
    return habitat_sim.Simulator(habitat_sim.Configuration(sim_cfg, [agent_cfg]))


def _make_pathfinder(sim, seed: int):
    """Recompute the navmesh exactly like HabitatInterface._make_navgraph."""
    import habitat_sim

    pf = habitat_sim.nav.PathFinder()
    settings = habitat_sim.NavMeshSettings()
    settings.agent_radius = AGENT_RADIUS
    ok = sim.recompute_navmesh(pf, settings)
    if not ok:
        raise RuntimeError("recompute_navmesh failed")
    pf.seed(int(seed))
    return pf


def _island_of(pf, pt):
    """Island index of a navmesh point, robust to habitat_sim version."""
    if hasattr(pf, "get_island"):
        return int(pf.get_island(pt))
    n = getattr(pf, "num_islands", 1)
    best_i, best_d = -1, float("inf")
    for i in range(int(n)):
        try:
            s = pf.snap_point(pt, island_index=i)
        except TypeError:
            # Very old habitat_sim without island support: treat the whole
            # navmesh as one island rather than crash.
            return 0
        d = _dist3(s, pt)
        if math.isfinite(d) and d < best_d:
            best_i, best_d = i, d
    return best_i if best_d < 0.05 else -1


def _dist3(a, b):
    try:
        return math.sqrt(sum((float(a[i]) - float(b[i])) ** 2 for i in range(3)))
    except (TypeError, ValueError):
        return float("inf")


def _dist_xz(a, b):
    return math.hypot(float(a[0]) - float(b[0]), float(a[2]) - float(b[2]))


def _target_island_for_band(pf, band):
    """Largest island of the floor band.

    Islands may span floors (stairs), so rank by island_area weighted by
    the fraction of the island's navmesh vertices inside the band.
    """
    lo, hi, _n = band
    verts = pf.build_navmesh_vertices()
    per_island = {}
    for v in verts:
        i = _island_of(pf, v)
        if i < 0:
            continue
        tot, inb = per_island.get(i, (0, 0))
        in_band = (lo - 0.3) <= float(v[1]) <= (hi + 0.3)
        per_island[i] = (tot + 1, inb + (1 if in_band else 0))
    best_i, best_w = -1, -1.0
    for i, (tot, inb) in sorted(per_island.items()):
        if inb == 0:
            continue
        try:
            area = float(pf.island_area(i)) if hasattr(pf, "island_area") else float(tot)
        except TypeError:
            area = float(tot)
        w = area * (inb / tot)
        if w > best_w + 1e-9:
            best_i, best_w = i, w
    if best_i < 0:
        raise RuntimeError(f"no navmesh island intersects band {band}")
    in_band_area = best_w
    total_area = (float(pf.navigable_area)
                  if hasattr(pf, "navigable_area") else 0.0)
    frac = (in_band_area / total_area) if total_area > 0 else 1.0
    return best_i, in_band_area, frac


def _scene_bboxes(sim):
    """Object aabbs worth avoiding, as (name, minv, maxv) in habitat frame."""
    out = []
    ss = sim.semantic_scene
    if ss is None:
        return out
    for obj in getattr(ss, "objects", []) or []:
        if obj is None or obj.category is None:
            continue
        try:
            name = str(obj.category.name()).lower()
        except TypeError:
            name = str(obj.category.name).lower()
        if any(s in name for s in BBOX_SKIP_SUBSTRINGS):
            continue
        try:
            aabb = obj.aabb
            c = aabb.center() if callable(aabb.center) else aabb.center
            s = aabb.sizes() if callable(getattr(aabb, "sizes", None)) else getattr(aabb, "sizes", None)
            if s is None:
                mn, mx = aabb.min, aabb.max
            else:
                mn = [float(c[i]) - float(s[i]) / 2.0 for i in range(3)]
                mx = [float(c[i]) + float(s[i]) / 2.0 for i in range(3)]
            mn = [float(v) for v in mn]
            mx = [float(v) for v in mx]
        except (AttributeError, TypeError, ValueError):
            continue
        if not all(math.isfinite(v) for v in mn + mx):
            continue
        if (mx[0] - mn[0]) * (mx[2] - mn[2]) > BBOX_MAX_FOOTPRINT_M2:
            continue
        out.append((name, mn, mx))
    return out


def _bbox_check(bboxes, pt):
    """Object-bbox test for the agent slab at pt.

    Returns (inside_deep, min_horizontal_distance, max_penetration_m).
    ``inside_deep`` is True only when the point penetrates some bbox
    deeper than BBOX_PENETRATION_TOL (raw HM3D aabbs are loose, see the
    constant's comment).
    """
    px, py, pz = float(pt[0]), float(pt[1]), float(pt[2])
    z_lo, z_hi = py + BODY_Z_LO, py + BODY_Z_HI
    min_d = float("inf")
    max_pen = 0.0
    for _name, mn, mx in bboxes:
        if mx[1] < z_lo or mn[1] > z_hi:
            continue
        dx = max(mn[0] - px, 0.0, px - mx[0])
        dz = max(mn[2] - pz, 0.0, pz - mx[2])
        d = math.hypot(dx, dz)
        if d == 0.0:
            pen = min(px - mn[0], mx[0] - px, pz - mn[2], mx[2] - pz)
            max_pen = max(max_pen, pen)
        min_d = min(min_d, d)
    if min_d == float("inf"):
        min_d = BBOX_NORM
    return max_pen > BBOX_PENETRATION_TOL, min_d, max_pen


def _free_range_profile(pf, pt, island, n_bins=N_YAW_BINS,
                        max_range=FREE_RANGE_MAX, step=FREE_RANGE_STEP):
    """Free walkable range along each yaw bin, staying on the pose's floor."""
    px, py, pz = float(pt[0]), float(pt[1]), float(pt[2])
    prof = []
    for b in range(n_bins):
        a = yaw_bin_angle(b, n_bins)
        ux, uz = -math.sin(a), -math.cos(a)
        free = 0.0
        d = step
        while d <= max_range + 1e-9:
            probe = [px + d * ux, py, pz + d * uz]
            try:
                snap = pf.snap_point(probe, island_index=island)
            except TypeError:
                snap = pf.snap_point(probe)
            if (_dist_xz(snap, probe) > 0.25
                    or not math.isfinite(float(snap[1]))
                    or abs(float(snap[1]) - py) > 0.5):
                break
            free = d
            d += step
        prof.append(free)
    return prof


def _clearance(pf, pt, max_search=5.0):
    try:
        return float(pf.distance_to_closest_obstacle(pt, max_search))
    except TypeError:
        return float(pf.distance_to_closest_obstacle(pt, max_search_radius=max_search))


def _render(sim, pt, angle_rad, png_path):
    import numpy as np
    from habitat_sim.utils.common import quat_from_angle_axis
    import habitat_sim

    agent = sim.get_agent(0)
    state = habitat_sim.AgentState()
    state.position = np.array([float(pt[0]), float(pt[1]), float(pt[2])],
                              dtype=np.float32)
    state.rotation = quat_from_angle_axis(float(angle_rad),
                                          np.array([0.0, 1.0, 0.0]))
    agent.set_state(state)
    obs = sim.get_sensor_observations()
    rgb = np.asarray(obs["color"])[:, :, :3]
    os.makedirs(os.path.dirname(png_path), exist_ok=True)
    try:
        from PIL import Image
        Image.fromarray(rgb).save(png_path)
    except ImportError:
        import imageio
        imageio.imwrite(png_path, rgb)


def _measure_point(pf, bboxes, pt, island=None):
    """All quality metrics for one navmesh point."""
    if island is None:
        island = _island_of(pf, pt)
    clearance = _clearance(pf, pt)
    island_radius = float(pf.island_radius(pt))
    prof = _free_range_profile(pf, pt, island)
    yaw, cone_mean = choose_yaw(prof)
    inside, bbox_d, bbox_pen = _bbox_check(bboxes, pt)
    return {
        "clearance_m": clearance,
        "island_radius_m": island_radius,
        "island": island,
        "openness_m": cone_mean,
        "best_yaw_rad": yaw,
        "mean_free_range_m": sum(prof) / len(prof),
        "inside_bbox": inside,
        "bbox_dist_m": bbox_d,
        "bbox_pen_m": bbox_pen,
    }


# ---------------------------------------------------------------------------
# measure: quality envelope of the existing 49 poses
# ---------------------------------------------------------------------------

def cmd_measure(args):
    pose_rows = _load_pose_rows()
    by_scene = {}
    for r in pose_rows:
        scene = r["scene_floor"].rsplit("_", 1)[0]
        by_scene.setdefault(scene, []).append(r)

    out_dir = resolve(OUT_DIR)
    os.makedirs(out_dir, exist_ok=True)
    metrics_path = os.path.join(out_dir, "existing_pose_metrics.csv")
    rows_out = []
    for scene in sorted(by_scene):
        print(f"[measure] {scene} ({len(by_scene[scene])} poses)", flush=True)
        sim = None
        try:
            sim = _make_sim(scene)
            pf = _make_pathfinder(sim, pair_seed(args.seed, scene))
            bboxes = _scene_bboxes(sim)
            for r in by_scene[scene]:
                pt = [float(r["init_x"]), float(r["init_y"]), float(r["init_z"])]
                snap = pf.snap_point(pt)
                snap_d = _dist3(snap, pt)
                use = snap if (math.isfinite(snap_d) and snap_d < 0.5) else pt
                m = _measure_point(pf, bboxes, use)
                m.update({
                    "scene_floor": r["scene_floor"],
                    "snap_dist_m": snap_d if math.isfinite(snap_d) else -1.0,
                    "is_navigable": bool(pf.is_navigable(use)),
                    "error": "",
                })
                rows_out.append(m)
        except Exception as e:  # keep the batch going
            traceback.print_exc()
            for r in by_scene[scene]:
                rows_out.append({"scene_floor": r["scene_floor"],
                                 "error": f"{type(e).__name__}: {e}"})
        finally:
            if sim is not None:
                sim.close(destroy=True)

    cols = ["scene_floor", "clearance_m", "island_radius_m", "island",
            "openness_m", "best_yaw_rad", "mean_free_range_m", "inside_bbox",
            "bbox_dist_m", "bbox_pen_m", "snap_dist_m", "is_navigable",
            "error"]
    with open(metrics_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for m in rows_out:
            w.writerow({c: m.get(c, "") for c in cols})

    ok = [m for m in rows_out if not m.get("error")]

    def pct(vals, q):
        if not vals:
            return None
        vals = sorted(vals)
        idx = min(len(vals) - 1, max(0, int(round(q / 100.0 * (len(vals) - 1)))))
        return vals[idx]

    def stats(key):
        vals = [float(m[key]) for m in ok if key in m]
        return {"min": min(vals), "p10": pct(vals, 10), "median": pct(vals, 50),
                "p90": pct(vals, 90), "max": max(vals)} if vals else {}

    envelope = {
        "n_poses": len(ok),
        "n_errors": len(rows_out) - len(ok),
        "clearance_m": stats("clearance_m"),
        "island_radius_m": stats("island_radius_m"),
        "openness_m": stats("openness_m"),
        "mean_free_range_m": stats("mean_free_range_m"),
        "bbox_dist_m": stats("bbox_dist_m"),
        "bbox_pen_m": stats("bbox_pen_m"),
        "snap_dist_m": stats("snap_dist_m"),
        "inside_bbox_count": sum(1 for m in ok if m.get("inside_bbox")),
    }
    env_path = os.path.join(out_dir, "envelope.json")
    with open(env_path, "w") as f:
        json.dump(envelope, f, indent=2)
    print(json.dumps(envelope, indent=2))
    print(f"[measure] wrote {metrics_path} and {env_path}")


def _min_clearance_from_envelope(out_dir):
    """Calibrate the clearance gate to the existing poses' envelope."""
    env_path = os.path.join(out_dir, "envelope.json")
    default = 0.3
    if not os.path.exists(env_path):
        return default
    with open(env_path) as f:
        env = json.load(f)
    p10 = (env.get("clearance_m") or {}).get("p10")
    if p10 is None:
        return default
    # Never demand more than 0.3 m, never accept less than the measured
    # 10th percentile of the poses the benchmark already runs with
    # (floored at 0.15 m so a degenerate envelope cannot zero the gate).
    return max(0.15, min(default, float(p10)))


# ---------------------------------------------------------------------------
# collect: candidate generation for the missing pairs
# ---------------------------------------------------------------------------

CAND_COLS = [
    "scene_floor", "cand", "init_x", "init_y", "init_z", "init_angle",
    "clearance_m", "island_radius_m", "island", "island_area_m2",
    "island_frac", "openness_m", "mean_free_range_m", "bbox_dist_m",
    "bbox_pen_m", "inside_bbox", "band_lo", "band_hi", "n_bands", "band_votes",
    "anchor_heights", "score", "recommended", "png", "status", "error",
    "git_head",
]


def cmd_collect(args):
    bench_rows = _load_bench_rows()
    pose_rows = _load_pose_rows()
    pairs = missing_pairs(bench_rows, [r["scene_floor"] for r in pose_rows])
    if args.pairs:
        want = set(args.pairs.split(","))
        unknown = want - set(pairs)
        if unknown:
            raise SystemExit(f"not missing pairs: {sorted(unknown)}")
        pairs = [p for p in pairs if p in want]
    print(f"[collect] {len(pairs)} pairs, seed {args.seed}, K={args.k}")

    out_dir = resolve(OUT_DIR)
    os.makedirs(out_dir, exist_ok=True)
    min_clear = _min_clearance_from_envelope(out_dir)
    print(f"[collect] clearance gate: {min_clear:.2f} m")

    all_rows = []
    for sf in pairs:
        all_rows.extend(_collect_pair(sf, bench_rows, out_dir, args, min_clear))
        _write_candidates_csv(out_dir, all_rows)  # checkpoint after each pair

    _write_candidates_csv(out_dir, all_rows)
    review_path = _write_review(out_dir, all_rows, min_clear, args)
    print(f"[collect] wrote {os.path.join(out_dir, 'candidates.csv')}")
    print(f"[collect] wrote {review_path}")


def _collect_pair(sf, bench_rows, out_dir, args, min_clear):
    scene = sf.rsplit("_", 1)[0]
    seed = pair_seed(args.seed, sf)
    print(f"[collect] {sf} (seed {seed})", flush=True)
    sim = None
    try:
        sim = _make_sim(scene)
        pf = _make_pathfinder(sim, seed)
        bboxes = _scene_bboxes(sim)

        verts = pf.build_navmesh_vertices()
        bands = cluster_heights([float(v[1]) for v in verts])
        anchors = anchor_heights_for_pair(bench_rows, sf)
        band_i, votes = pick_floor_band(bands, anchors)
        band = bands[band_i]
        island, island_area, island_frac = _target_island_for_band(pf, band)

        def sample(pool_target, check_bbox, taken):
            got, rejected, tries = [], 0, 0
            max_tries = 800
            while len(got) < pool_target and tries < max_tries:
                tries += 1
                try:
                    p = pf.get_random_navigable_point(island_index=island)
                except TypeError:
                    p = pf.get_random_navigable_point()
                p = [float(p[0]), float(p[1]), float(p[2])]
                if not all(math.isfinite(v) for v in p):
                    continue
                if not (band[0] - 0.3 <= p[1] <= band[1] + 0.3):
                    continue
                if _island_of(pf, p) != island:
                    continue
                if any(_dist_xz(p, q) < 0.5 for q in taken + got):
                    continue
                if _clearance(pf, p) < min_clear:
                    rejected += 1
                    continue
                if check_bbox:
                    inside, _d, _pen = _bbox_check(bboxes, p)
                    if inside:
                        rejected += 1
                        continue
                got.append(p)
            return got, rejected, tries

        accepted, rejected, tries = sample(args.pool, True, [])
        relaxed_from = len(accepted)
        if len(accepted) < args.k:
            # Tight floor: the existing 49 poses themselves sit deep inside
            # loose HM3D aabbs (29 of 49 deeper than the tolerance), so a
            # bbox-relaxed second pass is still within the measured
            # envelope. Rows sampled here are flagged in the status column.
            extra, _rej2, _t2 = sample(args.k - len(accepted), False, accepted)
            accepted += extra

        if not accepted:
            raise RuntimeError(
                f"no candidate passed the gates after {tries} tries "
                f"(band {band}, island {island}, {rejected} rejected)")

        picks = select_diverse(accepted, args.k)
        rows = []
        for k, idx in enumerate(picks):
            p = accepted[idx]
            m = _measure_point(pf, bboxes, p, island=island)
            score = composite_score(m["clearance_m"], m["openness_m"],
                                    island_frac, m["bbox_dist_m"])
            png_rel = os.path.join("pose_candidates", sf, f"cand_{k}.png")
            png_abs = os.path.join(out_dir, sf, f"cand_{k}.png")
            _render(sim, p, m["best_yaw_rad"], png_abs)
            rows.append({
                "scene_floor": sf, "cand": f"cand_{k}",
                "init_x": round(p[0], 4), "init_y": round(p[1], 4),
                "init_z": round(p[2], 4),
                "init_angle": round(m["best_yaw_rad"], 4),
                "clearance_m": round(m["clearance_m"], 3),
                "island_radius_m": round(m["island_radius_m"], 2),
                "island": island,
                "island_area_m2": round(island_area, 2),
                "island_frac": round(island_frac, 3),
                "openness_m": round(m["openness_m"], 2),
                "mean_free_range_m": round(m["mean_free_range_m"], 2),
                "bbox_dist_m": round(m["bbox_dist_m"], 3),
                "bbox_pen_m": round(m["bbox_pen_m"], 3),
                "inside_bbox": m["inside_bbox"],
                "band_lo": round(band[0], 3), "band_hi": round(band[1], 3),
                "n_bands": len(bands),
                "band_votes": "/".join(str(v) for v in votes),
                "anchor_heights": ";".join(f"{h:.2f}" for h in anchors),
                "score": round(score, 4), "recommended": "",
                "png": png_rel,
                "status": "ok" if idx < relaxed_from else "ok_relaxed_bbox",
                "error": "",
                "git_head": git_head(),
            })
        best = max(rows, key=lambda r: (r["score"], r["cand"]))
        best["recommended"] = "yes"
        return rows
    except Exception as e:
        traceback.print_exc()
        return [{c: "" for c in CAND_COLS} | {
            "scene_floor": sf, "cand": "", "status": "error",
            "error": f"{type(e).__name__}: {e}", "git_head": git_head(),
        }]
    finally:
        if sim is not None:
            sim.close(destroy=True)


def _write_candidates_csv(out_dir, rows):
    path = os.path.join(out_dir, "candidates.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CAND_COLS)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in CAND_COLS})


def _write_review(out_dir, rows, min_clear, args):
    env_path = os.path.join(out_dir, "envelope.json")
    env_txt = ""
    if os.path.exists(env_path):
        with open(env_path) as f:
            env = json.load(f)

        def fmt(key):
            s = env.get(key) or {}
            if not s:
                return "n/a"
            return (f"min {s['min']:.2f} / p10 {s['p10']:.2f} / "
                    f"median {s['median']:.2f} / max {s['max']:.2f}")
        env_txt = (
            f"Measured envelope of the existing {env.get('n_poses', '?')} poses "
            f"(quality bar the candidates are calibrated against):\n\n"
            f"- clearance from walls and obstacles (m): {fmt('clearance_m')}\n"
            f"- island radius (m): {fmt('island_radius_m')}\n"
            f"- openness, best 90 degree cone mean free range (m): {fmt('openness_m')}\n"
            f"- distance to nearest object bbox (m): {fmt('bbox_dist_m')}\n"
            f"- poses inside an object bbox: {env.get('inside_bbox_count', '?')}\n")

    by_pair = {}
    for r in rows:
        by_pair.setdefault(r["scene_floor"], []).append(r)

    lines = [
        "# Init pose review sheet (board item 6)",
        "",
        f"Provenance: generated at repo commit {git_head()} by "
        "src/scripts/collect_init_poses.py.",
        "",
        f"Seeded run: seed {args.seed}, K={args.k}, clearance gate "
        f"{min_clear:.2f} m. Re-running with the same seed reproduces "
        "these exact candidates.",
        "",
        "How to review: open each pair's PNGs (paths are relative to "
        "datasets/explore-eqa/), check exactly one ACCEPT box per pair "
        "(the RECOMMENDED row is the best composite score, override "
        "freely), then run the finalize command at the bottom.",
        "",
    ]
    if env_txt:
        lines += [env_txt]
    for sf in sorted(by_pair):
        prs = by_pair[sf]
        lines.append(f"## {sf}")
        lines.append("")
        err_rows = [r for r in prs if r["status"] == "error"]
        if err_rows:
            lines.append(f"SCENE FAILED: {err_rows[0]['error']}")
            lines.append("")
            continue
        r0 = prs[0]
        lines.append(
            f"Floor band [{r0['band_lo']}, {r0['band_hi']}] of "
            f"{r0['n_bands']} bands (votes {r0['band_votes']}), anchors at "
            f"y = {r0['anchor_heights']}, island {r0['island']} "
            f"(area {r0['island_area_m2']} m2, {r0['island_frac']} of scene).")
        lines.append("")
        lines.append("| cand | recommended | x | y | z | yaw | clearance_m | "
                     "island_radius_m | openness_m | bbox_dist_m | score | "
                     "image | ACCEPT | REJECT |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for r in prs:
            rec = "RECOMMENDED" if r["recommended"] == "yes" else ""
            lines.append(
                f"| {r['cand']} | {rec} | {r['init_x']} | {r['init_y']} | "
                f"{r['init_z']} | {r['init_angle']} | {r['clearance_m']} | "
                f"{r['island_radius_m']} | {r['openness_m']} | "
                f"{r['bbox_dist_m']} | {r['score']} | {r['png']} | [ ] | [ ] |")
        lines.append("")
    lines += [
        "## Finalize",
        "",
        "After checking one ACCEPT box per pair, run inside the container:",
        "",
        "    python3 -m src.scripts.collect_init_poses finalize --from-review",
        "",
        "This writes datasets/explore-eqa/scene_init_poses_semantic_only_v2.csv "
        "(the original 49 rows verbatim plus the approved 15) and prints its "
        "sha256 for splits/MANIFEST.json. The original CSV is never edited.",
        "",
        "Approved by: ____________  Date: ____________",
        "",
    ]
    path = os.path.join(out_dir, "pose_review.md")
    with open(path, "w") as f:
        f.write("\n".join(lines))
    return path


# ---------------------------------------------------------------------------
# finalize: build the v2 CSV from the approved candidates
# ---------------------------------------------------------------------------

def cmd_finalize(args):
    out_dir = resolve(OUT_DIR)
    cand_path = os.path.join(out_dir, "candidates.csv")
    with open(cand_path, newline="") as f:
        cand_rows = [r for r in csv.DictReader(f)
                     if r["status"].startswith("ok")]

    if args.approved:
        with open(resolve(args.approved), newline="") as f:
            selections = {r["scene_floor"]: r["cand"]
                          for r in csv.DictReader(f)}
    elif args.from_review:
        with open(os.path.join(out_dir, "pose_review.md")) as f:
            selections = parse_review_selections(f.read())
    else:
        raise SystemExit("finalize needs --approved <csv> or --from-review")

    bench_rows = _load_bench_rows()
    pose_rows = _load_pose_rows()
    need = missing_pairs(bench_rows, [r["scene_floor"] for r in pose_rows])
    missing_sel = [p for p in need if p not in selections]
    if missing_sel:
        raise SystemExit(
            f"no approved candidate for {len(missing_sel)} pairs: {missing_sel}")
    extra = sorted(set(selections) - set(need))
    if extra:
        raise SystemExit(f"approved pairs that are not missing: {extra}")

    by_key = {(r["scene_floor"], r["cand"]): r for r in cand_rows}
    new_rows = []
    for sf in need:
        key = (sf, selections[sf])
        if key not in by_key:
            raise SystemExit(f"approved candidate not found: {key}")
        r = by_key[key]
        new_rows.append((sf, float(r["init_x"]), float(r["init_y"]),
                         float(r["init_z"]), float(r["init_angle"])))

    src_path = resolve(POSES_CSV)
    with open(src_path) as f:
        original_text = f.read()
    v2_text = build_v2_text(original_text, new_rows)
    v2_path = resolve(V2_CSV)
    with open(v2_path, "w") as f:
        f.write(v2_text)

    # Sanity: v2 must cover every bench pair exactly once.
    with open(v2_path, newline="") as f:
        v2_pairs = [r["scene_floor"] for r in csv.DictReader(f)]
    bench_pairs = {f"{r['scene']}_{int(float(r['floor']))}" for r in bench_rows}
    assert len(v2_pairs) == len(set(v2_pairs)), "duplicate pair in v2"
    uncovered = bench_pairs - set(v2_pairs)
    assert not uncovered, f"v2 does not cover: {sorted(uncovered)}"

    sha = hashlib.sha256(v2_text.encode("utf-8")).hexdigest()
    print(f"[finalize] wrote {v2_path} ({len(v2_pairs)} rows)")
    print(f"[finalize] sha256 {sha}")
    print("[finalize] add this hash to splits/MANIFEST.json (do not edit "
          "the original scene_init_poses_semantic_only_new.csv)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    ap_m = sub.add_parser("measure", help="measure the existing 49 poses")
    ap_m.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap_m.set_defaults(func=cmd_measure)

    ap_c = sub.add_parser("collect", help="generate candidates for missing pairs")
    ap_c.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap_c.add_argument("--k", type=int, default=DEFAULT_K)
    ap_c.add_argument("--pool", type=int, default=40,
                      help="accepted samples to draw before diverse pick")
    ap_c.add_argument("--pairs", default="",
                      help="comma-separated scene_floor subset")
    ap_c.set_defaults(func=cmd_collect)

    ap_f = sub.add_parser("finalize", help="write the v2 CSV from approvals")
    ap_f.add_argument("--approved", default="",
                      help="CSV with columns scene_floor,cand")
    ap_f.add_argument("--from-review", action="store_true",
                      help="read ACCEPT checkboxes from pose_review.md")
    ap_f.set_defaults(func=cmd_finalize)

    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
