#!/usr/bin/env python3
"""
Print the Habitat-Sim semantic scene graph as a tree (depth-first), and audit object AABBs.

Examples:
  # Full tree (truncated children), plus an audit summary
  python src/tools/print_semantic_tree.py \
    --scene-dir 00006-HkseAnWCgqk --scene-id HkseAnWCgqk --audit

  # Only objects that contain 'chair' and show AABB min/max explicitly
  python src/tools/print_semantic_tree.py \
    --scene-dir 00006-HkseAnWCgqk --scene-id HkseAnWCgqk \
    --only-objects --filter chair --show-aabb

Options:
  --max-children INT   Truncate long child lists per node (default: 50)
  --depth INT          Maximum recursion depth (default: 8)
  --show-aabb          Also print AABB min/max (center is printed by default if present)
  --filter ""          Substring filter on object category (lowercase), applied to objects
  --only-objects       Skip printing levels/regions, print only objects (still audits tree)
  --audit              Print a summary of object AABB coverage (has vs missing)
"""

import argparse
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Optional, Iterable, Tuple, Any, List, Set, Dict

import numpy as np

from src.envs.habitat_interface import HabitatInterface

DATASET_BASE_PATH = "/datasets/hm3d/train"

# ----------------- small helpers -----------------
def call_maybe(x):
    return x() if callable(x) else x

def safe_center(aabb) -> Optional[np.ndarray]:
    """Return center of AABB if available and sane."""
    if aabb is None:
        return None
    try:
        c = call_maybe(aabb.center)
    except Exception:
        try:
            c = (np.asarray(aabb.max, dtype=np.float32) + np.asarray(aabb.min, dtype=np.float32)) / 2.0
        except Exception:
            return None
    c = np.asarray(c, dtype=np.float32)
    if c.shape[-1] == 3 and np.isfinite(c).all():
        return c
    return None

def safe_cat_name(cat) -> Optional[str]:
    """Lowercased category name, if present."""
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

def aabb_tuple(aabb) -> Optional[Tuple[List[float], List[float]]]:
    """Return (min, max) lists if available."""
    try:
        mn = np.asarray(aabb.min, dtype=np.float32).tolist()
        mx = np.asarray(aabb.max, dtype=np.float32).tolist()
        return mn, mx
    except Exception:
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

# Known child-list attributes in Habitat semantic structures
CHILD_ATTRS = ("levels", "regions", "objects", "children", "nodes", "parts", "segments")

def iter_child_lists(node) -> Iterable[Tuple[str, list]]:
    """Yield (attr_name, list) for known child containers that are non-empty."""
    for attr in CHILD_ATTRS:
        try:
            val = getattr(node, attr, None)
        except Exception:
            val = None
        if isinstance(val, (list, tuple)) and len(val) > 0:
            yield attr, list(val)

def is_semantic_object(node: Any) -> bool:
    """Heuristic: objects typically have aabb + category and are not regions (which often have 'regions')."""
    has_obj_bits = hasattr(node, "aabb") and hasattr(node, "category")
    has_region = hasattr(node, "regions")
    return bool(has_obj_bits and not has_region)

def node_header(node, show_minmax: bool = False) -> str:
    parts = []
    # type name
    parts.append(node.__class__.__name__)
    # id
    nid = getattr(node, "id", None)
    if nid is not None:
        parts.append(f"id={nid}")
    # category (objects only)
    if is_semantic_object(node):
        cname = safe_cat_name(getattr(node, "category", None))
        if cname:
            parts.append(f"cat={cname}")
    # center (if aabb present)
    ctr = safe_center(getattr(node, "aabb", None))
    if ctr is not None:
        parts.append(f"center={np.round(ctr, 3).tolist()}")
    else:
        if is_semantic_object(node):
            parts.append("center=None")
    # explicit min/max
    if show_minmax:
        ab = aabb_tuple(getattr(node, "aabb", None))
        if ab:
            mn, mx = ab
            parts.append(f"aabb_min={np.round(np.array(mn),3).tolist()}")
            parts.append(f"aabb_max={np.round(np.array(mx),3).tolist()}")
        else:
            if is_semantic_object(node):
                parts.append("aabb=None")
    return " ".join(parts)

def print_tree(node: Any,
               indent: int,
               max_children: int,
               depth_left: int,
               show_aabb: bool,
               obj_filter: str,
               only_objects: bool,
               seen: Set[int]):
    if node is None or depth_left < 0:
        return
    # avoid cycles by object identity
    oid = id(node)
    if oid in seen:
        print("  " * indent + "(…seen…)")
        return
    seen.add(oid)

    # filter logic for objects only
    if is_semantic_object(node):
        cname = safe_cat_name(getattr(node, "category", None)) or ""
        if obj_filter and obj_filter not in cname:
            return
        # print object headers
        print("  " * indent + node_header(node, show_minmax=show_aabb))
    else:
        # non-object nodes (levels/regions/…)
        if not only_objects:
            print("  " * indent + node_header(node, show_minmax=False))

    # children
    for attr, lst in iter_child_lists(node):
        if not only_objects:
            print("  " * indent + f"  {attr} (count={len(lst)}):")
        for i, child in enumerate(lst[:max_children]):
            print_tree(child, indent + (0 if only_objects else 2),
                       max_children, depth_left - 1, show_aabb,
                       obj_filter, only_objects, seen)
        if len(lst) > max_children and not only_objects:
            print("  " * (indent + 2) + f"... ({len(lst) - max_children} more)")

def audit_object_aabbs(scene) -> Dict[str, Any]:
    """Walk the tree and collect AABB coverage stats for objects."""
    stats = {
        "total_objects": 0,
        "with_aabb": 0,
        "without_aabb": 0,
        "examples_missing": [],  # (id, category) pairs
        "examples_present": [],  # (id, category) pairs
    }

    def walk(n):
        if is_semantic_object(n):
            stats["total_objects"] += 1
            has = getattr(n, "aabb", None) is not None and (safe_center(getattr(n, "aabb", None)) is not None or aabb_tuple(getattr(n, "aabb", None)) is not None)
            cname = safe_cat_name(getattr(n, "category", None)) or "unknown"
            nid = getattr(n, "id", None)
            if has:
                stats["with_aabb"] += 1
                if len(stats["examples_present"]) < 5:
                    stats["examples_present"].append((nid, cname))
            else:
                stats["without_aabb"] += 1
                if len(stats["examples_missing"]) < 5:
                    stats["examples_missing"].append((nid, cname))
        for _, lst in iter_child_lists(n):
            for c in lst:
                walk(c)

    walk(scene)
    return stats

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene-dir", required=True)
    ap.add_argument("--scene-id", required=True)
    ap.add_argument("--max-children", type=int, default=50)
    ap.add_argument("--depth", type=int, default=8)
    ap.add_argument("--show-aabb", action="store_true")
    ap.add_argument("--filter", type=str, default="", help="Substring on object category (lowercase)")
    ap.add_argument("--only-objects", action="store_true", help="Print only objects (skip levels/regions)")
    ap.add_argument("--audit", action="store_true", help="Print AABB coverage summary for objects")
    args = ap.parse_args()

    glb = os.path.join(DATASET_BASE_PATH, args.scene_dir, f"{args.scene_id}.basis.glb")
    assert os.path.exists(glb), f"Missing: {glb}"

    hab = HabitatInterface(glb, cfg=make_hab_cfg(), device="cpu")
    sim = hab._sim

    scene = sim.semantic_scene
    if not scene or not getattr(scene, "objects", None):
        print("[FATAL] No semantic scene or objects.")
        try:
            sim.close(destroy=True)
        except Exception:
            pass
        return

    # Root summary
    n_levels = len(getattr(scene, "levels", []) or [])
    n_regions = len(getattr(scene, "regions", []) or [])
    n_objects = len(getattr(scene, "objects", []) or [])
    print(f"\n=== SemanticScene ===  levels={n_levels} regions={n_regions} objects={n_objects}\n")

    # Optional audit
    if args.audit:
        stats = audit_object_aabbs(scene)
        tot = stats["total_objects"]
        w = stats["with_aabb"]
        wo = stats["without_aabb"]
        pct = (100.0 * w / tot) if tot > 0 else 0.0
        print("=== AABB Audit (objects) ===")
        print(f"total={tot} | with_aabb={w} | without_aabb={wo} | coverage={pct:.1f}%")
        if stats["examples_present"]:
            print("  examples_with_aabb (id, category):", stats["examples_present"])
        if stats["examples_missing"]:
            print("  examples_without_aabb (id, category):", stats["examples_missing"])
        print()

    # Print tree (objects-only or full)
    print_tree(scene, indent=0, max_children=args.max_children, depth_left=args.depth,
               show_aabb=args.show_aabb, obj_filter=(args.filter.strip().lower()),
               only_objects=args.only_objects, seen=set())

    # Cleanup
    try:
        sim.close(destroy=True)
    except Exception:
        pass

if __name__ == "__main__":
    main()
