"""Build the frozen GT_object -> Hydra node-ID mapping (scaffold).

The strict object-selection rule (metrics.md 2.3) scores exact node-ID
matches against a mapping that is built ONCE offline, released with the
benchmark, and versioned by hash. This script builds that mapping.

What it needs, and where each part can run:

  HOST OK      Loading the frozen bench split (src.splits.load_split),
               canonicalization and synonym mapping
               (src.evals.object_selection), reading already-exported
               scene-graph JSON files, writing the output JSON.

  CONTAINER    Producing the scene-graph JSON files in the first place.
  ONLY         Hydra scene graphs are built inside the Habitat docker
               image (flux04/mapg_habitat:src); export them there with
               one JSON per scene:  <graphs-dir>/<scene>.json  containing
               {"nodes": [{"node_id": int, "label": str,
                           "centroid": [x, y, z]}, ...]}
               The import of any Hydra/Habitat code is guarded below and
               never needed on the host: this script only READS the
               exported JSON.

Resolution rule per bench row (deterministic):
  1. Canonicalize the GT object label (NFKC, lowercase, articles and
     punctuation stripped, singularized, synonyms_v1 applied).
  2. Exact match against canonicalized node labels of that scene's graph.
  3. Ties (several same-label instances) break by smaller centroid
     distance to the row's anchor centroid (anchor_center_x/y/z in the
     bench CSV), then lower node ID.
  4. No match: the entry is recorded under "unresolved" for manual
     review. A mapping with unresolved entries is written but is NOT
     fit to freeze; resolve by hand and rebuild.

Output: splits/node_id_map_v1.json with per-scene entries and a sha256
self-hash in _meta (computed over the file content with the hash field
empty, then filled in). The script refuses to overwrite an existing
mapping unless --force is given: the mapping is frozen once released.

Usage:
  # Host, no graphs needed: show what would be done.
  python3 -m src.scripts.build_node_id_map --graphs-dir results/run1/graphs --dry-run

  # After exporting graphs from the container:
  python3 -m src.scripts.build_node_id_map --graphs-dir results/run1/graphs
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from src.evals.object_selection import canonicalize, load_synonyms  # noqa: E402
from src.splits import load_split  # noqa: E402

DEFAULT_OUT = _REPO_ROOT / "splits" / "node_id_map_v1.json"

# Container-only imports would go here, guarded. None are needed as long
# as scene graphs are consumed as exported JSON (the intended flow).
try:  # pragma: no cover - container only
    import hydra_python  # noqa: F401
    HAVE_HYDRA = True
except ImportError:
    HAVE_HYDRA = False


def _self_hash(payload: dict) -> str:
    """sha256 of the canonical serialization with the hash field empty."""
    payload["_meta"]["sha256"] = ""
    blob = json.dumps(payload, indent=2, sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _load_scene_graph(graphs_dir: Path, scene: str) -> list:
    path = graphs_dir / f"{scene}.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"No exported scene graph for '{scene}' at {path}. Export it "
            "from the container first (see module docstring)."
        )
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data["nodes"]


def _resolve(label: str, nodes: list, anchor_xyz, synonyms: dict):
    """Resolve one GT label to a node ID, or None if no exact match."""
    canon = canonicalize(label, synonyms)
    hits = [n for n in nodes if canonicalize(n["label"], synonyms) == canon]
    if not hits:
        return None

    def dist(node):
        c = node.get("centroid")
        if c is None or anchor_xyz is None:
            return 0.0
        return sum((a - b) ** 2 for a, b in zip(c, anchor_xyz))

    hits.sort(key=lambda n: (dist(n), int(n["node_id"])))
    return int(hits[0]["node_id"])


def build(split_name: str, graphs_dir: Path, synonyms: dict) -> dict:
    rows = load_split(split_name)
    scenes: dict = {}
    unresolved: list = []
    graph_cache: dict = {}

    for row in rows:
        scene = row["scene"]
        if scene not in graph_cache:
            graph_cache[scene] = _load_scene_graph(graphs_dir, scene)
        nodes = graph_cache[scene]
        anchor = None
        if row.get("anchor_center_x"):
            anchor = [
                float(row["anchor_center_x"]),
                float(row["anchor_center_y"]),
                float(row["anchor_center_z"]),
            ]
        for col in ("GT Object 1", "GT Object 2"):
            label = (row.get(col) or "").strip()
            if not label:
                continue
            node_id = _resolve(label, nodes, anchor, synonyms)
            entry = scenes.setdefault(scene, {})
            key = canonicalize(label, synonyms)
            if node_id is None:
                unresolved.append({"scene": scene, "label": label})
            elif key in entry and entry[key] != node_id:
                # Same canonical label resolved to two different nodes
                # across rows of one scene: needs a human decision.
                unresolved.append(
                    {"scene": scene, "label": label,
                     "conflict": [entry[key], node_id]}
                )
            else:
                entry[key] = node_id

    return {
        "_meta": {
            "version": "node_id_map_v1",
            "split": split_name,
            "sha256": "",
            "frozen": False,
            "note": "Set frozen true and pin in MANIFEST.json aux_files "
                    "only after unresolved is empty.",
        },
        "scenes": scenes,
        "unresolved": unresolved,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--split", default="bench_v1_98")
    ap.add_argument("--graphs-dir", type=Path, required=True,
                    help="Directory of per-scene exported graph JSONs")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--dry-run", action="store_true",
                    help="Host-safe: report what would be done, write nothing")
    ap.add_argument("--force", action="store_true",
                    help="Allow overwriting an existing frozen mapping")
    args = ap.parse_args(argv)

    if args.out.exists() and not args.force:
        print(f"REFUSING to overwrite existing mapping at {args.out}. "
              "The node-ID map is frozen once released; rerun with --force "
              "only if you intend to rebuild it.", file=sys.stderr)
        return 2

    if args.dry_run:
        rows = load_split(args.split)
        scenes = sorted({r["scene"] for r in rows})
        labels = sorted({
            (r.get(c) or "").strip()
            for r in rows for c in ("GT Object 1", "GT Object 2")
            if (r.get(c) or "").strip()
        })
        graphs_ok = args.graphs_dir.is_dir()
        print(f"[dry-run] split '{args.split}': {len(rows)} rows, "
              f"{len(scenes)} scenes, {len(labels)} distinct GT labels.")
        print(f"[dry-run] graphs dir {args.graphs_dir}: "
              f"{'found' if graphs_ok else 'MISSING (export from container)'}")
        print(f"[dry-run] hydra importable here: {HAVE_HYDRA} "
              "(not required when graphs are pre-exported JSON)")
        print(f"[dry-run] would resolve each GT label per scene and write "
              f"{args.out} with a sha256 self-hash.")
        return 0

    synonyms = load_synonyms()
    payload = build(args.split, args.graphs_dir, synonyms)
    payload["_meta"]["sha256"] = _self_hash(payload)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")
    n_ok = sum(len(v) for v in payload["scenes"].values())
    n_bad = len(payload["unresolved"])
    print(f"Wrote {args.out}: {n_ok} resolved entries, {n_bad} unresolved. "
          f"sha256={payload['_meta']['sha256']}")
    if n_bad:
        print("Mapping has unresolved entries and is NOT fit to freeze.",
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
