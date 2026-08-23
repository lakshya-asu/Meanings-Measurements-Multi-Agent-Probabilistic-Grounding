"""Container/host path resolution (gate 3 of the research harness).

Inside Docker, docker-compose mounts ``./`` at ``/workspace`` and
``./datasets`` at ``/datasets``, so code historically hardcodes absolute
``/datasets/...`` paths. Those paths break on any host checkout. This
module maps them onto the local checkout so the same config and code run
in both places:

- ``/datasets/...`` is kept as-is when it exists (in the container),
  otherwise remapped to ``<repo>/datasets/...``.
- The env var ``MAPG_DATA_ROOT`` overrides the datasets root entirely.
- Relative paths are resolved against the repo root, so repo-relative
  config values like ``splits/bench_v1_98.csv`` work regardless of the
  current working directory.

Stdlib only, so it imports anywhere, including inside the Habitat image.
"""

from __future__ import annotations

import os
from pathlib import Path

# Repo root is one level up from this file (src/paths.py).
REPO_ROOT = Path(__file__).resolve().parent.parent

# The container mount point for the datasets volume.
CONTAINER_DATA_ROOT = "/datasets"

# Env var that overrides where the datasets tree lives on the host.
DATA_ROOT_ENV = "MAPG_DATA_ROOT"


def data_root() -> Path:
    """Return the datasets root: $MAPG_DATA_ROOT if set, else <repo>/datasets."""
    override = os.environ.get(DATA_ROOT_ENV, "").strip()
    if override:
        return Path(override)
    return REPO_ROOT / "datasets"


def resolve_data_path(p) -> str:
    """Resolve a config path so it works both in and out of the container.

    Rules, in order:
    1. ``/datasets/...`` paths: if $MAPG_DATA_ROOT is set, remap onto it.
       Otherwise keep the path when it exists (we are in the container),
       else remap onto ``<repo>/datasets/...``.
    2. Other absolute paths are returned unchanged.
    3. Relative paths are resolved against the repo root.
    """
    p = str(p)
    if p == CONTAINER_DATA_ROOT or p.startswith(CONTAINER_DATA_ROOT + "/"):
        rel = p[len(CONTAINER_DATA_ROOT):].lstrip("/")
        if os.environ.get(DATA_ROOT_ENV, "").strip():
            return str(data_root() / rel) if rel else str(data_root())
        if os.path.exists(p):
            return p
        return str(data_root() / rel) if rel else str(data_root())
    if os.path.isabs(p):
        return p
    return str(REPO_ROOT / p)
