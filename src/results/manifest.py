"""Run manifest builder (gate 4 of the research harness).

Every benchmark run gets a run.json that pins exactly what produced the
numbers: git SHA (with dirty flag and branch), a sha256 content hash of
the resolved config, the seed, the split name and its pinned SHA, model
pins, container flag and hostname. The manifest is required: the
results store refuses to record episodes for a run that was never
registered with one.

Git information is captured with subprocess git and fails HARD if git
is unavailable. The old wandb path swallowed every exception, which is
how runs without a recorded SHA happened. Not here.

Stdlib only. OmegaConf configs are flattened by duck typing (items /
list), never by importing omegaconf.
"""

from __future__ import annotations

import hashlib
import json
import os
import socket
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from src.splits import SplitIntegrityError, split_sha

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent


class ManifestError(RuntimeError):
    """Raised when a run manifest cannot be built (e.g. git missing)."""


def _to_plain(obj: Any) -> Any:
    """Recursively turn cfg-like objects into plain JSON types.

    Handles dicts and OmegaConf DictConfig (both have .items), lists,
    tuples and ListConfig (iterable, not str), and primitives. Anything
    else becomes its str form so hashing never fails.
    """
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if hasattr(obj, "items"):
        try:
            return {str(k): _to_plain(v) for k, v in obj.items()}
        except Exception:
            return str(obj)
    if isinstance(obj, (list, tuple)):
        return [_to_plain(v) for v in obj]
    # ListConfig and other sequence-likes.
    if hasattr(obj, "__iter__") and not isinstance(obj, (str, bytes)):
        try:
            return [_to_plain(v) for v in obj]
        except Exception:
            return str(obj)
    return str(obj)


def config_hash(cfg: Any) -> str:
    """sha256 of the resolved config, serialized canonically.

    Canonical means: plain types only, keys sorted, no whitespace
    variation. The same config always hashes the same, regardless of
    key order in the yaml.
    """
    plain = _to_plain(cfg)
    blob = json.dumps(plain, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _git(args: list, repo_root: Path) -> str:
    """Run a git command in repo_root. Hard fail with a clear message."""
    cmd = ["git", "-C", str(repo_root)] + args
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=30, check=False
        )
    except FileNotFoundError:
        raise ManifestError(
            "git is not installed or not on PATH. A run manifest requires "
            "the exact commit SHA; refusing to start a run without it."
        )
    except subprocess.TimeoutExpired:
        raise ManifestError(f"git command timed out: {' '.join(cmd)}")
    if proc.returncode != 0:
        raise ManifestError(
            f"git command failed ({' '.join(cmd)}): {proc.stderr.strip()}. "
            "A run manifest requires git metadata; fix the repo state "
            "before running."
        )
    return proc.stdout.strip()


def git_info(repo_root: Path | None = None) -> Dict[str, Any]:
    """Commit SHA, dirty flag and branch via subprocess git. No fallbacks."""
    root = Path(repo_root) if repo_root is not None else _REPO_ROOT
    sha = _git(["rev-parse", "HEAD"], root)
    branch = _git(["rev-parse", "--abbrev-ref", "HEAD"], root)
    status = _git(["status", "--porcelain"], root)
    return {
        "git_sha": sha,
        "git_sha_short": sha[:8],
        "git_branch": branch,
        "git_dirty": bool(status),
    }


def build_manifest(cfg: Any, seed: int, split_name: str) -> Dict[str, Any]:
    """Build the run manifest dict for one benchmark run.

    run_id is timestamp + short git SHA + seed, so ids sort by start
    time and two runs of the same commit and seed still get distinct
    ids. The split SHA comes from the pinned splits/MANIFEST.json; if
    the split name is not pinned there (legacy csv paths), the SHA is
    recorded as None with an explicit warning instead of crashing the
    run.
    """
    git = git_info()
    started = datetime.now(timezone.utc)
    seed_int = int(seed) if seed is not None else 0
    run_id = "{}_{}_s{}".format(
        started.strftime("%Y%m%d_%H%M%S"), git["git_sha_short"], seed_int
    )

    warnings = []
    try:
        pinned_split_sha = split_sha(str(split_name))
    except SplitIntegrityError as e:
        pinned_split_sha = None
        warnings.append(f"split '{split_name}' has no pinned SHA: {e}")

    plain_cfg = _to_plain(cfg)
    model_pins = None
    model_aliases = None
    if isinstance(plain_cfg, dict):
        model_pins = plain_cfg.get("model_pins")
        vlm = plain_cfg.get("vlm")
        if isinstance(vlm, dict):
            model_aliases = {
                k: vlm.get(k) for k in ("name", "model", "model_name") if k in vlm
            }

    manifest = {
        "run_id": run_id,
        "seed": seed_int,
        "split": str(split_name),
        "split_sha256": pinned_split_sha,
        "config_sha256": config_hash(cfg),
        "model_pins": model_pins,
        "model_aliases": model_aliases,
        "in_container": os.path.exists("/.dockerenv"),
        "hostname": socket.gethostname(),
        "start_time_utc": started.isoformat(),
        # Renderer provenance. This container is CUDA-capable but
        # render-incapable: gpu_device_id 0 raises "unable to find CUDA
        # device 0 among 1 EGL devices" and only software mesa works.
        # The v1 98-query numbers predate the move to WSL and were very
        # likely produced on a real GPU renderer, so which renderer
        # drew the pixels is a scientific fact about the run, not an
        # ops detail: the VLM arms consume those pixels. Recorded on
        # every run so a renderer confound is detectable later instead
        # of being argued about from memory.
        "renderer": renderer_info(cfg),
        **git,
    }
    if warnings:
        manifest["warnings"] = warnings
    return manifest


def renderer_info(cfg: Any) -> Dict[str, Any]:
    """Which renderer this run will use, and the GL string if available.

    Never raises and never constructs a simulator: this runs at manifest
    build time, before the run, and must not be able to break it.
    """
    plain = _to_plain(cfg)
    gpu = None
    if isinstance(plain, dict):
        hab = plain.get("habitat")
        if isinstance(hab, dict):
            gpu = hab.get("sim_gpu")
        if gpu is None:
            gpu = plain.get("sim_gpu")
    try:
        gpu_int = int(gpu) if gpu is not None else None
    except (TypeError, ValueError):
        gpu_int = None

    info: Dict[str, Any] = {
        "sim_gpu": gpu_int,
        "mode": "software" if gpu_int is not None and gpu_int < 0 else "gpu",
        "gl_renderer": os.environ.get("MAPG_GL_RENDERER"),
    }
    return info


def set_coverage(manifest: Dict[str, Any], attempted: int, limit: int,
                 uncovered_skipped) -> Dict[str, Any]:
    """Record what the run actually covered (janus condition c6).

    Console output is not evidence. The paper gets written from
    manifests months later, so a partial run has to be self-describing
    from its artifacts. Call this on every exit path, including the
    cost-cap breach path, where the console coverage line never prints.
    """
    missing = sorted({str(p) for p in (uncovered_skipped or [])})
    manifest["coverage"] = {
        "episodes_attempted": int(attempted),
        "limit": int(limit or 0),
        "skipped_no_init_pose": len(uncovered_skipped or []),
        "skipped_scene_floors": missing,
        "partial": bool(missing) or bool(limit),
    }
    return manifest


def write_manifest(manifest: Dict[str, Any], out_dir) -> Path:
    """Write manifest to out_dir/run.json atomically (temp file + rename).

    A crash mid-write can never leave a truncated run.json: the data is
    fully written and fsynced to a temp file first, then renamed into
    place with os.replace, which is atomic on POSIX.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    final_path = out / "run.json"
    tmp_path = out / ".run.json.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, default=str)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, final_path)
    return final_path
