"""Frozen, SHA-pinned benchmark splits (gate 2 of the research harness).

Split CSVs live in ``splits/`` at the repo root and are pinned in
``splits/MANIFEST.json``. ``load_split`` verifies the file's SHA256 against
the pinned value on every load, so a run can never silently use a tampered
or wrong file. ``split_sha`` exposes the pinned hash for run manifests.

Stdlib only (csv, hashlib, json, pathlib) so this module imports anywhere,
including inside the Habitat docker image, without pandas.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

# Repo root is one level up from this file (src/splits.py).
_REPO_ROOT = Path(__file__).resolve().parent.parent
_MANIFEST_PATH = _REPO_ROOT / "splits" / "MANIFEST.json"


class SplitIntegrityError(RuntimeError):
    """Raised when a split file is missing or fails SHA256 verification."""


def _load_manifest(manifest_path: Path | None = None) -> dict:
    path = Path(manifest_path) if manifest_path is not None else _MANIFEST_PATH
    if not path.is_file():
        raise SplitIntegrityError(
            "SPLIT MANIFEST MISSING: expected the split manifest at "
            f"'{path}'. Without it no benchmark split can be verified. "
            "Restore splits/MANIFEST.json from git before running anything."
        )
    with open(path, encoding="utf-8") as f:
        manifest = json.load(f)
    return manifest


def _entry_for(name: str, manifest_path: Path | None = None) -> tuple[dict, Path]:
    manifest = _load_manifest(manifest_path)
    splits = manifest.get("splits", {})
    if name not in splits:
        known = ", ".join(sorted(splits)) or "none"
        raise SplitIntegrityError(
            f"UNKNOWN SPLIT '{name}': not present in the split manifest. "
            f"Known splits: {known}."
        )
    entry = splits[name]
    base = (Path(manifest_path).parent.parent if manifest_path is not None
            else _REPO_ROOT)
    return entry, base / entry["path"]


def _sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def split_sha(name: str, manifest_path: Path | None = None) -> str:
    """Return the pinned SHA256 for split ``name`` (for run manifests)."""
    entry, _ = _entry_for(name, manifest_path)
    return entry["sha256"]


def load_split(name: str, manifest_path: Path | None = None) -> list[dict]:
    """Load split ``name`` as a list of row dicts, verifying its SHA256.

    Raises SplitIntegrityError if the file is missing or its SHA256 does
    not match the value pinned in splits/MANIFEST.json.
    """
    entry, csv_path = _entry_for(name, manifest_path)

    if not csv_path.is_file():
        raise SplitIntegrityError(
            f"SPLIT FILE MISSING: split '{name}' should be at '{csv_path}' "
            "but no such file exists. The benchmark split is frozen; "
            "restore it from git. Do NOT substitute another CSV."
        )

    actual = _sha256_of(csv_path)
    expected = entry["sha256"]
    if actual != expected:
        raise SplitIntegrityError(
            f"SPLIT INTEGRITY FAILURE for '{name}': file '{csv_path}' has "
            f"sha256 {actual} but the manifest pins {expected}. The file "
            "has been modified or replaced. This split is FROZEN "
            f"(date_frozen={entry.get('date_frozen', 'unknown')}); results "
            "from a mismatched file are not comparable. Restore the exact "
            "file from git before running."
        )

    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    expected_rows = entry.get("rows")
    if expected_rows is not None and len(rows) != expected_rows:
        raise SplitIntegrityError(
            f"SPLIT ROW COUNT MISMATCH for '{name}': parsed {len(rows)} "
            f"rows but the manifest records {expected_rows}."
        )
    return rows
