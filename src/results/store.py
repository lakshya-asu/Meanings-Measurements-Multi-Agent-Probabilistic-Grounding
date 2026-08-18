"""SQLite results store (gate 4 of the research harness).

One row per episode, keyed by (run_id, qid), written with INSERT OR
REPLACE inside a transaction. WAL mode keeps writes atomic and
crash-safe: a killed run leaves every already-recorded episode intact,
unlike the legacy flat JSON that is rewritten whole on every episode.

A run must be registered with start_run(manifest) before any episode
can be recorded. That makes it impossible to produce rows that are not
tied to a manifest (git SHA, config hash, seed, split SHA).

Stdlib only: sqlite3, json, datetime. Numpy values are handled by duck
typing (tolist / __float__) and a default=str fallback, never by
importing numpy.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# Scalar columns pulled out of the episode dict for cheap SQL queries.
# The full dict is always kept verbatim in the "final" JSON column.
_INT_COLS = ("num_steps", "vlm_calls")
_REAL_COLS = ("traj_length", "episode_time_sec", "final_confidence")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id     TEXT PRIMARY KEY,
    manifest   TEXT NOT NULL,
    status     TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS episodes (
    run_id           TEXT NOT NULL,
    method           TEXT,
    backend          TEXT,
    split            TEXT,
    seed             INTEGER,
    qid              TEXT NOT NULL,
    scene            TEXT,
    success          INTEGER,
    num_steps        INTEGER,
    vlm_calls        INTEGER,
    traj_length      REAL,
    episode_time_sec REAL,
    final_confidence REAL,
    target_x         REAL,
    target_y         REAL,
    target_z         REAL,
    final            TEXT NOT NULL,
    created_at       TEXT NOT NULL,
    PRIMARY KEY (run_id, qid)
);

CREATE TABLE IF NOT EXISTS calls (
    run_id            TEXT NOT NULL,
    qid               TEXT NOT NULL,
    call_idx          INTEGER NOT NULL,
    step_idx          INTEGER,
    role              TEXT,
    model_name        TEXT,
    prompt_tokens     INTEGER,
    completion_tokens INTEGER,
    is_retry          INTEGER,
    latency_ms        REAL,
    created_at        TEXT NOT NULL,
    PRIMARY KEY (run_id, qid, call_idx)
);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_default(value: Any) -> Any:
    """Fallback for values json cannot serialize (numpy types etc.)."""
    if hasattr(value, "tolist"):
        try:
            return value.tolist()
        except Exception:
            pass
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return str(value)


def _as_int(value: Any) -> Optional[int]:
    """Coerce to int or return None. Never raises."""
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_real(value: Any) -> Optional[float]:
    """Coerce to float or return None. Never raises."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _point_xyz(row: Dict[str, Any]) -> tuple:
    """Pull target_point_xyz out of the row as three floats or Nones."""
    point = row.get("target_point_xyz")
    if point is None and isinstance(row.get("final_pred"), dict):
        point = row["final_pred"].get("target_point_xyz")
    if hasattr(point, "tolist"):
        try:
            point = point.tolist()
        except Exception:
            point = None
    if isinstance(point, (list, tuple)) and len(point) == 3:
        return (_as_real(point[0]), _as_real(point[1]), _as_real(point[2]))
    return (None, None, None)


class ResultsStore:
    """One-row-per-episode SQLite store with a runs (manifest) table."""

    def __init__(self, db_path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    # ------------------------------------------------------------------
    # Runs
    # ------------------------------------------------------------------
    def start_run(self, manifest: Dict[str, Any]) -> str:
        """Register a run from its manifest. Returns the run_id.

        Must be called before record_episode for that run_id. Re-calling
        with the same run_id (a resumed run) refreshes the manifest and
        sets the status back to running.
        """
        if not isinstance(manifest, dict) or not manifest.get("run_id"):
            raise ValueError(
                "start_run needs a manifest dict with a non-empty 'run_id'. "
                "Build one with src.results.manifest.build_manifest."
            )
        run_id = str(manifest["run_id"])
        now = _now()
        with self._conn:
            self._conn.execute(
                "INSERT INTO runs (run_id, manifest, status, created_at, updated_at) "
                "VALUES (?, ?, 'running', ?, ?) "
                "ON CONFLICT(run_id) DO UPDATE SET "
                "manifest=excluded.manifest, status='running', updated_at=excluded.updated_at",
                (run_id, json.dumps(manifest, default=_json_default), now, now),
            )
        return run_id

    def finish_run(self, run_id: str, status: str = "complete") -> None:
        """Mark a run complete or aborted."""
        if status not in ("running", "complete", "aborted"):
            raise ValueError(f"unknown run status {status!r}")
        with self._conn:
            self._conn.execute(
                "UPDATE runs SET status=?, updated_at=? WHERE run_id=?",
                (status, _now(), run_id),
            )

    def run_status(self, run_id: str) -> Optional[str]:
        cur = self._conn.execute("SELECT status FROM runs WHERE run_id=?", (run_id,))
        row = cur.fetchone()
        return row["status"] if row is not None else None

    def runs(self) -> List[Dict[str, Any]]:
        """All registered runs, manifests parsed back to dicts."""
        cur = self._conn.execute(
            "SELECT run_id, manifest, status, created_at, updated_at "
            "FROM runs ORDER BY created_at"
        )
        out = []
        for r in cur.fetchall():
            out.append(
                {
                    "run_id": r["run_id"],
                    "manifest": json.loads(r["manifest"]),
                    "status": r["status"],
                    "created_at": r["created_at"],
                    "updated_at": r["updated_at"],
                }
            )
        return out

    # ------------------------------------------------------------------
    # Episodes
    # ------------------------------------------------------------------
    def record_episode(self, run_id: str, row: Dict[str, Any]) -> None:
        """Write one episode row. INSERT OR REPLACE keyed by (run_id, qid).

        Raises RuntimeError if the run_id was never registered with
        start_run: no manifest, no results.
        """
        if not isinstance(row, dict):
            raise TypeError(f"episode row must be a dict, got {type(row).__name__}")
        if self.run_status(run_id) is None:
            raise RuntimeError(
                f"run_id {run_id!r} was never registered. Call "
                "store.start_run(manifest) before recording episodes; a run "
                "without a manifest is not a valid result."
            )

        qid = row.get("qid") or row.get("experiment_id")
        if not qid:
            raise ValueError("episode row needs a 'qid' (or 'experiment_id') key")

        success = row.get("success")
        success_int = None if success is None else (1 if bool(success) else 0)
        tx, ty, tz = _point_xyz(row)

        values = {
            "run_id": run_id,
            "method": row.get("method", "unknown"),
            "backend": row.get("backend", "unknown"),
            "split": row.get("split", "unknown"),
            "seed": _as_int(row.get("seed")),
            "qid": str(qid),
            "scene": row.get("scene"),
            "success": success_int,
            "num_steps": _as_int(row.get("num_steps")),
            "vlm_calls": _as_int(row.get("vlm_calls")),
            "traj_length": _as_real(row.get("traj_length")),
            "episode_time_sec": _as_real(row.get("episode_time_sec")),
            "final_confidence": _as_real(row.get("final_confidence")),
            "target_x": tx,
            "target_y": ty,
            "target_z": tz,
            "final": json.dumps(row, default=_json_default),
            "created_at": _now(),
        }
        cols = ", ".join(values)
        marks = ", ".join("?" for _ in values)
        with self._conn:
            self._conn.execute(
                f"INSERT OR REPLACE INTO episodes ({cols}) VALUES ({marks})",
                tuple(values.values()),
            )

    def episodes(self, run_id: str) -> List[Dict[str, Any]]:
        """All episode rows for a run, 'final' parsed back to a dict."""
        cur = self._conn.execute(
            "SELECT * FROM episodes WHERE run_id=? ORDER BY qid", (run_id,)
        )
        out = []
        for r in cur.fetchall():
            d = dict(r)
            d["final"] = json.loads(d["final"])
            out.append(d)
        return out

    # ------------------------------------------------------------------
    # Per-call rows (MAPG-02)
    # ------------------------------------------------------------------
    def record_calls(self, run_id: str, qid: str, calls: Any) -> None:
        """Replace the per-call LLM rows for one episode.

        ``calls`` is an iterable of dicts (CallLog.rows()) or objects
        with the CallRecord fields; call_idx is assigned from
        iteration order. Keyed (run_id, qid, call_idx): re-recording
        an episode replaces its rows, matching record_episode.

        Raises RuntimeError if the run_id was never registered with
        start_run, same contract as record_episode.
        """
        if self.run_status(run_id) is None:
            raise RuntimeError(
                f"run_id {run_id!r} was never registered. Call "
                "store.start_run(manifest) before recording calls; a run "
                "without a manifest is not a valid result."
            )
        if not qid:
            raise ValueError("record_calls needs a non-empty qid")

        def _get(rec: Any, key: str) -> Any:
            if isinstance(rec, dict):
                return rec.get(key)
            return getattr(rec, key, None)

        now = _now()
        rows = []
        for idx, rec in enumerate(calls):
            is_retry = _get(rec, "is_retry")
            rows.append(
                (
                    str(run_id),
                    str(qid),
                    idx,
                    _as_int(_get(rec, "step_idx")),
                    _get(rec, "role"),
                    _get(rec, "model_name"),
                    _as_int(_get(rec, "prompt_tokens")),
                    _as_int(_get(rec, "completion_tokens")),
                    None if is_retry is None else (1 if bool(is_retry) else 0),
                    _as_real(_get(rec, "latency_ms")),
                    now,
                )
            )
        with self._conn:
            self._conn.execute(
                "DELETE FROM calls WHERE run_id=? AND qid=?", (str(run_id), str(qid))
            )
            if rows:
                self._conn.executemany(
                    "INSERT INTO calls (run_id, qid, call_idx, step_idx, role, "
                    "model_name, prompt_tokens, completion_tokens, is_retry, "
                    "latency_ms, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    rows,
                )

    def calls(self, run_id: str, qid: Optional[str] = None) -> List[Dict[str, Any]]:
        """Per-call rows for a run (optionally one episode), in order."""
        if qid is None:
            cur = self._conn.execute(
                "SELECT * FROM calls WHERE run_id=? ORDER BY qid, call_idx",
                (run_id,),
            )
        else:
            cur = self._conn.execute(
                "SELECT * FROM calls WHERE run_id=? AND qid=? ORDER BY call_idx",
                (run_id, str(qid)),
            )
        return [dict(r) for r in cur.fetchall()]

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "ResultsStore":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
