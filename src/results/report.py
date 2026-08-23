"""Minimal inspection CLI for the gate 4 results database.

Usage:
    python3 -m src.results.report --db path/to/results.sqlite

Prints every run with its status and episode count, then per-run means
of the numeric columns grouped by (method, backend, seed). Stdlib only.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys

_NUMERIC = (
    "success",
    "num_steps",
    "vlm_calls",
    "traj_length",
    "episode_time_sec",
    "final_confidence",
)


def _fmt(value) -> str:
    if value is None:
        return "-"
    return f"{value:.3f}"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Inspect a gate 4 results db")
    parser.add_argument("--db", required=True, help="path to the sqlite database")
    args = parser.parse_args(argv)

    conn = sqlite3.connect(args.db)
    conn.row_factory = sqlite3.Row

    runs = conn.execute(
        "SELECT r.run_id, r.status, r.created_at, r.manifest, "
        "(SELECT COUNT(*) FROM episodes e WHERE e.run_id = r.run_id) AS n "
        "FROM runs r ORDER BY r.created_at"
    ).fetchall()

    if not runs:
        print("no runs recorded")
        return 0

    for r in runs:
        manifest = json.loads(r["manifest"])
        print(f"run {r['run_id']}")
        print(f"  status={r['status']} episodes={r['n']}")
        print(
            "  git={} dirty={} split={} split_sha={} config={}".format(
                manifest.get("git_sha_short", "?"),
                manifest.get("git_dirty", "?"),
                manifest.get("split", "?"),
                str(manifest.get("split_sha256"))[:12],
                str(manifest.get("config_sha256"))[:12],
            )
        )
        avg_cols = ", ".join(f"AVG({c}) AS {c}" for c in _NUMERIC)
        groups = conn.execute(
            f"SELECT method, backend, seed, COUNT(*) AS n, {avg_cols} "
            "FROM episodes WHERE run_id=? GROUP BY method, backend, seed",
            (r["run_id"],),
        ).fetchall()
        for g in groups:
            means = " ".join(f"{c}={_fmt(g[c])}" for c in _NUMERIC)
            print(
                f"  [{g['method']} | {g['backend']} | seed={g['seed']}] "
                f"n={g['n']} {means}"
            )
        print()

    conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
