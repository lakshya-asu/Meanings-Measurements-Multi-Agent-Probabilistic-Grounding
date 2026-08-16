"""Shared, stdlib-only pieces of the bench-v2-150 authoring toolchain (MAPG-07).

Three tools import this module:

- ``src/scripts/author_bench_v2.py``      : authoring plan, draft templates,
                                            assembly, spot-check draw, and the
                                            D6 test-retest noise-floor passes.
- ``src/scripts/validate_bench_rows.py``  : the row-level validator.
- ``src/scripts/render_query_context.py`` : per-query visual context.

What lives here is exactly the part that all three must agree on: the frozen
30-column v1 schema, the byte-level prefix rule, the annotation-tool
constants, the predicate vocabulary, and the number formatting used when a
new row is written back out as CSV text.

Nothing here imports habitat_sim, numpy, or pandas, so it is importable on
the host and inside the container alike, and the pure logic is unit tested
in ``tests/test_bench_authoring.py``.

Freeze rule this module encodes (protocol section 9 step 3): the 150-row
file must begin with the exact bytes of ``splits/bench_v1_98.csv``. That is
checked by comparing BYTES, never parsed rows, because a reparse-and-rewrite
round trip can silently normalize quoting, spacing, or line endings and still
produce "equal" rows.
"""

from __future__ import annotations

import csv
import hashlib
import io
import math
import os
from typing import Dict, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# Frozen v1 facts (measured from splits/bench_v1_98.csv, protocol section 1)
# ---------------------------------------------------------------------------

V1_SPLIT_NAME = "bench_v1_98"
V1_CSV_REL = "splits/bench_v1_98.csv"
V1_SHA256 = "ae3e2429eb08876577f1d871c0108252cc236fa53bf2c6985f1f70f1755e1a8f"
V1_ROWS = 98
V1_BYTES = 27549

# The deliverable. The tools refuse to write this path without an explicit
# freeze flag: freezing is a human gate (protocol section 9 step 7).
FROZEN_V2_REL = "splits/bench_v2_150.csv"
TARGET_ROWS = 150
NEW_ROWS = TARGET_ROWS - V1_ROWS  # 52

# MAPG-05 deliverable. V12 keys on it; there is no fallback to the 49-row
# file on purpose, see validate_bench_rows.check_pose_pairs.
POSE_CSV_V2 = "/datasets/explore-eqa/scene_init_poses_semantic_only_v2.csv"

# The 30 columns, in order. Adding or reordering any of them breaks the
# byte-identical prefix, so this list is frozen with the split.
V1_COLUMNS: Tuple[str, ...] = (
    "scene", "floor", "distance_m", "predicate", "msp_question", "ann_ok",
    "ann_ts", "ann_pos_x", "ann_pos_y", "ann_pos_z", "ann_yaw_rad",
    "ann_m", "ann_n", "ann_scale_x", "ann_scale_y", "ann_scale_z",
    "ann_aabb_min_x", "ann_aabb_min_y", "ann_aabb_min_z",
    "ann_aabb_max_x", "ann_aabb_max_y", "ann_aabb_max_z", "ann_volume",
    "anchor_sid", "anchor_label",
    "anchor_center_x", "anchor_center_y", "anchor_center_z",
    "GT Object 1", "GT Object 2",
)

# Columns that must hold a finite number on every row.
NUMERIC_COLUMNS: Tuple[str, ...] = (
    "floor", "distance_m", "ann_ok", "ann_ts",
    "ann_pos_x", "ann_pos_y", "ann_pos_z", "ann_yaw_rad",
    "ann_m", "ann_n", "ann_scale_x", "ann_scale_y", "ann_scale_z",
    "ann_aabb_min_x", "ann_aabb_min_y", "ann_aabb_min_z",
    "ann_aabb_max_x", "ann_aabb_max_y", "ann_aabb_max_z", "ann_volume",
    "anchor_sid",
    "anchor_center_x", "anchor_center_y", "anchor_center_z",
)

# The annotation tool's marker constants, identical on all 98 v1 rows
# (protocol section 1.4). New rows carry the same values verbatim.
ANN_TOOL_CONSTANTS: Dict[str, str] = {
    "ann_m": "2",
    "ann_n": "2",
    "ann_scale_x": "0.15",
    "ann_scale_y": "0.15",
    "ann_scale_z": "0.15",
    "ann_volume": "0.01418652",
}

# The v1 ann_aabb_* columns are the marker mesh's measured box, so their
# half extents wobble between 0.1495 and 0.1500 m rather than sitting at an
# exact constant. A synthesized box uses the nominal half extent; the
# validator only requires the box to contain ann_pos and to have an extent
# inside MARKER_EXTENT_RANGE_M.
MARKER_HALF_EXTENT_M = 0.15
MARKER_EXTENT_RANGE_M = (0.29, 0.31)

# Predicate column vocabulary, and the relation label each one maps to in
# src/parsing/metric_literal.infer_relation. "towards" and "from" are v1
# legacy singletons with no entry in the frozen phrase table: V10 cannot be
# evaluated for them, so they are reported as not-applicable rather than
# guessed at.
PREDICATE_TO_RELATION: Dict[str, Optional[str]] = {
    "in front of": "in_front_of",
    "right of": "right_of",
    "left of": "left_of",
    "behind": "behind",
    "above": "above",
    "below": "below",
    "between": "between",
    "near": "near",
    "towards": None,
    "from": None,
}

# Predicates a NEW row may use (protocol 2.1: towards and from stay at 1).
NEW_ROW_PREDICATES: Tuple[str, ...] = (
    "in front of", "right of", "left of", "behind",
    "above", "below", "between", "near",
)

HORIZONTAL_PREDICATES: Tuple[str, ...] = (
    "in front of", "right of", "left of", "behind", "near", "towards", "from",
)
VERTICAL_PREDICATES: Tuple[str, ...] = ("above", "below")

# Commanded-distance grid a new row may use (protocol 2.2, plus 0 for the
# between rows which carry no literal).
NEW_ROW_D0_GRID: Tuple[float, ...] = (
    0.0, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0,
)

# Parser plausible range, mirrored from src/parsing/metric_literal.py so the
# validator can report a range finding without importing a habitat-side
# module. Kept in sync by test_bench_authoring.
PLAUSIBLE_RANGE_M = (0.1, 10.0)

# Same epsilon as src/evals/success.py and src/tools/offset_metric.py.
EPS = 1e-9


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

def repo_root() -> str:
    """Repo root, one level up from src/ (mirrors collect_init_poses.py)."""
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def repo_path(rel: str) -> str:
    return os.path.join(repo_root(), rel)


# ---------------------------------------------------------------------------
# Bytes, hashes, and the prefix rule
# ---------------------------------------------------------------------------

def sha256_bytes(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()


def read_bytes(path: str) -> bytes:
    with open(path, "rb") as f:
        return f.read()


def line_index_of_offset(blob: bytes, offset: int) -> int:
    """1-based FILE line number containing byte ``offset`` (line 1 = header)."""
    if offset < 0:
        return 0
    return blob.count(b"\n", 0, min(offset, len(blob))) + 1


def prefix_report(candidate: bytes, frozen: bytes) -> Dict[str, object]:
    """Byte-level prefix check of ``candidate`` against the frozen v1 file.

    Returns a dict with ok, the frozen length, the first differing byte
    offset (None when the prefix matches), the 1-based file line that offset
    falls in, the 1-based DATA row index of that line, and the sha256 of the
    candidate's first len(frozen) bytes. Nothing is parsed as CSV: the whole
    point of the check is that the bytes are identical.
    """
    n = len(frozen)
    head = candidate[:n]
    out: Dict[str, object] = {
        "prefix_bytes": n,
        "candidate_bytes": len(candidate),
        "prefix_sha256": sha256_bytes(head),
        "frozen_sha256": sha256_bytes(frozen),
        "ok": False,
        "first_diff_offset": None,
        "first_diff_file_line": None,
        "first_diff_data_row": None,
        "reason": "",
    }
    if len(candidate) < n:
        out["reason"] = (
            "candidate is shorter than the frozen v1 file "
            f"({len(candidate)} < {n} bytes), so it cannot carry the prefix"
        )
        # Still locate the first difference inside the shared span.
        limit = len(candidate)
    else:
        limit = n

    diff = None
    for i in range(limit):
        if candidate[i] != frozen[i]:
            diff = i
            break
    if diff is None and len(candidate) >= n:
        out["ok"] = True
        return out
    if diff is None:
        # Truncated but otherwise identical.
        out["first_diff_offset"] = limit
        diff = limit
    else:
        out["first_diff_offset"] = diff
    file_line = line_index_of_offset(frozen, diff)
    out["first_diff_file_line"] = file_line
    out["first_diff_data_row"] = file_line - 1 if file_line > 1 else 0
    if not out["reason"]:
        out["reason"] = (
            f"byte {diff} differs (file line {file_line}, data row "
            f"{out['first_diff_data_row']}): frozen has "
            f"{frozen[diff:diff + 1]!r}, candidate has "
            f"{candidate[diff:diff + 1]!r}"
        )
    return out


def assemble_v2_bytes(frozen: bytes, new_lines: Sequence[str]) -> bytes:
    """Frozen v1 bytes, then the new rows, appended in binary (section 9.2).

    ``new_lines`` are CSV data lines WITHOUT a trailing newline. The frozen
    file already ends with a newline; if a future frozen file ever did not,
    one is added so the first new row cannot be glued onto the last v1 row.
    The frozen span is copied byte for byte and never reserialized.
    """
    out = bytearray(frozen)
    if out and not out.endswith(b"\n"):
        out.extend(b"\n")
    for line in new_lines:
        if "\n" in line or "\r" in line:
            raise ValueError(f"new row contains a line break: {line!r}")
        out.extend(line.encode("utf-8"))
        out.extend(b"\n")
    return bytes(out)


# ---------------------------------------------------------------------------
# CSV text helpers
# ---------------------------------------------------------------------------

def parse_csv_text(text: str) -> Tuple[List[str], List[Dict[str, str]]]:
    """Parse CSV text into (header, row dicts). No type coercion."""
    reader = csv.DictReader(io.StringIO(text, newline=""))
    rows = list(reader)
    header = list(reader.fieldnames or [])
    return header, rows


def csv_line(values: Sequence[str]) -> str:
    """One CSV data line, quoted exactly like python's csv writer, no newline."""
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(list(values))
    return buf.getvalue().rstrip("\n")


def row_to_line(row: Dict[str, object], columns: Sequence[str] = V1_COLUMNS) -> str:
    """Serialize a row dict to a CSV data line in the frozen column order.

    Missing keys raise: a silently blank column is exactly the kind of hole
    that would sail past a reviewer and fail at freeze time.
    """
    missing = [c for c in columns if c not in row]
    if missing:
        raise KeyError(f"row is missing columns: {missing}")
    extra = [k for k in row if k not in columns]
    if extra:
        raise KeyError(f"row has columns outside the frozen schema: {extra}")
    return csv_line([str(row[c]) for c in columns])


def fmt_coord(v: float, ndigits: int = 6) -> str:
    """Format a coordinate the way the v1 rows do: up to 6 dp, zeros trimmed."""
    r = round(float(v), ndigits)
    if r == 0:
        r = 0.0
    if r == int(r):
        return str(int(r))
    return repr(r)


def fmt_distance(v: float) -> str:
    """Format distance_m the way v1 does: '0', '0.5', '1', '1.5', '2'."""
    return fmt_coord(v, 3)


def marker_aabb(ann_pos: Sequence[float], half: float = MARKER_HALF_EXTENT_M):
    """Nominal marker box around ann_pos as (min_xyz, max_xyz)."""
    p = [float(ann_pos[0]), float(ann_pos[1]), float(ann_pos[2])]
    return ([c - half for c in p], [c + half for c in p])


# ---------------------------------------------------------------------------
# Geometry shared by the validator and the render helper
# ---------------------------------------------------------------------------

def dist3(a: Sequence[float], b: Sequence[float]) -> float:
    return math.sqrt(sum((float(a[i]) - float(b[i])) ** 2 for i in range(3)))


def dist_horizontal(a: Sequence[float], b: Sequence[float]) -> float:
    """Habitat y is up, so the horizontal distance uses x and z (metrics.md)."""
    return math.hypot(float(a[0]) - float(b[0]), float(a[2]) - float(b[2]))


def elevation_deg(anchor: Sequence[float], point: Sequence[float]) -> Optional[float]:
    """Elevation of the anchor-to-point ray above horizontal, in degrees.

    Returns None when the two points coincide within EPS (the direction, and
    therefore the elevation, is undefined; that is V13's case).
    """
    dx = float(point[0]) - float(anchor[0])
    dy = float(point[1]) - float(anchor[1])
    dz = float(point[2]) - float(anchor[2])
    horiz = math.hypot(dx, dz)
    if math.sqrt(dx * dx + dy * dy + dz * dz) <= EPS:
        return None
    return math.degrees(math.atan2(abs(dy), horiz))


def scene_floor_key(scene: str, floor: object) -> str:
    """'<scene>_<int floor>', the key used by the init-pose CSV."""
    return f"{scene}_{int(float(floor))}"


# ---------------------------------------------------------------------------
# Text style rules (protocol 4.2)
# ---------------------------------------------------------------------------

EM_DASH_CHARS = ("—", "–", "−")


def text_style_problems(value: str) -> List[str]:
    """House and protocol style violations in one field's text."""
    problems = []
    if value != value.strip():
        problems.append("leading or trailing whitespace")
    try:
        value.encode("ascii")
    except UnicodeEncodeError:
        problems.append("non-ascii characters")
    for ch in EM_DASH_CHARS:
        if ch in value:
            problems.append("em dash or en dash")
            break
    return problems
