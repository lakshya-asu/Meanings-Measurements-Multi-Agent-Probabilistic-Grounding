#!/usr/bin/env python3
"""Row-level validator for the bench-v2-150 split (MAPG-07, protocol section 6).

    python3 -m src.scripts.validate_bench_rows --csv <file> --mode {strict,report}

Strict mode (the default) exits nonzero on any hard failure and on any hard
check that could not be run. Report mode records findings and exits zero; it
is the mode protocol section 6.3 prescribes for the frozen 98, whose known
flaws must be documented, never fixed.

Every finding names the offending 1-based DATA row index (row 1 is the first
row after the header, matching the loader's row identity), its scene, and
what is wrong.

What is checked
---------------

File level:

  P1  the v1 prefix is byte-identical to splits/bench_v1_98.csv. BYTES are
      compared, not parsed rows, because a reparse-and-rewrite round trip
      can normalize quoting or line endings and still look "equal".
  P2  the data row count is exactly 150.
  P3  the header is exactly the 30 v1 columns, in order.
  P4  UTF-8 without BOM, LF line endings, file ends with a newline.

Row level (ids are the protocol's):

  V1   schema and typing; ann_ok = 1
  V2   GT navigable: ann_pos snaps to the navmesh within 0.2 m (horizontal
       predicates, between, near; skipped for above/below)          [sim]
  V3   reachability: a navigable point within 1.0 m horizontal of the
       scoring GT, so SR@1.0m is achievable at all                  [sim]
  V4   d0 round trip: parse_metric_literal(question).value_m == distance_m
  V5   anchor: sid present in the scene graph, category consistent with
       anchor_label, label lowercase with no trailing whitespace [sim: sid]
  V6   ann_yaw_rad in (-pi, pi]
  V7   radial consistency: |dist(ann_pos, anchor) - distance_m| <= 1.5 m
  V8   horizontal predicates: elevation of the anchor-to-point ray <= 60 deg
  V9   above/below: elevation >= 45 deg
  V10  predicate column agrees with infer_relation(question)
  V11  uniqueness: no duplicate (scene, msp_question); no reuse of a v1
       (scene, anchor_sid, predicate) triple
  V12  scene_floor is one of the pairs in the frozen v2 pose file
  V13  EPS guard: dist(ann_pos, anchor_center) > 1e-9 on nonzero-d0 rows
  D5   zero-distance rows carry ann_pos, so they are scoreable at all
  GT   src.evals.success.gt_xyz_from_row resolves the row to a point
  S1   style: ascii, no em dashes, no trailing whitespace, "Where is" form
  R1   predicate and distance inside the protocol's stated ranges

The GT check imports the REAL scoring function. This validator deliberately
does not carry a second opinion about how a row scores: if
``gt_xyz_from_row`` cannot resolve a row, the row is unscoreable, full stop.
The D5 check then verifies that the function's answer for a zero-distance
row is the annotated point itself, which is what makes a between row
meaningfully scored rather than scored against the anchor's center.

Rows inside the frozen v1 prefix are always evaluated in report mode, even
when the run is strict: they are byte-frozen and their known flaws (5
predicate mismatches, the row-53 distance mismatch, label whitespace) are on
the record by design. Their findings carry severity "frozen".

Sim-backed checks need habitat_sim, the scenes, and the navmesh, so they run
only in the container:

    docker exec -w /workspace mapg_dev python3 -m src.scripts.validate_bench_rows \\
        --csv splits/bench_v2_150.csv --mode strict --out validator_report.json

On the host, pass --no-sim. A strict run with --no-sim still exits nonzero,
because a strict pass that skipped V2, V3, and the V5 graph lookup is not
the freeze gate it looks like.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import traceback
from typing import Any, Dict, List, Optional, Sequence

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.evals.success import EPS as SUCCESS_EPS  # noqa: E402
from src.evals.success import gt_xyz_from_row  # noqa: E402
from src.parsing.metric_literal import infer_relation, parse_metric_literal  # noqa: E402
from src.scripts.bench_v2_common import (  # noqa: E402
    ANN_TOOL_CONSTANTS,
    EPS,
    HORIZONTAL_PREDICATES,
    MARKER_EXTENT_RANGE_M,
    NEW_ROW_D0_GRID,
    NEW_ROW_PREDICATES,
    NUMERIC_COLUMNS,
    PLAUSIBLE_RANGE_M,
    POSE_CSV_V2,
    PREDICATE_TO_RELATION,
    TARGET_ROWS,
    V1_COLUMNS,
    V1_CSV_REL,
    V1_ROWS,
    VERTICAL_PREDICATES,
    dist3,
    dist_horizontal,
    elevation_deg,
    parse_csv_text,
    prefix_report,
    read_bytes,
    repo_path,
    scene_floor_key,
    text_style_problems,
)

# Thresholds, all from protocol sections 5.2, 6.1 and 6.2.
SNAP_TOL_M = 0.2            # V2
REACH_TOL_M = 1.0           # V3, the primary endpoint tau
RADIAL_WARN_M = 0.5         # V7 warn
RADIAL_FAIL_M = 1.5         # V7 fail
TILT_WARN_DEG = 30.0        # V8 warn
TILT_FAIL_DEG = 60.0        # V8 fail
VERTICAL_WARN_DEG = 60.0    # V9 warn
VERTICAL_FAIL_DEG = 45.0    # V9 fail

# Severities, most severe first.
FAIL = "fail"
NOT_RUN = "not_run"
WARN = "warn"
FROZEN = "frozen"
NOTE = "note"
SKIP = "skip"

# Hard checks whose absence makes a strict run incomplete.
SIM_CHECKS = ("V2", "V3", "V5_graph")


# ---------------------------------------------------------------------------
# Findings
# ---------------------------------------------------------------------------

def finding(check: str, severity: str, message: str,
            row: Optional[int] = None, scene: str = "",
            detail: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "check": check,
        "severity": severity,
        "row": row,
        "scene": scene,
        "message": message,
    }
    if detail:
        out["detail"] = detail
    return out


def describe(f: Dict[str, Any]) -> str:
    where = "file" if f.get("row") is None else f"row {f['row']}"
    scene = f" {f['scene']}" if f.get("scene") else ""
    return f"[{f['severity'].upper():7}] {f['check']:<8} {where}{scene}: {f['message']}"


# ---------------------------------------------------------------------------
# Small typed accessors (CSV rows are strings)
# ---------------------------------------------------------------------------

def _num(row: Dict[str, str], key: str) -> Optional[float]:
    raw = (row.get(key) or "").strip()
    if raw == "":
        return None
    try:
        v = float(raw)
    except ValueError:
        return None
    if math.isnan(v) or math.isinf(v):
        return None
    return v


def _vec(row: Dict[str, str], prefix: str) -> Optional[List[float]]:
    parts = [_num(row, prefix + "_" + a) for a in ("x", "y", "z")]
    if any(p is None for p in parts):
        return None
    return [float(p) for p in parts]  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# File-level checks (pure)
# ---------------------------------------------------------------------------

def check_file_bytes(candidate: bytes, frozen: bytes,
                     expect_rows: int = TARGET_ROWS) -> List[Dict[str, Any]]:
    """P1 to P4. Pure: bytes in, findings out."""
    out: List[Dict[str, Any]] = []

    if candidate.startswith(b"\xef\xbb\xbf"):
        out.append(finding("P4", FAIL, "file starts with a UTF-8 BOM; the "
                                       "frozen prefix has none"))
    if b"\r\n" in candidate:
        idx = candidate.find(b"\r\n")
        out.append(finding("P4", FAIL,
                           f"CRLF line ending at byte {idx}; the split is LF only"))
    try:
        candidate.decode("utf-8")
    except UnicodeDecodeError as e:
        out.append(finding("P4", FAIL, f"file is not valid UTF-8: {e}"))
    if candidate and not candidate.endswith(b"\n"):
        out.append(finding("P4", FAIL, "file does not end with a newline, so "
                                       "appending a row would glue it to the last one"))

    rep = prefix_report(candidate, frozen)
    if rep["ok"]:
        out.append(finding("P1", NOTE,
                           f"v1 prefix byte-identical over {rep['prefix_bytes']} "
                           f"bytes, sha256 {rep['prefix_sha256']}"))
    else:
        out.append(finding("P1", FAIL,
                           "v1 prefix is NOT byte-identical: " + str(rep["reason"]),
                           row=rep.get("first_diff_data_row") or None,
                           detail=rep))

    n_rows = candidate.count(b"\n")
    if candidate and not candidate.endswith(b"\n"):
        n_rows += 1
    n_data = max(0, n_rows - 1)
    if n_data != expect_rows:
        out.append(finding("P2", FAIL,
                           f"file has {n_data} data rows, expected exactly {expect_rows}"))
    else:
        out.append(finding("P2", NOTE, f"{n_data} data rows"))
    return out


def check_header(header: Sequence[str]) -> List[Dict[str, Any]]:
    """P3: exactly the 30 v1 columns, in order."""
    if list(header) == list(V1_COLUMNS):
        return [finding("P3", NOTE, f"header matches the frozen {len(V1_COLUMNS)} columns")]
    out = []
    missing = [c for c in V1_COLUMNS if c not in header]
    extra = [c for c in header if c not in V1_COLUMNS]
    if missing:
        out.append(finding("P3", FAIL, f"header is missing columns: {missing}"))
    if extra:
        out.append(finding("P3", FAIL, f"header has columns outside the schema: {extra}"))
    if not missing and not extra:
        out.append(finding("P3", FAIL,
                           "header has the right columns in the wrong order: "
                           f"got {list(header)}"))
    return out


# ---------------------------------------------------------------------------
# Row-level checks (pure)
# ---------------------------------------------------------------------------

def check_row(idx: int, row: Dict[str, str]) -> List[Dict[str, Any]]:
    """All host-runnable checks for one row. ``idx`` is 1-based."""
    out: List[Dict[str, Any]] = []
    scene = (row.get("scene") or "").strip()
    question = row.get("msp_question") or ""
    predicate = (row.get("predicate") or "").strip()

    def add(check: str, severity: str, message: str, detail=None):
        out.append(finding(check, severity, message, row=idx, scene=scene, detail=detail))

    # --- V1 schema and typing -------------------------------------------
    for col in V1_COLUMNS:
        if col not in row:
            add("V1", FAIL, f"column '{col}' is absent from the row")
    for col in NUMERIC_COLUMNS:
        if col in row and _num(row, col) is None:
            add("V1", FAIL,
                f"column '{col}' does not parse as a finite number: "
                f"{row.get(col)!r}")
    ann_ok = _num(row, "ann_ok")
    if ann_ok is not None and ann_ok != 1:
        add("V1", FAIL, f"ann_ok is {row.get('ann_ok')!r}, must be 1")
    for col, want in ANN_TOOL_CONSTANTS.items():
        got = (row.get(col) or "").strip()
        if got != want:
            add("V1", FAIL,
                f"annotation-tool constant '{col}' is {got!r}, all v1 rows "
                f"carry {want!r}")
    if not scene:
        add("V1", FAIL, "scene is empty")

    ann = _vec(row, "ann_pos")
    anchor = _vec(row, "anchor_center")
    d0 = _num(row, "distance_m")

    # Marker box sanity: it must contain ann_pos and be a ~0.3 m cube.
    lo = _vec(row, "ann_aabb_min")
    hi = _vec(row, "ann_aabb_max")
    if lo is not None and hi is not None:
        for i, axis in enumerate("xyz"):
            extent = hi[i] - lo[i]
            if not (MARKER_EXTENT_RANGE_M[0] <= extent <= MARKER_EXTENT_RANGE_M[1]):
                add("V1", FAIL,
                    f"marker box extent on {axis} is {extent:.6f} m, outside "
                    f"the v1 range {MARKER_EXTENT_RANGE_M}")
            if ann is not None and not (lo[i] - 1e-6 <= ann[i] <= hi[i] + 1e-6):
                add("V1", FAIL,
                    f"marker box does not contain ann_pos on {axis}: "
                    f"[{lo[i]}, {hi[i]}] excludes {ann[i]}")

    # --- GT resolvable, through the REAL scoring function ----------------
    gt = gt_xyz_from_row(row)
    if gt is None:
        add("GT", FAIL,
            "src.evals.success.gt_xyz_from_row cannot resolve this row to a "
            "GT point, so the episode can never be scored; check "
            "distance_m, ann_pos_*, anchor_center_*")
    else:
        add("GT", NOTE, "scoring GT resolves to "
                        f"[{gt[0]:.6f}, {gt[1]:.6f}, {gt[2]:.6f}]",
            detail={"gt_xyz": gt})

    # --- D5 zero-distance rule (protocol 6.4) ----------------------------
    if d0 is not None and abs(d0) <= SUCCESS_EPS:
        if ann is None:
            add("D5", FAIL,
                "zero-distance row has no ann_pos, so gt_xyz_from_row falls "
                "through and the row is unscoreable; per D5 a zero-distance "
                "row must carry ann_pos_x/y/z")
        elif gt is None or max(abs(gt[i] - ann[i]) for i in range(3)) > 1e-9:
            add("D5", FAIL,
                "zero-distance row does not score against its annotated "
                f"point: gt_xyz_from_row returned {gt}, ann_pos is {ann}; "
                "the D5 rule in src/evals/success.py is not taking effect")
        else:
            add("D5", NOTE, "zero-distance row scores against ann_pos, per D5")

    # --- V13 EPS guard ---------------------------------------------------
    if d0 is not None and abs(d0) > SUCCESS_EPS and ann is not None and anchor is not None:
        sep = dist3(ann, anchor)
        if sep <= EPS:
            add("V13", FAIL,
                f"ann_pos coincides with anchor_center (separation {sep:.3e} m "
                "<= 1e-9), so the offset direction is undefined and the "
                "scoring GT degenerates to the anchor center")

    # --- V4 d0 round trip -------------------------------------------------
    parsed = parse_metric_literal(question)
    if d0 is None:
        pass  # already reported by V1
    elif abs(d0) <= SUCCESS_EPS:
        if parsed.value_m is not None:
            add("V4", FAIL,
                f"distance_m is 0 but the question carries a metric literal "
                f"({parsed.raw!r} = {parsed.value_m} m); a zero-distance row "
                "must have no literal to round trip against")
        else:
            add("V4", NOTE, "no metric literal, distance_m 0, consistent")
    elif parsed.value_m is None:
        add("V4", FAIL,
            f"distance_m is {d0} but parse_metric_literal found no literal in "
            f"the question: {question!r}")
    elif abs(parsed.value_m - d0) > 1e-9:
        add("V4", FAIL,
            f"d0 round trip fails: question says {parsed.raw!r} "
            f"({parsed.value_m} m) but distance_m is {d0}")
    else:
        add("V4", NOTE, f"d0 round trip ok via {parsed.raw!r}")
    for w in parsed.warnings:
        add("V4", WARN, f"parser warning: {w}")

    # --- V5 anchor label (the sid lookup needs the scene graph) ----------
    label = row.get("anchor_label") or ""
    if not label.strip():
        add("V5", FAIL, "anchor_label is empty")
    else:
        if label != label.strip():
            add("V5", FAIL, f"anchor_label has surrounding whitespace: {label!r}")
        if label.strip() != label.strip().lower():
            add("V5", FAIL, f"anchor_label is not lowercase: {label!r}")
        if label.strip().lower() not in question.lower():
            add("V5", WARN,
                f"anchor_label {label.strip()!r} does not appear verbatim in "
                "the question; check that the anchor phrase really names it")
    sid = (row.get("anchor_sid") or "").strip()
    if sid and not sid.lstrip("-").isdigit():
        add("V5", FAIL, f"anchor_sid is not an integer: {sid!r}")

    # --- V6 yaw range -----------------------------------------------------
    yaw = _num(row, "ann_yaw_rad")
    if yaw is not None and not (-math.pi < yaw <= math.pi):
        add("V6", FAIL, f"ann_yaw_rad {yaw} is outside (-pi, pi]")

    # --- V7 radial consistency -------------------------------------------
    if d0 is not None and abs(d0) > SUCCESS_EPS and ann is not None and anchor is not None:
        radial = abs(dist3(ann, anchor) - d0)
        if radial > RADIAL_FAIL_M:
            add("V7", FAIL,
                f"annotated point sits {dist3(ann, anchor):.3f} m from the "
                f"anchor but the command says {d0} m (off by {radial:.3f} m, "
                f"limit {RADIAL_FAIL_M} m)")
        elif radial > RADIAL_WARN_M:
            add("V7", WARN,
                f"annotated point is {radial:.3f} m off the commanded "
                f"distance (warn above {RADIAL_WARN_M} m)")

    # --- V8 / V9 ray direction -------------------------------------------
    if ann is not None and anchor is not None and predicate:
        elev = elevation_deg(anchor, ann)
        if elev is None:
            pass  # V13 already covers the degenerate case
        elif predicate in HORIZONTAL_PREDICATES:
            if elev > TILT_FAIL_DEG:
                add("V8", FAIL,
                    f"horizontal predicate {predicate!r} but the anchor-to-point "
                    f"ray is tilted {elev:.1f} deg above horizontal (limit "
                    f"{TILT_FAIL_DEG} deg); the projected GT loses most of its "
                    "horizontal offset")
            elif elev > TILT_WARN_DEG:
                add("V8", WARN,
                    f"anchor-to-point ray tilted {elev:.1f} deg for a horizontal "
                    f"predicate (warn above {TILT_WARN_DEG} deg)")
        elif predicate in VERTICAL_PREDICATES:
            if elev < VERTICAL_FAIL_DEG:
                add("V9", FAIL,
                    f"vertical predicate {predicate!r} but the anchor-to-point "
                    f"ray is only {elev:.1f} deg above horizontal (needs "
                    f"{VERTICAL_FAIL_DEG} deg)")
            elif elev < VERTICAL_WARN_DEG:
                add("V9", WARN,
                    f"anchor-to-point ray is {elev:.1f} deg for a vertical "
                    f"predicate (warn below {VERTICAL_WARN_DEG} deg)")

    # --- V10 predicate agrees with the question text ---------------------
    if predicate not in PREDICATE_TO_RELATION:
        add("V10", FAIL,
            f"predicate {predicate!r} is not in the frozen vocabulary "
            f"{sorted(PREDICATE_TO_RELATION)}")
    else:
        want = PREDICATE_TO_RELATION[predicate]
        got = infer_relation(question)
        if want is None:
            add("V10", NOTE,
                f"predicate {predicate!r} is a v1 legacy singleton with no "
                "entry in the frozen phrase table; V10 not applicable")
        elif got != want:
            add("V10", FAIL,
                f"predicate column says {predicate!r} (relation {want}) but "
                f"infer_relation reads {got!r} from the question: {question!r}")

    # --- R1 protocol ranges ----------------------------------------------
    if predicate and predicate not in NEW_ROW_PREDICATES:
        add("R1", FAIL,
            f"predicate {predicate!r} is not one of the 8 a new row may use: "
            f"{list(NEW_ROW_PREDICATES)}")
    if d0 is not None:
        if not any(abs(d0 - g) <= 1e-9 for g in NEW_ROW_D0_GRID):
            add("R1", FAIL,
                f"distance_m {d0} is not on the protocol grid "
                f"{list(NEW_ROW_D0_GRID)}")
        if abs(d0) > SUCCESS_EPS and not (
                PLAUSIBLE_RANGE_M[0] <= d0 <= PLAUSIBLE_RANGE_M[1]):
            add("R1", FAIL,
                f"distance_m {d0} is outside the parser's plausible range "
                f"{PLAUSIBLE_RANGE_M}")

    # --- S1 style ---------------------------------------------------------
    for col in V1_COLUMNS:
        value = row.get(col)
        if value is None:
            continue
        for problem in text_style_problems(value):
            add("S1", FAIL, f"column '{col}' has {problem}: {value!r}")
    q = question.strip()
    if q:
        if not q.lower().startswith("where is"):
            add("S1", FAIL,
                f"question is not in the 'Where is ...?' form used by all 98 "
                f"v1 rows: {question!r}")
        if not q.endswith("?"):
            add("S1", FAIL, f"question does not end with a question mark: {question!r}")
    else:
        add("S1", FAIL, "msp_question is empty")

    return out


def check_uniqueness(rows: Sequence[Dict[str, str]], new_from: int) -> List[Dict[str, Any]]:
    """V11: duplicate queries anywhere, and v1 triples reused by new rows."""
    out: List[Dict[str, Any]] = []
    seen: Dict[Any, int] = {}
    for i, row in enumerate(rows, start=1):
        key = ((row.get("scene") or "").strip(),
               (row.get("msp_question") or "").strip())
        if key in seen:
            out.append(finding(
                "V11", FAIL,
                f"duplicate query: same (scene, msp_question) as row {seen[key]}; "
                f"question {key[1]!r}",
                row=i, scene=key[0]))
        else:
            seen[key] = i

    v1_triples: Dict[Any, int] = {}
    for i, row in enumerate(rows[:new_from - 1], start=1):
        triple = ((row.get("scene") or "").strip(),
                  (row.get("anchor_sid") or "").strip(),
                  (row.get("predicate") or "").strip())
        v1_triples.setdefault(triple, i)
    for i, row in enumerate(rows[new_from - 1:], start=new_from):
        triple = ((row.get("scene") or "").strip(),
                  (row.get("anchor_sid") or "").strip(),
                  (row.get("predicate") or "").strip())
        if triple in v1_triples:
            out.append(finding(
                "V11", FAIL,
                "reuses a v1 (scene, anchor_sid, predicate) triple already "
                f"used by row {v1_triples[triple]}: {triple}",
                row=i, scene=triple[0]))
    return out


def check_pose_pairs(rows: Sequence[Dict[str, str]],
                     pose_pairs: Optional[Sequence[str]]) -> List[Dict[str, Any]]:
    """V12: every row's scene_floor exists in the frozen pose set."""
    if pose_pairs is None:
        return [finding("V12", NOT_RUN,
                        "no pose file available, so scene_floor coverage was "
                        "not checked; pass --pose-csv with the frozen v2 pose "
                        "file (MAPG-05)")]
    known = set(pose_pairs)
    out: List[Dict[str, Any]] = []
    for i, row in enumerate(rows, start=1):
        scene = (row.get("scene") or "").strip()
        floor = (row.get("floor") or "").strip()
        try:
            key = scene_floor_key(scene, floor)
        except (TypeError, ValueError):
            out.append(finding("V12", FAIL,
                               f"floor {floor!r} is not a number, so the "
                               "scene_floor key cannot be formed",
                               row=i, scene=scene))
            continue
        if key not in known:
            out.append(finding("V12", FAIL,
                               f"scene_floor {key!r} has no start pose in the "
                               "frozen pose file; new rows must sit on an "
                               "existing pair (protocol section 3)",
                               row=i, scene=scene))
    if not out:
        out.append(finding("V12", NOTE,
                           f"all {len(rows)} rows sit on one of "
                           f"{len(known)} frozen scene_floor pairs"))
    return out


def load_pose_pairs(path: str) -> List[str]:
    import csv as _csv
    with open(path, newline="", encoding="utf-8") as f:
        return [r["scene_floor"] for r in _csv.DictReader(f) if r.get("scene_floor")]


# ---------------------------------------------------------------------------
# Sim-backed checks (container only)
# ---------------------------------------------------------------------------

def have_habitat() -> bool:
    try:
        import habitat_sim  # noqa: F401
    except Exception:
        return False
    return True


def sim_checks(rows: Sequence[Dict[str, str]],
               row_indices: Sequence[int]) -> List[Dict[str, Any]]:
    """V2, V3 and the V5 scene-graph half, for the given 1-based rows.

    Scenes are loaded once each and processed in a stable order. Any scene
    that fails to load produces a NOT_RUN finding per affected row rather
    than a silent pass.
    """
    import habitat_sim  # noqa: F401  (import here so the host path stays clean)

    from src.scripts.collect_init_poses import (  # reuse the frozen navmesh setup
        _make_pathfinder,
        _make_sim,
    )

    out: List[Dict[str, Any]] = []
    by_scene: Dict[str, List[int]] = {}
    for i in row_indices:
        by_scene.setdefault((rows[i - 1].get("scene") or "").strip(), []).append(i)

    for scene in sorted(by_scene):
        sim = None
        try:
            sim = _make_sim(scene)
            pf = _make_pathfinder(sim, 0)
            objects = _semantic_objects(sim)
        except Exception as e:
            traceback.print_exc()
            for i in by_scene[scene]:
                out.append(finding("V2", NOT_RUN,
                                   f"scene failed to load: {type(e).__name__}: {e}",
                                   row=i, scene=scene))
                out.append(finding("V3", NOT_RUN, "scene failed to load",
                                   row=i, scene=scene))
                out.append(finding("V5_graph", NOT_RUN, "scene failed to load",
                                   row=i, scene=scene))
            if sim is not None:
                sim.close(destroy=True)
            continue

        try:
            for i in by_scene[scene]:
                out.extend(_sim_check_row(i, rows[i - 1], pf, objects, scene))
        finally:
            sim.close(destroy=True)
    return out


def _semantic_objects(sim) -> Dict[str, Dict[str, Any]]:
    """{sid: {'category': str, 'center': [x,y,z]}} for the scene's objects.

    HM3D object ids look like '0_1_12'; the bench's anchor_sid column holds
    the trailing integer. Both forms are indexed so a future annotation tool
    that stores the full id still resolves.
    """
    out: Dict[str, Dict[str, Any]] = {}
    ss = getattr(sim, "semantic_scene", None)
    if ss is None:
        return out
    for obj in getattr(ss, "objects", []) or []:
        if obj is None:
            continue
        try:
            cat = obj.category
            name = cat.name() if callable(cat.name) else cat.name
            name = str(name).strip().lower()
        except Exception:
            name = ""
        try:
            aabb = obj.aabb
            c = aabb.center() if callable(aabb.center) else aabb.center
            center = [float(c[0]), float(c[1]), float(c[2])]
        except Exception:
            center = [float("nan")] * 3
        raw_id = str(getattr(obj, "id", ""))
        entry = {"category": name, "center": center, "id": raw_id}
        out[raw_id] = entry
        tail = raw_id.rsplit("_", 1)[-1]
        out.setdefault(tail, entry)
        sem = getattr(obj, "semantic_id", None)
        if sem is not None:
            out.setdefault(str(int(sem)), entry)
    return out


def _sim_check_row(idx: int, row: Dict[str, str], pf, objects, scene: str):
    out: List[Dict[str, Any]] = []

    def add(check, severity, message, detail=None):
        out.append(finding(check, severity, message, row=idx, scene=scene, detail=detail))

    ann = _vec(row, "ann_pos")
    predicate = (row.get("predicate") or "").strip()

    # V2: the annotated point must sit on (or within 0.2 m of) the navmesh,
    # except for above/below whose targets are free 3D points.
    if predicate in VERTICAL_PREDICATES:
        add("V2", SKIP, f"skipped for vertical predicate {predicate!r}")
    elif ann is None:
        add("V2", FAIL, "ann_pos is missing, cannot test navigability")
    else:
        snap = pf.snap_point(ann)
        d = dist3(snap, ann)
        if not math.isfinite(d):
            add("V2", FAIL, "ann_pos does not snap to the navmesh at all")
        elif d > SNAP_TOL_M:
            add("V2", FAIL,
                f"ann_pos is {d:.3f} m from the navmesh, over the {SNAP_TOL_M} m "
                "snap tolerance; the annotated point is not standable")

    # V3: some navigable point within 1.0 m horizontal of the scoring GT.
    gt = gt_xyz_from_row(row)
    if gt is None:
        add("V3", FAIL, "no scoring GT, so reachability cannot be established")
    else:
        snap = pf.snap_point(gt)
        d = dist_horizontal(snap, gt)
        if not math.isfinite(d):
            add("V3", FAIL, "the scoring GT has no navigable point anywhere near it")
        elif d > REACH_TOL_M:
            add("V3", FAIL,
                f"nearest navigable point is {d:.3f} m horizontally from the "
                f"scoring GT, over tau = {REACH_TOL_M} m; no embodied agent can "
                "succeed on this row")
        else:
            add("V3", NOTE, f"navigable point {d:.3f} m from the scoring GT")

    # V5 graph half: the sid must exist and its category must match the label.
    sid = (row.get("anchor_sid") or "").strip()
    label = (row.get("anchor_label") or "").strip().lower()
    node = objects.get(sid)
    if node is None:
        add("V5_graph", FAIL,
            f"anchor_sid {sid!r} is not an object id in this scene's graph")
    else:
        cat = node["category"]
        if label and cat and label not in cat and cat not in label:
            add("V5_graph", FAIL,
                f"anchor_sid {sid} is category {cat!r} but anchor_label says "
                f"{label!r}")
        else:
            add("V5_graph", NOTE, f"anchor_sid {sid} resolves to {cat!r}")
        center = _vec(row, "anchor_center")
        if center is not None and all(math.isfinite(c) for c in node["center"]):
            off = dist3(center, node["center"])
            if off > 0.5:
                add("V5_graph", WARN,
                    f"anchor_center is {off:.2f} m from the graph node's aabb "
                    "center; check the row was not hand-typed")

    # Distractor count for the sidecar (protocol section 8), reported not gated.
    if node is not None and node["category"]:
        n = sum(1 for k, v in objects.items()
                if k == v["id"] and v["category"] == node["category"])
        add("META", NOTE, f"distractor_count {n} for category {node['category']!r}",
            detail={"distractor_count": n, "category": node["category"]})
    return out


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def run(csv_path: str, mode: str, v1_path: str, pose_csv: Optional[str],
        use_sim: bool, expect_rows: int, new_from: Optional[int]) -> Dict[str, Any]:
    candidate = read_bytes(csv_path)
    frozen = read_bytes(v1_path)

    findings: List[Dict[str, Any]] = list(
        check_file_bytes(candidate, frozen, expect_rows=expect_rows))

    try:
        text = candidate.decode("utf-8")
    except UnicodeDecodeError:
        text = candidate.decode("utf-8", errors="replace")
    header, rows = parse_csv_text(text)
    findings.extend(check_header(header))

    if new_from is None:
        # The frozen prefix is the first V1_ROWS rows when the prefix check
        # passed; otherwise nothing is treated as frozen and every row is
        # hard-checked, which is the strict reading.
        prefix_ok = any(f["check"] == "P1" and f["severity"] == NOTE for f in findings)
        new_from = V1_ROWS + 1 if prefix_ok else 1

    for i, row in enumerate(rows, start=1):
        findings.extend(check_row(i, row))
    findings.extend(check_uniqueness(rows, new_from))

    pose_pairs = None
    if pose_csv:
        resolved = pose_csv
        if not os.path.isabs(resolved):
            resolved = repo_path(resolved)
        if os.path.exists(resolved):
            pose_pairs = load_pose_pairs(resolved)
        else:
            findings.append(finding(
                "V12", NOT_RUN,
                f"pose file {resolved} does not exist. It is the MAPG-05 "
                "deliverable; V12 keys on it and there is deliberately no "
                "fallback to the 49-row scene_init_poses_semantic_only_new.csv, "
                "which covers only part of the 64 pairs"))
    findings.extend(check_pose_pairs(rows, pose_pairs))

    sim_ran = False
    if use_sim:
        if not have_habitat():
            for check in SIM_CHECKS:
                findings.append(finding(
                    check, NOT_RUN,
                    "habitat_sim is not importable here; run this validator "
                    "inside the mapg_dev container, or pass --no-sim for a "
                    "host-side report"))
        else:
            findings.extend(sim_checks(rows, list(range(1, len(rows) + 1))))
            sim_ran = True
    else:
        for check in SIM_CHECKS:
            findings.append(finding(
                check, NOT_RUN,
                "--no-sim was passed, so this hard check did not run"))

    # Rows inside the frozen prefix are report-only (protocol 6.3).
    for f in findings:
        if f.get("row") and f["row"] < new_from and f["severity"] in (FAIL, WARN):
            f["frozen_v1"] = True
            f["original_severity"] = f["severity"]
            f["severity"] = FROZEN

    gt_by_row = {}
    for f in findings:
        if f["check"] == "GT" and f.get("detail", {}).get("gt_xyz"):
            gt_by_row[f["row"]] = f["detail"]["gt_xyz"]

    counts: Dict[str, int] = {}
    for f in findings:
        counts[f["severity"]] = counts.get(f["severity"], 0) + 1

    hard_fails = [f for f in findings if f["severity"] == FAIL]
    not_run = [f for f in findings if f["severity"] == NOT_RUN]

    return {
        "meta": {
            "csv": os.path.abspath(csv_path),
            "v1_csv": os.path.abspath(v1_path),
            "mode": mode,
            "expect_rows": expect_rows,
            "new_from_row": new_from,
            "rows_parsed": len(rows),
            "sim_checks_ran": sim_ran,
            "pose_csv": pose_csv or "",
            "success_eps": SUCCESS_EPS,
        },
        "counts": counts,
        "n_hard_fails": len(hard_fails),
        "n_not_run": len(not_run),
        "metric_corrected": {str(k): v for k, v in sorted(gt_by_row.items())},
        "findings": findings,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--csv", required=True, help="the split CSV to validate")
    ap.add_argument("--mode", choices=("strict", "report"), default="strict")
    ap.add_argument("--v1-csv", default=V1_CSV_REL,
                    help="the frozen prefix to compare bytes against")
    ap.add_argument("--pose-csv", default=POSE_CSV_V2,
                    help="frozen v2 init-pose CSV for V12")
    ap.add_argument("--no-sim", action="store_true",
                    help="skip the habitat-backed checks (V2, V3, V5 graph). "
                         "A strict run still exits nonzero: an incomplete "
                         "strict pass is not a freeze gate.")
    ap.add_argument("--expect-rows", type=int, default=TARGET_ROWS,
                    help="expected data-row count (150 for the real split; "
                         "lower values are for dry runs and are flagged)")
    ap.add_argument("--new-from", type=int, default=None,
                    help="1-based row index where the new rows start "
                         "(default: 99 when the v1 prefix verifies)")
    ap.add_argument("--out", default="",
                    help="write the JSON report here")
    ap.add_argument("--quiet", action="store_true",
                    help="print only failures and the summary")
    args = ap.parse_args(argv)

    csv_path = args.csv if os.path.isabs(args.csv) else repo_path(args.csv)
    v1_path = args.v1_csv if os.path.isabs(args.v1_csv) else repo_path(args.v1_csv)

    report = run(csv_path=csv_path, mode=args.mode, v1_path=v1_path,
                 pose_csv=args.pose_csv, use_sim=not args.no_sim,
                 expect_rows=args.expect_rows, new_from=args.new_from)

    if args.expect_rows != TARGET_ROWS:
        print(f"NOTE: --expect-rows {args.expect_rows} is not the frozen 150. "
              "This run is a dry run and is not a freeze gate.")

    show = {FAIL, NOT_RUN, WARN} if args.quiet else None
    for f in report["findings"]:
        if show is None or f["severity"] in show:
            print(describe(f))

    if args.out:
        out_path = args.out if os.path.isabs(args.out) else repo_path(args.out)
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
            f.write("\n")
        print(f"[validate] wrote {out_path}")

    counts = report["counts"]
    print(f"[validate] {report['meta']['rows_parsed']} rows, "
          f"new rows from {report['meta']['new_from_row']}, "
          f"{report['n_hard_fails']} hard fails, "
          f"{counts.get(WARN, 0)} warnings, "
          f"{counts.get(FROZEN, 0)} findings on frozen v1 rows, "
          f"{report['n_not_run']} checks not run.")

    if args.mode == "report":
        print("[validate] report mode: exiting 0 by design (protocol 6.3).")
        return 0

    if report["n_hard_fails"]:
        print(f"[validate] STRICT FAIL: {report['n_hard_fails']} hard failures.")
        return 1
    if report["n_not_run"]:
        print(f"[validate] STRICT INCOMPLETE: {report['n_not_run']} hard checks "
              "did not run. A strict pass that skipped checks is not a freeze "
              "gate; rerun in the container with the pose file present.")
        return 2
    print("[validate] STRICT PASS.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
