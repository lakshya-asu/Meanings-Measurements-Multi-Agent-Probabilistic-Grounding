#!/usr/bin/env python3
"""Authoring tooling for bench-v2-150 (MAPG-07, protocol sections 2 to 4 and 7).

The 52 new queries are authored by humans and fleet drafters; this script is
the scaffolding around that work. It never invents a question and it never
freezes a split.

Subcommands
-----------

  plan        Turn the protocol's distribution tables into 52 concrete slots:
              which scene, which floor, which predicate, which commanded
              distance, which phrasing quota, which cross-cutting overlays.
              Seeded, so re-running reproduces the same brief.

  template    Emit one draft CSV per receiving scene, with the exact 30-column
              v1 header and the slot fields pre-filled. Annotation columns are
              left blank for the in-container annotation flow.

  assemble    Concatenate the frozen v1 bytes and the draft rows into a
              candidate v2 file, then verify the byte-identical prefix before
              anything is written. Refuses the canonical frozen path unless
              --freeze is passed, because freezing is a human gate.

  spotcheck   The lakshya 10-row draw, random.Random(20260822), reproducible
              by janus (protocol 4.1).

  noise-floor Test-retest passes (D6, protocol section 7): row selection,
              blind worksheets, and recording. See the blindness section.

Drafts live OUTSIDE the repo (protocol 4.4), so --out-dir is required and has
no default: nothing in this repo may hardcode a path to the draft directory.

Blindness (D6)
--------------

The noise floor is test-retest: one annotator, the same 20 rows, twice, at
least a week apart, blind to his first pass. Blindness cannot be an honor
rule when it is the same human both times, so it is structural here:

  1. Redaction at the source. The worksheet is built from
     ``redact_for_blind(row)``, which drops ann_pos_*, ann_yaw_rad,
     ann_aabb_*, ann_ts, ann_ok and the marker constants. The annotator sees
     scene, floor, and the question. The primary GT never reaches the
     worksheet either, so pass 1 is blind to the bench's own annotation.
  2. No defaults. Response fields are emitted empty. There is no prefill,
     no "last value", no placeholder derived from anything already stored.
  3. Opaque, per-pass item ids. The worksheet is keyed by item_01..item_20,
     permuted with a pass-specific seed, so item_07 in pass 2 is a different
     row than item_07 in pass 1 and the ORDER carries no signal. The
     item-to-row mapping lives in a separate keymap file the annotator does
     not open; only ``record`` reads it.
  4. Value-blind gating. Opening pass 2 checks that pass 1 is complete and
     at least 7 days old. That check reads only row_idx, pass_id and ann_ts
     through ``gate_projection``; pass-1 coordinates are never loaded into
     the pass-2 code path at all.
  5. No diff view. This tool has no command that shows one pass against the
     other, or against the primary GT. Comparing the passes is the analysis
     step, done once both are banked, outside this tool.
  6. Idempotent, non-echoing recording. Re-recording an (row_idx, pass_id)
     that already exists is refused, and the refusal names the item, never
     the stored value.
  7. Blind renders. The worksheet points at a render directory produced by
     ``render_query_context.py --mode blind``, which redacts the row before
     any drawing code sees it, so no marker can be drawn.

Run in the container for consistency with the rest of the toolchain:

    docker exec -w /workspace mapg_dev python3 -m src.scripts.author_bench_v2 plan \\
        --out-dir /drafts
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.scripts.bench_v2_common import (  # noqa: E402
    ANN_TOOL_CONSTANTS,
    FROZEN_V2_REL,
    NEW_ROWS,
    TARGET_ROWS,
    V1_COLUMNS,
    V1_CSV_REL,
    V1_ROWS,
    VERTICAL_PREDICATES,
    assemble_v2_bytes,
    fmt_distance,
    parse_csv_text,
    prefix_report,
    read_bytes,
    repo_path,
    row_to_line,
    sha256_bytes,
)

# Seeds, all fixed by the protocol so janus can reproduce every draw.
PLAN_SEED = 20260818          # tooling seed for the slot brief
SPOTCHECK_SEED = 20260822     # protocol 4.1
NOISE_SELECT_SEED = 20260825  # protocol section 7
SPOTCHECK_N = 10
NOISE_FLOOR_N = 20
MIN_PASS_GAP_DAYS = 7         # protocol section 7: sittings a week apart

# Protocol 2.1: predicates for the 52 new rows.
PREDICATE_TARGETS: Dict[str, int] = {
    "in front of": 12,
    "right of": 5,
    "behind": 8,
    "left of": 5,
    "above": 8,
    "below": 8,
    "between": 3,
    "near": 3,
}

# Protocol 2.2: commanded distances for the 49 literal rows.
D0_TARGETS: Dict[float, int] = {
    0.5: 6, 0.75: 3, 1.0: 10, 1.5: 6, 2.0: 8, 2.5: 5, 3.0: 5, 4.0: 3, 5.0: 3,
}

# Protocol 2.2 phrasing quotas, as (d0, count, hint).
PHRASING_QUOTAS: Tuple[Tuple[float, int, str], ...] = (
    (0.5, 2, "centimeters, e.g. '50 centimeters'"),
    (0.75, 3, "digits, e.g. '0.75 meters' or '75 centimeters'"),
    (1.0, 2, "word number, e.g. 'one meter'"),
    (1.5, 2, "'a meter and a half'"),
    (2.0, 1, "word number, e.g. 'two meters'"),
    (2.5, 2, "'two and a half meters'"),
)

# Protocol 2.3 overlays.
OVERLAY_QUOTAS: Dict[str, int] = {
    "intrinsic_frame": 10,      # 6 in front of, 4 behind
    "occlusion": 10,
    "distractor": 16,           # a floor, not a cap
    "modifier": 10,             # a floor, not a cap
}
INTRINSIC_SPLIT: Dict[str, int] = {"in front of": 6, "behind": 4}

# Engineering constraint the protocol does not state: a 3, 4 or 5 m VERTICAL
# offset is not an indoor query. Vertical slots are capped so the brief never
# asks a drafter to author an impossible row. Flagged in the MAPG-07 report.
VERTICAL_MAX_D0_M = 2.5

# Protocol section 3: the receiving scenes are the 14 one-query and the 12
# two-query scenes, identified by their leading scene number.
EXPECTED_ONE_QUERY = (
    "00506", "00386", "00326", "00166", "00324", "00203", "00404", "00323",
    "00529", "00397", "00304", "00414", "00299", "00258",
)
EXPECTED_TWO_QUERY = (
    "00256", "00366", "00388", "00207", "00245", "00313", "00135", "00720",
    "00669", "00606", "00035", "00537",
)

# Protocol section 7 composition of the 20 test-retest rows.
NOISE_STRATA_V1: Tuple[Tuple[str, int], ...] = (
    ("in front of", 3), ("right of", 2), ("left of", 1), ("behind", 1),
    ("above", 1), ("below", 1), ("between", 1),
)
NOISE_STRATA_NEW: Tuple[Tuple[str, int], ...] = (
    ("in front of", 2), ("behind", 1), ("left of", 1), ("right of", 1),
    ("above", 2), ("below", 1), ("between", 1), ("near", 1),
)

# The noise-floor store (protocol section 7). Additive, does not touch the
# frozen split, so it may land after the Aug 22 freeze.
NOISE_STORE_REL = "splits/noise_floor_v2.csv"
NOISE_STORE_COLUMNS = (
    "row_idx", "pass_id", "ann_pos_x", "ann_pos_y", "ann_pos_z",
    "ann_yaw_rad", "ann_ts",
)

# Everything a blind worksheet may show. Any column not listed here is
# redacted, so adding a column to the split cannot silently widen the leak.
BLIND_VISIBLE_COLUMNS = ("scene", "floor", "msp_question")


# ---------------------------------------------------------------------------
# Pure logic: receiving scenes and the 52 slots
# ---------------------------------------------------------------------------

def scene_query_counts(v1_rows: Sequence[Dict[str, str]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for r in v1_rows:
        counts[r["scene"]] = counts.get(r["scene"], 0) + 1
    return counts


def scene_floors(v1_rows: Sequence[Dict[str, str]]) -> Dict[str, List[int]]:
    """Existing scene_floor pairs per scene. New rows may use only these."""
    out: Dict[str, set] = {}
    for r in v1_rows:
        out.setdefault(r["scene"], set()).add(int(float(r["floor"])))
    return {s: sorted(v) for s, v in out.items()}


def receiving_scenes(v1_rows: Sequence[Dict[str, str]]) -> List[str]:
    """The 26 scenes that receive +2, derived from the data and cross-checked.

    The protocol names them by leading scene number. Deriving the groups
    from the frozen rows and then comparing against those numbers catches a
    protocol-versus-data drift instead of silently trusting either one.
    """
    counts = scene_query_counts(v1_rows)
    ones = sorted(s for s, n in counts.items() if n == 1)
    twos = sorted(s for s, n in counts.items() if n == 2)
    got_one = tuple(sorted(s.split("-")[0] for s in ones))
    got_two = tuple(sorted(s.split("-")[0] for s in twos))
    if got_one != tuple(sorted(EXPECTED_ONE_QUERY)):
        raise ValueError(
            "one-query scenes in the frozen split do not match protocol "
            f"section 3: data has {got_one}, protocol has "
            f"{tuple(sorted(EXPECTED_ONE_QUERY))}")
    if got_two != tuple(sorted(EXPECTED_TWO_QUERY)):
        raise ValueError(
            "two-query scenes in the frozen split do not match protocol "
            f"section 3: data has {got_two}, protocol has "
            f"{tuple(sorted(EXPECTED_TWO_QUERY))}")
    return ones + twos


def _predicate_pool() -> List[str]:
    pool: List[str] = []
    for pred in sorted(PREDICATE_TARGETS):
        pool.extend([pred] * PREDICATE_TARGETS[pred])
    return pool


def _d0_pool() -> List[float]:
    pool: List[float] = []
    for d0 in sorted(D0_TARGETS):
        pool.extend([d0] * D0_TARGETS[d0])
    return pool


def pair_predicates_with_d0(predicates: Sequence[str], d0s: Sequence[float],
                            rng: random.Random) -> List[Tuple[str, float]]:
    """Pair non-between predicates with commanded distances.

    Vertical predicates are served first, from the distances at or below
    VERTICAL_MAX_D0_M, so no drafter is handed "5 meters above the shelf".
    Raises when the vertical demand cannot be met, rather than quietly
    handing out an impossible slot.
    """
    remaining = sorted(d0s)
    out: List[Tuple[str, float]] = []
    verticals = [p for p in predicates if p in VERTICAL_PREDICATES]
    others = [p for p in predicates if p not in VERTICAL_PREDICATES]

    eligible = [d for d in remaining if d <= VERTICAL_MAX_D0_M]
    if len(eligible) < len(verticals):
        raise ValueError(
            f"{len(verticals)} vertical slots need a distance at or below "
            f"{VERTICAL_MAX_D0_M} m but only {len(eligible)} are on offer")
    rng.shuffle(eligible)
    for pred in verticals:
        d0 = eligible.pop()
        remaining.remove(d0)
        out.append((pred, d0))

    rng.shuffle(remaining)
    rng.shuffle(others)
    for pred, d0 in zip(others, remaining):
        out.append((pred, d0))
    out.sort(key=lambda t: (t[0], t[1]))
    return out


def assign_slots_to_scenes(pairs: Sequence[Tuple[str, float]],
                           scenes: Sequence[str],
                           floors: Dict[str, List[int]],
                           rng: random.Random) -> List[Dict[str, Any]]:
    """Two slots per scene, preferring two different predicates per scene.

    Deterministic given ``rng``. When the pool leaves no alternative the
    duplicate is taken and flagged, so the plan says so out loud instead of
    silently producing two identical predicates in one scene.
    """
    pool = list(pairs)
    rng.shuffle(pool)
    slots: List[Dict[str, Any]] = []
    for scene in scenes:
        scene_floor_list = floors.get(scene) or [0]
        chosen: List[Tuple[str, float]] = []
        for j in range(2):
            pick = None
            for k, cand in enumerate(pool):
                if not chosen or cand[0] != chosen[0][0]:
                    pick = k
                    break
            duplicate = pick is None
            if pick is None:
                pick = 0
            cand = pool.pop(pick)
            chosen.append(cand)
            slots.append({
                "slot_id": "",
                "scene": scene,
                "floor": scene_floor_list[j % len(scene_floor_list)],
                "predicate": cand[0],
                "distance_m": cand[1],
                "duplicate_predicate_in_scene": duplicate,
            })
    if pool:
        raise ValueError(f"{len(pool)} slots left unassigned")
    slots.sort(key=lambda s: (s["scene"], s["floor"], s["predicate"], s["distance_m"]))
    for i, slot in enumerate(slots, start=1):
        slot["slot_id"] = f"slot_{i:02d}"
    return slots


def apply_phrasing_quotas(slots: List[Dict[str, Any]]) -> None:
    """Attach the section 2.2 phrasing hints to the first N slots per cell."""
    for slot in slots:
        slot["phrasing"] = "digits"
    for d0, count, hint in PHRASING_QUOTAS:
        cell = [s for s in slots if abs(s["distance_m"] - d0) < 1e-9]
        for s in cell[:count]:
            s["phrasing"] = hint


def apply_overlays(slots: List[Dict[str, Any]], rng: random.Random) -> None:
    """Attach the section 2.3 overlay assignments, deterministically."""
    for slot in slots:
        slot["overlays"] = []

    for pred, n in sorted(INTRINSIC_SPLIT.items()):
        cell = [s for s in slots if s["predicate"] == pred]
        for s in rng.sample(cell, min(n, len(cell))):
            s["overlays"].append("intrinsic_frame")

    horizontal = [s for s in slots if s["predicate"] not in VERTICAL_PREDICATES]
    for s in rng.sample(horizontal, min(OVERLAY_QUOTAS["occlusion"], len(horizontal))):
        s["overlays"].append("occlusion")
    for s in rng.sample(slots, min(OVERLAY_QUOTAS["distractor"], len(slots))):
        s["overlays"].append("distractor")
    for s in rng.sample(slots, min(OVERLAY_QUOTAS["modifier"], len(slots))):
        s["overlays"].append("modifier")
    for s in slots:
        s["overlays"] = sorted(set(s["overlays"]))
        s["needs_sightline_check"] = s["distance_m"] >= 5.0


def build_slots(v1_rows: Sequence[Dict[str, str]],
                seed: int = PLAN_SEED) -> List[Dict[str, Any]]:
    """The 52-slot authoring brief. Deterministic for a given seed."""
    scenes = receiving_scenes(v1_rows)
    if len(scenes) * 2 != NEW_ROWS:
        raise ValueError(
            f"{len(scenes)} receiving scenes give {len(scenes) * 2} slots, "
            f"but {NEW_ROWS} new rows are required")
    preds = _predicate_pool()
    if len(preds) != NEW_ROWS:
        raise ValueError(f"predicate targets sum to {len(preds)}, not {NEW_ROWS}")
    d0s = _d0_pool()
    n_between = PREDICATE_TARGETS["between"]
    if len(d0s) != NEW_ROWS - n_between:
        raise ValueError(
            f"d0 targets sum to {len(d0s)}, expected {NEW_ROWS - n_between} "
            "literal rows")

    rng = random.Random(seed)
    literal_preds = [p for p in preds if p != "between"]
    pairs = pair_predicates_with_d0(literal_preds, d0s, rng)
    pairs = pairs + [("between", 0.0)] * n_between

    slots = assign_slots_to_scenes(pairs, scenes, scene_floors(v1_rows), rng)
    apply_phrasing_quotas(slots)
    apply_overlays(slots, rng)
    return slots


def slot_summary(slots: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    preds: Dict[str, int] = {}
    d0s: Dict[str, int] = {}
    overlays: Dict[str, int] = {}
    for s in slots:
        preds[s["predicate"]] = preds.get(s["predicate"], 0) + 1
        key = fmt_distance(s["distance_m"])
        d0s[key] = d0s.get(key, 0) + 1
        for o in s.get("overlays", []):
            overlays[o] = overlays.get(o, 0) + 1
    return {"n_slots": len(slots), "predicates": preds, "distances": d0s,
            "overlays": overlays}


# ---------------------------------------------------------------------------
# Pure logic: draft rows
# ---------------------------------------------------------------------------

def blank_draft_row(slot: Dict[str, Any]) -> Dict[str, str]:
    """A draft row in the frozen 30-column schema, annotation fields blank.

    The annotation flow fills ann_ts, ann_pos_*, ann_yaw_rad and the marker
    box; the drafter fills msp_question, anchor_sid, anchor_label,
    anchor_center_* and the two GT object columns. Nothing here is invented.
    """
    row = {c: "" for c in V1_COLUMNS}
    row["scene"] = slot["scene"]
    row["floor"] = str(int(slot["floor"]))
    row["distance_m"] = fmt_distance(slot["distance_m"])
    row["predicate"] = slot["predicate"]
    row["ann_ok"] = "1"
    row.update(ANN_TOOL_CONSTANTS)
    return row


def draft_rows_from_csv(text: str) -> List[Dict[str, str]]:
    """Read a draft CSV, requiring the exact frozen header."""
    header, rows = parse_csv_text(text)
    if list(header) != list(V1_COLUMNS):
        raise ValueError(
            "draft CSV header does not match the frozen 30-column schema; "
            f"got {list(header)}")
    return rows


def spot_check_draw(row_ids: Sequence[Any], n: int = SPOTCHECK_N,
                    seed: int = SPOTCHECK_SEED) -> List[Any]:
    """The reproducible human spot-check draw (protocol 4.1)."""
    ids = list(row_ids)
    if n > len(ids):
        raise ValueError(f"cannot draw {n} rows from {len(ids)}")
    return random.Random(seed).sample(ids, n)


# ---------------------------------------------------------------------------
# Pure logic: D6 test-retest noise floor
# ---------------------------------------------------------------------------

def redact_for_blind(row: Dict[str, str]) -> Dict[str, str]:
    """Everything the annotator may see for a blind pass, and nothing else.

    Allowlist, not denylist: a column added to the split in future is
    redacted by default rather than leaking until someone remembers to add
    it to a denylist.
    """
    return {c: (row.get(c) or "") for c in BLIND_VISIBLE_COLUMNS}


def select_noise_floor_rows(rows: Sequence[Dict[str, str]], new_from: int,
                            seed: int = NOISE_SELECT_SEED) -> List[int]:
    """The 20 test-retest rows: 10 v1 and 10 new, stratified by predicate.

    Returns 1-based data-row indices, sorted. Selection uses only the
    predicate column and the row index, never any annotation value.
    """
    rng = random.Random(seed)
    picked: List[int] = []
    for strata, lo, hi in ((NOISE_STRATA_V1, 1, new_from - 1),
                           (NOISE_STRATA_NEW, new_from, len(rows))):
        for predicate, count in strata:
            eligible = [
                i for i in range(lo, hi + 1)
                if (rows[i - 1].get("predicate") or "").strip() == predicate
            ]
            if len(eligible) < count:
                raise ValueError(
                    f"stratum {predicate!r} in rows {lo}..{hi} has "
                    f"{len(eligible)} rows, needs {count}")
            picked.extend(rng.sample(eligible, count))
    if len(picked) != NOISE_FLOOR_N:
        raise ValueError(f"strata sum to {len(picked)} rows, expected {NOISE_FLOOR_N}")
    if len(set(picked)) != len(picked):
        raise ValueError("a row was selected twice across strata")
    return sorted(picked)


def pass_item_order(row_indices: Sequence[int], pass_id: int,
                    seed: int = NOISE_SELECT_SEED) -> List[int]:
    """Presentation order of the rows for one pass.

    The permutation is seeded per pass, so the position of a row in pass 2
    carries no information about where it sat in pass 1. Deterministic, so a
    resumed session shows the same order it started with.
    """
    if pass_id not in (1, 2):
        raise ValueError(f"pass_id must be 1 or 2, got {pass_id}")
    order = sorted(row_indices)
    rng = random.Random(seed * 100 + pass_id)
    rng.shuffle(order)
    if pass_id == 2:
        first = pass_item_order(row_indices, 1, seed)
        if order == first:
            # Astronomically unlikely, but an identical order would make
            # position a cue. Rotate rather than accept it.
            order = order[1:] + order[:1]
    return order


def gate_projection(records: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Strip stored records down to what pass-2 gating may see.

    Only row_idx, pass_id and ann_ts survive. Coordinates never enter the
    pass-2 code path, so no gating bug can turn into a blindness leak.
    """
    out = []
    for r in records:
        out.append({
            "row_idx": int(r["row_idx"]),
            "pass_id": int(r["pass_id"]),
            "ann_ts": float(r["ann_ts"]) if str(r.get("ann_ts", "")).strip() else 0.0,
        })
    return out


def pass_gate(gate_records: Sequence[Dict[str, Any]], pass_id: int,
              row_indices: Sequence[int], now: float,
              min_gap_days: int = MIN_PASS_GAP_DAYS) -> Tuple[bool, str]:
    """May pass ``pass_id`` be opened? Value-blind by construction.

    ``gate_records`` must already have been through gate_projection.
    """
    want = set(int(i) for i in row_indices)
    done1 = {r["row_idx"] for r in gate_records if r["pass_id"] == 1}
    if pass_id == 1:
        return True, "pass 1 is always open"
    missing = sorted(want - done1)
    if missing:
        return False, (
            f"pass 1 is incomplete: {len(missing)} of {len(want)} rows have no "
            f"pass-1 record (rows {missing}). Finish pass 1 first.")
    last = max((r["ann_ts"] for r in gate_records if r["pass_id"] == 1),
               default=0.0)
    gap_days = (now - last) / 86400.0
    if gap_days < min_gap_days:
        return False, (
            f"pass 1 finished {gap_days:.2f} days ago; the protocol requires "
            f"at least {min_gap_days} days between sittings. Earliest open: "
            f"{time.strftime('%Y-%m-%d %H:%M', time.gmtime(last + min_gap_days * 86400))} UTC.")
    return True, f"pass 1 complete and {gap_days:.1f} days old"


def merge_noise_records(existing: Sequence[Dict[str, Any]],
                        new: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Append-only merge of noise-floor records.

    A duplicate (row_idx, pass_id) is refused. The error names the row, never
    the stored coordinates: an "existing value is X, new value is Y" message
    would be a blindness leak dressed up as a helpful diagnostic.
    """
    out = [dict(r) for r in existing]
    seen = {(int(r["row_idx"]), int(r["pass_id"])) for r in out}
    for r in new:
        key = (int(r["row_idx"]), int(r["pass_id"]))
        if key in seen:
            raise ValueError(
                f"row {key[0]} already has a pass-{key[1]} record. Records are "
                "append-only; delete nothing and re-annotate nothing, because "
                "a second look at an already-answered row is no longer blind.")
        seen.add(key)
        out.append(dict(r))
    out.sort(key=lambda r: (int(r["pass_id"]), int(r["row_idx"])))
    return out


def worksheet_text(pass_id: int, items: Sequence[Tuple[str, Dict[str, str]]],
                   render_dir: str, done_items: Sequence[str] = ()) -> str:
    """The blind worksheet an annotator fills in.

    ``items`` is [(item_id, redacted_row)]. Every response field is empty.
    Raises if a redacted row somehow still carries an annotation column, so
    a future refactor cannot quietly widen what the worksheet prints.
    """
    for item_id, row in items:
        leaked = [k for k in row if k not in BLIND_VISIBLE_COLUMNS]
        if leaked:
            raise ValueError(
                f"{item_id} carries columns that must not reach a blind "
                f"worksheet: {leaked}")
    done = set(done_items)
    lines = [
        f"# Test-retest noise floor, pass {pass_id} (D6, protocol section 7)",
        "",
        "You are annotating blind. This sheet deliberately does not show the "
        "benchmark's own annotation, and it does not show any earlier pass of "
        "your own. Item numbers are shuffled for this pass, so their order "
        "means nothing.",
        "",
        f"Renders for each item are in {render_dir} (blind mode: no markers).",
        "",
        "For each item: open the scene in the container annotation tool, place "
        "the point the question asks for, and write the values into the "
        "response CSV next to this sheet. Leave nothing prefilled.",
        "",
        "| item | scene | floor | question | status |",
        "|---|---|---|---|---|",
    ]
    for item_id, row in items:
        status = "recorded" if item_id in done else "TODO"
        question = row.get("msp_question", "").replace("|", "\\|")
        lines.append(
            f"| {item_id} | {row.get('scene', '')} | {row.get('floor', '')} | "
            f"{question} | {status} |")
    lines += [
        "",
        "## Response CSV",
        "",
        "Fill `responses.csv` next to this sheet with one line per item:",
        "",
        "    item_id,ann_pos_x,ann_pos_y,ann_pos_z,ann_yaw_rad,ann_ts",
        "",
        "Then run, in the container:",
        "",
        f"    python3 -m src.scripts.author_bench_v2 noise-floor record --pass {pass_id} \\",
        "        --session-dir <this directory> --store splits/noise_floor_v2.csv",
        "",
        "Recording is append-only. A row that already has a record for this "
        "pass is refused, because a second look at an answered row is no "
        "longer blind.",
        "",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------

def _load_v1_rows(path: str) -> List[Dict[str, str]]:
    _header, rows = parse_csv_text(read_bytes(path).decode("utf-8"))
    return rows


def _read_csv_rows(path: str) -> List[Dict[str, str]]:
    if not os.path.exists(path):
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write_csv_rows(path: str, columns: Sequence[str],
                    rows: Sequence[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(columns), lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in columns})


def _require_out_dir(args) -> str:
    if not args.out_dir:
        raise SystemExit(
            "--out-dir is required. Protocol 4.4 keeps drafts outside the "
            "repo, so no default path is baked in here.")
    os.makedirs(args.out_dir, exist_ok=True)
    return args.out_dir


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_plan(args) -> int:
    out_dir = _require_out_dir(args)
    v1_rows = _load_v1_rows(repo_path(args.v1_csv))
    slots = build_slots(v1_rows, seed=args.seed)
    summary = slot_summary(slots)

    cols = ["slot_id", "scene", "floor", "predicate", "distance_m", "phrasing",
            "overlays", "needs_sightline_check", "duplicate_predicate_in_scene"]
    rows = []
    for s in slots:
        rows.append({
            "slot_id": s["slot_id"], "scene": s["scene"], "floor": s["floor"],
            "predicate": s["predicate"],
            "distance_m": fmt_distance(s["distance_m"]),
            "phrasing": s["phrasing"],
            "overlays": ";".join(s["overlays"]),
            "needs_sightline_check": "yes" if s["needs_sightline_check"] else "",
            "duplicate_predicate_in_scene": "yes" if s["duplicate_predicate_in_scene"] else "",
        })
    _write_csv_rows(os.path.join(out_dir, "slots.csv"), cols, rows)

    lines = [
        "# bench-v2-150 authoring brief (MAPG-07)",
        "",
        f"Seed {args.seed}. Re-running this command reproduces the same 52 "
        "slots exactly.",
        "",
        f"Slots: {summary['n_slots']}. Predicates: "
        + ", ".join(f"{k} {v}" for k, v in sorted(summary["predicates"].items())),
        "",
        "Distances: "
        + ", ".join(f"{k} m x{v}" for k, v in sorted(summary["distances"].items(),
                                                     key=lambda t: float(t[0]))),
        "",
        "Overlays: "
        + ", ".join(f"{k} {v}" for k, v in sorted(summary["overlays"].items())),
        "",
        "Each slot is one query to author. The scene and floor are fixed by "
        "the per-scene allocation; the predicate, distance and phrasing come "
        "from the distribution tables. If a slot cannot be authored in its "
        "scene, use the swap rule (protocol section 3) and log the reason.",
        "",
        "| slot | scene | floor | predicate | d0 (m) | phrasing | overlays | notes |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        notes = []
        if r["needs_sightline_check"]:
            notes.append("verify a sightline over 5 m before authoring")
        if r["duplicate_predicate_in_scene"]:
            notes.append("second slot repeats this scene's predicate")
        lines.append(
            f"| {r['slot_id']} | {r['scene']} | {r['floor']} | {r['predicate']} "
            f"| {r['distance_m']} | {r['phrasing']} | {r['overlays']} | "
            f"{'; '.join(notes)} |")
    lines += [
        "",
        "## Next steps",
        "",
        "1. `template --out-dir <dir>` writes one draft CSV per scene.",
        "2. Author the questions and annotate in the container.",
        "3. `assemble --drafts <dir> --out <candidate.csv>` then "
        "`validate_bench_rows --csv <candidate.csv> --mode strict`.",
        "",
    ]
    path = os.path.join(out_dir, "authoring-plan.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"[plan] wrote {path} and {os.path.join(out_dir, 'slots.csv')}")
    return 0


def cmd_template(args) -> int:
    out_dir = _require_out_dir(args)
    v1_rows = _load_v1_rows(repo_path(args.v1_csv))
    slots = build_slots(v1_rows, seed=args.seed)
    by_scene: Dict[str, List[Dict[str, Any]]] = {}
    for s in slots:
        by_scene.setdefault(s["scene"], []).append(s)

    wanted = set(args.scenes.split(",")) if args.scenes else None
    written = 0
    for scene, scene_slots in sorted(by_scene.items()):
        if wanted and scene not in wanted:
            continue
        path = os.path.join(out_dir, f"{scene}.csv")
        if os.path.exists(path) and not args.force:
            print(f"[template] {path} exists, skipping (use --force to overwrite)")
            continue
        rows = [blank_draft_row(s) for s in scene_slots]
        with open(path, "w", encoding="utf-8", newline="") as f:
            f.write(",".join(_quote_header(V1_COLUMNS)) + "\n")
            for row in rows:
                f.write(row_to_line(row) + "\n")
        written += 1
    print(f"[template] wrote {written} draft CSVs into {out_dir}")
    print("[template] fill msp_question, anchor_sid, anchor_label, "
          "anchor_center_* and the GT object columns; the annotation flow "
          "fills ann_ts, ann_pos_*, ann_yaw_rad and the marker box.")
    return 0


def _quote_header(columns: Sequence[str]) -> List[str]:
    return [c for c in columns]


def cmd_assemble(args) -> int:
    frozen = read_bytes(repo_path(args.v1_csv))
    draft_dir = args.drafts
    if not draft_dir or not os.path.isdir(draft_dir):
        raise SystemExit(f"--drafts must be an existing directory, got {draft_dir!r}")

    new_lines: List[str] = []
    sources: List[str] = []
    for name in sorted(os.listdir(draft_dir)):
        if not name.endswith(".csv"):
            continue
        with open(os.path.join(draft_dir, name), encoding="utf-8", newline="") as f:
            rows = draft_rows_from_csv(f.read())
        for row in rows:
            new_lines.append(row_to_line({c: (row.get(c) or "") for c in V1_COLUMNS}))
            sources.append(name)
    print(f"[assemble] {len(new_lines)} draft rows from {len(set(sources))} files")

    out_rel = args.out
    out_path = out_rel if os.path.isabs(out_rel) else repo_path(out_rel)
    canonical = os.path.abspath(repo_path(FROZEN_V2_REL))
    if os.path.abspath(out_path) == canonical and not args.freeze:
        raise SystemExit(
            f"refusing to write {FROZEN_V2_REL} without --freeze. Freezing is "
            "the human gate in protocol section 9: strict validator pass, "
            "spot-check verdicts, distribution tables, then janus sign-off.")

    blob = assemble_v2_bytes(frozen, new_lines)
    rep = prefix_report(blob, frozen)
    if not rep["ok"]:
        raise SystemExit(f"[assemble] PREFIX BROKEN, nothing written: {rep['reason']}")

    n_data = blob.count(b"\n") - 1
    if args.expect_rows and n_data != args.expect_rows:
        raise SystemExit(
            f"[assemble] {n_data} data rows, expected {args.expect_rows}. "
            "Nothing written.")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "wb") as f:
        f.write(blob)
    print(f"[assemble] wrote {out_path}")
    print(f"[assemble] rows {n_data}, bytes {len(blob)}, "
          f"sha256 {sha256_bytes(blob)}")
    print(f"[assemble] prefix {rep['prefix_bytes']} bytes, "
          f"prefix sha256 {rep['prefix_sha256']}")
    print("[assemble] now run: python3 -m src.scripts.validate_bench_rows "
          f"--csv {out_rel} --mode strict")
    return 0


def cmd_spotcheck(args) -> int:
    _header, rows = parse_csv_text(read_bytes(
        args.csv if os.path.isabs(args.csv) else repo_path(args.csv)).decode("utf-8"))
    lo = args.new_from
    ids = list(range(lo, len(rows) + 1))
    draw = spot_check_draw(ids, n=args.n, seed=args.seed)
    print(f"[spotcheck] seed {args.seed}, {args.n} of {len(ids)} rows "
          f"({lo}..{len(rows)})")
    for row_idx in sorted(draw):
        row = rows[row_idx - 1]
        print(f"  row {row_idx}  {row.get('scene', '')}  "
              f"{row.get('predicate', '')}  {row.get('msp_question', '')}")
    return 0


def _session_dir(base: str, pass_id: int) -> str:
    return os.path.join(base, f"pass{pass_id}")


def cmd_noise_select(args) -> int:
    out_dir = _require_out_dir(args)
    _header, rows = parse_csv_text(read_bytes(
        args.csv if os.path.isabs(args.csv) else repo_path(args.csv)).decode("utf-8"))
    picked = select_noise_floor_rows(rows, new_from=args.new_from, seed=args.seed)
    payload = {
        "seed": args.seed,
        "new_from_row": args.new_from,
        "row_indices": picked,
        "note": "1-based data-row indices into the bench split. No annotation "
                "values are stored here on purpose.",
    }
    path = os.path.join(out_dir, "noise_floor_rows.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")
    print(f"[noise-floor] selected rows {picked}")
    print(f"[noise-floor] wrote {path}")
    return 0


def cmd_noise_worksheet(args) -> int:
    out_dir = _require_out_dir(args)
    sel_path = os.path.join(out_dir, "noise_floor_rows.json")
    if not os.path.exists(sel_path):
        raise SystemExit(f"run 'noise-floor select' first; {sel_path} is missing")
    with open(sel_path, encoding="utf-8") as f:
        picked = json.load(f)["row_indices"]

    store_path = args.store if os.path.isabs(args.store) else repo_path(args.store)
    gate_records = gate_projection(_read_csv_rows(store_path))
    ok, why = pass_gate(gate_records, args.pass_id, picked, now=time.time(),
                        min_gap_days=args.min_gap_days)
    if not ok:
        raise SystemExit(f"[noise-floor] pass {args.pass_id} is not open: {why}")
    print(f"[noise-floor] pass {args.pass_id} gate: {why}")

    _header, rows = parse_csv_text(read_bytes(
        args.csv if os.path.isabs(args.csv) else repo_path(args.csv)).decode("utf-8"))
    order = pass_item_order(picked, args.pass_id, seed=args.seed)
    items = [(f"item_{i:02d}", redact_for_blind(rows[r - 1]))
             for i, r in enumerate(order, start=1)]
    keymap = {f"item_{i:02d}": r for i, r in enumerate(order, start=1)}

    done_rows = {r["row_idx"] for r in gate_records if r["pass_id"] == args.pass_id}
    done_items = [k for k, v in keymap.items() if v in done_rows]

    sess = _session_dir(out_dir, args.pass_id)
    os.makedirs(sess, exist_ok=True)
    render_dir = os.path.join(sess, "renders")
    with open(os.path.join(sess, "worksheet.md"), "w", encoding="utf-8") as f:
        f.write(worksheet_text(args.pass_id, items, render_dir, done_items))
    with open(os.path.join(sess, "_keymap.json"), "w", encoding="utf-8") as f:
        json.dump({"pass_id": args.pass_id, "items": keymap,
                   "note": "For the record command only. Opening this file "
                           "while annotating does not break blindness by "
                           "itself, but it is not part of the annotator's "
                           "workflow."}, f, indent=2)
        f.write("\n")
    resp = os.path.join(sess, "responses.csv")
    if not os.path.exists(resp):
        _write_csv_rows(resp,
                        ["item_id", "ann_pos_x", "ann_pos_y", "ann_pos_z",
                         "ann_yaw_rad", "ann_ts"],
                        [{"item_id": k} for k, _ in items])
    print(f"[noise-floor] wrote {sess}/worksheet.md, _keymap.json, responses.csv")
    print(f"[noise-floor] {len(done_items)} of {len(items)} items already recorded "
          "for this pass")
    print("[noise-floor] render the blind context with: python3 -m "
          "src.scripts.render_query_context --mode blind --rows "
          + ",".join(str(r) for r in order) + f" --out-dir {render_dir}")
    return 0


def cmd_noise_record(args) -> int:
    sess = args.session_dir
    if not sess or not os.path.isdir(sess):
        raise SystemExit(f"--session-dir must be an existing directory, got {sess!r}")
    with open(os.path.join(sess, "_keymap.json"), encoding="utf-8") as f:
        keymap_payload = json.load(f)
    if int(keymap_payload["pass_id"]) != args.pass_id:
        raise SystemExit(
            f"session directory is for pass {keymap_payload['pass_id']}, "
            f"not pass {args.pass_id}")
    keymap = keymap_payload["items"]

    responses = _read_csv_rows(os.path.join(sess, "responses.csv"))
    new: List[Dict[str, Any]] = []
    for r in responses:
        item = (r.get("item_id") or "").strip()
        if item not in keymap:
            raise SystemExit(f"response references unknown item {item!r}")
        vals = [(r.get(c) or "").strip() for c in
                ("ann_pos_x", "ann_pos_y", "ann_pos_z", "ann_yaw_rad", "ann_ts")]
        if not any(vals):
            continue
        if not all(vals):
            raise SystemExit(
                f"{item} is partially filled; every response field is required")
        yaw = float(vals[3])
        if not (-math.pi < yaw <= math.pi):
            raise SystemExit(f"{item} has ann_yaw_rad {yaw} outside (-pi, pi]")
        new.append({
            "row_idx": int(keymap[item]), "pass_id": args.pass_id,
            "ann_pos_x": vals[0], "ann_pos_y": vals[1], "ann_pos_z": vals[2],
            "ann_yaw_rad": vals[3], "ann_ts": vals[4],
        })

    store_path = args.store if os.path.isabs(args.store) else repo_path(args.store)
    existing = _read_csv_rows(store_path)
    merged = merge_noise_records(existing, new)
    _write_csv_rows(store_path, NOISE_STORE_COLUMNS, merged)
    print(f"[noise-floor] recorded {len(new)} items into {store_path} "
          f"({len(merged)} records total)")
    print("[noise-floor] this tool has no command that compares the two "
          "passes; that comparison belongs to the metrics analysis, after "
          "both passes are banked.")
    return 0


def cmd_noise_status(args) -> int:
    store_path = args.store if os.path.isabs(args.store) else repo_path(args.store)
    gate_records = gate_projection(_read_csv_rows(store_path))
    for pass_id in (1, 2):
        n = sum(1 for r in gate_records if r["pass_id"] == pass_id)
        print(f"[noise-floor] pass {pass_id}: {n} of {NOISE_FLOOR_N} rows recorded")
    print("[noise-floor] counts only; this command never prints an annotated "
          "value, in either pass.")
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--v1-csv", default=V1_CSV_REL)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("plan", help="write the 52-slot authoring brief")
    p.add_argument("--out-dir", default="")
    p.add_argument("--seed", type=int, default=PLAN_SEED)
    p.set_defaults(func=cmd_plan)

    p = sub.add_parser("template", help="write per-scene draft CSVs")
    p.add_argument("--out-dir", default="")
    p.add_argument("--seed", type=int, default=PLAN_SEED)
    p.add_argument("--scenes", default="", help="comma-separated subset")
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=cmd_template)

    p = sub.add_parser("assemble", help="v1 bytes plus draft rows, prefix checked")
    p.add_argument("--drafts", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--expect-rows", type=int, default=TARGET_ROWS)
    p.add_argument("--freeze", action="store_true",
                   help="allow writing the canonical splits/bench_v2_150.csv")
    p.set_defaults(func=cmd_assemble)

    p = sub.add_parser("spotcheck", help="the reproducible 10-row human draw")
    p.add_argument("--csv", required=True)
    p.add_argument("--new-from", type=int, default=V1_ROWS + 1)
    p.add_argument("--n", type=int, default=SPOTCHECK_N)
    p.add_argument("--seed", type=int, default=SPOTCHECK_SEED)
    p.set_defaults(func=cmd_spotcheck)

    nf = sub.add_parser("noise-floor", help="D6 test-retest passes")
    nfs = nf.add_subparsers(dest="nf_cmd", required=True)

    q = nfs.add_parser("select", help="draw the 20 stratified rows")
    q.add_argument("--csv", required=True)
    q.add_argument("--out-dir", default="")
    q.add_argument("--new-from", type=int, default=V1_ROWS + 1)
    q.add_argument("--seed", type=int, default=NOISE_SELECT_SEED)
    q.set_defaults(func=cmd_noise_select)

    q = nfs.add_parser("worksheet", help="build a blind worksheet for one pass")
    q.add_argument("--csv", required=True)
    q.add_argument("--out-dir", default="")
    q.add_argument("--pass", dest="pass_id", type=int, required=True, choices=(1, 2))
    q.add_argument("--store", default=NOISE_STORE_REL)
    q.add_argument("--seed", type=int, default=NOISE_SELECT_SEED)
    q.add_argument("--min-gap-days", type=int, default=MIN_PASS_GAP_DAYS)
    q.set_defaults(func=cmd_noise_worksheet)

    q = nfs.add_parser("record", help="append a pass's responses to the store")
    q.add_argument("--session-dir", required=True)
    q.add_argument("--pass", dest="pass_id", type=int, required=True, choices=(1, 2))
    q.add_argument("--store", default=NOISE_STORE_REL)
    q.set_defaults(func=cmd_noise_record)

    q = nfs.add_parser("status", help="per-pass record counts, values never shown")
    q.add_argument("--store", default=NOISE_STORE_REL)
    q.set_defaults(func=cmd_noise_status)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
