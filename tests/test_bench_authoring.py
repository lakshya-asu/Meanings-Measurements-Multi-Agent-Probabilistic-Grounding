"""Unit tests for the pure logic of the MAPG-07 bench-v2-150 tooling.

Habitat-free, like tests/test_collect_poses.py: the byte-level prefix rule,
the row-level checks, the authoring slot builder, the D6 blindness
guarantees, and the camera projection are all exercised with plain python
data. The habitat-backed half of the validator (V2, V3, the V5 graph
lookup) and the actual rendering are covered by the in-container dry run.
"""

import math

import pytest

from src.evals.success import gt_xyz_from_row
from src.scripts import validate_bench_rows as V
from src.scripts.author_bench_v2 import (
    D0_TARGETS,
    NOISE_FLOOR_N,
    PREDICATE_TARGETS,
    VERTICAL_MAX_D0_M,
    blank_draft_row,
    build_slots,
    gate_projection,
    merge_noise_records,
    pair_predicates_with_d0,
    pass_gate,
    pass_item_order,
    receiving_scenes,
    redact_for_blind,
    select_noise_floor_rows,
    slot_summary,
    spot_check_draw,
    worksheet_text,
)
from src.scripts.bench_v2_common import ANN_TOOL_CONSTANTS


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def good_row(**overrides):
    """A synthetic new row that passes every host-side check."""
    from src.scripts.bench_v2_common import marker_aabb

    ann = [1.5, 0.0, 0.0]
    lo, hi = marker_aabb(ann)
    row = {
        "scene": "00506-QVAA6zecMHu",
        "floor": "0",
        "distance_m": "1.5",
        "predicate": "in front of",
        "msp_question": "Where is 1.5 meters in front of the blue sofa?",
        "ann_ok": "1",
        "ann_ts": "1772444800.5",
        "ann_pos_x": str(ann[0]), "ann_pos_y": str(ann[1]), "ann_pos_z": str(ann[2]),
        "ann_yaw_rad": "0.5",
        "ann_aabb_min_x": str(lo[0]), "ann_aabb_min_y": str(lo[1]),
        "ann_aabb_min_z": str(lo[2]),
        "ann_aabb_max_x": str(hi[0]), "ann_aabb_max_y": str(hi[1]),
        "ann_aabb_max_z": str(hi[2]),
        "anchor_sid": "10",
        "anchor_label": "blue sofa",
        "anchor_center_x": "0", "anchor_center_y": "0", "anchor_center_z": "0",
        "GT Object 1": "wall",
        "GT Object 2": "table",
    }
    row.update(ANN_TOOL_CONSTANTS)
    row.update(overrides)
    return row


def fails(findings, check=None):
    out = [f for f in findings if f["severity"] == V.FAIL]
    if check:
        out = [f for f in out if f["check"] == check]
    return out


def load_v1_rows():
    from src.scripts.bench_v2_common import (
        V1_CSV_REL, parse_csv_text, read_bytes, repo_path,
    )
    _h, rows = parse_csv_text(read_bytes(repo_path(V1_CSV_REL)).decode("utf-8"))
    return rows


# ---------------------------------------------------------------------------
# bench_v2_common: the byte-level prefix rule
# ---------------------------------------------------------------------------

FROZEN = b"a,b\n1,2\n3,4\n"


def test_prefix_report_accepts_exact_prefix():
    from src.scripts.bench_v2_common import prefix_report
    rep = prefix_report(FROZEN + b"5,6\n", FROZEN)
    assert rep["ok"] is True
    assert rep["first_diff_offset"] is None
    assert rep["prefix_bytes"] == len(FROZEN)


def test_prefix_report_names_the_offending_row():
    from src.scripts.bench_v2_common import prefix_report
    tampered = b"a,b\n1,2\n3,9\n5,6\n"
    rep = prefix_report(tampered, FROZEN)
    assert rep["ok"] is False
    # File line 3 is data row 2.
    assert rep["first_diff_file_line"] == 3
    assert rep["first_diff_data_row"] == 2


def test_prefix_report_rejects_truncation():
    from src.scripts.bench_v2_common import prefix_report
    rep = prefix_report(FROZEN[:-3], FROZEN)
    assert rep["ok"] is False
    assert "shorter" in rep["reason"]


def test_prefix_report_catches_whitespace_only_change():
    # A reparse-and-rewrite round trip is exactly the failure the byte check
    # exists for: same parsed rows, different bytes.
    from src.scripts.bench_v2_common import prefix_report
    rep = prefix_report(b"a,b\n1,2\n3, 4\n", FROZEN)
    assert rep["ok"] is False


def test_assemble_keeps_the_prefix_byte_identical():
    from src.scripts.bench_v2_common import assemble_v2_bytes, prefix_report
    blob = assemble_v2_bytes(FROZEN, ["5,6", "7,8"])
    assert blob.startswith(FROZEN)
    assert prefix_report(blob, FROZEN)["ok"]
    assert blob.count(b"\n") == 5


def test_assemble_adds_a_separator_when_the_frozen_file_lacks_one():
    from src.scripts.bench_v2_common import assemble_v2_bytes
    blob = assemble_v2_bytes(b"a,b\n1,2", ["5,6"])
    assert blob == b"a,b\n1,2\n5,6\n"


def test_assemble_rejects_an_embedded_newline():
    from src.scripts.bench_v2_common import assemble_v2_bytes
    with pytest.raises(ValueError):
        assemble_v2_bytes(FROZEN, ["5,6\n7,8"])


def test_row_to_line_requires_the_exact_schema():
    from src.scripts.bench_v2_common import V1_COLUMNS, row_to_line
    row = {c: "" for c in V1_COLUMNS}
    assert row_to_line(row).count(",") == len(V1_COLUMNS) - 1
    del row["floor"]
    with pytest.raises(KeyError):
        row_to_line(row)
    row["floor"] = ""
    row["extra"] = "x"
    with pytest.raises(KeyError):
        row_to_line(row)


def test_row_to_line_quotes_like_the_csv_module():
    from src.scripts.bench_v2_common import csv_line
    assert csv_line(["a", "b,c", 'd"e']) == 'a,"b,c","d""e"'


def test_number_formatting_matches_v1_style():
    from src.scripts.bench_v2_common import fmt_coord, fmt_distance
    assert fmt_distance(1.0) == "1"
    assert fmt_distance(0.5) == "0.5"
    assert fmt_distance(0.0) == "0"
    assert fmt_coord(12.2803341234) == "12.280334"


def test_elevation_and_horizontal_distance():
    from src.scripts.bench_v2_common import dist_horizontal, elevation_deg
    assert elevation_deg([0, 0, 0], [1, 0, 0]) == pytest.approx(0.0)
    assert elevation_deg([0, 0, 0], [0, 1, 0]) == pytest.approx(90.0)
    assert elevation_deg([0, 0, 0], [1, 1, 0]) == pytest.approx(45.0)
    # Below counts the same as above: it is the ray's tilt, not its sign.
    assert elevation_deg([0, 0, 0], [0, -1, 0]) == pytest.approx(90.0)
    assert elevation_deg([0, 0, 0], [0, 0, 0]) is None
    # y is up, so horizontal distance ignores it.
    assert dist_horizontal([0, 5, 0], [3, -9, 4]) == pytest.approx(5.0)


def test_text_style_problems():
    from src.scripts.bench_v2_common import text_style_problems
    assert text_style_problems("table ") == ["leading or trailing whitespace"]
    assert "em dash or en dash" in text_style_problems("a — b")
    assert text_style_problems("plain text") == []


# ---------------------------------------------------------------------------
# validator: file level
# ---------------------------------------------------------------------------

def test_check_file_bytes_on_a_clean_append():
    from src.scripts.bench_v2_common import assemble_v2_bytes
    blob = assemble_v2_bytes(FROZEN, ["5,6"])
    got = V.check_file_bytes(blob, FROZEN, expect_rows=3)
    assert not fails(got)


def test_check_file_bytes_flags_row_count_bom_and_crlf():
    got = V.check_file_bytes(FROZEN, FROZEN, expect_rows=150)
    assert [f["check"] for f in fails(got)] == ["P2"]
    assert "expected exactly 150" in fails(got)[0]["message"]

    got = V.check_file_bytes(b"\xef\xbb\xbf" + FROZEN, FROZEN, expect_rows=2)
    checks = {f["check"] for f in fails(got)}
    assert "P4" in checks and "P1" in checks

    got = V.check_file_bytes(b"a,b\r\n1,2\n3,4\n", FROZEN, expect_rows=2)
    assert any("CRLF" in f["message"] for f in fails(got))


def test_check_file_bytes_flags_a_missing_trailing_newline():
    got = V.check_file_bytes(FROZEN[:-1], FROZEN, expect_rows=2)
    assert any("does not end with a newline" in f["message"] for f in fails(got))


def test_check_header():
    from src.scripts.bench_v2_common import V1_COLUMNS
    assert not fails(V.check_header(V1_COLUMNS))
    assert fails(V.check_header(list(V1_COLUMNS) + ["oops"]))
    swapped = list(V1_COLUMNS)
    swapped[0], swapped[1] = swapped[1], swapped[0]
    msg = fails(V.check_header(swapped))[0]["message"]
    assert "wrong order" in msg


def test_frozen_v1_file_is_its_own_prefix():
    from src.scripts.bench_v2_common import (
        V1_CSV_REL, V1_ROWS, V1_SHA256, read_bytes, repo_path, sha256_bytes,
    )
    blob = read_bytes(repo_path(V1_CSV_REL))
    assert sha256_bytes(blob) == V1_SHA256
    got = V.check_file_bytes(blob, blob, expect_rows=V1_ROWS)
    assert not fails(got)


# ---------------------------------------------------------------------------
# validator: row level
# ---------------------------------------------------------------------------

def test_good_row_has_no_failures():
    assert fails(V.check_row(99, good_row())) == []


def test_every_finding_names_its_row_and_scene():
    for f in V.check_row(123, good_row(ann_yaw_rad="9.9")):
        assert f["row"] == 123
        assert f["scene"] == "00506-QVAA6zecMHu"
        assert f["message"]


def test_gt_check_uses_the_real_scoring_function():
    row = good_row()
    got = [f for f in V.check_row(99, row) if f["check"] == "GT"]
    assert got[0]["detail"]["gt_xyz"] == gt_xyz_from_row(row)


def test_unresolvable_gt_fails():
    row = good_row(anchor_center_x="", anchor_center_y="", anchor_center_z="",
                   ann_pos_x="", ann_pos_y="", ann_pos_z="")
    assert fails(V.check_row(99, row), "GT")


def test_zero_distance_row_without_ann_pos_fails_d5():
    row = good_row(distance_m="0", msp_question="Where is between the sofa and the tv?",
                   predicate="between", anchor_label="sofa",
                   ann_pos_x="", ann_pos_y="", ann_pos_z="",
                   ann_aabb_min_x="", ann_aabb_min_y="", ann_aabb_min_z="",
                   ann_aabb_max_x="", ann_aabb_max_y="", ann_aabb_max_z="")
    d5 = fails(V.check_row(99, row), "D5")
    assert d5 and "unscoreable" in d5[0]["message"]
    # And it is genuinely unscoreable, per the real function.
    assert gt_xyz_from_row(row) is None


def test_zero_distance_row_with_ann_pos_scores_against_it():
    row = good_row(distance_m="0", predicate="between",
                   msp_question="Where is between the blue sofa and the tv?")
    findings = V.check_row(99, row)
    assert fails(findings, "D5") == []
    assert gt_xyz_from_row(row) == [1.5, 0.0, 0.0]


def test_zero_distance_row_with_a_literal_fails_v4():
    row = good_row(distance_m="0", predicate="between",
                   msp_question="Where is 2 meters between the blue sofa and the tv?")
    assert fails(V.check_row(99, row), "V4")


def test_d0_round_trip_mismatch_fails_like_v1_row_53():
    row = good_row(distance_m="1.2",
                   msp_question="Where is 1 meter in front of the blue sofa?")
    got = fails(V.check_row(99, row), "V4")
    assert got and "round trip fails" in got[0]["message"]


def test_missing_literal_fails_v4():
    row = good_row(msp_question="Where is in front of the blue sofa?")
    assert fails(V.check_row(99, row), "V4")


def test_predicate_text_disagreement_fails_v10():
    row = good_row(predicate="left of")
    got = fails(V.check_row(99, row), "V10")
    assert got and "infer_relation" in got[0]["message"]


def test_legacy_predicates_are_not_applicable_for_v10_but_fail_r1():
    row = good_row(predicate="towards")
    findings = V.check_row(99, row)
    assert fails(findings, "V10") == []
    assert fails(findings, "R1")


def test_yaw_out_of_range_fails_v6():
    assert fails(V.check_row(99, good_row(ann_yaw_rad="3.2")), "V6")
    assert fails(V.check_row(99, good_row(ann_yaw_rad=str(-math.pi))), "V6")
    assert fails(V.check_row(99, good_row(ann_yaw_rad=str(math.pi))), "V6") == []


def test_radial_consistency_warns_then_fails():
    warn = V.check_row(99, good_row(ann_pos_x="2.4"))
    assert fails(warn, "V7") == []
    assert any(f["check"] == "V7" and f["severity"] == V.WARN for f in warn)
    bad = V.check_row(99, good_row(ann_pos_x="4.0"))
    assert fails(bad, "V7")


def test_horizontal_tilt_fails_v8():
    # A point straight above the anchor with a horizontal predicate.
    row = good_row(ann_pos_x="0.1", ann_pos_y="1.5")
    got = fails(V.check_row(99, row), "V8")
    assert got and "tilted" in got[0]["message"]


def test_vertical_direction_fails_v9():
    row = good_row(predicate="above", distance_m="1",
                   msp_question="Where is 1 meter above the blue sofa?",
                   ann_pos_x="1.0", ann_pos_y="0.1")
    assert fails(V.check_row(99, row), "V9")


def test_vertical_direction_passes_when_the_ray_is_vertical():
    row = good_row(predicate="above", distance_m="1",
                   msp_question="Where is 1 meter above the blue sofa?",
                   ann_pos_x="0.0", ann_pos_y="1.0")
    assert fails(V.check_row(99, row), "V9") == []


def test_eps_guard_fails_v13():
    row = good_row(ann_pos_x="0", ann_pos_y="0", ann_pos_z="0")
    assert fails(V.check_row(99, row), "V13")


def test_anchor_label_whitespace_and_case_fail_v5():
    assert fails(V.check_row(99, good_row(anchor_label="blue sofa ")), "V5")
    assert fails(V.check_row(99, good_row(anchor_label="Blue Sofa")), "V5")


def test_ann_ok_and_tool_constants_are_enforced():
    assert fails(V.check_row(99, good_row(ann_ok="0")), "V1")
    assert fails(V.check_row(99, good_row(ann_scale_x="0.2")), "V1")


def test_marker_box_must_contain_ann_pos():
    row = good_row(ann_aabb_min_x="99.0", ann_aabb_max_x="99.3")
    assert fails(V.check_row(99, row), "V1")


def test_off_grid_distance_fails_r1():
    row = good_row(distance_m="1.2",
                   msp_question="Where is 1.2 meters in front of the blue sofa?")
    got = fails(V.check_row(99, row), "R1")
    assert got and "grid" in got[0]["message"]


def test_style_rules_fail_s1():
    assert fails(V.check_row(99, good_row(anchor_label="sofa—blue")), "S1")
    assert fails(V.check_row(99, good_row(
        msp_question="How far is 1.5 meters in front of the blue sofa?")), "S1")
    assert fails(V.check_row(99, good_row(
        msp_question="Where is 1.5 meters in front of the blue sofa.")), "S1")


def test_uniqueness_flags_duplicates_and_reused_triples():
    rows = [good_row(), good_row(anchor_sid="11"), good_row()]
    got = V.check_uniqueness(rows, new_from=2)
    dupes = [f for f in got if "duplicate query" in f["message"]]
    assert [f["row"] for f in dupes] == [2, 3]
    reused = [f for f in got if "triple" in f["message"]]
    assert [f["row"] for f in reused] == [3]


def test_pose_pair_check_reports_and_refuses_to_guess():
    rows = [good_row(), good_row(floor="7")]
    got = V.check_pose_pairs(rows, ["00506-QVAA6zecMHu_0"])
    bad = fails(got, "V12")
    assert len(bad) == 1 and bad[0]["row"] == 2
    assert "00506-QVAA6zecMHu_7" in bad[0]["message"]
    # No pose file at all is NOT_RUN, never a silent pass.
    assert V.check_pose_pairs(rows, None)[0]["severity"] == V.NOT_RUN


def test_validator_eps_matches_the_scoring_module():
    from src.evals.success import EPS as scoring_eps
    from src.scripts.bench_v2_common import EPS
    assert EPS == scoring_eps


def test_plausible_range_mirrors_the_parser():
    from src.parsing.metric_literal import PLAUSIBLE_RANGE_M as parser_range
    from src.scripts.bench_v2_common import PLAUSIBLE_RANGE_M
    assert PLAUSIBLE_RANGE_M == parser_range


# ---------------------------------------------------------------------------
# authoring: slots
# ---------------------------------------------------------------------------

def test_receiving_scenes_matches_the_protocol_table():
    scenes = receiving_scenes(load_v1_rows())
    assert len(scenes) == 26


def test_receiving_scenes_rejects_a_split_that_drifted():
    rows = [{"scene": "99999-nope", "floor": "0"}]
    with pytest.raises(ValueError):
        receiving_scenes(rows)


def test_build_slots_realizes_the_protocol_distribution():
    slots = build_slots(load_v1_rows())
    summary = slot_summary(slots)
    assert summary["n_slots"] == 52
    assert summary["predicates"] == PREDICATE_TARGETS
    want = {}
    for d0, n in D0_TARGETS.items():
        want[("%g" % d0)] = n
    want["0"] = PREDICATE_TARGETS["between"]
    assert summary["distances"] == want


def test_build_slots_is_deterministic_and_seed_sensitive():
    rows = load_v1_rows()
    a = build_slots(rows, seed=20260818)
    b = build_slots(rows, seed=20260818)
    c = build_slots(rows, seed=1)
    assert a == b
    assert a != c
    # Same seed, same distribution regardless.
    assert slot_summary(a)["predicates"] == slot_summary(c)["predicates"]


def test_every_scene_gets_exactly_two_slots_on_an_existing_floor():
    rows = load_v1_rows()
    slots = build_slots(rows)
    per_scene = {}
    for s in slots:
        per_scene[s["scene"]] = per_scene.get(s["scene"], 0) + 1
    assert set(per_scene.values()) == {2}
    existing = {}
    for r in rows:
        existing.setdefault(r["scene"], set()).add(int(float(r["floor"])))
    for s in slots:
        assert s["floor"] in existing[s["scene"]]


def test_vertical_slots_never_get_an_impossible_distance():
    for s in build_slots(load_v1_rows()):
        if s["predicate"] in ("above", "below"):
            assert s["distance_m"] <= VERTICAL_MAX_D0_M


def test_pair_predicates_raises_when_verticals_cannot_be_served():
    rng = __import__("random").Random(0)
    with pytest.raises(ValueError):
        pair_predicates_with_d0(["above", "below"], [5.0, 5.0], rng)


def test_overlay_quotas_are_met():
    slots = build_slots(load_v1_rows())
    counts = slot_summary(slots)["overlays"]
    assert counts["intrinsic_frame"] == 10
    assert counts["occlusion"] == 10
    assert counts["distractor"] >= 16
    assert counts["modifier"] >= 10
    intrinsic = [s for s in slots if "intrinsic_frame" in s["overlays"]]
    assert sum(1 for s in intrinsic if s["predicate"] == "in front of") == 6
    assert sum(1 for s in intrinsic if s["predicate"] == "behind") == 4


def test_blank_draft_row_is_schema_shaped_and_invents_nothing():
    from src.scripts.bench_v2_common import V1_COLUMNS, row_to_line
    slot = build_slots(load_v1_rows())[0]
    row = blank_draft_row(slot)
    assert set(row) == set(V1_COLUMNS)
    assert row["msp_question"] == ""
    assert row["ann_pos_x"] == ""
    assert row["anchor_label"] == ""
    assert row["ann_scale_x"] == "0.15"
    row_to_line(row)  # must serialize


def test_spot_check_draw_is_reproducible():
    ids = list(range(99, 151))
    assert spot_check_draw(ids) == spot_check_draw(ids)
    assert len(set(spot_check_draw(ids))) == 10
    with pytest.raises(ValueError):
        spot_check_draw([1, 2], n=10)


# ---------------------------------------------------------------------------
# D6: test-retest blindness
# ---------------------------------------------------------------------------

def noise_rows():
    """98 v1-shaped rows plus 52 new-shaped rows, predicates only."""
    v1_preds = (["in front of"] * 33 + ["right of"] * 18 + ["behind"] * 13
                + ["left of"] * 11 + ["above"] * 10 + ["between"] * 6
                + ["below"] * 4 + ["towards"] + ["near"] + ["from"])
    new_preds = []
    for p, n in PREDICATE_TARGETS.items():
        new_preds.extend([p] * n)
    rows = []
    for i, p in enumerate(v1_preds + new_preds, start=1):
        rows.append({
            "scene": f"scene_{i % 41:02d}", "floor": "0", "predicate": p,
            "msp_question": f"Where is 1 meter {p} the thing {i}?",
            "ann_pos_x": "1.0", "ann_pos_y": "2.0", "ann_pos_z": "3.0",
            "ann_yaw_rad": "0.4", "ann_ts": "1772444800",
        })
    return rows


def test_noise_floor_selection_is_stratified_and_reproducible():
    rows = noise_rows()
    picked = select_noise_floor_rows(rows, new_from=99)
    assert picked == select_noise_floor_rows(rows, new_from=99)
    assert len(picked) == NOISE_FLOOR_N == 20
    assert sum(1 for i in picked if i < 99) == 10
    assert sum(1 for i in picked if i >= 99) == 10
    got = {}
    for i in picked:
        p = rows[i - 1]["predicate"]
        got[p] = got.get(p, 0) + 1
    assert got["between"] == 2  # one from v1, one new
    assert got["near"] == 1


def test_noise_floor_selection_refuses_a_thin_stratum():
    rows = noise_rows()
    for r in rows[:98]:
        if r["predicate"] == "between":
            r["predicate"] = "near"
    with pytest.raises(ValueError):
        select_noise_floor_rows(rows, new_from=99)


def test_redaction_is_an_allowlist():
    row = dict(good_row())
    row["some_future_column"] = "secret"
    blind = redact_for_blind(row)
    assert set(blind) == {"scene", "floor", "msp_question"}
    assert "secret" not in "".join(blind.values())
    for key in ("ann_pos_x", "ann_yaw_rad", "ann_ts", "anchor_center_x"):
        assert key not in blind


def test_pass_two_order_does_not_correlate_with_pass_one():
    rows = list(range(1, 21))
    p1 = pass_item_order(rows, 1)
    p2 = pass_item_order(rows, 2)
    assert sorted(p1) == sorted(p2) == rows
    assert p1 != p2
    # Deterministic, so a resumed session shows the order it started with.
    assert p2 == pass_item_order(rows, 2)
    assert pass_item_order(rows, 1) == p1


def test_pass_item_order_rejects_a_bogus_pass():
    with pytest.raises(ValueError):
        pass_item_order([1, 2, 3], 3)


def test_gate_projection_drops_every_coordinate():
    records = [{"row_idx": "5", "pass_id": "1", "ann_pos_x": "1.25",
                "ann_pos_y": "2", "ann_pos_z": "3", "ann_yaw_rad": "0.4",
                "ann_ts": "100.0"}]
    got = gate_projection(records)
    assert got == [{"row_idx": 5, "pass_id": 1, "ann_ts": 100.0}]
    assert "1.25" not in str(got)


def test_pass_two_is_gated_on_completeness_and_a_week():
    rows = [1, 2, 3]
    day = 86400.0
    assert pass_gate([], 1, rows, now=0.0)[0] is True

    partial = gate_projection([{"row_idx": 1, "pass_id": 1, "ann_ts": 0.0}])
    ok, why = pass_gate(partial, 2, rows, now=100 * day)
    assert ok is False and "incomplete" in why

    full = gate_projection([{"row_idx": i, "pass_id": 1, "ann_ts": 10 * day}
                            for i in rows])
    ok, why = pass_gate(full, 2, rows, now=13 * day)
    assert ok is False and "at least 7 days" in why
    ok, why = pass_gate(full, 2, rows, now=18 * day)
    assert ok is True


def test_records_are_append_only_and_the_refusal_leaks_nothing():
    existing = [{"row_idx": 5, "pass_id": 1, "ann_pos_x": "1.2345",
                 "ann_pos_y": "0", "ann_pos_z": "0", "ann_yaw_rad": "0",
                 "ann_ts": "1"}]
    added = merge_noise_records(
        existing, [{"row_idx": 5, "pass_id": 2, "ann_pos_x": "9", "ann_pos_y": "9",
                    "ann_pos_z": "9", "ann_yaw_rad": "0", "ann_ts": "2"}])
    assert len(added) == 2
    with pytest.raises(ValueError) as e:
        merge_noise_records(existing, [dict(existing[0])])
    assert "1.2345" not in str(e.value)
    assert "already has a pass-1 record" in str(e.value)


def test_worksheet_shows_no_annotation_value_and_no_defaults():
    rows = noise_rows()
    items = [(f"item_{i:02d}", redact_for_blind(rows[i - 1])) for i in range(1, 4)]
    text = worksheet_text(2, items, "/drafts/pass2/renders")
    # The column names appear once, in the empty response template. No
    # annotated VALUE from any pass, and no primary GT, may appear.
    for value in ("1.0", "2.0", "3.0", "0.4", "1772444800"):
        assert value not in text
    assert "item_id,ann_pos_x,ann_pos_y,ann_pos_z,ann_yaw_rad,ann_ts" in text
    assert "TODO" in text
    assert "blind" in text.lower()


def test_worksheet_refuses_a_row_that_was_not_redacted():
    with pytest.raises(ValueError) as e:
        worksheet_text(2, [("item_01", good_row())], "/tmp/r")
    assert "must not reach a blind worksheet" in str(e.value)


def test_worksheet_marks_recorded_items_without_showing_them():
    rows = noise_rows()
    items = [(f"item_{i:02d}", redact_for_blind(rows[i - 1])) for i in range(1, 4)]
    text = worksheet_text(2, items, "/r", done_items=["item_02"])
    line = [ln for ln in text.splitlines() if ln.startswith("| item_02")][0]
    assert "recorded" in line
    assert "0.4" not in line


# ---------------------------------------------------------------------------
# render helper: camera and view planning
# ---------------------------------------------------------------------------

def test_projection_puts_a_point_dead_ahead_at_the_centre():
    from src.scripts.render_query_context import project_to_pixel
    uv = project_to_pixel([0, 0, -3], [0, 0, 0], yaw=0.0, pitch=0.0,
                          width=640, height=480)
    assert uv == pytest.approx((320.0, 240.0))


def test_projection_puts_a_point_to_the_right_right_of_centre():
    from src.scripts.render_query_context import project_to_pixel
    u, v = project_to_pixel([1, 0, -3], [0, 0, 0], 0.0, 0.0)
    assert u > 320.0 and v == pytest.approx(240.0)


def test_projection_returns_none_behind_and_outside_the_frame():
    from src.scripts.render_query_context import project_to_pixel
    assert project_to_pixel([0, 0, 3], [0, 0, 0], 0.0, 0.0) is None
    assert project_to_pixel([50, 0, -1], [0, 0, 0], 0.0, 0.0) is None


def test_projection_follows_yaw():
    from src.scripts.render_query_context import project_to_pixel
    # Turning to face -x puts a point at -x dead centre.
    uv = project_to_pixel([-3, 0, 0], [0, 0, 0], yaw=math.pi / 2, pitch=0.0)
    assert uv == pytest.approx((320.0, 240.0))


def test_projection_follows_pitch_downwards():
    from src.scripts.render_query_context import project_to_pixel
    uv = project_to_pixel([0, -3, 0], [0, 0, 0], yaw=0.0,
                          pitch=math.radians(-90))
    assert uv == pytest.approx((320.0, 240.0))


def test_crosshair_pixels_clip_to_the_frame():
    from src.scripts.render_query_context import crosshair_pixels
    px = crosshair_pixels(0.0, 0.0, 640, 480, arm=5)
    assert all(0 <= x < 640 and 0 <= y < 480 for x, y in px)
    assert (0, 0) in px


def test_orbit_viewpoints_face_the_anchor():
    from src.scripts.render_query_context import orbit_viewpoints, project_to_pixel
    anchor = [2.0, 1.0, -3.0]
    vps = orbit_viewpoints(anchor)
    assert len(vps) == 8
    for _label, pos, yaw in vps:
        uv = project_to_pixel(anchor, pos, yaw, 0.0)
        assert uv is not None
        assert uv[0] == pytest.approx(320.0, abs=1.0)


def test_blind_plan_has_no_anchor_views_and_no_markers():
    from src.scripts.render_query_context import markers_for_row, plan_views
    blind = redact_for_blind(good_row())
    pose = {"init_x": 0.0, "init_y": 0.0, "init_z": 0.0, "init_angle": 0.0}
    views = plan_views(blind, "blind", pose)
    assert [v["name"] for v in views] == ["start_pose"]
    assert markers_for_row(blind, "blind") == {}
    assert markers_for_row(good_row(), "author") == {}


def test_review_plan_marks_the_real_scoring_gt():
    from src.scripts.render_query_context import markers_for_row, plan_views
    row = good_row()
    views = plan_views(row, "review", None)
    names = [v["name"] for v in views]
    assert "topdown" in names and len(names) == 9
    markers = markers_for_row(row, "review")
    assert markers["scoring_gt"] == gt_xyz_from_row(row)
    assert markers["ann_pos"] == [1.5, 0.0, 0.0]


def test_plan_views_rejects_an_unknown_mode():
    from src.scripts.render_query_context import plan_views
    with pytest.raises(ValueError):
        plan_views(good_row(), "peek", None)
