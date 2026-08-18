"""Tests for src/evals/success.py (item 5: GT-checked success metrics).

Worked examples are hand-computed. Habitat convention: y is up, so the
horizontal distance d_h uses x and z only.
"""

import math

from src.evals.success import (
    DESCRIPTIVE_TAUS,
    PRIMARY_TAU,
    euclidean_error,
    gt_success,
    gt_xyz_from_row,
    horizontal_error,
    object_answer_success,
    score_episode,
)


# ----------------------------------------------------------------------
# horizontal_error / euclidean_error worked examples
# ----------------------------------------------------------------------

def test_horizontal_error_hand_computed():
    # dx = 3, dz = 4 -> 3-4-5 triangle in the horizontal plane.
    pred = [4.0, 7.0, 6.0]
    gt = [1.0, 2.0, 2.0]
    assert horizontal_error(pred, gt) == 5.0


def test_horizontal_error_projects_out_y():
    # Points differing only in y (the up axis) have zero horizontal error.
    pred = [1.5, 10.0, -2.0]
    gt = [1.5, 0.0, -2.0]
    assert horizontal_error(pred, gt) == 0.0
    # But the 3D distance sees the vertical gap.
    assert euclidean_error(pred, gt) == 10.0


def test_euclidean_error_hand_computed():
    # dx = 3, dy = 4, dz = 12 -> sqrt(9 + 16 + 144) = 13.
    pred = [3.0, 4.0, 12.0]
    gt = [0.0, 0.0, 0.0]
    assert euclidean_error(pred, gt) == 13.0


def test_errors_accept_tuples_and_are_symmetric():
    a = (2.0, 1.0, 0.0)
    b = (0.0, 5.0, 0.0)
    assert horizontal_error(a, b) == horizontal_error(b, a) == 2.0
    assert euclidean_error(a, b) == euclidean_error(b, a) == math.sqrt(4.0 + 16.0)


# ----------------------------------------------------------------------
# gt_success: tau boundary and missing prediction
# ----------------------------------------------------------------------

def test_gt_success_tau_boundary_exact_is_success():
    gt = [0.0, 0.0, 0.0]
    # d_h exactly 1.0 m: success (<= is the preregistered rule).
    assert gt_success([1.0, 5.0, 0.0], gt, tau=1.0) is True
    # Just over the boundary: failure.
    assert gt_success([1.0000001, 5.0, 0.0], gt, tau=1.0) is False


def test_gt_success_uses_horizontal_not_3d():
    gt = [0.0, 0.0, 0.0]
    # 3D distance is huge, horizontal is 0.5: still a success at 1 m.
    assert gt_success([0.5, 20.0, 0.0], gt, tau=1.0) is True


def test_gt_success_default_tau_is_primary():
    assert PRIMARY_TAU == 1.0
    assert DESCRIPTIVE_TAUS == (0.5, 2.0)
    assert gt_success([0.9, 0.0, 0.0], [0.0, 0.0, 0.0]) is True


def test_gt_success_none_pred_returns_none():
    # A missing prediction is a failure for SR purposes, but the pure
    # function returns None (distances are undefined); score_episode is
    # responsible for storing the explicit False.
    assert gt_success(None, [0.0, 0.0, 0.0], tau=1.0) is None


# ----------------------------------------------------------------------
# gt_xyz_from_row: column convention
# ----------------------------------------------------------------------

def test_gt_from_row_prefers_metric_corrected_columns():
    # The columns eval_offset_distances.py reads today. CSV rows hold
    # strings; the parser must coerce them.
    row = {
        "metric_corrected_x": "1.5",
        "metric_corrected_y": "0.2",
        "metric_corrected_z": "-3.0",
        "anchor_center_x": "99.0",
        "anchor_center_y": "99.0",
        "anchor_center_z": "99.0",
    }
    assert gt_xyz_from_row(row) == [1.5, 0.2, -3.0]


def test_gt_from_row_recomputes_offset_metric_formula():
    # bench_v1_98.csv shape: no metric_corrected columns, but the
    # ingredients are there. anchor (0,0,0), ann_pos (2,0,0), d = 3
    # -> gt = anchor + 3 * unit(ann - anchor) = (3, 0, 0).
    row = {
        "anchor_center_x": "0.0",
        "anchor_center_y": "0.0",
        "anchor_center_z": "0.0",
        "ann_pos_x": "2.0",
        "ann_pos_y": "0.0",
        "ann_pos_z": "0.0",
        "distance_m": "3.0",
    }
    assert gt_xyz_from_row(row) == [3.0, 0.0, 0.0]


def test_gt_from_row_diagonal_direction():
    # anchor (1,1,1), ann (1+3,1,1+4) -> unit (0.6, 0, 0.8), d = 10
    # -> gt = (7, 1, 9).
    row = {
        "anchor_center_x": "1",
        "anchor_center_y": "1",
        "anchor_center_z": "1",
        "ann_pos_x": "4",
        "ann_pos_y": "1",
        "ann_pos_z": "5",
        "distance_m": "10",
    }
    gt = gt_xyz_from_row(row)
    assert gt is not None
    assert math.isclose(gt[0], 7.0)
    assert math.isclose(gt[1], 1.0)
    assert math.isclose(gt[2], 9.0)


def test_gt_from_row_degenerate_direction_matches_offset_metric():
    # ann_pos == anchor_center: direction undefined; offset_metric.py
    # leaves the point at the anchor (metric_corrected_ok = False).
    row = {
        "anchor_center_x": "2.0",
        "anchor_center_y": "0.5",
        "anchor_center_z": "1.0",
        "ann_pos_x": "2.0",
        "ann_pos_y": "0.5",
        "ann_pos_z": "1.0",
        "distance_m": "3.0",
    }
    assert gt_xyz_from_row(row) == [2.0, 0.5, 1.0]


def test_gt_from_row_zero_distance_scores_against_ann_pos():
    # Decision D5: a between-style row carries distance_m = 0, and its
    # answer is the annotated point itself. The offset formula would
    # degenerate this to the anchor center (3.0, 0.5, 1.0), which is a
    # different point and the wrong target.
    row = {
        "anchor_center_x": "3.0",
        "anchor_center_y": "0.5",
        "anchor_center_z": "1.0",
        "ann_pos_x": "1.25",
        "ann_pos_y": "0.5",
        "ann_pos_z": "-2.0",
        "distance_m": "0.0",
        "predicate": "between",
    }
    assert gt_xyz_from_row(row) == [1.25, 0.5, -2.0]


def test_gt_from_row_zero_distance_beats_stale_metric_corrected():
    # A metric_corrected_* column derived by the offline offset tool
    # holds the anchor center for a zero-distance row. D5 wins.
    row = {
        "metric_corrected_x": "3.0",
        "metric_corrected_y": "0.5",
        "metric_corrected_z": "1.0",
        "anchor_center_x": "3.0",
        "anchor_center_y": "0.5",
        "anchor_center_z": "1.0",
        "ann_pos_x": "1.25",
        "ann_pos_y": "0.5",
        "ann_pos_z": "-2.0",
        "distance_m": "0",
    }
    assert gt_xyz_from_row(row) == [1.25, 0.5, -2.0]


def test_gt_from_row_zero_distance_without_ann_pos_falls_through():
    # No annotated point to fall back on: the ordinary path applies and
    # the row is simply unscoreable.
    row = {
        "anchor_center_x": "3.0",
        "anchor_center_y": "0.5",
        "anchor_center_z": "1.0",
        "distance_m": "0.0",
    }
    assert gt_xyz_from_row(row) is None


def test_gt_from_row_nonzero_distance_still_uses_offset_formula():
    # Guard against the D5 branch swallowing ordinary offset rows.
    row = {
        "anchor_center_x": "0.0",
        "anchor_center_y": "0.0",
        "anchor_center_z": "0.0",
        "ann_pos_x": "5.0",
        "ann_pos_y": "0.0",
        "ann_pos_z": "0.0",
        "distance_m": "2.0",
    }
    assert gt_xyz_from_row(row) == [2.0, 0.0, 0.0]


def test_gt_from_row_missing_columns_returns_none():
    assert gt_xyz_from_row({}) is None
    assert gt_xyz_from_row({"anchor_center_x": "1.0"}) is None
    assert gt_xyz_from_row({"scene": "00410-v7DzfFFEpsD", "floor": "1"}) is None


# ----------------------------------------------------------------------
# score_episode: fake episode end to end
# ----------------------------------------------------------------------

def _bench_like_row():
    # Shaped like a splits/bench_v1_98.csv row (all string values).
    # anchor (0,0,0), ann (5,0,0), d = 4 -> gt = (4, 0, 0).
    return {
        "scene": "00000-testScene",
        "floor": "0",
        "distance_m": "4.0",
        "predicate": "in front of",
        "anchor_center_x": "0.0",
        "anchor_center_y": "0.0",
        "anchor_center_z": "0.0",
        "ann_pos_x": "5.0",
        "ann_pos_y": "0.0",
        "ann_pos_z": "0.0",
    }


def test_score_episode_success_case():
    # pred (4.3, 1.7, 0.4): d_h = sqrt(0.09 + 0.16) = 0.5, success at
    # 1.0 and at 0.5 (boundary), and at 2.0. d_3 also includes dy = 1.7.
    final_pred = {"target_point_xyz": [4.3, 1.7, 0.4], "confidence": 0.2}
    out = score_episode(final_pred, _bench_like_row())
    assert math.isclose(out["d_h"], 0.5)
    assert math.isclose(out["d_3"], math.sqrt(0.09 + 1.7 * 1.7 + 0.16))
    assert out["success_gt_1m"] is True
    assert out["success_gt_0p5m"] is True
    assert out["success_gt_2m"] is True
    assert out["gt_xyz"] == [4.0, 0.0, 0.0]
    # O-O node matching is a stub until item 6.
    assert out["success_gt_node"] is None


def test_score_episode_between_thresholds():
    # pred (5.5, 0.0, 0.0): d_h = 1.5 -> fails 1.0 and 0.5, passes 2.0.
    final_pred = {"target_point_xyz": [5.5, 0.0, 0.0]}
    out = score_episode(final_pred, _bench_like_row())
    assert math.isclose(out["d_h"], 1.5)
    assert out["success_gt_1m"] is False
    assert out["success_gt_0p5m"] is False
    assert out["success_gt_2m"] is True


def test_score_episode_legacy_key_prediction():
    # Legacy planner key: normalize_prediction inside score_episode must
    # pick it up (multi_agent_msp_planner wrote "target_location").
    final_pred = {"target_location": [4.0, 0.0, 0.0], "confidence": 0.95}
    out = score_episode(final_pred, _bench_like_row())
    assert out["d_h"] == 0.0
    assert out["success_gt_1m"] is True


def test_score_episode_missing_prediction_is_failure():
    # No prediction: distances None, success False (failure to answer
    # is failure), never None while GT exists.
    for pred in (None, {}, {"target_point_xyz": None}):
        out = score_episode(pred, _bench_like_row())
        assert out["d_h"] is None
        assert out["d_3"] is None
        assert out["success_gt_1m"] is False
        assert out["success_gt_0p5m"] is False
        assert out["success_gt_2m"] is False


def test_score_episode_missing_gt_scores_nothing():
    # Row with no GT columns (e.g. a grapheqa question): everything None.
    out = score_episode({"target_point_xyz": [1.0, 2.0, 3.0]}, {"scene": "x"})
    assert out["gt_xyz"] is None
    assert out["d_h"] is None
    assert out["d_3"] is None
    assert out["success_gt_1m"] is None
    assert out["success_gt_0p5m"] is None
    assert out["success_gt_2m"] is None


def test_object_answer_success_is_a_stub():
    # Item 6 replaces this hook; until then it must return None, never
    # False, so O-O episodes are not miscounted as failures.
    assert object_answer_success({"chosen_id": "object_12"}, {"anchor_sid": "12"}) is None
