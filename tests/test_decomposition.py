"""Tests for src/evals/decomposition.py (item 8: error decomposition).

Worked examples are hand-computed. Habitat convention: y is up, so every
distance is horizontal (x and z only) and the anchor sits at the origin
in most examples to keep the arithmetic readable.

Row helper convention: the bench CSV carries anchor_center_x/y/z,
ann_pos_x/y/z, distance_m; gt_xyz_from_row (item 5) turns those into the
GT point anchor + distance_m * unit(ann_pos - anchor).
"""

import math

from src.evals.decomposition import (
    FIELDS,
    FRAME_FLIP_DEG,
    RATIO_BAND,
    anchor_hat_from_pred,
    decompose_episode,
    mirrored_gt,
    summarize,
)


def make_row(anchor, ann_pos, distance_m):
    """Bench-style row dict from an anchor point, an annotated point and
    a commanded distance. The GT point derived by gt_xyz_from_row is
    anchor + distance_m * unit(ann_pos - anchor)."""
    row = {"distance_m": distance_m}
    for prefix, vec in (("anchor_center", anchor), ("ann_pos", ann_pos)):
        for axis, v in zip(("x", "y", "z"), vec):
            row[f"{prefix}_{axis}"] = v
    return row


def make_pred(xyz, pdf_anchor=None):
    """Final-prediction dict with the canonical point key, optionally
    carrying pdf_params x0/y0/z0 (the system's anchor, a_hat)."""
    pred = {"target_point_xyz": list(xyz) if xyz is not None else None}
    if pdf_anchor is not None:
        pred["pdf_params"] = {
            "x0": pdf_anchor[0],
            "y0": pdf_anchor[1],
            "z0": pdf_anchor[2],
        }
    return pred


# ----------------------------------------------------------------------
# Hand-computed worked example: anchor at origin, GT along +x
# ----------------------------------------------------------------------

def test_worked_example_orthogonal_prediction():
    # Anchor at origin, ann_pos [4, 0, 0], d_cmd 2 -> GT point [2, 0, 0].
    # Prediction [0, 0, 2]: same distance from the anchor, 90 degrees off.
    row = make_row([0.0, 0.0, 0.0], [4.0, 0.0, 0.0], 2.0)
    out = decompose_episode(make_pred([0.0, 0.0, 2.0], pdf_anchor=[0.0, 0.0, 0.0]), row)

    # e_r = |dist(pred, anchor) - d_cmd| = |2 - 2| = 0.
    assert out["e_r"] == 0.0
    assert out["e_r_reason"] is None
    # e_theta = angle([0,0,2], [2,0,0]) = 90 degrees.
    assert math.isclose(out["e_theta_deg"], 90.0, abs_tol=1e-9)
    # 90 < 150: not a frame flip.
    assert out["frame_flip"] is False
    # System anchor equals GT anchor: e_a = 0.
    assert out["e_a"] == 0.0
    # ratio = 2 / 2 = 1, inside the band.
    assert out["ratio"] == 1.0
    assert out["ratio_in_band"] is True
    # d_h(pred, gt) = sqrt(8) > 1 and the mirrored GT [-2, 0, 0] is just
    # as far, so the oracle does not rescue this one.
    assert out["success_best_of_frames_1m"] is False


def test_worked_example_radial_error_nonzero():
    # Same geometry but d_cmd 3: prediction still 2 m from the anchor,
    # so e_r = |2 - 3| = 1 and ratio = 2/3.
    row = make_row([0.0, 0.0, 0.0], [4.0, 0.0, 0.0], 3.0)
    out = decompose_episode(make_pred([0.0, 0.0, 2.0]), row)
    assert math.isclose(out["e_r"], 1.0, abs_tol=1e-9)
    assert math.isclose(out["ratio"], 2.0 / 3.0, abs_tol=1e-12)
    assert out["ratio_in_band"] is True


def test_distances_are_horizontal():
    # A prediction 2 m from the anchor horizontally but 5 m up: the y
    # component must not enter e_r or the ratio.
    row = make_row([0.0, 0.0, 0.0], [4.0, 0.0, 0.0], 2.0)
    out = decompose_episode(make_pred([0.0, 5.0, 2.0]), row)
    assert out["e_r"] == 0.0
    assert out["ratio"] == 1.0
    # e_a horizontal too: a_hat directly above the GT anchor is 0 m off.
    out2 = decompose_episode(
        make_pred([0.0, 0.0, 2.0], pdf_anchor=[0.0, 9.0, 0.0]), row
    )
    assert out2["e_a"] == 0.0


# ----------------------------------------------------------------------
# Mirrored frame: the flip rescues SR
# ----------------------------------------------------------------------

def test_mirrored_gt_construction():
    # Reflect through the anchor horizontally; y kept from the GT point.
    assert mirrored_gt([2.0, 0.7, 0.0], [0.0, 0.0, 0.0]) == [-2.0, 0.7, 0.0]
    assert mirrored_gt([3.0, 1.0, 5.0], [1.0, 0.0, 2.0]) == [-1.0, 1.0, -1.0]


def test_frame_flip_rescued_by_best_of_frames():
    # GT [2, 0, 0], prediction [-2, 0, 0]: exactly the opposite frame.
    row = make_row([0.0, 0.0, 0.0], [4.0, 0.0, 0.0], 2.0)
    out = decompose_episode(make_pred([-2.0, 0.0, 0.0]), row)

    # e_theta = 180 >= 150: frame flip.
    assert math.isclose(out["e_theta_deg"], 180.0, abs_tol=1e-9)
    assert out["frame_flip"] is True
    # Plain SR fails: d_h(pred, gt) = 4 m. The mirrored GT is [-2, 0, 0],
    # d_h(pred, mirror) = 0 m, so the best-of-frames oracle succeeds.
    assert out["success_best_of_frames_1m"] is True
    assert out["success_best_of_frames_1m_reason"] is None
    # Radial and ratio components are perfect: the error is pure frame.
    assert out["e_r"] == 0.0
    assert out["ratio"] == 1.0


def test_frame_flip_threshold_boundary_inclusive():
    # Prediction at exactly 150 degrees from the GT direction, same
    # 2 m radius: yaw 150 deg -> [2cos150, 0, 2sin150].
    row = make_row([0.0, 0.0, 0.0], [4.0, 0.0, 0.0], 2.0)
    ang = math.radians(FRAME_FLIP_DEG)
    pred = [2.0 * math.cos(ang), 0.0, 2.0 * math.sin(ang)]
    out = decompose_episode(make_pred(pred), row)
    assert math.isclose(out["e_theta_deg"], 150.0, abs_tol=1e-9)
    assert out["frame_flip"] is True


# ----------------------------------------------------------------------
# Ratio band edges, inclusive at exactly 0.5 and 2.0
# ----------------------------------------------------------------------

def test_ratio_band_edges_inclusive():
    row = make_row([0.0, 0.0, 0.0], [4.0, 0.0, 0.0], 2.0)
    # dist(pred, anchor) = 1 -> ratio exactly 0.5: inside.
    lo = decompose_episode(make_pred([1.0, 0.0, 0.0]), row)
    assert lo["ratio"] == RATIO_BAND[0]
    assert lo["ratio_in_band"] is True
    # dist = 4 -> ratio exactly 2.0: inside.
    hi = decompose_episode(make_pred([4.0, 0.0, 0.0]), row)
    assert hi["ratio"] == RATIO_BAND[1]
    assert hi["ratio_in_band"] is True
    # Just outside on both sides.
    below = decompose_episode(make_pred([0.98, 0.0, 0.0]), row)
    assert below["ratio_in_band"] is False
    above = decompose_episode(make_pred([4.02, 0.0, 0.0]), row)
    assert above["ratio_in_band"] is False


def test_ratio_undefined_for_nonpositive_command():
    row = make_row([0.0, 0.0, 0.0], [4.0, 0.0, 0.0], 0.0)
    out = decompose_episode(make_pred([1.0, 0.0, 0.0]), row)
    assert out["ratio"] is None
    assert out["ratio_in_band"] is None
    assert "not positive" in out["ratio_reason"]
    # e_r is still defined: |1 - 0| = 1.
    assert out["e_r"] == 1.0


# ----------------------------------------------------------------------
# Anchor error and its source
# ----------------------------------------------------------------------

def test_anchor_error_hand_computed():
    # GT anchor at [1, 0, 2], system anchor at [4, 0, 6]: 3-4-5 triangle
    # in the horizontal plane -> e_a = 5.
    row = make_row([1.0, 0.0, 2.0], [5.0, 0.0, 2.0], 2.0)
    out = decompose_episode(
        make_pred([3.0, 0.0, 2.0], pdf_anchor=[4.0, 0.0, 6.0]), row
    )
    assert out["e_a"] == 5.0
    assert out["e_a_reason"] is None


def test_anchor_hat_sources_and_missing():
    # Canonical x0/y0/z0.
    a, reason = anchor_hat_from_pred({"pdf_params": {"x0": 1.0, "y0": 2.0, "z0": 3.0}})
    assert a == [1.0, 2.0, 3.0] and reason is None
    # mu_x/mu_y/mu_z fallback.
    a, reason = anchor_hat_from_pred({"pdf_params": {"mu_x": 4.0, "mu_y": 5.0, "mu_z": 6.0}})
    assert a == [4.0, 5.0, 6.0] and reason is None
    # No pdf_params at all.
    a, reason = anchor_hat_from_pred({"target_point_xyz": [0.0, 0.0, 0.0]})
    assert a is None and "no pdf_params" in reason
    # pdf_params without anchor fields.
    a, reason = anchor_hat_from_pred({"pdf_params": {"kappa": 2.0}})
    assert a is None and "no usable anchor" in reason


def test_e_a_missing_reason_counted_not_fabricated():
    # Prediction without pdf_params: e_a None with a reason, everything
    # else still computed.
    row = make_row([0.0, 0.0, 0.0], [4.0, 0.0, 0.0], 2.0)
    out = decompose_episode(make_pred([2.0, 0.0, 0.0]), row)
    assert out["e_a"] is None
    assert "no pdf_params" in out["e_a_reason"]
    assert out["e_r"] == 0.0


# ----------------------------------------------------------------------
# Missing anchor and missing prediction
# ----------------------------------------------------------------------

def test_missing_anchor():
    # Row with a directly given GT point but no anchor columns.
    row = {
        "metric_corrected_x": 2.0,
        "metric_corrected_y": 0.0,
        "metric_corrected_z": 0.0,
        "distance_m": 2.0,
    }
    out = decompose_episode(make_pred([2.0, 0.0, 0.4]), row)
    for name in ("e_r", "e_theta_deg", "e_a", "ratio", "ratio_in_band", "frame_flip"):
        assert out[name] is None, name
        assert out[name + "_reason"] is not None, name
    # Best-of-frames falls back to the GT frame only, with a reason:
    # d_h = 0.4 <= 1.0 m.
    assert out["success_best_of_frames_1m"] is True
    assert "mirrored frame undefined" in out["success_best_of_frames_1m_reason"]


def test_missing_prediction():
    row = make_row([0.0, 0.0, 0.0], [4.0, 0.0, 0.0], 2.0)
    for pred in (None, {}, {"target_point_xyz": None}):
        out = decompose_episode(pred, row)
        for name in ("e_r", "e_theta_deg", "ratio", "ratio_in_band", "frame_flip"):
            assert out[name] is None, name
            assert out[name + "_reason"] is not None, name
        # Failure to answer is failure for the binary oracle.
        assert out["success_best_of_frames_1m"] is False
        assert "no predicted point" in out["success_best_of_frames_1m_reason"]


def test_missing_gt_scores_nothing_binary():
    out = decompose_episode(make_pred([1.0, 0.0, 0.0]), {})
    assert out["success_best_of_frames_1m"] is None
    assert "no GT point" in out["success_best_of_frames_1m_reason"]
    assert out["e_theta_deg"] is None


def test_all_fields_always_present():
    out = decompose_episode(None, {})
    for name in FIELDS:
        assert name in out
        assert name + "_reason" in out


# ----------------------------------------------------------------------
# summarize
# ----------------------------------------------------------------------

def test_summarize_counts_and_rates():
    row = make_row([0.0, 0.0, 0.0], [4.0, 0.0, 0.0], 2.0)
    rows = [
        # Perfect prediction with a_hat: everything defined.
        decompose_episode(make_pred([2.0, 0.0, 0.0], pdf_anchor=[0.0, 0.0, 0.0]), row),
        # Frame-flipped prediction, rescued by the oracle, no a_hat.
        decompose_episode(make_pred([-2.0, 0.0, 0.0]), row),
        # Out-of-band prediction: ratio 3.0, no a_hat.
        decompose_episode(make_pred([6.0, 0.0, 0.0]), row),
        # Missing prediction: only the binary oracle is defined (False).
        decompose_episode(None, row),
    ]
    s = summarize(rows)
    assert s["n"] == 4

    # e_r over the 3 rows with predictions: values 0, 0, 4.
    assert s["e_r_n"] == 3 and s["e_r_missing"] == 1
    assert math.isclose(s["e_r_mean"], 4.0 / 3.0, abs_tol=1e-12)
    assert s["e_r_median"] == 0.0

    # e_theta over 3 rows: 0, 180, 0.
    assert s["e_theta_deg_n"] == 3 and s["e_theta_deg_missing"] == 1
    assert math.isclose(s["e_theta_deg_median"], 0.0, abs_tol=1e-9)

    # e_a exists only on the first row.
    assert s["e_a_n"] == 1 and s["e_a_missing"] == 3
    assert s["e_a_mean"] == 0.0

    # ratio over 3 rows: 1, 1, 3 -> band rate 2/3.
    assert s["ratio_n"] == 3 and s["ratio_missing"] == 1
    assert math.isclose(s["ratio_median"], 1.0, abs_tol=1e-12)
    assert math.isclose(s["ratio_band_rate"], 2.0 / 3.0, abs_tol=1e-12)
    assert s["ratio_band_n"] == 3 and s["ratio_band_missing"] == 1

    # Frame flips: 1 of 3 scored rows.
    assert math.isclose(s["frame_flip_rate"], 1.0 / 3.0, abs_tol=1e-12)
    assert s["frame_flip_n"] == 3 and s["frame_flip_missing"] == 1

    # Best-of-frames: successes on rows 1 and 2, failures on rows 3
    # (4 m off in both frames) and 4 (missing prediction) -> 2/4.
    assert s["sr_best_of_frames_1m"] == 0.5
    assert s["best_of_frames_n"] == 4 and s["best_of_frames_missing"] == 0


def test_summarize_empty_and_all_missing():
    s = summarize([])
    assert s["n"] == 0
    assert s["e_r_mean"] is None and s["e_r_median"] is None
    assert s["frame_flip_rate"] is None
    assert s["sr_best_of_frames_1m"] is None

    # Rows where nothing could be computed at all.
    s2 = summarize([decompose_episode(None, {})])
    assert s2["n"] == 1
    assert s2["e_r_n"] == 0 and s2["e_r_missing"] == 1
    assert s2["best_of_frames_n"] == 0 and s2["best_of_frames_missing"] == 1


def test_summarize_median_even_count():
    row = make_row([0.0, 0.0, 0.0], [4.0, 0.0, 0.0], 2.0)
    rows = [
        decompose_episode(make_pred([1.0, 0.0, 0.0]), row),  # e_r = 1
        decompose_episode(make_pred([5.0, 0.0, 0.0]), row),  # e_r = 3
    ]
    s = summarize(rows)
    assert s["e_r_median"] == 2.0
