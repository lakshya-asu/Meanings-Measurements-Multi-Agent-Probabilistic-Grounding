"""Tests for src/evals/angular.py (harness item 7).

All expected values are hand computed. Frame: Habitat world, y up,
yaw = atan2(z, x) from +x toward +z, pitch = elevation above the x-z
plane.
"""

import math

from src.evals.angular import (
    angular_error_from_vectors,
    angular_errors_from_points,
    pitch_error_deg,
    wrap_deg,
    yaw_error_deg,
    yaw_pitch_from_vector,
)


class TestWrapDeg:
    def test_identity_inside_range(self):
        assert wrap_deg(0.0) == 0.0
        assert wrap_deg(90.0) == 90.0
        assert wrap_deg(-90.0) == -90.0

    def test_half_turn_maps_to_plus_180(self):
        # Interval is (-180, 180], so both boundaries land on +180.
        assert wrap_deg(180.0) == 180.0
        assert wrap_deg(-180.0) == 180.0

    def test_wraps_beyond_half_turn(self):
        assert wrap_deg(190.0) == -170.0
        assert wrap_deg(-190.0) == 170.0
        assert wrap_deg(360.0) == 0.0
        assert wrap_deg(720.0 + 10.0) == 10.0


class TestYawErrorDeg:
    def test_wrap_around_359_vs_1_is_2(self):
        # 359 deg and 1 deg are 2 deg apart across the wrap, not 358.
        err, reason = yaw_error_deg(359.0, 1.0)
        assert reason is None
        assert math.isclose(err, 2.0)
        err, reason = yaw_error_deg(1.0, 359.0)
        assert math.isclose(err, 2.0)

    def test_plain_difference(self):
        err, reason = yaw_error_deg(30.0, -15.0)
        assert reason is None
        assert math.isclose(err, 45.0)

    def test_max_is_180(self):
        err, _ = yaw_error_deg(0.0, 180.0)
        assert math.isclose(err, 180.0)

    def test_none_input_gives_none_with_reason(self):
        err, reason = yaw_error_deg(None, 10.0)
        assert err is None
        assert "None" in reason

    def test_nan_input_gives_none_with_reason(self):
        err, reason = yaw_error_deg(float("nan"), 10.0)
        assert err is None
        assert "NaN" in reason


class TestPitchErrorDeg:
    def test_elevation_difference(self):
        # 30 deg up vs 15 deg down: 45 deg apart.
        err, reason = pitch_error_deg(30.0, -15.0)
        assert reason is None
        assert math.isclose(err, 45.0)

    def test_up_vs_down_is_180(self):
        # Straight up vs straight down: 180, not folded to 0.
        err, _ = pitch_error_deg(90.0, -90.0)
        assert math.isclose(err, 180.0)

    def test_none_input(self):
        err, reason = pitch_error_deg(10.0, None)
        assert err is None
        assert reason is not None


class TestYawPitchFromVector:
    def test_plus_x_axis(self):
        (yaw, pitch), reason = yaw_pitch_from_vector([1.0, 0.0, 0.0])
        assert reason is None
        assert math.isclose(yaw, 0.0, abs_tol=1e-12)
        assert math.isclose(pitch, 0.0, abs_tol=1e-12)

    def test_plus_z_axis_is_yaw_90(self):
        # In this frame yaw is measured from +x toward +z, so +z is 90.
        (yaw, pitch), reason = yaw_pitch_from_vector([0.0, 0.0, 1.0])
        assert reason is None
        assert math.isclose(yaw, 90.0)
        assert math.isclose(pitch, 0.0, abs_tol=1e-12)

    def test_x_axis_vs_z_axis_yaw_geometry(self):
        # [1,0,0] vs [0,0,1]: both level, yaws 0 and 90, so the yaw error
        # is 90 and the full 3D angle is also 90.
        (yaw_a, _), _ = yaw_pitch_from_vector([1.0, 0.0, 0.0])
        (yaw_b, _), _ = yaw_pitch_from_vector([0.0, 0.0, 1.0])
        err, _ = yaw_error_deg(yaw_a, yaw_b)
        assert math.isclose(err, 90.0)
        full, _ = angular_error_from_vectors([1.0, 0.0, 0.0], [0.0, 0.0, 1.0])
        assert math.isclose(full, 90.0)

    def test_45_degree_climb(self):
        # Level distance 1 (along x), rise 1 (along y): a 45 deg climb.
        (yaw, pitch), reason = yaw_pitch_from_vector([1.0, 1.0, 0.0])
        assert reason is None
        assert math.isclose(yaw, 0.0, abs_tol=1e-12)
        assert math.isclose(pitch, 45.0)

    def test_straight_up_vector(self):
        # Vertical: pitch 90, yaw takes the documented atan2(0,0)=0
        # convention.
        (yaw, pitch), reason = yaw_pitch_from_vector([0.0, 2.0, 0.0])
        assert reason is None
        assert math.isclose(pitch, 90.0)
        assert math.isclose(yaw, 0.0, abs_tol=1e-12)

    def test_zero_vector_returns_none(self):
        result, reason = yaw_pitch_from_vector([0.0, 0.0, 0.0])
        assert result is None
        assert "zero" in reason

    def test_none_and_nan_and_shape(self):
        for bad in (None, [1.0, 2.0], [1.0, float("nan"), 0.0], "abc"):
            result, reason = yaw_pitch_from_vector(bad)
            assert result is None, bad
            assert isinstance(reason, str) and reason


class TestAngularErrorFromVectors:
    def test_opposite_vectors_are_180(self):
        err, reason = angular_error_from_vectors([1.0, 2.0, 3.0], [-1.0, -2.0, -3.0])
        assert reason is None
        assert math.isclose(err, 180.0)

    def test_identical_vectors_are_0(self):
        # Parallel vectors: acos of a clamped dot product can leave a
        # tiny floating point residue, hence the loose tolerance.
        err, _ = angular_error_from_vectors([0.3, 0.4, 0.5], [0.6, 0.8, 1.0])
        assert math.isclose(err, 0.0, abs_tol=1e-3)

    def test_zero_vector_returns_none(self):
        err, reason = angular_error_from_vectors([0.0, 0.0, 0.0], [1.0, 0.0, 0.0])
        assert err is None
        assert "zero" in reason
        err, reason = angular_error_from_vectors([1.0, 0.0, 0.0], [0.0, 0.0, 0.0])
        assert err is None
        assert "zero" in reason

    def test_nan_component_returns_none(self):
        err, reason = angular_error_from_vectors([1.0, 0.0, float("nan")], [1.0, 0.0, 0.0])
        assert err is None
        assert "NaN" in reason


class TestAngularErrorsFromPoints:
    def test_worked_example(self):
        # Anchor at origin. GT 2 m along +x, level. Prediction 2 m away
        # at yaw 90 (along +z), level. Yaw error 90, pitch error 0, full
        # angle 90.
        out = angular_errors_from_points([0.0, 0.0, 2.0], [2.0, 0.0, 0.0], [0.0, 0.0, 0.0])
        assert out["reason"] is None
        assert math.isclose(out["yaw_error_deg"], 90.0)
        assert math.isclose(out["pitch_error_deg"], 0.0, abs_tol=1e-12)
        assert math.isclose(out["angular_error_deg"], 90.0)

    def test_climb_example(self):
        # GT level along +x; prediction along +x but climbing 45 deg.
        out = angular_errors_from_points([1.0, 1.0, 0.0], [2.0, 0.0, 0.0], [0.0, 0.0, 0.0])
        assert out["reason"] is None
        assert math.isclose(out["yaw_error_deg"], 0.0, abs_tol=1e-12)
        assert math.isclose(out["pitch_error_deg"], 45.0)
        assert math.isclose(out["angular_error_deg"], 45.0)

    def test_nonzero_anchor(self):
        # Same geometry translated by the anchor position.
        a = [5.0, 1.0, -3.0]
        out = angular_errors_from_points(
            [a[0], a[1], a[2] + 2.0], [a[0] + 2.0, a[1], a[2]], a
        )
        assert out["reason"] is None
        assert math.isclose(out["yaw_error_deg"], 90.0)

    def test_pred_at_anchor_is_unavailable(self):
        out = angular_errors_from_points([0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 0.0, 0.0])
        assert out["yaw_error_deg"] is None
        assert out["pitch_error_deg"] is None
        assert out["angular_error_deg"] is None
        assert "pred direction" in out["reason"]

    def test_missing_pred_is_unavailable(self):
        out = angular_errors_from_points(None, [2.0, 0.0, 0.0], [0.0, 0.0, 0.0])
        assert out["yaw_error_deg"] is None
        assert out["reason"] is not None

    def test_never_raises_on_garbage(self):
        out = angular_errors_from_points("x", {"a": 1}, [1, 2])
        assert out["yaw_error_deg"] is None
        assert out["reason"] is not None
