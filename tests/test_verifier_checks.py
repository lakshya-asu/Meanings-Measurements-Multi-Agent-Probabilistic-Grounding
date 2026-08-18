"""Tests for src/verification/checks.py (P0 fix 3: verifier that verifies).

Stdlib only: the navmesh is injected as a fake snap function, no
habitat required. Covers the four programmatic checks, the skip
semantics, and the run_checks aggregation.
"""

import math

from src.verification.checks import (
    schema_valid,
    in_scene_bounds,
    on_navmesh,
    distance_consistency,
    run_checks,
    failed_reasons,
    CHECK_NAMES,
)


# ---------------------------------------------------------------------------
# schema_valid
# ---------------------------------------------------------------------------

def test_schema_valid_pass():
    ok, reason = schema_valid({"theta": 1.2, "phi": math.pi / 2}, ("theta", "phi"))
    assert ok, reason


def test_schema_valid_missing_field():
    ok, reason = schema_valid({"theta": 1.2}, ("theta", "phi"))
    assert not ok
    assert "phi" in reason


def test_schema_valid_none_field():
    ok, _ = schema_valid({"theta": 1.2, "phi": None}, ("theta", "phi"))
    assert not ok


def test_schema_valid_non_finite():
    ok, reason = schema_valid({"theta": float("nan"), "phi": 0.0}, ("theta", "phi"))
    assert not ok
    assert "theta" in reason
    ok, _ = schema_valid({"theta": float("inf"), "phi": 0.0}, ("theta", "phi"))
    assert not ok


def test_schema_valid_not_a_dict():
    ok, _ = schema_valid(["theta"], ("theta",))
    assert not ok


# ---------------------------------------------------------------------------
# in_scene_bounds (scene AABB + 5 m margin)
# ---------------------------------------------------------------------------

_MIN = [-2.0, 0.0, -3.0]
_MAX = [4.0, 2.5, 5.0]


def test_bounds_inside():
    ok, reason = in_scene_bounds([1.0, 1.0, 1.0], _MIN, _MAX)
    assert ok, reason


def test_bounds_inside_margin():
    # 5 m margin: x = 8.9 is within max 4.0 + 5.0.
    ok, _ = in_scene_bounds([8.9, 1.0, 1.0], _MIN, _MAX)
    assert ok


def test_bounds_outside_margin():
    ok, reason = in_scene_bounds([9.1, 1.0, 1.0], _MIN, _MAX)
    assert not ok
    assert "axis 0" in reason


def test_bounds_outside_negative():
    ok, _ = in_scene_bounds([1.0, 1.0, -8.1], _MIN, _MAX)
    assert not ok


def test_bounds_invalid_point():
    ok, _ = in_scene_bounds([float("nan"), 0.0, 0.0], _MIN, _MAX)
    assert not ok
    ok, _ = in_scene_bounds(None, _MIN, _MAX)
    assert not ok


# ---------------------------------------------------------------------------
# on_navmesh (fake snap function injected)
# ---------------------------------------------------------------------------

def _snap_identity(p):
    return list(p)


def _snap_offset(dy):
    def fn(p):
        return [p[0], p[1] + dy, p[2]]
    return fn


def test_navmesh_snap_within_tolerance():
    ok, reason, skipped = on_navmesh([1.0, 0.5, 1.0], _snap_identity)
    assert ok and not skipped, reason


def test_navmesh_snap_at_edge():
    ok, _, skipped = on_navmesh([1.0, 0.5, 1.0], _snap_offset(0.19))
    assert ok and not skipped


def test_navmesh_snap_too_far():
    ok, reason, skipped = on_navmesh([1.0, 0.5, 1.0], _snap_offset(0.5))
    assert not ok and not skipped
    assert "tolerance" in reason


def test_navmesh_snap_failure():
    ok, _, skipped = on_navmesh([1.0, 0.5, 1.0], lambda p: None)
    assert not ok and not skipped


def test_navmesh_snap_raises_is_fail():
    def broken(p):
        raise RuntimeError("no navmesh loaded")
    ok, reason, skipped = on_navmesh([1.0, 0.5, 1.0], broken)
    assert not ok and not skipped
    assert "raised" in reason


def test_navmesh_skipped_outside_container():
    ok, reason, skipped = on_navmesh([1.0, 0.5, 1.0], None)
    assert ok and skipped
    assert "skipped" in reason


# ---------------------------------------------------------------------------
# distance_consistency (|r - d0| <= 3 sigma_m)
# ---------------------------------------------------------------------------

def test_distance_exact():
    ok, _, skipped = distance_consistency([2.0, 0.0, 0.0], [0.0, 0.0, 0.0], 2.0, 0.15)
    assert ok and not skipped


def test_distance_within_three_sigma():
    # r = 2.4, d0 = 2.0, err 0.4 < 3 * 0.15 = 0.45
    ok, _, skipped = distance_consistency([2.4, 0.0, 0.0], [0.0, 0.0, 0.0], 2.0, 0.15)
    assert ok and not skipped


def test_distance_outside_three_sigma():
    # r = 2.5, err 0.5 > 0.45
    ok, reason, skipped = distance_consistency([2.5, 0.0, 0.0], [0.0, 0.0, 0.0], 2.0, 0.15)
    assert not ok and not skipped
    assert "sigma_m" in reason


def test_distance_skipped_when_no_literal():
    # d0 None: the metric kernel was omitted, nothing to check.
    ok, reason, skipped = distance_consistency([9.0, 9.0, 9.0], [0.0, 0.0, 0.0], None, 0.15)
    assert ok and skipped
    assert "skipped" in reason


def test_distance_skipped_when_sigma_unusable():
    ok, _, skipped = distance_consistency([2.0, 0.0, 0.0], [0.0, 0.0, 0.0], 2.0, None)
    assert ok and skipped
    ok, _, skipped = distance_consistency([2.0, 0.0, 0.0], [0.0, 0.0, 0.0], 2.0, 0.0)
    assert ok and skipped


def test_distance_invalid_prediction_fails():
    ok, _, skipped = distance_consistency(None, [0.0, 0.0, 0.0], 2.0, 0.15)
    assert not ok and not skipped


# ---------------------------------------------------------------------------
# run_checks aggregation
# ---------------------------------------------------------------------------

def _good_kwargs():
    return dict(
        spatial_payload={"theta": 0.4, "phi": math.pi / 2},
        required_fields=("theta", "phi"),
        prediction_xyz=[2.0, 0.5, 0.0],
        anchor_xyz=[0.0, 0.5, 0.0],
        d0_m=2.0,
        sigma_m=0.15,
        scene_min=_MIN,
        scene_max=_MAX,
        navmesh_snap_fn=_snap_identity,
    )


def test_run_checks_all_pass():
    result = run_checks(**_good_kwargs())
    assert result["all_ok"]
    for name in CHECK_NAMES:
        assert result[name]["ok"], result[name]
        assert not result[name]["skipped"]
        assert isinstance(result[name]["reason"], str)
    assert failed_reasons(result) == "no failed checks"


def test_run_checks_one_failure_fails_gate():
    kwargs = _good_kwargs()
    kwargs["prediction_xyz"] = [40.0, 0.5, 0.0]  # out of bounds and radius
    result = run_checks(**kwargs)
    assert not result["all_ok"]
    assert not result["in_scene_bounds"]["ok"]
    assert not result["distance_consistency"]["ok"]
    summary = failed_reasons(result)
    assert "in_scene_bounds" in summary
    assert "distance_consistency" in summary


def test_run_checks_skips_do_not_fail_gate():
    kwargs = _good_kwargs()
    kwargs["navmesh_snap_fn"] = None  # outside the container
    kwargs["d0_m"] = None  # no metric literal
    result = run_checks(**kwargs)
    assert result["all_ok"]
    assert result["on_navmesh"]["skipped"]
    assert result["distance_consistency"]["skipped"]


def test_run_checks_records_all_four_booleans():
    result = run_checks(**_good_kwargs())
    assert set(CHECK_NAMES).issubset(result.keys())
    for name in CHECK_NAMES:
        assert set(("ok", "reason", "skipped")).issubset(result[name].keys())


def test_run_checks_schema_failure():
    kwargs = _good_kwargs()
    kwargs["spatial_payload"] = {"theta": 0.4}  # phi missing
    result = run_checks(**kwargs)
    assert not result["all_ok"]
    assert not result["schema_valid"]["ok"]


def test_run_checks_without_prediction_skips_geometry():
    result = run_checks(
        spatial_payload={"theta": 0.4, "phi": 1.5},
        required_fields=("theta", "phi"),
        prediction_xyz=None,
    )
    assert result["all_ok"]
    assert result["in_scene_bounds"]["skipped"]
    assert result["on_navmesh"]["skipped"]
    assert result["distance_consistency"]["skipped"]
