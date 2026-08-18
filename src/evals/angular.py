"""Angular error metrics for MAPG evaluation (harness item 7).

Single canonical implementation of the yaw and pitch angular errors the
paper reports (Table I) and of the full 3D angular error e_theta from the
metric spec (flux-work/mapg-paper/research/metrics.md, section 2.1):

    e_theta = angle(p_hat - a, p_gt - a) in degrees, in [0, 180]

where a is the anchor centroid, p_hat the predicted point and p_gt the
ground-truth point. Yaw error and pitch error are the horizontal and
vertical decomposition of that same pair of direction vectors.

Frame and angle conventions (Habitat world frame, y is up):

- yaw(v) = degrees(atan2(v_z, v_x)): the angle of the vector's projection
  onto the horizontal x-z plane, measured from the +x axis toward the +z
  axis, in (-180, 180]. This matches the repo's own world-angle
  convention theta_world in src/planners/vlm_planner_msp.py
  (_unit_dir_from_theta_phi: x = cos(theta) * sin(phi),
  z = sin(theta) * sin(phi)), which is also the convention the dataset
  column ann_yaw_rad (anchor-front yaw, world radians) is compared
  against in that planner.
- pitch(v) = degrees(atan2(v_y, hypot(v_x, v_z))): elevation above the
  horizontal plane, in [-90, 90]. Positive is up. The planners carry the
  polar angle phi measured from +y (phi = pi/2 is level); the relation is
  pitch_deg = 90 - degrees(phi).
- yaw error = |wrap_deg(pred_yaw - gt_yaw)| in [0, 180].
- pitch error = |pred_pitch - gt_pitch|. With pitches as elevations in
  [-90, 90] this lies in [0, 180]. metrics.md does not define a separate
  pitch range; its angular errors live in [0, 180] (section 2.1), so the
  [0, 180] elevation-difference form is used rather than folding to
  [0, 90], which would silently equate an up error with a down error.

Degenerate-input policy (metrics.md section 4: angular error for a failed
query is undefined and must not be imputed): every public function except
wrap_deg returns a (value, reason) tuple. On success reason is None; on a
degenerate input (None, NaN, infinity, wrong shape, zero vector) the
value is None and reason is a short string. Nothing here ever raises.

Stdlib math only on purpose: importable and testable outside the
container, no numpy.

TODO(orientation adapter): if a future results schema carries an explicit
predicted orientation (for example pdf_params theta0/phi0 from
src/msp/pdf.py, or a predicted facing yaw) instead of a predicted point,
add an adapter here that converts it to the conventions above and feeds
yaw_error_deg / pitch_error_deg. The current evaluator derives both
orientations from point geometry (anchor -> point vectors), so no adapter
is needed for the bench CSV, whose only GT orientation column is
ann_yaw_rad (anchor-front yaw; the GT target direction is derived from
anchor and GT point instead).
"""

import math
from typing import Any, Optional, Sequence, Tuple

# Vectors shorter than this are treated as zero (no direction).
_EPS = 1e-9

VecResult = Tuple[Optional[Tuple[float, float]], Optional[str]]
NumResult = Tuple[Optional[float], Optional[str]]


def wrap_deg(a: float) -> float:
    """Wrap an angle in degrees to the interval (-180, 180].

    Pure helper; expects a finite number. NaN propagates as NaN.
    """
    r = a % 360.0
    if r > 180.0:
        r -= 360.0
    return r


def _bad_number(x: Any) -> Optional[str]:
    """Return a reason string if x is not a usable finite number."""
    if x is None:
        return "input is None"
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        return "input is not a number (got %s)" % type(x).__name__
    if math.isnan(x):
        return "input is NaN"
    if math.isinf(x):
        return "input is infinite"
    return None


def _coerce_vec3(v: Any, name: str) -> Tuple[Optional[Tuple[float, float, float]], Optional[str]]:
    """Validate v as a 3-vector of finite numbers. Never raises."""
    if v is None:
        return None, "%s is None" % name
    try:
        items = list(v)
    except TypeError:
        return None, "%s is not a sequence" % name
    if len(items) != 3:
        return None, "%s does not have 3 components (got %d)" % (name, len(items))
    out = []
    for c in items:
        reason = _bad_number(c)
        if reason is not None:
            return None, "%s component: %s" % (name, reason)
        out.append(float(c))
    return (out[0], out[1], out[2]), None


def yaw_error_deg(pred_yaw: Any, gt_yaw: Any) -> NumResult:
    """Absolute wrapped yaw difference in degrees, in [0, 180].

    Inputs are yaw angles in degrees (any real value; wrapping handles
    turns). Returns (error, None) or (None, reason).
    """
    for label, val in (("pred_yaw", pred_yaw), ("gt_yaw", gt_yaw)):
        reason = _bad_number(val)
        if reason is not None:
            return None, "%s: %s" % (label, reason)
    return abs(wrap_deg(float(pred_yaw) - float(gt_yaw))), None


def pitch_error_deg(pred_pitch: Any, gt_pitch: Any) -> NumResult:
    """Absolute pitch (elevation) difference in degrees.

    Inputs are elevations in [-90, 90] (positive up), so the result lies
    in [0, 180]. See the module docstring for why [0, 180] and not
    [0, 90]. Returns (error, None) or (None, reason).
    """
    for label, val in (("pred_pitch", pred_pitch), ("gt_pitch", gt_pitch)):
        reason = _bad_number(val)
        if reason is not None:
            return None, "%s: %s" % (label, reason)
    return abs(float(pred_pitch) - float(gt_pitch)), None


def yaw_pitch_from_vector(v: Any) -> VecResult:
    """Yaw and pitch in degrees of a 3-vector in the Habitat frame (y up).

    yaw = degrees(atan2(v_z, v_x)), in (-180, 180], measured in the x-z
    plane from +x toward +z. pitch = degrees(atan2(v_y, hypot(v_x, v_z))),
    in [-90, 90], elevation above the horizontal plane.

    For a vertical vector (v_x = v_z = 0, v_y != 0) the yaw is not
    defined by the geometry; atan2(0, 0) = 0 is kept as the convention,
    so a straight-up vector reports (yaw=0.0, pitch=90.0).

    Returns ((yaw_deg, pitch_deg), None) on success, or (None, reason)
    for None input, wrong shape, NaN or infinite components, or a zero
    (or near-zero, < 1e-9) vector, which has no direction.
    """
    vec, reason = _coerce_vec3(v, "vector")
    if reason is not None:
        return None, reason
    x, y, z = vec
    if math.sqrt(x * x + y * y + z * z) < _EPS:
        return None, "zero vector has no direction"
    yaw = math.degrees(math.atan2(z, x))
    pitch = math.degrees(math.atan2(y, math.hypot(x, z)))
    return (yaw, pitch), None


def angular_error_from_vectors(pred_v: Any, gt_v: Any) -> NumResult:
    """Full 3D angle between two direction vectors, in degrees [0, 180].

    This is e_theta from metrics.md section 2.1 when called with
    (p_hat - a, p_gt - a). Returns (angle_deg, None) or (None, reason)
    on degenerate input (see yaw_pitch_from_vector).
    """
    a, reason = _coerce_vec3(pred_v, "pred vector")
    if reason is not None:
        return None, reason
    b, reason = _coerce_vec3(gt_v, "gt vector")
    if reason is not None:
        return None, reason
    na = math.sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2])
    nb = math.sqrt(b[0] * b[0] + b[1] * b[1] + b[2] * b[2])
    if na < _EPS:
        return None, "pred vector is zero, no direction"
    if nb < _EPS:
        return None, "gt vector is zero, no direction"
    dot = (a[0] * b[0] + a[1] * b[1] + a[2] * b[2]) / (na * nb)
    dot = max(-1.0, min(1.0, dot))
    return math.degrees(math.acos(dot)), None


def angular_errors_from_points(pred_p: Any, gt_p: Any, anchor_p: Any):
    """Yaw, pitch and full angular error of a prediction versus GT.

    Both orientations are derived from point geometry: the predicted
    direction is (pred_p - anchor_p) and the GT direction is
    (gt_p - anchor_p), per metrics.md section 2.1. All three inputs are
    3-vectors in the Habitat world frame.

    Returns a dict:
        yaw_error_deg:     float or None
        pitch_error_deg:   float or None
        angular_error_deg: float or None
        reason:            None, or why the errors are unavailable

    All three errors are None together whenever any input is degenerate
    (missing point, prediction or GT coincident with the anchor). Never
    raises.
    """
    unavailable = {
        "yaw_error_deg": None,
        "pitch_error_deg": None,
        "angular_error_deg": None,
        "reason": None,
    }

    pred, reason = _coerce_vec3(pred_p, "pred point")
    if reason is None:
        gt, reason = _coerce_vec3(gt_p, "gt point")
    if reason is None:
        anchor, reason = _coerce_vec3(anchor_p, "anchor point")
    if reason is not None:
        unavailable["reason"] = reason
        return unavailable

    pred_dir = (pred[0] - anchor[0], pred[1] - anchor[1], pred[2] - anchor[2])
    gt_dir = (gt[0] - anchor[0], gt[1] - anchor[1], gt[2] - anchor[2])

    pred_yp, reason = yaw_pitch_from_vector(pred_dir)
    if reason is not None:
        unavailable["reason"] = "pred direction: %s" % reason
        return unavailable
    gt_yp, reason = yaw_pitch_from_vector(gt_dir)
    if reason is not None:
        unavailable["reason"] = "gt direction: %s" % reason
        return unavailable

    yaw_err, _ = yaw_error_deg(pred_yp[0], gt_yp[0])
    pitch_err, _ = pitch_error_deg(pred_yp[1], gt_yp[1])
    full_err, _ = angular_error_from_vectors(pred_dir, gt_dir)
    return {
        "yaw_error_deg": yaw_err,
        "pitch_error_deg": pitch_err,
        "angular_error_deg": full_err,
        "reason": None,
    }
