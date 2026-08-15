"""Programmatic verifier checks (P0 fix 3: a verifier that verifies).

Replaces the old fail-open verifier, which defaulted to PASS whenever
its own LLM call crashed (claude_verifier_agent.py:85 and its three
siblings), so a crashed verifier was indistinguishable from an
approval. Verification is now programmatic first: four independently
logged booleans with reasons, computed here with no LLM and no habitat
dependency. The optional LLM critique (cfg ``verifier.llm``, default
off per the ablation design) runs only after these checks pass, and an
LLM crash is a recorded FAIL, never a silent PASS.

The four checks (each returns ok, reason, and a skipped flag; a
skipped check never fails the gate but is always recorded):

1. schema_valid: the upstream structured output has the required
   fields and finite numeric values.
2. in_scene_bounds: the proposed target lies inside the scene AABB
   inflated by 5 m.
3. on_navmesh: the target snaps to the navmesh within 0.2 m. The snap
   function is injectable; outside the container (no habitat) pass
   snap_fn=None and the check is recorded as skipped.
4. distance_consistency: |dist(pred, anchor) - d0| <= 3 sigma_m, with
   d0 from the deterministic parser (src/parsing/metric_literal.py).
   Skipped when the query has no metric literal (d0 None).

This module is stdlib only on purpose: importable and testable outside
the container. Points are any sequence of 3 numbers.
"""

import math
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

DEFAULT_BOUNDS_MARGIN_M = 5.0
DEFAULT_NAVMESH_TOLERANCE_M = 0.2
DEFAULT_DISTANCE_K = 3.0

CHECK_NAMES = (
    "schema_valid",
    "in_scene_bounds",
    "on_navmesh",
    "distance_consistency",
)


def _to_xyz(value: Any) -> Optional[Tuple[float, float, float]]:
    """Coerce value into (x, y, z) floats, or None. Never raises."""
    if value is None:
        return None
    if hasattr(value, "tolist"):
        try:
            value = value.tolist()
        except Exception:
            return None
    try:
        items = list(value)
    except TypeError:
        return None
    if len(items) != 3:
        return None
    try:
        xyz = (float(items[0]), float(items[1]), float(items[2]))
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(v) for v in xyz):
        return None
    return xyz


def _dist(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> float:
    return math.sqrt(sum((ai - bi) ** 2 for ai, bi in zip(a, b)))


def schema_valid(payload: Any, required_fields: Sequence[str]) -> Tuple[bool, str]:
    """Payload is a dict with every required field present and finite.

    Numeric required fields must be finite numbers; non-numeric ones
    must simply be present and non-None.
    """
    if not isinstance(payload, dict):
        return False, "payload is not a dict (got %s)" % type(payload).__name__
    for key in required_fields:
        if key not in payload or payload[key] is None:
            return False, "missing required field %r" % key
        value = payload[key]
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            if not math.isfinite(float(value)):
                return False, "field %r is not finite: %r" % (key, value)
    return True, "all required fields present and finite"


def in_scene_bounds(
    point: Any,
    scene_min: Any,
    scene_max: Any,
    margin_m: float = DEFAULT_BOUNDS_MARGIN_M,
) -> Tuple[bool, str]:
    """Point inside the scene AABB inflated by margin_m on every axis."""
    p = _to_xyz(point)
    if p is None:
        return False, "prediction is not a valid finite 3-vector: %r" % (point,)
    lo = _to_xyz(scene_min)
    hi = _to_xyz(scene_max)
    if lo is None or hi is None:
        return False, "scene bounds are not valid 3-vectors"
    for axis, (v, a, b) in enumerate(zip(p, lo, hi)):
        if v < a - margin_m or v > b + margin_m:
            return False, (
                "axis %d value %.3f outside [%.3f, %.3f] (AABB + %.1f m margin)"
                % (axis, v, a - margin_m, b + margin_m, margin_m)
            )
    return True, "inside scene AABB + %.1f m margin" % margin_m


def on_navmesh(
    point: Any,
    snap_fn: Optional[Callable[[Sequence[float]], Any]],
    tolerance_m: float = DEFAULT_NAVMESH_TOLERANCE_M,
) -> Tuple[bool, str, bool]:
    """Point snaps to the navmesh within tolerance_m.

    snap_fn(point) must return the snapped point (any 3-sequence) or
    None when the snap fails. With snap_fn=None the check is SKIPPED
    (ok=True, skipped=True) and the reason records why, so runs outside
    the container stay honest about what was not checked.
    """
    if snap_fn is None:
        return True, "navmesh unavailable (no snap function injected); check skipped", True
    p = _to_xyz(point)
    if p is None:
        return False, "prediction is not a valid finite 3-vector: %r" % (point,), False
    try:
        snapped = snap_fn(p)
    except Exception as e:
        return False, "navmesh snap raised: %s" % e, False
    s = _to_xyz(snapped)
    if s is None:
        return False, "navmesh snap failed (no navigable point returned)", False
    d = _dist(p, s)
    if d > tolerance_m:
        return False, "snap distance %.3f m exceeds tolerance %.2f m" % (d, tolerance_m), False
    return True, "snapped within %.3f m (tolerance %.2f m)" % (d, tolerance_m), False


def distance_consistency(
    point: Any,
    anchor: Any,
    d0_m: Optional[float],
    sigma_m: Optional[float],
    k: float = DEFAULT_DISTANCE_K,
) -> Tuple[bool, str, bool]:
    """|dist(point, anchor) - d0| <= k * sigma_m when both exist.

    SKIPPED (ok=True, skipped=True) when the query carries no metric
    literal (d0_m is None: the metric kernel was omitted, there is
    nothing to be consistent with) or when sigma_m or the anchor is
    unavailable.
    """
    if d0_m is None:
        return True, "no metric literal (d0 is None); check skipped", True
    if sigma_m is None or not math.isfinite(float(sigma_m)) or float(sigma_m) <= 0:
        return True, "sigma_m unavailable; check skipped", True
    a = _to_xyz(anchor)
    if a is None:
        return True, "anchor position unavailable; check skipped", True
    p = _to_xyz(point)
    if p is None:
        return False, "prediction is not a valid finite 3-vector: %r" % (point,), False
    r = _dist(p, a)
    err = abs(r - float(d0_m))
    limit = k * float(sigma_m)
    if err > limit:
        return False, (
            "radial error %.3f m exceeds %g * sigma_m = %.3f m (r=%.3f, d0=%.3f)"
            % (err, k, limit, r, float(d0_m))
        ), False
    return True, (
        "radial error %.3f m within %g * sigma_m = %.3f m" % (err, k, limit)
    ), False


def run_checks(
    *,
    spatial_payload: Any = None,
    required_fields: Sequence[str] = ("theta", "phi"),
    prediction_xyz: Any = None,
    anchor_xyz: Any = None,
    d0_m: Optional[float] = None,
    sigma_m: Optional[float] = None,
    scene_min: Any = None,
    scene_max: Any = None,
    bounds_margin_m: float = DEFAULT_BOUNDS_MARGIN_M,
    navmesh_snap_fn: Optional[Callable[[Sequence[float]], Any]] = None,
    navmesh_tolerance_m: float = DEFAULT_NAVMESH_TOLERANCE_M,
    distance_k: float = DEFAULT_DISTANCE_K,
) -> Dict[str, Any]:
    """Run all four checks; return per-check records plus all_ok.

    Result shape:
        {"schema_valid": {"ok": bool, "reason": str, "skipped": bool},
         "in_scene_bounds": {...}, "on_navmesh": {...},
         "distance_consistency": {...}, "all_ok": bool}

    A skipped check is recorded but never fails the gate. Checks whose
    inputs are absent (no scene bounds, no prediction yet) are skipped
    rather than failed, so the gate only fires on positive evidence of
    a bad output.
    """
    result: Dict[str, Any] = {}

    if spatial_payload is None:
        result["schema_valid"] = {
            "ok": True,
            "reason": "no payload provided; check skipped",
            "skipped": True,
        }
    else:
        ok, reason = schema_valid(spatial_payload, required_fields)
        result["schema_valid"] = {"ok": ok, "reason": reason, "skipped": False}

    if prediction_xyz is None or scene_min is None or scene_max is None:
        result["in_scene_bounds"] = {
            "ok": True,
            "reason": "prediction or scene bounds unavailable; check skipped",
            "skipped": True,
        }
    else:
        ok, reason = in_scene_bounds(prediction_xyz, scene_min, scene_max, bounds_margin_m)
        result["in_scene_bounds"] = {"ok": ok, "reason": reason, "skipped": False}

    if prediction_xyz is None:
        result["on_navmesh"] = {
            "ok": True,
            "reason": "no prediction point; check skipped",
            "skipped": True,
        }
    else:
        ok, reason, skipped = on_navmesh(prediction_xyz, navmesh_snap_fn, navmesh_tolerance_m)
        result["on_navmesh"] = {"ok": ok, "reason": reason, "skipped": skipped}

    if prediction_xyz is None:
        result["distance_consistency"] = {
            "ok": True,
            "reason": "no prediction point; check skipped",
            "skipped": True,
        }
    else:
        ok, reason, skipped = distance_consistency(
            prediction_xyz, anchor_xyz, d0_m, sigma_m, distance_k
        )
        result["distance_consistency"] = {"ok": ok, "reason": reason, "skipped": skipped}

    result["all_ok"] = all(result[name]["ok"] for name in CHECK_NAMES)
    return result


def failed_reasons(checks: Dict[str, Any]) -> str:
    """Human-readable summary of the failed checks in a run_checks dict."""
    parts = []
    for name in CHECK_NAMES:
        rec = checks.get(name)
        if isinstance(rec, dict) and not rec.get("ok", True):
            parts.append("%s: %s" % (name, rec.get("reason", "")))
    return "; ".join(parts) if parts else "no failed checks"
