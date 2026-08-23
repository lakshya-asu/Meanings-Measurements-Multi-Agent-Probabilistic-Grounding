"""Canonical final-prediction schema for MAPG planners.

Every planner ends an episode with a "final prediction" dict. Historically each
planner used its own key for the predicted 3D point (Habitat world frame):

- multi_agent_msp_planner.py wrote "target_location"
- vlm_planner_msp.py wrote "target_xyz_hab"
- vlm_planner_benchmark_gemini.py wrote "target_point_xyz"
- some pipelines wrote "selected_object_xyz" or variants

The evaluator (src/evals/eval_offset_distances.py) could only read some of
these, so episodes from other planners scored as NaN. This module defines one
canonical key and a converter that maps every legacy key onto it.

Canonical key: "target_point_xyz", a list of 3 floats in the Habitat world
frame, or None when the planner produced no point.

This module is stdlib only on purpose: no habitat, no numpy. It must be
importable and testable outside the container. Numpy arrays are handled by
duck typing (tolist or plain iteration), never by importing numpy.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# The one key everything downstream should read.
CANONICAL_POINT_KEY = "target_point_xyz"

# Legacy keys that may hold the predicted point, in priority order.
# The canonical key comes first so already-normalized dicts stay untouched.
POINT_KEY_PRIORITY = (
    "target_point_xyz",
    "target_location",
    "target_xyz_hab",
    "selected_object_xyz",
    "selected_object_center_xyz",
    "selected_center_xyz",
)

WARNINGS_KEY = "schema_warnings"


@dataclass
class FinalPrediction:
    """Spec for the canonical final-prediction record.

    This dataclass documents the fields the runners already produce. Runners
    keep emitting plain dicts; this class is the reference shape and can be
    used to build well-formed records in new code.

    Required fields:
        action_type: what the planner decided to do at the end, for example
            "answer", "goto_object", "goto_frontier", "lookaround".
        chosen_id: scene-graph id of the chosen object or frontier, or a
            sentinel such as "POINT_GUESS" or "" when nothing was chosen.
        confidence: planner confidence in [0, 1].
        target_point_xyz: predicted 3D point [x, y, z] in the Habitat world
            frame, or None when the planner produced no point.

    Optional fields:
        pdf_params: parameters of the spatial PDF used to place the point.
        top_k_objects: ranked candidate objects, each a dict with at least
            an id and a position.
        thought: free-text reasoning from the planner.
        schema_warnings: strings recorded by normalize_prediction when the
            input was malformed or the point was missing.
    """

    action_type: str
    chosen_id: str
    confidence: float
    target_point_xyz: Optional[List[float]]
    pdf_params: Optional[Dict[str, Any]] = None
    top_k_objects: Optional[List[Dict[str, Any]]] = None
    thought: str = ""
    schema_warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action_type": self.action_type,
            "chosen_id": self.chosen_id,
            "confidence": self.confidence,
            "target_point_xyz": self.target_point_xyz,
            "pdf_params": self.pdf_params,
            "top_k_objects": self.top_k_objects,
            "thought": self.thought,
            "schema_warnings": list(self.schema_warnings),
        }


def _coerce_xyz(value: Any) -> Optional[List[float]]:
    """Try to turn value into [x, y, z] floats. Return None if impossible.

    Accepts a list or tuple of 3 numbers, a dict with x/y/z keys (any case),
    and anything array-like: objects with a tolist method (numpy arrays) or
    plain iterables of 3 numbers. Never raises.
    """
    if value is None:
        return None

    # Dict form: {"x": ..., "y": ..., "z": ...}
    if isinstance(value, dict):
        lowered = {str(k).lower(): v for k, v in value.items()}
        try:
            return [float(lowered["x"]), float(lowered["y"]), float(lowered["z"])]
        except (KeyError, TypeError, ValueError):
            return None

    # Numpy array duck typing: tolist gives plain Python numbers.
    if hasattr(value, "tolist"):
        try:
            value = value.tolist()
        except Exception:
            return None

    # Strings are iterable but never a valid point.
    if isinstance(value, (str, bytes)):
        return None

    # List, tuple, or any other iterable of 3 numbers.
    try:
        items = list(value)
    except TypeError:
        return None
    if len(items) != 3:
        return None
    try:
        return [float(items[0]), float(items[1]), float(items[2])]
    except (TypeError, ValueError):
        return None


def _add_warning(out: Dict[str, Any], msg: str) -> None:
    """Append msg to out[schema_warnings], skipping duplicates.

    Deduping keeps normalize_prediction idempotent: running it twice does
    not stack the same warning twice.
    """
    warnings = out.get(WARNINGS_KEY)
    if not isinstance(warnings, list):
        warnings = []
        out[WARNINGS_KEY] = warnings
    if msg not in warnings:
        warnings.append(msg)


def normalize_prediction(pred: Any) -> Dict[str, Any]:
    """Map a planner's final-prediction dict onto the canonical schema.

    Returns a shallow copy of pred with the canonical key
    "target_point_xyz" set to [x, y, z] floats or None. All original fields
    are kept as-is, legacy keys included. Never raises on malformed input:
    problems are recorded as strings in "schema_warnings" instead.

    The point is taken from the first key in POINT_KEY_PRIORITY whose value
    coerces to a 3-vector. Values may be a list or tuple of 3 numbers, a
    dict with x/y/z keys, or a numpy array (handled without importing
    numpy). Empty lists and None are treated as "no point here" and the
    search moves on to the next key.
    """
    if not isinstance(pred, dict):
        out: Dict[str, Any] = {CANONICAL_POINT_KEY: None}
        _add_warning(out, f"prediction is not a dict (got {type(pred).__name__})")
        return out

    out = dict(pred)

    point: Optional[List[float]] = None
    source_key: Optional[str] = None
    for key in POINT_KEY_PRIORITY:
        if key not in out:
            continue
        raw = out[key]
        # None and empty containers are deliberate "no point" sentinels
        # in the planners, not malformed data. Just keep looking.
        if raw is None or (isinstance(raw, (list, tuple)) and len(raw) == 0):
            continue
        point = _coerce_xyz(raw)
        if point is not None:
            source_key = key
            break
        _add_warning(out, f"key {key!r} present but not a valid 3-vector: {raw!r}")

    out[CANONICAL_POINT_KEY] = point
    if point is None:
        _add_warning(out, "no target point found under any known key")
    elif source_key != CANONICAL_POINT_KEY:
        # Not a problem, just provenance. Useful when auditing runs.
        _add_warning(out, f"target_point_xyz filled from legacy key {source_key!r}")

    return out
