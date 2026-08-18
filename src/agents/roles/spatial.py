"""Spatial role: intrinsic front vector (image required).

Post-processing preserves the legacy claude/openai behavior exactly:
theta_cam -> theta_world via the agent yaw, and phi resolved from
question keywords ("above" / "below") rather than the model's
phi_radians. (The legacy gemini copy used the model phi; unification
keeps the majority/claude behavior, and the model phi remains
available in the validated payload if an ablation wants it.) The
success event ("KernelParams") is logged on every backend; legacy
claude skipped it while gemini logged it, which put different ledgers
in front of the verifier per backend.
"""

from typing import Any, Dict
import math
import os

from src.agents.prompts import spatial as prompt
from src.agents.roles._shared import chunked_parts_with_image, sans_usage
from src.agents.schemas import SPATIAL_SCHEMA, try_validate


class SpatialRole:
    role = "spatial"

    def __init__(self, backend):
        self.backend = backend

    @property
    def model_name(self):
        return self.backend.model_name

    def process(self, blackboard, anchor_obj: Dict[str, Any]) -> Dict[str, Any]:
        if not blackboard.current_image_path or not os.path.exists(
            blackboard.current_image_path
        ):
            blackboard.append_event(
                "Spatial", "Error", "No image for spatial kernel", "FAIL"
            )
            return {"ok": False, "error": "No image available."}

        # MAPG-10: the stable scene-graph chunk carries the cache mark;
        # concatenated text is identical to prompt.render (golden-tested).
        system, chunks = prompt.render_parts(blackboard, anchor_obj)
        parts = chunked_parts_with_image(chunks, blackboard.current_image_path)
        try:
            parsed, usage, _latency_ms = self.backend.send(
                system, parts, SPATIAL_SCHEMA
            )
        except Exception as e:
            blackboard.append_event("Spatial", "Error", str(e), "FAIL")
            return {"ok": False, "error": str(e)}
        ok, coerced, errors = try_validate("spatial", parsed)
        if not ok:
            error_msg = f"schema_invalid: {'; '.join(errors)}"
            blackboard.append_event("Spatial", "SchemaInvalid", error_msg, "FAIL")
            return {
                "ok": False,
                "error": error_msg,
                "schema_invalid": True,
                "usage": usage,
            }

        # Convert camera theta to world theta (legacy behavior).
        theta_cam = float(coerced["theta_radians"])
        two_pi = 2.0 * math.pi
        theta_world = (blackboard.agent_yaw_rad + theta_cam) % two_pi

        q_lower = blackboard.question.lower()
        if "above" in q_lower:
            phi_val = 0.0
        elif "below" in q_lower:
            phi_val = 3.14
        else:
            phi_val = 1.57

        out = {
            "ok": True,
            "theta": theta_world,
            "theta_cam": theta_cam,
            "agent_yaw": blackboard.agent_yaw_rad,
            "phi": phi_val,
            "phi_radians_model": float(coerced["phi_radians"]),
            "kappa": 0.0,
            "target_frontier_id": coerced.get("target_frontier_id", "NONE"),
            "reasoning": coerced["reasoning"],
            "usage": usage,
        }
        blackboard.append_event("Spatial", "KernelParams", sans_usage(out), "PASS")
        return out
