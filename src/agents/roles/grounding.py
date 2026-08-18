"""Grounding role: anchors -> scene-graph object ids (image attached)."""

from typing import Any, Dict

from src.agents.prompts import grounding as prompt
from src.agents.roles._shared import user_parts_with_image
from src.agents.schemas import GROUNDING_SCHEMA, try_validate


class GroundingRole:
    role = "grounding"

    def __init__(self, backend):
        self.backend = backend

    @property
    def model_name(self):
        return self.backend.model_name

    def process(self, blackboard, orchestrator_output: Dict[str, Any]) -> Dict[str, Any]:
        system, user = prompt.render(blackboard, orchestrator_output)
        parts = user_parts_with_image(user, blackboard.current_image_path)
        try:
            parsed, usage, _latency_ms = self.backend.send(
                system, parts, GROUNDING_SCHEMA
            )
        except Exception as e:
            error_msg = f"Visual grounding failed: {e}"
            blackboard.append_event("Grounding", "Error", error_msg, "FAIL")
            return {"error": error_msg, "needs_exploration": True}
        ok, coerced, errors = try_validate("grounding", parsed)
        if not ok:
            error_msg = f"schema_invalid: {'; '.join(errors)}"
            blackboard.append_event("Grounding", "SchemaInvalid", error_msg, "FAIL")
            return {
                "error": error_msg,
                "needs_exploration": True,
                "schema_invalid": True,
                "usage": usage,
            }
        blackboard.append_event("Grounding", "MatchObjects", coerced, "PASS")
        return {**coerced, "usage": usage}
