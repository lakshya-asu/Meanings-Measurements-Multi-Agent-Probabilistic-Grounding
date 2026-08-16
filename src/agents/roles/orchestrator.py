"""Orchestrator role: query -> typed decomposition (no image)."""

from typing import Any, Dict

from src.agents.base import text_part
from src.agents.prompts import orchestrator as prompt
from src.agents.schemas import ORCHESTRATOR_SCHEMA, try_validate


class OrchestratorRole:
    role = "orchestrator"

    def __init__(self, backend):
        self.backend = backend

    @property
    def model_name(self):
        return self.backend.model_name

    def process(self, blackboard) -> Dict[str, Any]:
        system, user = prompt.render(blackboard)
        try:
            parsed, usage, _latency_ms = self.backend.send(
                system, [text_part(user)], ORCHESTRATOR_SCHEMA
            )
        except Exception as e:
            error_msg = f"Failed to orchestrate query: {e}"
            blackboard.append_event("Orchestrator", "Error", error_msg, "FAIL")
            return {"error": error_msg}
        ok, coerced, errors = try_validate("orchestrator", parsed)
        if not ok:
            error_msg = f"schema_invalid: {'; '.join(errors)}"
            blackboard.append_event("Orchestrator", "SchemaInvalid", error_msg, "FAIL")
            return {"error": error_msg, "schema_invalid": True, "usage": usage}
        blackboard.append_event("Orchestrator", "ParseQuery", coerced, "PASS")
        return {**coerced, "usage": usage}
