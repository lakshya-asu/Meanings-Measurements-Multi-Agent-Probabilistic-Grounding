"""QA role: the MCQ/EQA fast path (image attached on every backend)."""

from typing import Any, Dict

from src.agents.prompts import qa as prompt
from src.agents.roles._shared import chunked_parts_with_image, sans_usage
from src.agents.schemas import QA_SCHEMA, try_validate


class QaRole:
    role = "qa"

    def __init__(self, backend):
        self.backend = backend

    @property
    def model_name(self):
        return self.backend.model_name

    def process(self, blackboard) -> Dict[str, Any]:
        # MAPG-10: the stable question+scene-graph chunk carries the
        # cache mark; concatenated text is identical to prompt.render.
        system, chunks = prompt.render_parts(blackboard)
        parts = chunked_parts_with_image(chunks, blackboard.current_image_path)
        try:
            parsed, usage, _latency_ms = self.backend.send(system, parts, QA_SCHEMA)
        except Exception as e:
            error_msg = f"Failed to infer MCQ QA: {e}"
            blackboard.append_event("QA", "Error", error_msg, "FAIL")
            return {"ok": False, "error": error_msg}
        ok, coerced, errors = try_validate("qa", parsed)
        if not ok:
            error_msg = f"schema_invalid: {'; '.join(errors)}"
            blackboard.append_event("QA", "SchemaInvalid", error_msg, "FAIL")
            return {
                "ok": False,
                "error": error_msg,
                "schema_invalid": True,
                "usage": usage,
            }

        out = {
            "ok": True,
            "prior_hypothesis": coerced.get("prior_hypothesis", ""),
            "hypothesis_likelihood": coerced.get("hypothesis_likelihood", "low"),
            "action_type": coerced.get("action_type", "lookaround"),
            "chosen_id": coerced.get("chosen_id", "NONE"),
            "answer": coerced.get("answer", ""),
            "confidence": float(coerced.get("confidence", 0.0)),
            "reasoning": coerced.get("reasoning", ""),
            "usage": usage,
        }
        blackboard.append_event("QA", out["action_type"], sans_usage(out), "PASS")
        return out
