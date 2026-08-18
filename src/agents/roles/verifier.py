"""Verifier role: programmatic checks first, optional LLM critique last.

Behavior contract from a844d6b (P0 fix 3), preserved exactly:

- the planner computes the four programmatic checks
  (src/verification/checks.py) and passes them into ``process``; any
  failed check is a FAIL with reasons and no LLM call;
- the LLM critique is optional (``llm_enabled``, cfg verifier.llm,
  default off per the ablation design) and only runs after the checks
  pass;
- an LLM exception, and now also a schema-invalid LLM reply, is a
  recorded FAIL, never a silent PASS (fail closed);
- the returned dict always carries ``checks`` and ``llm_used`` so the
  planner's CallLog record_if can keep programmatic-only verifications
  out of the call count.
"""

from typing import Any, Dict, Optional

from src.agents.prompts import verifier as prompt
from src.agents.roles._shared import sans_usage, user_parts_with_image
from src.agents.schemas import VERIFIER_SCHEMA, try_validate
from src.verification.checks import failed_reasons


class VerifierRole:
    role = "verifier"

    def __init__(self, backend, llm_enabled: bool = False):
        self.backend = backend
        self.llm_enabled = bool(llm_enabled)

    @property
    def model_name(self):
        return self.backend.model_name

    def process(
        self, blackboard, checks: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        checks = checks if isinstance(checks, dict) else None

        if checks is not None and not checks.get("all_ok", False):
            parsed = {
                "status": "FAIL",
                "feedback": f"Programmatic checks failed: {failed_reasons(checks)}",
                "reasoning": "programmatic verification",
                "checks": checks,
                "llm_used": False,
            }
            blackboard.append_event("Verifier", "Critique", parsed, "FAIL")
            return parsed

        if not self.llm_enabled:
            parsed = {
                "status": "PASS",
                "feedback": "",
                "reasoning": (
                    "Programmatic checks passed; LLM critique disabled by config "
                    "(verifier.llm=false)." if checks is not None else
                    "No programmatic checks provided and LLM critique disabled by "
                    "config (verifier.llm=false)."
                ),
                "checks": checks,
                "llm_used": False,
            }
            blackboard.append_event("Verifier", "Critique", parsed, "PASS")
            return parsed

        system, user = prompt.render(blackboard)
        parts = user_parts_with_image(user, blackboard.current_image_path)
        try:
            reply, usage, _latency_ms = self.backend.send(
                system, parts, VERIFIER_SCHEMA
            )
            ok, coerced, errors = try_validate("verifier", reply)
            if not ok:
                raise ValueError(f"schema_invalid: {'; '.join(errors)}")
        except Exception as e:
            # Fail closed: an LLM crash or malformed reply is a
            # recorded FAIL, not an approval.
            parsed = {
                "status": "FAIL",
                "feedback": f"Verifier LLM error, fail closed: {e}",
                "reasoning": "",
                "checks": checks,
                "llm_used": True,
                "llm_error": str(e),
            }
            blackboard.append_event("Verifier", "Critique", parsed, "FAIL")
            return parsed

        parsed = dict(coerced)
        parsed["checks"] = checks
        parsed["llm_used"] = True
        parsed["usage"] = usage
        blackboard.append_event("Verifier", "Critique", sans_usage(parsed), parsed["status"])
        return parsed
