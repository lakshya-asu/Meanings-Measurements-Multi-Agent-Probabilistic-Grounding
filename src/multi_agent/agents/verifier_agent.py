import json
import mimetypes
from typing import Any, Dict, Optional

from .spatial_agent import encode_image

from ..blackboard import Blackboard
from src.verification.checks import failed_reasons

class VerifierAgent:
    """Verifier (Gemini): programmatic checks first, optional LLM last.

    P0 fix 3, see ClaudeVerifierAgent for the rationale. Fail-open
    default-PASS removed; LLM exceptions are recorded FAILs.
    """

    def __init__(self, model_name="models/gemini-3-pro-preview", llm_enabled: bool = False):
        self.model_name = model_name
        self.llm_enabled = bool(llm_enabled)
        self._model = None
        self._schema = None

    def _get_model(self):
        # Lazy: the genai client is only needed when the LLM critique runs.
        if self._model is None:
            import google.generativeai as genai
            self._model = genai.GenerativeModel(model_name=self.model_name)
            self._schema = genai.protos.Schema(
                type=genai.protos.Type.OBJECT,
                properties={
                    "reasoning": genai.protos.Schema(type=genai.protos.Type.STRING),
                    "status": genai.protos.Schema(type=genai.protos.Type.STRING, enum=["PASS", "FAIL"]),
                    "feedback": genai.protos.Schema(type=genai.protos.Type.STRING, description="If FAIL, explain what went wrong so the system can recover.")
                },
                required=["reasoning", "status", "feedback"]
            )
        return self._model

    def process(self, blackboard: Blackboard, checks: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
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

        prompt = f"""
        You are the Verifier Critic for a robotic spatial reasoning system.
        Review the current execution ledger and the provided image.

        Current Question: {blackboard.question}
        Agent current global yaw: {blackboard.agent_yaw_rad:.3f} rad.

        Ledger of actions taken so far in this step:
        {blackboard.get_ledger_str()}

        Task:
        1. Check if the Orchestrator's logic makes sense for the question.
        2. Look at the image: Does the Spatial Agent's calculated 'theta_cam' (egocentric front direction) logically align with the object visible in the image?
        CRITICAL RULE: The Spatial agent reasons in egocentric 'theta_cam' (negative=Right, positive=Left). The system mathematically converts this to a global 'theta' using the agent's yaw. Do NOT flag a contradiction just because the global 'theta' is positive while the text reasoning discusses a negative 'theta_cam'.
        3. CRITICAL RULE: Be lenient on exact label nomenclature. If the Orchestrator/Grounding agents selected a 'sofa' instead of a 'couch', or a 'monitor' instead of a 'tv', DO NOT fail them. Accept synonymous or functionally equivalent object labels.
        4. If an agent hallucinated or made a clear error in the visual mapping, output FAIL with feedback. Otherwise, output PASS.
        """

        messages = [{"role": "user", "parts": [{"text": prompt}]}]
        if blackboard.current_image_path:
            mime = mimetypes.guess_type(blackboard.current_image_path)[0] or "image/png"
            messages[0]["parts"].append({"inline_data": {"mime_type": mime, "data": encode_image(blackboard.current_image_path)}})

        try:
            import google.generativeai as genai
            model = self._get_model()
            resp = model.generate_content(
                messages,
                generation_config=genai.GenerationConfig(response_mime_type="application/json", temperature=0.1, response_schema=self._schema)
            )
            parsed = json.loads(resp.text)
            parsed["checks"] = checks
            parsed["llm_used"] = True
            blackboard.append_event("Verifier", "Critique", parsed, parsed["status"])
            return parsed
        except Exception as e:
            # Fail closed: an LLM crash is a recorded FAIL, not an approval.
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
