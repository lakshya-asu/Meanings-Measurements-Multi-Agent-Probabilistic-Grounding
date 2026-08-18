import os
import json
import base64
import mimetypes
from typing import Any, Dict, Literal, Optional
from pydantic import BaseModel, Field

from src.multi_agent.blackboard import Blackboard
from src.verification.checks import failed_reasons

def encode_image(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

class VerifierOutput(BaseModel):
    reasoning: str
    status: Literal["PASS", "FAIL"]
    feedback: str = Field(description="If FAIL, explain what went wrong so the system can recover.")

class ClaudeVerifierAgent:
    """Verifier: programmatic checks first, optional LLM critique last.

    P0 fix 3. The old version defaulted to PASS whenever its own LLM
    call crashed (fail-open), so a crashed verifier was indistinguishable
    from an approval. Now: the planner computes the four programmatic
    checks (src/verification/checks.py) and passes them in; any failed
    check is a FAIL with reasons. The LLM critique is optional
    (llm_enabled, cfg verifier.llm, default off per the ablation design)
    and an LLM exception is a recorded FAIL, never a silent PASS.
    """

    def __init__(self, model_name="claude-opus-4-6", llm_enabled: bool = False):
        self.model_name = model_name
        self.llm_enabled = bool(llm_enabled)
        self._client = None

    def _get_client(self):
        # Lazy: no API key is needed unless the LLM critique actually runs.
        if self._client is None:
            import anthropic
            if "CLAUDE_API_KEY" not in os.environ:
                raise RuntimeError("CLAUDE_API_KEY must be set in the environment.")
            self._client = anthropic.Anthropic(api_key=os.environ["CLAUDE_API_KEY"])
        return self._client

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
            blackboard.append_event("Verifier(Claude)", "Critique", parsed, "FAIL")
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
            blackboard.append_event("Verifier(Claude)", "Critique", parsed, "PASS")
            return parsed

        sys_prompt = f"""
        SYSTEM: You are the Verifier Critic for a robotic spatial reasoning system.

        CRITICAL INSTRUCTION: You MUST output exactly ONE valid JSON object matching the schema below. Do not include any other text.
        Schema:
        {json.dumps(VerifierOutput.model_json_schema(), indent=2)}
        """

        prompt = f"""
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

        content = [{"type": "text", "text": prompt}]
        if blackboard.current_image_path and os.path.exists(blackboard.current_image_path):
            mime = mimetypes.guess_type(blackboard.current_image_path)[0] or "image/png"
            b64_img = encode_image(blackboard.current_image_path)
            content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": mime,
                    "data": b64_img
                }
            })

        try:
            response = self._get_client().messages.create(
                model=self.model_name,
                max_tokens=2048,
                system=sys_prompt,
                messages=[
                    {"role": "user", "content": content}
                ]
            )
            text = response.content[0].text.strip()
            if "```json" in text:
                text = text.split("```json")[-1].split("```")[0].strip()
            elif "```" in text:
                text = text.split("```")[-1].split("```")[0].strip()

            parsed = json.loads(text)
            parsed["checks"] = checks
            parsed["llm_used"] = True
            blackboard.append_event("Verifier(Claude)", "Critique", parsed, parsed["status"])
            return parsed
        except Exception as e:
            # Fail closed: an LLM crash is a recorded FAIL, not an approval.
            parsed = {
                "status": "FAIL",
                "feedback": f"Verifier LLM error (Claude), fail closed: {e}",
                "reasoning": "",
                "checks": checks,
                "llm_used": True,
                "llm_error": str(e),
            }
            blackboard.append_event("Verifier(Claude)", "Critique", parsed, "FAIL")
            return parsed
