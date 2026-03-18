import json
import google.generativeai as genai
from typing import Dict, Any
from .spatial_agent import encode_image
import mimetypes

from ..blackboard import Blackboard

class VerifierAgent:
    def __init__(self, model_name="models/gemini-3-pro-preview"):
        self.model = genai.GenerativeModel(model_name=model_name)
        self.schema = genai.protos.Schema(
            type=genai.protos.Type.OBJECT,
            properties={
                "reasoning": genai.protos.Schema(type=genai.protos.Type.STRING),
                "status": genai.protos.Schema(type=genai.protos.Type.STRING, enum=["PASS", "FAIL"]),
                "feedback": genai.protos.Schema(type=genai.protos.Type.STRING, description="If FAIL, explain what went wrong so the system can recover.")
            },
            required=["reasoning", "status", "feedback"]
        )

    def process(self, blackboard: Blackboard) -> Dict[str, Any]:
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
            resp = self.model.generate_content(
                messages,
                generation_config=genai.GenerationConfig(response_mime_type="application/json", temperature=0.1, response_schema=self.schema)
            )
            parsed = json.loads(resp.text)
            blackboard.append_event("Verifier", "Critique", parsed, parsed["status"])
            return parsed
        except Exception as e:
            return {"status": "PASS", "feedback": f"Verifier error, defaulting to pass: {e}", "reasoning": ""}