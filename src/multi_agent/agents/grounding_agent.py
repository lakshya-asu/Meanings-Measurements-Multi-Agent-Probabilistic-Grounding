import os
import json
import base64
import mimetypes
import google.generativeai as genai
from typing import Dict, Any

from ..blackboard import Blackboard

def encode_image(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

class GroundingAgent:
    def __init__(self, model_name="models/gemini-3-pro-preview"):
        self.model = genai.GenerativeModel(model_name=model_name)
        
        self.schema = genai.protos.Schema(
            type=genai.protos.Type.OBJECT,
            properties={
                "reasoning": genai.protos.Schema(type=genai.protos.Type.STRING, description="Explain your visual verification of the modifiers."),
                "grounded_anchors": genai.protos.Schema(
                    type=genai.protos.Type.ARRAY,
                    items=genai.protos.Schema(
                        type=genai.protos.Type.OBJECT,
                        properties={
                            "anchor_label": genai.protos.Schema(type=genai.protos.Type.STRING),
                            "matched_object_id": genai.protos.Schema(type=genai.protos.Type.STRING, description="The exact ID from the scene graph. Use 'NONE' if not found."),
                            "confidence": genai.protos.Schema(type=genai.protos.Type.NUMBER)
                        }
                    )
                ),
                "needs_exploration": genai.protos.Schema(type=genai.protos.Type.BOOLEAN, description="True if a required anchor is missing from the scene graph AND the image.")
            },
            required=["reasoning", "grounded_anchors", "needs_exploration"]
        )

    def process(self, blackboard: Blackboard, orchestrator_output: Dict[str, Any]) -> Dict[str, Any]:
        available_objs = [{"id": o["id"], "name": o.get("name", ""), "pos": o.get("position")} for o in blackboard.available_objects]
        
        prompt = f"""
        You are the Visual Grounding Agent. Your task is to link semantic anchor descriptions to specific object IDs in the robot's current scene graph by looking at the image.
        
        Target Entity to find: {orchestrator_output.get("target_entity")}
        Anchors to map: {json.dumps(orchestrator_output.get("anchors", []))}
        
        Current Scene Graph Candidates:
        {json.dumps(available_objs, indent=2)}
        
        Agent Exact Position: {blackboard.agent_pose_hab}
        Agent Yaw (rad): {blackboard.agent_yaw_rad}
        
        Agent Current Semantic State: {blackboard.agent_semantic_state}
        
        GLOBAL FAILURE HISTORY:
        {blackboard.global_history}
        
        Task: 
        1. Find the best matching object ID for each anchor.
        2. Look at the provided image. If an anchor has a modifier (e.g., '2 seater', 'next to the wall'), you MUST use the image to verify which scene graph candidate visually matches that description.
        3. CRITICAL: Read the GLOBAL FAILURE HISTORY carefully. If the Verifier explicitly rejected a specific object_id for an anchor, you MUST BLACKLIST it and choose the next best candidate. Never return a previously failed ID.
        4. If an anchor is completely missing from the scene graph, set matched_object_id to 'NONE' and needs_exploration to true.
        5. CRITICAL: Be highly flexible with object labels. A 'couch' is a 'sofa', a 'tv' is a 'monitor', a 'desk' is a 'table'. Match based on visual function and synonyms rather than strictly requiring the exact string.
        """
        
        messages = [{"role": "user", "parts": [{"text": prompt}]}]
        
        # Attach the image so the agent can visually verify "2 seater"
        if blackboard.current_image_path and os.path.exists(blackboard.current_image_path):
            mime = mimetypes.guess_type(blackboard.current_image_path)[0] or "image/png"
            messages[0]["parts"].append({
                "inline_data": {
                    "mime_type": mime, 
                    "data": encode_image(blackboard.current_image_path)
                }
            })
        
        try:
            resp = self.model.generate_content(
                messages,
                generation_config=genai.GenerationConfig(
                    response_mime_type="application/json",
                    temperature=0.1,
                    response_schema=self.schema,
                ),
            )
            parsed = json.loads(resp.text)
            blackboard.append_event("Grounding", "MatchObjects", parsed, "PASS")
            return parsed
        except Exception as e:
            error_msg = f"Visual grounding failed: {e}"
            blackboard.append_event("Grounding", "Error", error_msg, "FAIL")
            return {"error": error_msg, "needs_exploration": True}