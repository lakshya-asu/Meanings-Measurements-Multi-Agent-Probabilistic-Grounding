import os
import json
import re
import base64
import mimetypes
import anthropic
from typing import Dict, Any, List

from pydantic import BaseModel, Field
from src.multi_agent.blackboard import Blackboard

def encode_image(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

class GroundedAnchor(BaseModel):
    anchor_label: str
    matched_object_id: str = Field(description="The exact ID from the scene graph. Use 'NONE' if not found.")
    confidence: float

class GroundingOutput(BaseModel):
    reasoning: str = Field(description="Explain your visual verification of the modifiers.")
    grounded_anchors: List[GroundedAnchor]
    needs_exploration: bool = Field(description="True if a required anchor is missing from the scene graph AND the image.")

class ClaudeGroundingAgent:
    def __init__(self, model_name="claude-opus-4-6"):
        if "CLAUDE_API_KEY" not in os.environ:
            raise RuntimeError("CLAUDE_API_KEY must be set in the environment.")
        self.model_name = model_name
        self.client = anthropic.Anthropic(api_key=os.environ["CLAUDE_API_KEY"])

    def process(self, blackboard: Blackboard, orchestrator_output: Dict[str, Any]) -> Dict[str, Any]:
        available_objs = [{"id": o.get("id"), "name": o.get("name", ""), "pos": o.get("position")} for o in blackboard.available_objects if isinstance(o, dict)]
        
        sys_prompt = f"""
        SYSTEM: You are the Visual Grounding Agent.
        YOUR GOAL: Link semantic anchor descriptions to specific object IDs in the robot's current scene graph by looking at the image.
        
        CRITICAL INSTRUCTION: You MUST output exactly ONE valid JSON object matching the schema below. Do not include any other text.
        Schema:
        {json.dumps(GroundingOutput.model_json_schema(), indent=2)}
        """
        
        prompt = f"""
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
            response = self.client.messages.create(
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
            blackboard.append_event("Grounding(Claude)", "MatchObjects", parsed, "PASS")
            return parsed
        except Exception as e:
            error_msg = f"Visual grounding failed (Claude): {e}"
            blackboard.append_event("Grounding(Claude)", "Error", error_msg, "FAIL")
            return {"error": error_msg, "needs_exploration": True}
