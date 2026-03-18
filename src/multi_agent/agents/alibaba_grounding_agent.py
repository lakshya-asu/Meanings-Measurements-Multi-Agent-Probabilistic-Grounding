import os
import json
import base64
import mimetypes
from typing import Dict, Any, List

from pydantic import BaseModel, Field
from openai import OpenAI
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

class AlibabaGroundingAgent:
    def __init__(self, model_name="qwen3-vl-plus"):
        if "ALIBABA_API_KEY" not in os.environ:
            raise RuntimeError("ALIBABA_API_KEY must be set in the environment.")
        self.model_name = model_name
        self.client = OpenAI(
            api_key=os.environ["ALIBABA_API_KEY"],
            base_url="https://dashscope-us.aliyuncs.com/compatible-mode/v1"
        )

    def process(self, blackboard: Blackboard, orchestrator_output: Dict[str, Any]) -> Dict[str, Any]:
        available_objs = [{"id": o.get("id"), "name": o.get("name", ""), "pos": o.get("position")} for o in blackboard.available_objects if isinstance(o, dict)]
        
        sys_prompt = """
        SYSTEM: You are the Visual Grounding Agent.
        YOUR GOAL: Link semantic anchor descriptions to specific object IDs in the robot's current scene graph by looking at the image.
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
                "type": "image_url",
                "image_url": {"url": f"data:{mime};base64,{b64_img}"}
            })
            
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": content}
        ]
        
        try:
            completion = self.client.beta.chat.completions.parse(
                model=self.model_name,
                messages=messages,
                response_format=GroundingOutput
            )
            parsed = completion.choices[0].message.parsed.model_dump()
            blackboard.append_event("Grounding(Alibaba)", "MatchObjects", parsed, "PASS")
            return parsed
        except Exception as e:
            error_msg = f"Visual grounding failed (Alibaba): {e}"
            blackboard.append_event("Grounding(Alibaba)", "Error", error_msg, "FAIL")
            return {"error": error_msg, "needs_exploration": True}
