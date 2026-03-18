import os
import json
import math
import base64
import mimetypes
from typing import Dict, Any

from pydantic import BaseModel, Field
from openai import OpenAI
from src.multi_agent.blackboard import Blackboard

def encode_image(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

class SpatialOutput(BaseModel):
    reasoning: str
    theta_radians: float = Field(description="Egocentric Azimuth of the intrinsic front vector")
    phi_radians: float = Field(description="Elevation of the intrinsic front vector. 0.0 rad = Up. 1.57 rad = Level. 3.14 rad = Down.")
    target_frontier_id: str = Field(description="If the object is visible but not grounded, output a frontier id towards the object. Otherwise 'NONE'.")

class OpenAISpatialAgent:
    def __init__(self, model_name="gpt-5.2-chat-latest"):
        if "OPENAI_API_KEY" not in os.environ:
            raise RuntimeError("OPENAI_API_KEY must be set in the environment.")
        self.model_name = model_name
        self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    def process(self, blackboard: Blackboard, anchor_obj: Dict[str, Any]) -> Dict[str, Any]:
        if not blackboard.current_image_path or not os.path.exists(blackboard.current_image_path):
            blackboard.append_event("Spatial(OpenAI)", "Error", "No image for spatial kernel", "FAIL")
            return {"ok": False, "error": "No image available."}

        sys_prompt = """
        SYSTEM: You are a Geometric Orientation Engine.
        YOUR GOAL: Identify the **INTRINSIC FRONT VECTOR** of the Reference Object relative to the Camera.
        CRITICAL RULES:
        1. Output only face orientation (functional front) of the object.
        2. IGNORE DISTANCE.
        3. Check GLOBAL FAILURE HISTORY. If your previous theta/phi values resulted in a rejection, provide an alternative orientation.
        4. IF the object is visible in the scene and the grounding agent is not able to ground it, select a frontier towards the object and then check the scene graph. Output this in 'target_frontier_id'. Use 'NONE' if no frontier is needed.
        
        CAMERA COORDINATES (Egocentric, top-down):
        THETA (azimuth):
          0.00 rad  = Straight ahead (center of image)
          +1.57 rad = LEFT of image
          -1.57 rad (or 4.71) = RIGHT of image
          3.14 rad  = behind camera
          
        PHI (elevation/tilt of the normal vector):
          0.00 rad = Straight UP (e.g. top of a table)
          1.57 rad = Level with the ground plane (looking straight out horizontally)
          3.14 rad = Straight DOWN (e.g. underside of a surface)
          
        (Example for a flat table, assuming the 'front' is its top surface normal: phi=0.0)
        (Example for a tv screen, assuming the screen faces horizontally out: phi=1.57)
        """
        
        prompt = f"""
        Reference Object: {anchor_obj.get("name", "object")} (ID: {anchor_obj.get("id")})
        Task: Where is the intrinsic front of this object in the provided image? Output ONLY the egocentric angles for its functional front face relative to the camera view.
        
        Agent Exact Position: {blackboard.agent_pose_hab}
        Agent Yaw (rad): {blackboard.agent_yaw_rad}
        
        Anchor Exact Position: {anchor_obj.get("position")}
        Anchor Exact Size: {anchor_obj.get("size")}
        
        Available Frontiers: {blackboard.available_frontiers}
        
        Environment Scene Graph (Topological Layout):
        {blackboard.scene_graph_str}
        
        GLOBAL FAILURE HISTORY (VERIFIER FEEDBACK):
        {blackboard.global_history}
        """
        
        mime = mimetypes.guess_type(blackboard.current_image_path)[0] or "image/png"
        b64_img = encode_image(blackboard.current_image_path)
        
        content = [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64_img}"}}
        ]
        
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": content}
        ]
        
        try:
            completion = self.client.beta.chat.completions.parse(
                model=self.model_name,
                messages=messages,
                response_format=SpatialOutput
            )
            parsed = completion.choices[0].message.parsed.model_dump()
            
            # Convert camera theta to world theta
            theta_cam = float(parsed["theta_radians"])
            two_pi = 2.0 * math.pi
            theta_world = (blackboard.agent_yaw_rad + theta_cam) % two_pi
            
            q_lower = blackboard.question.lower()
            if "above" in q_lower:
                phi_val = 0.0
            elif "below" in q_lower:
                phi_val = 3.14
            else:
                phi_val = 1.57
            
            out = {
                "ok": True,
                "theta": theta_world,         
                "theta_cam": theta_cam,       
                "agent_yaw": blackboard.agent_yaw_rad,
                "phi": phi_val,
                "kappa": 0.0, 
                "target_frontier_id": parsed.get("target_frontier_id", "NONE"),
                "reasoning": parsed["reasoning"]
            }
            return out
        except Exception as e:
            blackboard.append_event("Spatial(OpenAI)", "Error", str(e), "FAIL")
            return {"ok": False, "error": str(e)}
