import os
import json
import math
import base64
import mimetypes
import google.generativeai as genai
from typing import Dict, Any

from ..blackboard import Blackboard

def encode_image(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

class SpatialAgent:
    def __init__(self, model_name="models/gemini-3-pro-preview"):
        self.model = genai.GenerativeModel(model_name=model_name)
        self.schema = genai.protos.Schema(
            type=genai.protos.Type.OBJECT,
            properties={
                "reasoning": genai.protos.Schema(type=genai.protos.Type.STRING),
                "theta_radians": genai.protos.Schema(type=genai.protos.Type.NUMBER),
                "phi_radians": genai.protos.Schema(type=genai.protos.Type.NUMBER),
                "target_frontier_id": genai.protos.Schema(type=genai.protos.Type.STRING),
            },
            required=["reasoning", "theta_radians", "phi_radians", "target_frontier_id"],
        )

    def process(self, blackboard: Blackboard, anchor_obj: Dict[str, Any]) -> Dict[str, Any]:
        if not blackboard.current_image_path or not os.path.exists(blackboard.current_image_path):
            blackboard.append_event("Spatial", "Error", "No image for spatial kernel", "FAIL")
            return {"ok": False, "error": "No image available."}

        sys_prompt = """
        SYSTEM: You are a Geometric Orientation Engine.
        YOUR GOAL: Identify the **INTRINSIC FRONT VECTOR** of the Reference Object relative to the Camera.
        CRITICAL RULES:
        1. Output only face orientation (functional front) of the object.
        2. IGNORE DISTANCE.
        3. Check GLOBAL FAILURE HISTORY. If your previous theta/phi values resulted in a rejection, provide an alternative orientation (e.g., perhaps the 'front' is actually a different side).
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
        {sys_prompt}
        Reference Object: {anchor_obj.get("name", "object")} (ID: {anchor_obj.get("id")})
        Task: Where is the intrinsic front of this object in the provided image?
        
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
        messages = [
            {
                "role": "user",
                "parts": [
                    {"text": prompt},
                    {"inline_data": {"mime_type": mime, "data": encode_image(blackboard.current_image_path)}},
                ],
            }
        ]
        
        try:
            resp = self.model.generate_content(
                messages,
                generation_config=genai.GenerationConfig(response_mime_type="application/json", temperature=0.2, response_schema=self.schema)
            )
            d = json.loads(resp.text)
            
            # Convert camera theta to world theta
            theta_cam = float(d["theta_radians"])
            two_pi = 2.0 * math.pi
            theta_world = (blackboard.agent_yaw_rad + theta_cam) % two_pi
            
            out = {
                "ok": True,
                "theta": theta_world,         # World coordinate
                "theta_cam": theta_cam,       # Egocentric coordinate
                "agent_yaw": blackboard.agent_yaw_rad,
                "phi": float(d["phi_radians"]),
                "kappa": 0.0, # Placeholder, Engine calculates this
                "target_frontier_id": d.get("target_frontier_id", "NONE"),
                "reasoning": d["reasoning"]
            }
            blackboard.append_event("Spatial", "KernelParams", out, "PASS")
            return out
        except Exception as e:
            blackboard.append_event("Spatial", "Error", str(e), "FAIL")
            return {"ok": False, "error": str(e)}