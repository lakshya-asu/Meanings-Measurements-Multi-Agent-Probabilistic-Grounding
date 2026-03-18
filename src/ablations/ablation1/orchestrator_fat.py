import os
import json
import math
import base64
import mimetypes
import google.generativeai as genai
from typing import Dict, Any

from src.multi_agent.blackboard import Blackboard

def encode_image(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

class OrchestratorFatAgent:
    def __init__(self, model_name="models/gemini-2.5-pro"):
        if "GOOGLE_API_KEY" not in os.environ:
            raise RuntimeError("GOOGLE_API_KEY must be set in the environment.")
        genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
        self.model = genai.GenerativeModel(model_name=model_name)

    def process(self, blackboard: Blackboard) -> Dict[str, Any]:
        has_image = blackboard.current_image_path and os.path.exists(blackboard.current_image_path)
        
        frontier_ids = [str(f.get("id")) for f in blackboard.available_frontiers]
        if not frontier_ids:
            frontier_ids = ["NONE"]

        schema = genai.protos.Schema(
            type=genai.protos.Type.OBJECT,
            properties={
                "reasoning": genai.protos.Schema(type=genai.protos.Type.STRING, description="Think step-by-step about the grammar, spatial layout implied by the query, the logical frontier to explore if the objects are missing, and the visual front of the target."),
                "target_entity": genai.protos.Schema(type=genai.protos.Type.STRING, description="The main object or area the user wants to find or answer a question about."),
                "anchors": genai.protos.Schema(
                    type=genai.protos.Type.ARRAY,
                    description="List of reference objects used to locate the target.",
                    items=genai.protos.Schema(
                        type=genai.protos.Type.OBJECT,
                        properties={
                            "label": genai.protos.Schema(type=genai.protos.Type.STRING, description="CRITICAL: The base noun ONLY (e.g., 'sofa', 'chair', 'table')."),
                            "modifiers": genai.protos.Schema(type=genai.protos.Type.STRING, description="All adjectives and spatial hints (e.g., '2 seater', 'next to the wall')."),
                            "metric": genai.protos.Schema(type=genai.protos.Type.STRING, description="Explicit distances or metrics (e.g., '3.0 meters'). Leave empty if none.")
                        }
                    )
                ),
                "composition_logic": genai.protos.Schema(
                    type=genai.protos.Type.STRING,
                    enum=["none", "near", "between", "intrinsic_front", "intrinsic_back", "intrinsic_left", "intrinsic_right"],
                    description="The spatial relationship between the target and the anchors."
                ),
                "requires_logical_reasoning": genai.protos.Schema(
                    type=genai.protos.Type.BOOLEAN, 
                    description="Set to true if the question is multiple choice or asks for a specific factual answer."
                ),
                "theta_cam_radians": genai.protos.Schema(
                    type=genai.protos.Type.NUMBER,
                    description="The egocentric azimuth angle (radians) indicating the intrinsic 'front' of the primary anchor. 0 is straight ahead, positive is left, negative is right."
                ),
                "target_frontier_id": genai.protos.Schema(
                    type=genai.protos.Type.STRING, 
                    description="If the required anchor objects are not in the scene graph, select the most logical frontier ID to explore.",
                    enum=frontier_ids
                )
            },
            required=["reasoning", "target_entity", "anchors", "composition_logic", "requires_logical_reasoning", "theta_cam_radians", "target_frontier_id"]
        )

        sys_prompt = """
        You are the Fat Semantic Orchestrator for an ablated robotic spatial reasoning pipeline.
        You must perform the duties of the Orchestrator, Spatial Agent, and Logical Agent simultaneously.
        
        CRITICAL RULES:
        1. Parse the text query into targets and anchors. 
        2. Identify the 'intrinsic front' orientation (theta_cam_radians) of the anchoring object from the camera's perspective. Positive=Left, Negative=Right. Look at the image!
        3. If the anchor object is not found in the 'Scene Graph Candidates', use the 'Available Frontiers' to intelligently guess which frontier to explore based on the Scene Graph Topology. 
        4. Anchor 'label' MUST be a single base noun.
        5. Check GLOBAL FAILURE HISTORY to avoid repeating mistakes.
        """
        
        prompt = f"""
        {sys_prompt}
        
        Current Question: {blackboard.question}
        Mode: {blackboard.mode}
        
        Agent Exact Position: {blackboard.agent_pose_hab}
        Agent Yaw (rad): {blackboard.agent_yaw_rad}
        
        Scene Graph Candidates (with exact positions):
        {json.dumps([{{'id': o['id'], 'name': o.get('name', ''), 'position': o.get('position')}} for o in blackboard.available_objects], indent=2)}
        
        Available Frontiers:
        {json.dumps([{{'id': f['id'], 'position': f.get('position')}} for f in blackboard.available_frontiers], indent=2)}
        
        Environment Scene Graph (Topological Layout):
        {blackboard.scene_graph_str}
        
        Current Environment Semantic State (Agent Room Node):
        {blackboard.agent_semantic_state}
        
        GLOBAL FAILURE HISTORY (VERIFIER FEEDBACK):
        {blackboard.global_history}
        """
        
        parts = [{"text": prompt}]
        if has_image:
            mime = mimetypes.guess_type(blackboard.current_image_path)[0] or "image/png"
            parts.append({"inline_data": {"mime_type": mime, "data": encode_image(blackboard.current_image_path)}})

        messages = [{"role": "user", "parts": parts}]
        
        try:
            resp = self.model.generate_content(
                messages,
                generation_config=genai.GenerationConfig(
                    response_mime_type="application/json",
                    temperature=0.1,
                    response_schema=schema,
                ),
            )
            parsed = json.loads(resp.text)
            
            # Map theta_cam to global like spatial engine did
            import math
            theta_cam = parsed["theta_cam_radians"]
            cam_yaw = blackboard.agent_yaw_rad
            # Typical spatial transformation
            global_theta_rad = cam_yaw + theta_cam
            
            # Ensure within -pi to pi
            global_theta_rad = (global_theta_rad + math.pi) % (2 * math.pi) - math.pi
            
            parsed["theta_radians"] = global_theta_rad
            parsed["phi_radians"] = 1.57 # Flat elevation
            
            blackboard.append_event("OrchestratorFat", "ParseAndSpatial", parsed, "PASS")
            return parsed
        except Exception as e:
            error_msg = f"Failed to orchestrate fat query: {e}"
            blackboard.append_event("OrchestratorFat", "Error", error_msg, "FAIL")
            return {"error": error_msg}
