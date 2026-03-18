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

class LogicalAgent:
    def __init__(self, model_name="models/gemini-3-pro-preview"):
        self.model = genai.GenerativeModel(model_name=model_name)

    def process(self, blackboard: Blackboard) -> Dict[str, Any]:
        has_image = blackboard.current_image_path and os.path.exists(blackboard.current_image_path)
        
        frontier_ids = [str(f.get("id")) for f in blackboard.available_frontiers]
        if not frontier_ids:
             # Fallback if no frontiers are available
             blackboard.append_event("Logical", "Error", "No frontiers available for exploration.", "FAIL")
             return {"ok": False, "error": "No frontiers available"}

        schema = genai.protos.Schema(
            type=genai.protos.Type.OBJECT,
            properties={
                "reasoning": genai.protos.Schema(type=genai.protos.Type.STRING, description="Logical deduction steps explaining why choosing this particular frontier is most likely to lead to the target anchor object based on the scene graph."),
                "target_frontier_id": genai.protos.Schema(
                    type=genai.protos.Type.STRING, 
                    description="The Scene Graph node ID of the chosen frontier.",
                    enum=frontier_ids
                )
            },
            required=["reasoning", "target_frontier_id"],
        )

        sys_prompt = """
        SYSTEM: You are a Logical Exploration Engine for a robotic spatial reasoning pipeline.
        YOUR GOAL: The primary 'Anchor Object' required to answer the user's question has NOT been found in the current Scene Graph. You must select the MOST LOGICAL UNEXPLORED FRONTIER to navigate to, in order to expand the scene graph and find the target object.
        
        CRITICAL RULES:
        1. Examine the Question closely. What kind of object is needed to answer it? (e.g. 'Fridge', 'Bed', 'Sink')
        2. Examine the Environment Scene Graph. Which room/region makes the most semantic sense for this object to be located in? 
        3. Examine the list of Available Frontiers. Select the frontier that is mathematically/semantically closest to or inside the most likely room.
        4. If the user is asking about a 'Bed', choosing a frontier located inside a 'Bedroom' is far superior to a 'Bathroom' frontier.
        5. Check GLOBAL FAILURE HISTORY. Try to pick a new direction if a previous one did not yield results.
        6. You MUST return exactly one `target_frontier_id` from the provided Available Frontiers list.
        """
        
        prompt = f"""
        {sys_prompt}
        
        Question: {blackboard.question}
        
        Agent Exact Position: {blackboard.agent_pose_hab}
        Agent Yaw (rad): {blackboard.agent_yaw_rad}
        
        Available Frontiers:
        {json.dumps([{{'id': f['id'], 'position': f.get('position')}} for f in blackboard.available_frontiers], indent=2)}
        
        Environment Scene Graph (Topological Layout):
        {blackboard.scene_graph_str}
        
        Current Environment Semantic State (Agent Room Node):
        {blackboard.agent_semantic_state}
        
        GLOBAL FAILURE HISTORY:
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
                    temperature=0.2, 
                    response_schema=schema
                )
            )
            d = json.loads(resp.text)
            
            out = {
                "ok": True,
                "target_frontier_id": d["target_frontier_id"],
                "reasoning": d["reasoning"]
            }
            blackboard.append_event("Logical", "SelectFrontier", out, "PASS")
            return out
        except Exception as e:
            blackboard.append_event("Logical", "Error", str(e), "FAIL")
            return {"ok": False, "error": str(e)}
