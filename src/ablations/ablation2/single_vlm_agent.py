import os
import json
import base64
import mimetypes
import google.generativeai as genai
from typing import Dict, Any

from src.multi_agent.blackboard import Blackboard

def encode_image(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

class SingleVLMAgent:
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
            
        object_ids = [str(o.get("id")) for o in blackboard.available_objects]
        if not object_ids:
            object_ids = ["NONE"]

        all_ids = list(set(object_ids + frontier_ids + ["NONE"]))
        
        # Determine schema properties based on whether it's an MCQ
        properties = {
            "reasoning": genai.protos.Schema(
                type=genai.protos.Type.STRING, 
                description="Think step-by-step about what object or location to navigate to based on the current image and scene graph, or deduce the final answer."
            ),
            "action_type": genai.protos.Schema(
                type=genai.protos.Type.STRING,
                enum=["goto_object", "goto_frontier", "lookaround", "answer", "goto_xyz"],
                description="The action to take. 'goto_object' to go to a known object, 'goto_frontier' to explore, 'lookaround' to spin in place, 'answer' if the final answer is known, 'goto_xyz' to go to a specific continuous point in space."
            ),
            "chosen_id": genai.protos.Schema(
                type=genai.protos.Type.STRING, 
                description="The Node ID of the object or frontier to navigate to. Use 'NONE' if action_type is lookaround or answer or goto_xyz.",
                enum=all_ids
            ),
            "xyz_target": genai.protos.Schema(
                type=genai.protos.Type.ARRAY,
                description="If action_type is goto_xyz, provide [X, Y, Z] world coordinates. Target Y should usually be flat (e.g. match your current Y). Leave empty for other actions.",
                items=genai.protos.Schema(type=genai.protos.Type.NUMBER)
            ),
            "confidence": genai.protos.Schema(
                type=genai.protos.Type.NUMBER,
                description="Confidence score between 0.0 and 1.0 of the chosen action or answer."
            )
        }
        
        if getattr(blackboard, "choices", None):
            properties["answer"] = genai.protos.Schema(
                type=genai.protos.Type.STRING,
                enum=blackboard.choices,
                description="Pick exactly one exact answer from the choices provided when action_type is 'answer'."
            )
        else:
             properties["answer"] = genai.protos.Schema(
                type=genai.protos.Type.STRING,
                description="The text answer to the question if action_type is 'answer'. Otherwise leave empty."
            )

        schema = genai.protos.Schema(
            type=genai.protos.Type.OBJECT,
            properties=properties,
            required=["reasoning", "action_type", "chosen_id", "xyz_target", "confidence", "answer"]
        )

        sys_prompt = """
        You are an End-to-End Visual Language Model agent.
        You must decide the next navigation step for a robot or answer the user's explicit question based on the visual input and topological scene graph.
        
        CRITICAL RULES:
        1. If you need to explore a new area because the target is missing, set action_type="goto_frontier" and chosen_id to a logical frontier ID.
        2. If you need to move to an object to verify or fulfill the spatial goal, set action_type="goto_object" and chosen_id to an object ID.
        3. If you know the final answer or you reached the target location, set action_type="answer" and provide the "answer" text.
        4. If the user asked for a continuous coordinate (e.g. 3 meters in front of a sofa), and you found the sofa, you can set action_type="goto_xyz" and provide the metric [X, Y, Z] coordinates in "xyz_target".
        5. Check the GLOBAL FAILURE HISTORY to avoid repeating mistakes or looping between the same frontiers.
        """
        
        prompt = f"""
        {sys_prompt}
        
        Current Question: {blackboard.question}
        Mode: {blackboard.mode}
        {"Choices: " + json.dumps(blackboard.choices) if getattr(blackboard, "choices", None) else ""}
        
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
            blackboard.append_event("SingleVLM", "EndToEnd", parsed, "PASS")
            parsed["ok"] = True
            return parsed
        except Exception as e:
            error_msg = f"Failed to run SingleVLM agent: {e}"
            blackboard.append_event("SingleVLM", "Error", error_msg, "FAIL")
            return {"ok": False, "error": error_msg}
