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

class QaAgent:
    def __init__(self, model_name="models/gemini-3-pro-preview"):
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
        
        is_mcq = bool(getattr(blackboard, "choices", None))
        
        properties = {
            "prior_hypothesis": genai.protos.Schema(
                type=genai.protos.Type.STRING, 
                description="Based on the question, formulate a prior hypothesis of what the expected answer or target location should be."
            ),
            "hypothesis_likelihood": genai.protos.Schema(
                type=genai.protos.Type.STRING, 
                enum=["low", "medium", "high"],
                description="Given the current evidence in the scene graph and visual input, how likely is your prior hypothesis to be correct?"
            ),
            "reasoning": genai.protos.Schema(
                type=genai.protos.Type.STRING, 
                description="Break down the query to identify the required target object, map the answer choices to symbols, and deduce the logical next step."
            ),
            "action_type": genai.protos.Schema(
                type=genai.protos.Type.STRING,
                enum=["goto_object", "goto_frontier", "lookaround", "answer"],
                description="The action to take. 'goto_object' to go to a known object, 'goto_frontier' to explore for missing context, 'lookaround' to spin, 'answer' if the final answer is definitively known now."
            ),
            "chosen_id": genai.protos.Schema(
                type=genai.protos.Type.STRING, 
                description="The Node ID of the object or frontier to navigate to. Use 'NONE' if action_type is lookaround or answer.",
                enum=all_ids
            ),
            "confidence": genai.protos.Schema(
                type=genai.protos.Type.NUMBER,
                description="Confidence score between 0.0 and 1.0 of the chosen action or answer."
            )
        }
        
        if is_mcq:
            properties["answer"] = genai.protos.Schema(
                type=genai.protos.Type.STRING,
                enum=["A", "B", "C", "D", "NONE"],
                description="Pick exactly ONE option symbol (A, B, C, or D) from the choices provided when action_type is 'answer'. Never pick an option that does not exist. Use 'NONE' if no answer yet."
            )
        else:
            properties["answer"] = genai.protos.Schema(
                type=genai.protos.Type.STRING,
                description="The text answer to the question if action_type is 'answer'. Otherwise leave empty."
            )

        schema = genai.protos.Schema(
            type=genai.protos.Type.OBJECT,
            properties=properties,
            required=["prior_hypothesis", "hypothesis_likelihood", "reasoning", "action_type", "chosen_id", "confidence", "answer"]
        )

        sys_prompt = """
        SYSTEM: You are an excellent hierarchical graph planning agent. 
        Your goal is to navigate an unseen environment to confidently answer a multiple-choice question about the environment.
        As you explore the environment, your sensors are building a scene graph representation (in json format) and you have access to that scene graph.
        
        CRITICAL RULES:
        1. Parse the query to figure out what object or area is being referred to.
        2. Break down the answer choices into variables/symbols (A, B, C...).
        3. STRICT RULE: If an option contains the text "(DO NOT SELECT THIS OPTION)", you MUST NOT select it under any circumstances. It is a trap.
        4. Formulate a prior hypothesis for the question. What do you expect the answer to be based on the choices and current environment? Explain it in the `prior_hypothesis` field.
        5. Evaluate the evidence collected so far. In the `reasoning` field, explicitly discuss your prior hypothesis and its likelihood of being correct given the current scene graph and visual input.
        6. Set `hypothesis_likelihood` to "high" if you have a strong educated guess (even without absolute certainty), "medium" if you have some evidence, or "low" if you are completely guessing.
        7. If `hypothesis_likelihood` is "high", choose `action_type="answer"` right there. Provide EXACTLY the option symbol (A, B, C, or D) in the `answer` field.
        8. If you are uncertain (`hypothesis_likelihood` is not "high") and should explore more to ground your answer, set `action_type` to something else and `answer` to "NONE". You can take two kinds of steps: `goto_object` or `goto_frontier`.
        9. `action_type="goto_object"`: Navigates near a certain object in the scene graph. Choose this action to get a good view of the region around this object, if you think going near this object will help verify your hypothesis. Put its ID in `chosen_id`.
        10. `action_type="goto_frontier"`: If you think that going near any of the object nodes in the current scene graph will not provide you with any useful information to verify your hypothesis, choose this action to expand the scene graph. Put its ID in `chosen_id`.
        11. Report your numerical confidence (0.0 to 1.0) in the `confidence` field. Pay close attention to the GLOBAL FAILURE HISTORY to avoid repeating mistakes.
        """
        
        prompt = f"""
        {sys_prompt}
        
        Current Question: {blackboard.question}
        Mode: {blackboard.mode}
        {"Choices: " + json.dumps(blackboard.choices) if getattr(blackboard, "choices", None) else ""}
        
        Agent Exact Position: {blackboard.agent_pose_hab}
        Agent Yaw (rad): {blackboard.agent_yaw_rad}
        
        Scene Graph Candidates (with exact positions):
        {json.dumps([{'id': o.get('id', ''), 'name': o.get('name', ''), 'position': o.get('position')} for o in blackboard.available_objects if isinstance(o, dict)], indent=2)}
        
        Available Frontiers:
        {json.dumps([{'id': f.get('id', ''), 'position': f.get('position')} for f in blackboard.available_frontiers if isinstance(f, dict)], indent=2)}
        
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
            
            out = {
                "ok": True,
                "prior_hypothesis": parsed.get("prior_hypothesis", ""),
                "hypothesis_likelihood": parsed.get("hypothesis_likelihood", "low"),
                "action_type": parsed.get("action_type", "lookaround"),
                "chosen_id": parsed.get("chosen_id", "NONE"),
                "answer": parsed.get("answer", ""),
                "confidence": float(parsed.get("confidence", 0.0)),
                "reasoning": parsed.get("reasoning", "")
            }
            blackboard.append_event("QA", out["action_type"], out, "PASS")
            return out
        except Exception as e:
            error_msg = f"Failed to run QA agent: {e}"
            blackboard.append_event("QA", "Error", error_msg, "FAIL")
            return {"ok": False, "error": error_msg}
