import os
import json
from typing import Dict, Any, Literal
from pydantic import BaseModel, Field

from openai import OpenAI
from src.multi_agent.blackboard import Blackboard

class QaOutput(BaseModel):
    prior_hypothesis: str = Field(description="Based on the question, formulate a prior hypothesis of what the expected answer or target location should be.")
    hypothesis_likelihood: Literal["low", "medium", "high"] = Field(description="Given the current evidence in the scene graph and visual input, how likely is your prior hypothesis to be correct?")
    reasoning: str = Field(description="Break down the query to identify the required target object, map the answer choices to symbols, and deduce the logical next step.")
    action_type: Literal["goto_object", "goto_frontier", "lookaround", "answer"] = Field(description="The action to take. 'goto_object' to go to a known object, 'goto_frontier' to explore for missing context, 'lookaround' to spin, 'answer' if the final answer is definitively known now.")
    chosen_id: str = Field(description="The Node ID of the object or frontier to navigate to. Use 'NONE' if action_type is lookaround or answer.")
    confidence: float = Field(description="Confidence score between 0.0 and 1.0 of the chosen action or answer.")
    answer: Literal["A", "B", "C", "D", "NONE"] = Field(description="If action_type is 'answer', provide EXACTLY the option symbol (A, B, C, or D) from the choices provided. Otherwise use 'NONE'.")

class OpenAIQaAgent:
    def __init__(self, model_name="gpt-5.2-chat-latest"):
        if "OPENAI_API_KEY" not in os.environ:
            raise RuntimeError("OPENAI_API_KEY must be set in the environment.")
        self.model_name = model_name
        self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    def process(self, blackboard: Blackboard) -> Dict[str, Any]:
        frontier_ids = [str(f.get("id")) for f in blackboard.available_frontiers]
        if not frontier_ids:
            frontier_ids = ["NONE"]
            
        object_ids = [str(o.get("id")) for o in blackboard.available_objects]
        if not object_ids:
            object_ids = ["NONE"]

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
        
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": prompt}
        ]
        
        try:
            completion = self.client.beta.chat.completions.parse(
                model=self.model_name,
                messages=messages,
                response_format=QaOutput
            )
            parsed = completion.choices[0].message.parsed.model_dump()
            
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
            
            blackboard.append_event("QA(OpenAI)", out["action_type"], out, "PASS")
            return out
        except Exception as e:
            error_msg = f"Failed to infer MCQ QA (OpenAI): {e}"
            blackboard.append_event("QA(OpenAI)", "Error", error_msg, "FAIL")
            return {"ok": False, "error": error_msg}
