import os
import json
import re
import anthropic
from typing import Dict, Any
from pydantic import BaseModel, Field

from src.multi_agent.blackboard import Blackboard

class LogicalOutput(BaseModel):
    reasoning: str = Field(description="Logical deduction steps explaining why choosing this particular frontier is most likely to lead to the target anchor object based on the scene graph.")
    target_frontier_id: str = Field(description="The Scene Graph node ID of the chosen frontier. Must be exactly one of the IDs provided in Available Frontiers.")

class ClaudeLogicalAgent:
    def __init__(self, model_name="claude-opus-4-6"):
        if "CLAUDE_API_KEY" not in os.environ:
            raise RuntimeError("CLAUDE_API_KEY must be set in the environment.")
        self.model_name = model_name
        self.client = anthropic.Anthropic(api_key=os.environ["CLAUDE_API_KEY"])

    def process(self, blackboard: Blackboard) -> Dict[str, Any]:
        frontier_ids = [str(f.get("id")) for f in blackboard.available_frontiers]
        if not frontier_ids:
             blackboard.append_event("Logical(Claude)", "Error", "No frontiers available for exploration.", "FAIL")
             return {"ok": False, "error": "No frontiers available"}

        sys_prompt = f"""
        SYSTEM: You are a Logical Exploration Engine for a robotic spatial reasoning pipeline.
        YOUR GOAL: The primary 'Anchor Object' required to answer the user's question has NOT been found in the current Scene Graph. You must select the MOST LOGICAL UNEXPLORED FRONTIER to navigate to, in order to expand the scene graph and find the target object.
        
        CRITICAL RULES:
        1. Examine the Question closely. What kind of object is needed to answer it? (e.g. 'Fridge', 'Bed', 'Sink')
        2. Examine the Environment Scene Graph. Which room/region makes the most semantic sense for this object to be located in? 
        3. Examine the list of Available Frontiers. Select the frontier that is mathematically/semantically closest to or inside the most likely room.
        4. If the user is asking about a 'Bed', choosing a frontier located inside a 'Bedroom' is far superior to a 'Bathroom' frontier.
        5. Check GLOBAL FAILURE HISTORY. Try to pick a new direction if a previous one did not yield results.
        6. You MUST return exactly one `target_frontier_id` from the provided Available Frontiers list.
        
        CRITICAL INSTRUCTION: You MUST output exactly ONE valid JSON object matching the schema below. Do not include any other text.
        Schema:
        {json.dumps(LogicalOutput.model_json_schema(), indent=2)}
        """
        
        prompt = f"""
        Question: {blackboard.question}
        
        Agent Exact Position: {blackboard.agent_pose_hab}
        Agent Yaw (rad): {blackboard.agent_yaw_rad}
        
        Available Frontiers (CHOOSE ONE ID FROM HERE):
        {json.dumps([{"id": f["id"], "position": f.get("position")} for f in blackboard.available_frontiers], indent=2)}
        
        Environment Scene Graph (Topological Layout):
        {blackboard.scene_graph_str}
        
        Current Environment Semantic State (Agent Room Node):
        {blackboard.agent_semantic_state}
        
        GLOBAL FAILURE HISTORY (VERIFIER FEEDBACK):
        {blackboard.global_history}
        """
        
        try:
            response = self.client.messages.create(
                model=self.model_name,
                max_tokens=2048,
                system=sys_prompt,
                messages=[
                    {"role": "user", "content": [{"type": "text", "text": prompt}]}
                ]
            )
            text = response.content[0].text.strip()
            if "```json" in text:
                text = text.split("```json")[-1].split("```")[0].strip()
            elif "```" in text:
                text = text.split("```")[-1].split("```")[0].strip()
                
            parsed = json.loads(text)
            
            if parsed.get("target_frontier_id") not in frontier_ids:
                parsed["target_frontier_id"] = frontier_ids[0]
            
            blackboard.append_event("Logical(Claude)", "FrontierSelection", parsed, "PASS")
            parsed["ok"] = True
            return parsed
        except Exception as e:
            error_msg = f"Failed to infer logic (Claude): {e}"
            blackboard.append_event("Logical(Claude)", "Error", error_msg, "FAIL")
            return {"ok": False, "error": error_msg}
