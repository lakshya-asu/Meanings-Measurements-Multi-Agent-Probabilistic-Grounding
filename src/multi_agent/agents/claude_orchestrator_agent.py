import os
import json
import re
import anthropic
from enum import Enum
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field

from src.multi_agent.blackboard import Blackboard

class CompositionLogicEnum(str, Enum):
    none = "none"
    near = "near"
    between = "between"
    intrinsic_front = "intrinsic_front"
    intrinsic_back = "intrinsic_back"
    intrinsic_left = "intrinsic_left"
    intrinsic_right = "intrinsic_right"

class AnchorSchema(BaseModel):
    label: str = Field(description="CRITICAL: The base noun ONLY (e.g., 'sofa', 'chair', 'table'). NO ADJECTIVES.")
    modifiers: str = Field(description="All adjectives, sizes, colors, and spatial hints (e.g., '2 seater', 'next to the wall', 'wooden'). DO NOT put explicit distances here.")
    metric: str = Field(description="Explicit distances or metrics (e.g., '3.0 meters'). Leave empty if none.")

class OrchestratorOutput(BaseModel):
    reasoning: str = Field(description="Think step-by-step about the grammar and spatial layout implied by the query.")
    target_entity: str = Field(description="The main object or area the user wants to find or answer a question about.")
    anchors: List[AnchorSchema] = Field(description="List of reference objects used to locate the target.")
    composition_logic: CompositionLogicEnum = Field(description="The spatial relationship between the target and the anchors.")
    requires_logical_reasoning: bool = Field(description="Set to true if the question asks for a specific factual answer/complex deduction beyond just a target navigation location.")

class ClaudeOrchestratorAgent:
    def __init__(self, model_name="claude-opus-4-6"):
        if "CLAUDE_API_KEY" not in os.environ:
            raise RuntimeError("CLAUDE_API_KEY must be set in the environment.")
        self.model_name = model_name
        self.client = anthropic.Anthropic(api_key=os.environ["CLAUDE_API_KEY"])

    def process(self, blackboard: Blackboard) -> Dict[str, Any]:
        sys_prompt = f"""
        You are the Semantic Orchestrator for a robotic spatial reasoning pipeline.
        Your job is to deconstruct a user's query into a structured execution graph.
        
        CRITICAL RULES:
        1. The 'label' MUST be a single base noun that matches standard indoor scene graph categories (e.g., 'sofa', 'chair', 'bed', 'table').
        2. Any descriptive words (e.g., '2 seater', 'leather') or spatial hints (e.g., 'next to the wall', 'near the window') MUST go into the 'modifiers' field.
        3. Any explicit distances (e.g., '3.0 meters', '5 feet') MUST go ONLY in the 'metric' field.
        4. If the question demands an intelligent factual response or deduction rather than just a navigation coordinate, set `requires_logical_reasoning` to true.
        5. CRITICAL: Review the GLOBAL FAILURE HISTORY. If your exact previous parsing resulted in a failure downstream, CHOOSE A DIFFERENT INTERPRETATION (different anchor, modifier, or logic).
        
        Example 1: "Find the apple between the chair next to the wall and the 2 seater sofa."
        Target: apple. Anchors: [{{"label": "chair", "modifiers": "next to the wall", "metric": ""}}, {{"label": "sofa", "modifiers": "2 seater", "metric": ""}}]. Logic: between. Logical Reasoning: false.
        
        Example 2: "Where is the location 3.0 meters in front of the large TV?"
        Target: location. Anchors: [{{"label": "tv", "modifiers": "large", "metric": "3.0 meters"}}]. Logic: intrinsic_front. Logical Reasoning: false.
        
        Example 3: "Based on the items on the table, what room am I in? A) Kitchen B) Bathroom"
        Target: room identity. Anchors: [{{"label": "table", "modifiers": "", "metric": ""}}]. Logic: none. Logical Reasoning: true.
        
        CRITICAL INSTRUCTION: You MUST output exactly ONE valid JSON object matching the schema below. Do not include any other text.
        Schema:
        {json.dumps(OrchestratorOutput.model_json_schema(), indent=2)}
        """
        
        prompt = f"""
        Current Question: {blackboard.question}
        Mode: {blackboard.mode}
        
        Agent Exact Position: {blackboard.agent_pose_hab}
        Agent Yaw (rad): {blackboard.agent_yaw_rad}
        
        GLOBAL FAILURE HISTORY (VERIFIER FEEDBACK):
        {blackboard.global_history}
        
        Previous Execution Ledger (Use this to fix your parsing if the pipeline failed previously):
        {blackboard.get_ledger_str()}
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
            
            if isinstance(parsed.get('composition_logic'), Enum):
                 parsed['composition_logic'] = parsed['composition_logic'].value
                 
            blackboard.append_event("Orchestrator(Claude)", "ParseQuery", parsed, "PASS")
            return parsed
        except Exception as e:
            error_msg = f"Failed to orchestrate query (Claude): {e}"
            blackboard.append_event("Orchestrator(Claude)", "Error", error_msg, "FAIL")
            return {"error": error_msg}
