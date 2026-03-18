import os
import json
import google.generativeai as genai
from typing import Dict, Any

from ..blackboard import Blackboard

class OrchestratorAgent:
    def __init__(self, model_name="models/gemini-3-pro-preview"):
        if "GOOGLE_API_KEY" not in os.environ:
            raise RuntimeError("GOOGLE_API_KEY must be set in the environment.")
        genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
        self.model = genai.GenerativeModel(model_name=model_name)
        
        # Define the strict output schema
        self.schema = genai.protos.Schema(
            type=genai.protos.Type.OBJECT,
            properties={
                "reasoning": genai.protos.Schema(type=genai.protos.Type.STRING, description="Think step-by-step about the grammar and spatial layout implied by the query."),
                "target_entity": genai.protos.Schema(type=genai.protos.Type.STRING, description="The main object or area the user wants to find or answer a question about."),
                "anchors": genai.protos.Schema(
                    type=genai.protos.Type.ARRAY,
                    description="List of reference objects used to locate the target.",
                    items=genai.protos.Schema(
                        type=genai.protos.Type.OBJECT,
                        properties={
                            "label": genai.protos.Schema(type=genai.protos.Type.STRING, description="CRITICAL: The base noun ONLY (e.g., 'sofa', 'chair', 'table'). NO ADJECTIVES."),
                            "modifiers": genai.protos.Schema(type=genai.protos.Type.STRING, description="All adjectives, sizes, colors, and spatial hints (e.g., '2 seater', 'next to the wall', 'wooden'). DO NOT put explicit distances here."),
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
                    description="Set to true if the question asks for a specific factual answer/complex deduction beyond just a target navigation location."
                )
            },
            required=["reasoning", "target_entity", "anchors", "composition_logic", "requires_logical_reasoning"]
        )

    def process(self, blackboard: Blackboard) -> Dict[str, Any]:
        sys_prompt = """
        You are the Semantic Orchestrator for a robotic spatial reasoning pipeline.
        Your job is to deconstruct a user's query into a structured execution graph.
        
        CRITICAL RULES:
        1. The 'label' MUST be a single base noun that matches standard indoor scene graph categories (e.g., 'sofa', 'chair', 'bed', 'table').
        2. Any descriptive words (e.g., '2 seater', 'leather') or spatial hints (e.g., 'next to the wall', 'near the window') MUST go into the 'modifiers' field.
        3. Any explicit distances (e.g., '3.0 meters', '5 feet') MUST go ONLY in the 'metric' field.
        4. If the question demands an intelligent factual response or deduction rather than just a navigation coordinate, set `requires_logical_reasoning` to true.
        5. CRITICAL: Review the GLOBAL FAILURE HISTORY. If your exact previous parsing resulted in a failure downstream, CHOOSE A DIFFERENT INTERPRETATION (different anchor, modifier, or logic).
        
        Example 1: "Find the apple between the chair next to the wall and the 2 seater sofa."
        Target: apple. Anchors: [{"label": "chair", "modifiers": "next to the wall", "metric": ""}, {"label": "sofa", "modifiers": "2 seater", "metric": ""}]. Logic: between. Logical Reasoning: false.
        
        Example 2: "Where is the location 3.0 meters in front of the large TV?"
        Target: location. Anchors: [{"label": "tv", "modifiers": "large", "metric": "3.0 meters"}]. Logic: intrinsic_front. Logical Reasoning: false.
        
        Example 3: "Based on the items on the table, what room am I in? A) Kitchen B) Bathroom"
        Target: room identity. Anchors: [{"label": "table", "modifiers": "", "metric": ""}]. Logic: none. Logical Reasoning: true.
        """
        
        prompt = f"""
        {sys_prompt}
        
        Current Question: {blackboard.question}
        Mode: {blackboard.mode}
        
        Agent Exact Position: {blackboard.agent_pose_hab}
        Agent Yaw (rad): {blackboard.agent_yaw_rad}
        
        GLOBAL FAILURE HISTORY (VERIFIER FEEDBACK):
        {blackboard.global_history}
        
        Previous Execution Ledger (Use this to fix your parsing if the pipeline failed previously):
        {blackboard.get_ledger_str()}
        """
        
        messages = [{"role": "user", "parts": [{"text": prompt}]}]
        
        try:
            resp = self.model.generate_content(
                messages,
                generation_config=genai.GenerationConfig(
                    response_mime_type="application/json",
                    temperature=0.1,
                    response_schema=self.schema,
                ),
            )
            parsed = json.loads(resp.text)
            blackboard.append_event("Orchestrator", "ParseQuery", parsed, "PASS")
            return parsed
        except Exception as e:
            error_msg = f"Failed to orchestrate query: {e}"
            blackboard.append_event("Orchestrator", "Error", error_msg, "FAIL")
            return {"error": error_msg}