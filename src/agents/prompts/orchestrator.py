"""Orchestrator prompt: deconstruct the query into a typed plan.

Text lifted from the legacy claude family (the most complete variant:
the openai/alibaba copies had silently lost the schema instruction
block, so the four backends were not seeing the same prompt). The
schema block is rendered from schemas.ORCHESTRATOR_SCHEMA.
"""

import json
from typing import Tuple

from src.agents.schemas import ORCHESTRATOR_SCHEMA

SYSTEM = f"""
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
        {json.dumps(ORCHESTRATOR_SCHEMA, indent=2)}
        """


def render(blackboard) -> Tuple[str, str]:
    user = f"""
        Current Question: {blackboard.question}
        Mode: {blackboard.mode}

        Agent Exact Position: {blackboard.agent_pose_hab}
        Agent Yaw (rad): {blackboard.agent_yaw_rad}

        GLOBAL FAILURE HISTORY (VERIFIER FEEDBACK):
        {blackboard.global_history}

        Previous Execution Ledger (Use this to fix your parsing if the pipeline failed previously):
        {blackboard.get_ledger_str()}
        """
    return SYSTEM, user
