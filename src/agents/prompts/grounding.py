"""Grounding prompt: link anchor descriptions to scene-graph ids.

Text lifted from the legacy claude family. The image is attached by
the role as a separate request part; nothing in this text depends on
whether an image is present.

MAPG-10: the static Task rules moved from the per-step user text into
the system text (verbatim), so the cacheable stable prefix carries
them and the volatile user text is data only. The candidate list is
rendered by src/agents/serialization.serialize_candidates (one line
per object, 2 dp) instead of ``json.dumps(..., indent=2)``.
"""

import json
from typing import Any, Dict, Tuple

from src.agents.schemas import GROUNDING_SCHEMA
from src.agents.serialization import serialize_candidates

SYSTEM = f"""
        SYSTEM: You are the Visual Grounding Agent.
        YOUR GOAL: Link semantic anchor descriptions to specific object IDs in the robot's current scene graph by looking at the image.

        Task:
        1. Find the best matching object ID for each anchor.
        2. Look at the provided image. If an anchor has a modifier (e.g., '2 seater', 'next to the wall'), you MUST use the image to verify which scene graph candidate visually matches that description.
        3. CRITICAL: Read the GLOBAL FAILURE HISTORY carefully. If the Verifier explicitly rejected a specific object_id for an anchor, you MUST BLACKLIST it and choose the next best candidate. Never return a previously failed ID.
        4. If an anchor is completely missing from the scene graph, set matched_object_id to 'NONE' and needs_exploration to true.
        5. CRITICAL: Be highly flexible with object labels. A 'couch' is a 'sofa', a 'tv' is a 'monitor', a 'desk' is a 'table'. Match based on visual function and synonyms rather than strictly requiring the exact string.

        CRITICAL INSTRUCTION: You MUST output exactly ONE valid JSON object matching the schema below. Do not include any other text.
        Schema:
        {json.dumps(GROUNDING_SCHEMA, indent=2)}
        """


def render(blackboard, orchestrator_output: Dict[str, Any]) -> Tuple[str, str]:
    user = f"""
        Target Entity to find: {orchestrator_output.get("target_entity")}
        Anchors to map: {json.dumps(orchestrator_output.get("anchors", []))}

        Current Scene Graph Candidates (id name (x, y, z)):
        {serialize_candidates(blackboard.available_objects)}

        Agent Exact Position: {blackboard.agent_pose_hab}
        Agent Yaw (rad): {blackboard.agent_yaw_rad}

        Agent Current Semantic State: {blackboard.agent_semantic_state}

        GLOBAL FAILURE HISTORY:
        {blackboard.global_history}
        """
    return SYSTEM, user
