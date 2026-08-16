"""Spatial prompt: intrinsic front vector of the reference object.

Text lifted from the legacy claude family, plus the one clarifying
parenthetical the gemini family had grown in rule 3 (the only wording
improvement any non-claude copy carried). The image is attached by the
role as a separate request part.

MAPG-10: the static Task sentence moved into the system text
(verbatim), and the user text is reordered stable-first so a provider
byte-prefix cache can hit across steps: the scene graph block (whose
serializer emits stable nodes first, see src/agents/serialization.py)
leads, and the per-step volatile fields (anchor, frontiers, pose,
history) trail. ``render_parts`` exposes the stable/volatile boundary
so the claude backend can set a cache_control breakpoint on the stable
block; ``render`` returns the identical text as one string.
"""

import json
from typing import Any, Dict, List, Tuple

from src.agents.schemas import SPATIAL_SCHEMA
from src.agents.serialization import serialize_frontiers

SYSTEM = f"""
        SYSTEM: You are a Geometric Orientation Engine.
        YOUR GOAL: Identify the **INTRINSIC FRONT VECTOR** of the Reference Object relative to the Camera.
        CRITICAL RULES:
        1. Output only face orientation (functional front) of the object.
        2. IGNORE DISTANCE.
        3. Check GLOBAL FAILURE HISTORY. If your previous theta/phi values resulted in a rejection, provide an alternative orientation (e.g., perhaps the 'front' is actually a different side).
        4. IF the object is visible in the scene and the grounding agent is not able to ground it, select a frontier towards the object and then check the scene graph. Output this in 'target_frontier_id'. Use 'NONE' if no frontier is needed.

        CAMERA COORDINATES (Egocentric, top-down):
        THETA (azimuth):
          0.00 rad  = Straight ahead (center of image)
          +1.57 rad = LEFT of image
          -1.57 rad (or 4.71) = RIGHT of image
          3.14 rad  = behind camera

        PHI (elevation/tilt of the normal vector):
          0.00 rad = Straight UP (e.g. top of a table)
          1.57 rad = Level with the ground plane (looking straight out horizontally)
          3.14 rad = Straight DOWN (e.g. underside of a surface)

        (Example for a flat table, assuming the 'front' is its top surface normal: phi=0.0)
        (Example for a tv screen, assuming the screen faces horizontally out: phi=1.57)

        Task: Where is the intrinsic front of the Reference Object in the provided image? Output ONLY the egocentric angles for its functional front face relative to the camera view.

        CRITICAL INSTRUCTION: You MUST output exactly ONE valid JSON object matching the schema below. Do not include any other text.
        Schema:
        {json.dumps(SPATIAL_SCHEMA, indent=2)}
        """


def render_parts(
    blackboard, anchor_obj: Dict[str, Any]
) -> Tuple[str, List[Tuple[str, bool]]]:
    """(system, [(user_text, cacheable), ...]); concatenating the user
    texts reproduces ``render`` byte for byte."""
    stable = f"""
        Environment Scene Graph (Topological Layout):
        {blackboard.scene_graph_str}
        """
    volatile = f"""
        Reference Object: {anchor_obj.get("name", "object")} (ID: {anchor_obj.get("id")})

        Anchor Exact Position: {anchor_obj.get("position")}
        Anchor Exact Size: {anchor_obj.get("size")}

        Available Frontiers (id (x, y, z)):
        {serialize_frontiers(blackboard.available_frontiers)}

        Agent Exact Position: {blackboard.agent_pose_hab}
        Agent Yaw (rad): {blackboard.agent_yaw_rad}

        GLOBAL FAILURE HISTORY (VERIFIER FEEDBACK):
        {blackboard.global_history}
        """
    return SYSTEM, [(stable, True), (volatile, False)]


def render(blackboard, anchor_obj: Dict[str, Any]) -> Tuple[str, str]:
    system, parts = render_parts(blackboard, anchor_obj)
    return system, "".join(text for text, _cacheable in parts)
