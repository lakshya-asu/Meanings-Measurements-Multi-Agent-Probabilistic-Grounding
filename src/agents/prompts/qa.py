"""QA prompt: hypothesis-driven MCQ exploration (the EQA fast path).

Text lifted from the legacy claude family. The image is attached by
the role as a separate request part on every backend; the legacy
gemini/alibaba copies attached it while claude/openai silently did
not, one of the divergences MAPG-09 removes.

MAPG-10: the user text is reordered stable-first for prompt caching
(question and choices are episode-constant, the scene graph block is
append-mostly under the compact serializer; candidates, frontiers,
pose, semantic state and history are per-step volatile). Candidate and
frontier lists are compact lines (src/agents/serialization) instead of
``json.dumps(..., indent=2)``. ``render_parts`` exposes the
stable/volatile boundary for the claude cache_control breakpoint.
"""

import json
from typing import List, Tuple

from src.agents.schemas import QA_SCHEMA
from src.agents.serialization import serialize_candidates, serialize_frontiers

SYSTEM = f"""
        SYSTEM: You are an excellent hierarchical graph planning agent.
        Your goal is to navigate an unseen environment to confidently answer a multiple-choice question about the environment.
        As you explore the environment, your sensors are building a scene graph representation and you have access to that scene graph.

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

        CRITICAL INSTRUCTION: You MUST output exactly ONE valid JSON object matching the schema below. Do not include any other text.
        Schema:
        {json.dumps(QA_SCHEMA, indent=2)}
        """


def render_parts(blackboard) -> Tuple[str, List[Tuple[str, bool]]]:
    """(system, [(user_text, cacheable), ...]); concatenating the user
    texts reproduces ``render`` byte for byte."""
    stable = f"""
        Current Question: {blackboard.question}
        Mode: {blackboard.mode}
        {"Choices: " + json.dumps(blackboard.choices) if getattr(blackboard, "choices", None) else ""}

        Environment Scene Graph (Topological Layout):
        {blackboard.scene_graph_str}
        """
    volatile = f"""
        Scene Graph Candidates (id name (x, y, z)):
        {serialize_candidates(blackboard.available_objects)}

        Available Frontiers (id (x, y, z)):
        {serialize_frontiers(blackboard.available_frontiers)}

        Agent Exact Position: {blackboard.agent_pose_hab}
        Agent Yaw (rad): {blackboard.agent_yaw_rad}

        Current Environment Semantic State (Agent Room Node):
        {blackboard.agent_semantic_state}

        GLOBAL FAILURE HISTORY (VERIFIER FEEDBACK):
        {blackboard.global_history}
        """
    return SYSTEM, [(stable, True), (volatile, False)]


def render(blackboard) -> Tuple[str, str]:
    system, parts = render_parts(blackboard)
    return system, "".join(text for text, _cacheable in parts)
