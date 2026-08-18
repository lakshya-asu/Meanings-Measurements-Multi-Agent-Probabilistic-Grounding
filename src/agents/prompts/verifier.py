"""Verifier LLM-critique prompt.

Only rendered when the optional LLM critique runs (cfg verifier.llm,
default off): the programmatic checks (src/verification/checks.py)
gate every step without a prompt. Text lifted from the legacy claude
family; the openai/alibaba copies had lost the schema block and the
gemini copy folded the system line into the user text, both now
unified. The image is attached by the role as a separate request part.

MAPG-10: the static Task rules moved from the per-step user text into
the system text (verbatim), so the cacheable stable prefix carries
them and the volatile user text is data only.
"""

import json
from typing import Tuple

from src.agents.schemas import VERIFIER_SCHEMA

SYSTEM = f"""
        SYSTEM: You are the Verifier Critic for a robotic spatial reasoning system.

        Task:
        1. Check if the Orchestrator's logic makes sense for the question.
        2. Look at the image: Does the Spatial Agent's calculated 'theta_cam' (egocentric front direction) logically align with the object visible in the image?
        CRITICAL RULE: The Spatial agent reasons in egocentric 'theta_cam' (negative=Right, positive=Left). The system mathematically converts this to a global 'theta' using the agent's yaw. Do NOT flag a contradiction just because the global 'theta' is positive while the text reasoning discusses a negative 'theta_cam'.
        3. CRITICAL RULE: Be lenient on exact label nomenclature. If the Orchestrator/Grounding agents selected a 'sofa' instead of a 'couch', or a 'monitor' instead of a 'tv', DO NOT fail them. Accept synonymous or functionally equivalent object labels.
        4. If an agent hallucinated or made a clear error in the visual mapping, output FAIL with feedback. Otherwise, output PASS.

        CRITICAL INSTRUCTION: You MUST output exactly ONE valid JSON object matching the schema below. Do not include any other text.
        Schema:
        {json.dumps(VERIFIER_SCHEMA, indent=2)}
        """


def render(blackboard) -> Tuple[str, str]:
    user = f"""
        Review the current execution ledger and the provided image.

        Current Question: {blackboard.question}
        Agent current global yaw: {blackboard.agent_yaw_rad:.3f} rad.

        Ledger of actions taken so far in this step:
        {blackboard.get_ledger_str()}
        """
    return SYSTEM, user
