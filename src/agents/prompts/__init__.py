"""ONE prompt template per role, backend-independent (MAPG-09 / P1).

Every module exposes ``render(...) -> (system_text, user_text)``. The
returned strings are the complete prompt text for the role; backends
package them without editing a byte, and image attachment happens
OUTSIDE these strings (as a separate request part chosen by the role),
so the text every provider receives is identical by construction.
Golden snapshots in tests/golden/ pin the rendered output; regenerate
with ``python tests/golden/regen.py`` after any deliberate wording
change so the diff is explicit and reviewed.

Template provenance: the claude family carried the most complete
wording (schema instructions included), so its text is the base, with
the one clarifying addition the gemini family had grown (spatial rule
3 parenthetical). The schema block embedded in each system prompt is
rendered from src/agents/schemas.py, the same dict the validator
enforces.
"""

from src.agents.prompts import grounding, orchestrator, qa, spatial, verifier

RENDERERS = {
    "orchestrator": orchestrator.render,
    "grounding": grounding.render,
    "spatial": spatial.render,
    "verifier": verifier.render,
    "qa": qa.render,
}
