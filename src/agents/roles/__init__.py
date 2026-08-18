"""Role classes for the unified agent stack (MAPG-09).

Each role takes a backend instance, renders its single shared prompt,
sends it, validates the reply against the role schema, and returns the
same output contract the legacy per-backend classes returned, plus a
``usage`` dict so the planner's CallLog records real token counts.

The ``logical`` role is not ported: it was constructed and never
called (orphan; EXPERIMENT_PLAN section 1), and is deleted with the
legacy files.
"""

from src.agents.roles.grounding import GroundingRole
from src.agents.roles.orchestrator import OrchestratorRole
from src.agents.roles.qa import QaRole
from src.agents.roles.spatial import SpatialRole
from src.agents.roles.verifier import VerifierRole

ROLE_CLASSES = {
    "orchestrator": OrchestratorRole,
    "grounding": GroundingRole,
    "spatial": SpatialRole,
    "verifier": VerifierRole,
    "qa": QaRole,
}
