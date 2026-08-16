"""Factory for the unified agent stack (MAPG-09).

``create_role(role, provider)`` mirrors the legacy
``AgentFactory.create_agent`` call shape so the planner switch is a
one-line substitution. Backends can be shared across roles or injected
(tests use a fake backend instance).
"""

from typing import Any, Optional

from src.agents.backends import create_backend
from src.agents.roles import ROLE_CLASSES, VerifierRole


def create_role(
    role: str,
    provider: str = "claude",
    model_name: Optional[str] = None,
    backend: Optional[Any] = None,
    **kwargs: Any,
):
    role = str(role).lower().strip()
    if role == "logical":
        raise ValueError(
            "The 'logical' role was removed in MAPG-09: it was constructed "
            "but never called by any planner (orphan; EXPERIMENT_PLAN "
            "section 1, 'no orphan components')."
        )
    if role not in ROLE_CLASSES:
        raise ValueError(f"Unknown agent role: {role}")
    if backend is None:
        backend = create_backend(provider, model_name=model_name)
    if role == "verifier":
        return VerifierRole(backend, llm_enabled=bool(kwargs.get("llm_enabled", False)))
    return ROLE_CLASSES[role](backend)
