"""Factory for the unified agent stack (MAPG-09).

``create_role(role, provider)`` mirrors the legacy
``AgentFactory.create_agent`` call shape so the planner switch is a
one-line substitution. Backends can be shared across roles or injected
(tests use a fake backend instance).

MAPG-10 model tiering: cfg ``model_tiers`` maps each role to either
null (use the run's main model for that role's provider, no behavior
change: this is the default for every role) or a model name that
overrides the backend model for that role only. The safe demotion
candidates per the cost design (method-architecture-cost.md section
3) are orchestrator and verifier; grounding/spatial must not be
demoted without an ablation. CallLog already records model_name per
call, so tiered runs are fully attributable.
"""

from typing import Any, Dict, List, Optional, Tuple

from src.agents.backends import create_backend
from src.agents.roles import ROLE_CLASSES, VerifierRole

TIERABLE_ROLES = ("orchestrator", "grounding", "spatial", "verifier", "qa")


def resolve_model_tiers(value: Any) -> Tuple[Dict[str, Optional[str]], List[str]]:
    """Normalize cfg ``model_tiers`` to {role: model_name or None}.

    None / missing / empty-string / "null" entries mean "no override"
    (the role uses its provider's default model). Unknown role keys
    produce a warning and are ignored. Never raises.
    """
    tiers: Dict[str, Optional[str]] = {role: None for role in TIERABLE_ROLES}
    warnings: List[str] = []
    if value is None:
        return tiers, warnings
    items = value.items() if hasattr(value, "items") else None
    if items is None:
        warnings.append(
            f"model_tiers is not a mapping ({type(value).__name__}); ignored."
        )
        return tiers, warnings
    for key, model in items:
        role = str(key).lower().strip()
        if role not in tiers:
            warnings.append(
                f"model_tiers has unknown role {key!r}; ignored. "
                f"Valid roles: {TIERABLE_ROLES}."
            )
            continue
        if model is None:
            continue
        name = str(model).strip()
        if not name or name.lower() in ("null", "none", "~"):
            continue
        tiers[role] = name
    return tiers, warnings


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
