"""Model tiering resolution and wiring (MAPG-10).

cfg ``model_tiers``: null per role = the run's main model (default,
no behavior change); a model name overrides the backend model for that
role only.
"""

from types import SimpleNamespace

from src.agents.factory import TIERABLE_ROLES, create_role, resolve_model_tiers


def test_default_is_all_null():
    tiers, warnings = resolve_model_tiers(None)
    assert tiers == {role: None for role in TIERABLE_ROLES}
    assert warnings == []


def test_null_like_values_mean_no_override():
    tiers, warnings = resolve_model_tiers(
        {
            "orchestrator": None,
            "grounding": "",
            "spatial": "null",
            "verifier": "  ",
            "qa": "~",
        }
    )
    assert tiers == {role: None for role in TIERABLE_ROLES}
    assert warnings == []


def test_named_override_survives_and_missing_roles_default():
    tiers, warnings = resolve_model_tiers(
        {"verifier": "claude-haiku-4-5", "orchestrator": "claude-haiku-4-5"}
    )
    assert tiers["verifier"] == "claude-haiku-4-5"
    assert tiers["orchestrator"] == "claude-haiku-4-5"
    assert tiers["grounding"] is None
    assert tiers["spatial"] is None
    assert tiers["qa"] is None
    assert warnings == []


def test_unknown_role_warns_and_is_ignored():
    tiers, warnings = resolve_model_tiers({"logical": "claude-haiku-4-5"})
    assert tiers == {role: None for role in TIERABLE_ROLES}
    assert len(warnings) == 1
    assert "logical" in warnings[0]


def test_non_mapping_warns_and_defaults():
    tiers, warnings = resolve_model_tiers("claude-haiku-4-5")
    assert tiers == {role: None for role in TIERABLE_ROLES}
    assert len(warnings) == 1


def test_omegaconf_like_mapping_is_accepted():
    class MappingLike:
        def __init__(self, data):
            self._data = data

        def items(self):
            return self._data.items()

    tiers, warnings = resolve_model_tiers(
        MappingLike({"verifier": "claude-haiku-4-5"})
    )
    assert tiers["verifier"] == "claude-haiku-4-5"
    assert warnings == []


def test_create_role_constructs_backend_with_override_model():
    # No SDK needed: backends import their SDKs lazily at transport time.
    role = create_role("verifier", provider="claude",
                       model_name="claude-haiku-4-5")
    assert role.backend.model_name == "claude-haiku-4-5"
    # Null override keeps the provider default.
    role_default = create_role("verifier", provider="claude", model_name=None)
    assert role_default.backend.model_name == "claude-opus-4-6"


def test_create_role_override_reaches_call_accounting():
    from src.results.calls import model_name_of

    role = create_role("orchestrator", provider="claude",
                       model_name="claude-haiku-4-5")
    assert model_name_of(role) == "claude-haiku-4-5"
