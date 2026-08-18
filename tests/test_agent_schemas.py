"""Schema round-trip and counted-failure tests (MAPG-09).

The validator is the strictest superset of what any legacy family
enforced: required everywhere (openai .parse), enums (gemini protos),
numeric coercion (claude float()). A malformed payload is a counted
schema_invalid failure at the role level, never a silent fallback.
"""

import pytest

from src.agents.roles.grounding import GroundingRole
from src.agents.roles.orchestrator import OrchestratorRole
from src.agents.roles.qa import QaRole
from src.agents.roles.spatial import SpatialRole
from src.agents.roles.verifier import VerifierRole
from src.agents.schemas import ROLES, SCHEMAS, SchemaError, try_validate, validate
from tests.fake_backend import FakeBackend
from tests.golden.context import (
    ANCHOR_OBJ,
    ORCHESTRATOR_OUTPUT,
    make_mcq_blackboard,
    make_where_blackboard,
)

EXAMPLES = {
    "orchestrator": {
        "reasoning": "Front of the tv, 3 meters out.",
        "target_entity": "location",
        "anchors": [{"label": "tv", "modifiers": "large", "metric": "3.0 meters"}],
        "composition_logic": "intrinsic_front",
        "requires_logical_reasoning": False,
    },
    "grounding": {
        "reasoning": "The large tv matches object_1.",
        "grounded_anchors": [
            {"anchor_label": "tv", "matched_object_id": "object_1", "confidence": 0.9}
        ],
        "needs_exploration": False,
    },
    "spatial": {
        "reasoning": "Screen faces the camera slightly left.",
        "theta_radians": 0.3,
        "phi_radians": 1.57,
        "target_frontier_id": "NONE",
    },
    "verifier": {
        "reasoning": "theta_cam aligns with the visible tv front.",
        "status": "PASS",
        "feedback": "",
    },
    "qa": {
        "prior_hypothesis": "The room is a living room.",
        "hypothesis_likelihood": "high",
        "reasoning": "Sofa and tv are living-room evidence.",
        "action_type": "answer",
        "chosen_id": "NONE",
        "confidence": 0.9,
        "answer": "D",
    },
}


# ----------------------------------------------------------------------
# Round trips
# ----------------------------------------------------------------------
@pytest.mark.parametrize("role", list(ROLES))
def test_example_payload_round_trips(role):
    coerced = validate(role, EXAMPLES[role])
    # Idempotent: validating the coerced output changes nothing.
    assert validate(role, coerced) == coerced
    for key in SCHEMAS[role]["required"]:
        assert key in coerced


def test_numbers_are_coerced_to_float():
    payload = dict(EXAMPLES["spatial"], theta_radians=1, phi_radians="1.57")
    coerced = validate("spatial", payload)
    assert coerced["theta_radians"] == 1.0
    assert coerced["phi_radians"] == 1.57


def test_extra_keys_are_preserved():
    payload = dict(EXAMPLES["verifier"], extra_note="kept")
    assert validate("verifier", payload)["extra_note"] == "kept"


# ----------------------------------------------------------------------
# Malformed payloads are explicit failures with reasons
# ----------------------------------------------------------------------
@pytest.mark.parametrize("role", list(ROLES))
def test_missing_required_field_fails(role):
    payload = dict(EXAMPLES[role])
    dropped = SCHEMAS[role]["required"][0]
    del payload[dropped]
    with pytest.raises(SchemaError) as err:
        validate(role, payload)
    assert dropped in str(err.value)


def test_bad_enum_fails():
    with pytest.raises(SchemaError):
        validate("orchestrator", dict(EXAMPLES["orchestrator"], composition_logic="behind"))
    with pytest.raises(SchemaError):
        validate("verifier", dict(EXAMPLES["verifier"], status="MAYBE"))
    with pytest.raises(SchemaError):
        validate("qa", dict(EXAMPLES["qa"], answer="E"))


def test_bad_types_fail():
    with pytest.raises(SchemaError):
        validate("spatial", dict(EXAMPLES["spatial"], theta_radians="left"))
    with pytest.raises(SchemaError):
        validate("grounding", dict(EXAMPLES["grounding"], grounded_anchors="object_1"))
    with pytest.raises(SchemaError):
        validate("orchestrator", dict(EXAMPLES["orchestrator"], requires_logical_reasoning="yes"))


def test_nested_item_errors_carry_paths():
    payload = dict(
        EXAMPLES["grounding"],
        grounded_anchors=[{"anchor_label": "tv", "confidence": 0.9}],
    )
    ok, _, errors = try_validate("grounding", payload)
    assert not ok
    assert any("grounded_anchors[0].matched_object_id" in e for e in errors)


def test_non_dict_payload_fails():
    with pytest.raises(SchemaError):
        validate("qa", ["not", "an", "object"])


# ----------------------------------------------------------------------
# Role level: schema_invalid is a counted, logged failure
# ----------------------------------------------------------------------
def _broken(role):
    payload = dict(EXAMPLES[role])
    del payload["reasoning"]
    return payload


def test_orchestrator_counts_schema_invalid():
    bb = make_where_blackboard()
    fake = FakeBackend({"OrchestratorOutput": _broken("orchestrator")})
    out = OrchestratorRole(fake).process(bb)
    assert out.get("schema_invalid") is True
    assert "schema_invalid" in out["error"]
    assert out["usage"]["prompt_tokens"] == 111  # failed call still billed
    assert bb.event_ledger[-1]["status"] == "FAIL"
    assert bb.event_ledger[-1]["type"] == "SchemaInvalid"


def test_grounding_schema_invalid_requests_exploration():
    bb = make_where_blackboard()
    fake = FakeBackend({"GroundingOutput": _broken("grounding")})
    out = GroundingRole(fake).process(bb, ORCHESTRATOR_OUTPUT)
    assert out.get("schema_invalid") is True
    assert out["needs_exploration"] is True


def test_spatial_schema_invalid_is_not_ok(tmp_path):
    img = tmp_path / "img.png"
    img.write_bytes(b"png")
    bb = make_where_blackboard(image_path=str(img))
    fake = FakeBackend({"SpatialOutput": _broken("spatial")})
    out = SpatialRole(fake).process(bb, ANCHOR_OBJ)
    assert out["ok"] is False
    assert out.get("schema_invalid") is True


def test_qa_schema_invalid_is_not_ok():
    bb = make_mcq_blackboard()
    fake = FakeBackend({"QaOutput": _broken("qa")})
    out = QaRole(fake).process(bb)
    assert out["ok"] is False
    assert out.get("schema_invalid") is True


def test_verifier_llm_schema_invalid_fails_closed():
    bb = make_where_blackboard()
    fake = FakeBackend({"VerifierOutput": _broken("verifier")})
    checks = {"all_ok": True}
    out = VerifierRole(fake, llm_enabled=True).process(bb, checks=checks)
    assert out["status"] == "FAIL"
    assert "schema_invalid" in out["llm_error"]
    assert out["llm_used"] is True


def test_verifier_contract_from_planner_checks():
    """The a844d6b contract: failed checks -> FAIL without any LLM,
    passing checks with llm disabled -> PASS without any LLM."""
    bb = make_where_blackboard()
    fake = FakeBackend({})
    role = VerifierRole(fake, llm_enabled=False)
    failing = {
        "all_ok": False,
        "in_scene_bounds": {"ok": False, "reason": "outside AABB", "skipped": False},
    }
    out = role.process(bb, checks=failing)
    assert out["status"] == "FAIL"
    assert "in_scene_bounds" in out["feedback"]
    assert out["llm_used"] is False
    out = role.process(bb, checks={"all_ok": True})
    assert out["status"] == "PASS"
    assert out["llm_used"] is False
    assert fake.sent == []  # the LLM was never touched
