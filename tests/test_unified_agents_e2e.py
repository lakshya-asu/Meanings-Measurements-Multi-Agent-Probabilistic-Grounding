"""Fake-backend end-to-end steps through MultiAgentMSPPlanner (MAPG-09).

Exercises the real planner call sites with the unified roles wired to
a FakeBackend: the MCQ fast path (1 call) and the "where" path
(orchestrator + grounding, then + spatial with the programmatic
verifier), asserting that

- the planner routes through the unified stack (cfg agents_impl
  default) without any provider SDK or API key,
- CallLog rows carry REAL prompt/completion token counts from the
  backend usage (the MAPG-02 gap this ticket closes: token counts
  were None because the legacy agents dropped provider responses),
- the programmatic-only verifier stays out of the call count
  (record_if contract).

The planner module chain still imports legacy gemini planner modules
at package-import time, so absent host SDKs (google.generativeai,
quaternion) are stubbed in sys.modules before the import; the stubs
satisfy module-scope imports only and nothing in the exercised paths
calls into them.
"""

import os
import sys
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest

for _name in ("google", "google.generativeai", "quaternion"):
    if _name not in sys.modules:
        sys.modules[_name] = mock.MagicMock()

# vlm_planner_benchmark_gemini.py (imported by the planners package
# __init__) reads this at module scope; the value never reaches a real
# client here because google.generativeai is stubbed above.
os.environ.setdefault("GOOGLE_API_KEY", "test-not-a-real-key")

from src.planners.multi_agent_msp_planner import MultiAgentMSPPlanner  # noqa: E402
from tests.fake_backend import FakeBackend  # noqa: E402
from tests.golden.context import (  # noqa: E402
    AGENT_STATE,
    CHOICES,
    FRONTIERS,
    OBJECTS,
    QUESTION_MCQ,
    QUESTION_WHERE,
    SCENE_GRAPH_STR,
)

RESPONSES = {
    "OrchestratorOutput": {
        "reasoning": "Point 3.0 meters along the tv front.",
        "target_entity": "location",
        "anchors": [{"label": "tv", "modifiers": "", "metric": "3.0 meters"}],
        "composition_logic": "intrinsic_front",
        "requires_logical_reasoning": False,
    },
    "GroundingOutput": {
        "reasoning": "The tv is object_1.",
        "grounded_anchors": [
            {"anchor_label": "tv", "matched_object_id": "object_1", "confidence": 0.95}
        ],
        "needs_exploration": False,
    },
    "SpatialOutput": {
        "reasoning": "Screen faces the camera.",
        "theta_radians": 0.2,
        "phi_radians": 1.57,
        "target_frontier_id": "NONE",
    },
    "QaOutput": {
        "prior_hypothesis": "Living room.",
        "hypothesis_likelihood": "high",
        "reasoning": "Sofa and tv visible.",
        "action_type": "answer",
        "chosen_id": "NONE",
        "confidence": 0.9,
        "answer": "D",
    },
}


class StubSgSim:
    scene_graph_str = SCENE_GRAPH_STR

    def __init__(self):
        self._pos = {
            **{o["id"]: o["position"] for o in OBJECTS},
            **{f["id"]: f["position"] for f in FRONTIERS},
        }

    def get_current_semantic_state_str(self):
        return AGENT_STATE

    def get_position_from_id(self, node_id):
        return self._pos.get(str(node_id))


class _TestPlanner(MultiAgentMSPPlanner):
    """Scene data injected directly; everything else is the real code."""

    def _get_scene_data(self):
        self._size_provenance = {
            o["id"]: ("fixed_fallback", None) for o in OBJECTS
        }
        objects = [dict(o) for o in OBJECTS]
        frontiers = [dict(f) for f in FRONTIERS]
        return objects, frontiers


def _make_planner(tmp_path, question, choices=()):
    (tmp_path / "current_img_0.png").write_bytes(b"\x89PNG\r\n\x1a\nfake")
    cfg = SimpleNamespace(
        agents_impl="unified",
        top_k_objects=2,
        sigma_s_factor=0.5,
        sigma_m_factor=0.3,
        kappa_factor=10.0,
        flatten_semantic=False,
        pre_answer_conf_thresh=0.8,
    )
    planner = _TestPlanner(
        cfg,
        StubSgSim(),
        question,
        out_path=str(tmp_path),
        choices=list(choices),
    )
    fake = FakeBackend(RESPONSES)
    for role_obj in (
        planner.orchestrator,
        planner.grounder,
        planner.spatial,
        planner.verifier,
        planner.qa,
    ):
        role_obj.backend = fake
    return planner, fake


def test_planner_defaults_to_unified_stack(tmp_path):
    planner, _ = _make_planner(tmp_path, QUESTION_WHERE)
    assert planner.agents_impl == "unified"
    assert planner.logical is None  # orphan role not constructed
    from src.agents.roles import ROLE_CLASSES

    assert isinstance(planner.orchestrator, ROLE_CLASSES["orchestrator"])
    assert isinstance(planner.verifier, ROLE_CLASSES["verifier"])


def test_mcq_fast_path_is_one_call_with_real_tokens(tmp_path):
    planner, fake = _make_planner(tmp_path, QUESTION_MCQ, choices=CHOICES)
    _pose, target_id, is_conf, conf, extra = planner.get_next_action(
        agent_yaw_rad=0.5, agent_pos_hab=np.array([1.0, 0.5, -2.0], dtype=np.float32)
    )
    assert target_id == "D"
    assert is_conf is True
    assert extra["action_type"] == "answer"
    rows = planner.call_log.rows()
    assert planner.call_log.total() == 1
    assert rows[0]["role"] == "qa"
    assert rows[0]["model_name"] == "fake-model-1"
    # The MAPG-09 point: token counts are real, not None.
    assert rows[0]["prompt_tokens"] == 111
    assert rows[0]["completion_tokens"] == 22
    # The QA image rode as a separate part; exactly one call was sent.
    assert len(fake.sent) == 1
    _system, parts, _schema = fake.sent[0]
    assert [p["type"] for p in parts] == ["text", "image_path"]


def test_where_path_locks_anchor_then_runs_spatial_and_verifier(tmp_path):
    np.random.seed(0)
    planner, fake = _make_planner(tmp_path, QUESTION_WHERE)
    pose = np.array([1.0, 0.5, -2.0], dtype=np.float32)

    # Step 1: orchestrate + ground, lock the anchor, navigate to it.
    _t, target_id, _c, _conf, extra = planner.get_next_action(
        agent_yaw_rad=0.5, agent_pos_hab=pose
    )
    assert extra["action_type"] == "goto_object"
    assert target_id == "object_1"
    assert planner.locked_anchor_id == "object_1"
    assert planner.call_log.total() == 2
    assert [r["role"] for r in planner.call_log.rows()] == [
        "orchestrator",
        "grounding",
    ]

    # Step 2: anchor locked -> spatial runs, programmatic verifier
    # gates the step without an LLM call (llm disabled by default).
    _t, target_id, _c, _conf, extra = planner.get_next_action(
        agent_yaw_rad=0.5, agent_pos_hab=pose
    )
    roles = [r["role"] for r in planner.call_log.rows()]
    assert roles == ["orchestrator", "grounding", "orchestrator", "grounding", "spatial"]
    assert planner.call_log.total() == 5  # verifier llm_used=False: not a call
    assert all(r["prompt_tokens"] == 111 for r in planner.call_log.rows())
    assert all(r["completion_tokens"] == 22 for r in planner.call_log.rows())
    assert extra["verifier"]["llm_used"] is False
    assert extra["verifier"]["checks"] is not None
    assert extra["metric_parse"]["d0_used_m"] == 3.0
    assert extra["action_type"] in ("goto_object", "lookaround", "goto_frontier", "answer")


def test_agents_impl_unknown_falls_back_to_unified(tmp_path):
    (tmp_path / "current_img_0.png").write_bytes(b"png")
    cfg = SimpleNamespace(agents_impl="not-a-real-impl")
    planner = _TestPlanner(
        cfg, StubSgSim(), QUESTION_WHERE, out_path=str(tmp_path)
    )
    assert planner.agents_impl == "unified"


def test_agents_impl_legacy_raises_clearly(tmp_path):
    """The legacy files are gone (MAPG-09 commit 2); a config still
    naming them must fail loudly, not silently run something else."""
    cfg = SimpleNamespace(agents_impl="legacy")
    with pytest.raises(RuntimeError, match="MAPG-09"):
        _TestPlanner(cfg, StubSgSim(), QUESTION_WHERE, out_path=str(tmp_path))


def test_logical_role_is_gone():
    from src.agents.factory import create_role

    with pytest.raises(ValueError, match="logical"):
        create_role("logical", provider="claude")
