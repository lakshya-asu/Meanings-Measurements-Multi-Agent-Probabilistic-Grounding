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
from src.agents.base import BackendReplyError  # noqa: E402
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
    "RoomNameOutput": {"room": "living room"},
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


def _make_planner(tmp_path, question, choices=(), **kwargs):
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
        **kwargs,
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


def test_injected_pathfinder_reaches_masking_and_verifier(tmp_path):
    pathfinder = object()
    planner, _ = _make_planner(tmp_path, QUESTION_WHERE, pathfinder=pathfinder)
    assert planner._get_pathfinder() is pathfinder


def test_room_name_call_is_accounted_and_charged(tmp_path):
    charged = []
    planner, fake = _make_planner(
        tmp_path, QUESTION_WHERE, on_llm_call=lambda: charged.append(True)
    )
    assert planner.infer_room_name(["sofa", "television"]) == "living room"
    row = planner.call_log.rows()[0]
    assert row["role"] == "room_naming"
    assert row["prompt_tokens"] == 111
    assert row["completion_tokens"] == 22
    assert charged == [True]
    assert fake.sent[-1][2]["title"] == "RoomNameOutput"


def test_room_name_parse_failure_retries_once_with_exact_usage(tmp_path):
    class MalformedOnceBackend:
        model_name = "fake-model-1"

        def __init__(self):
            self.attempts = 0

        def send(self, _system, _parts, _schema):
            self.attempts += 1
            if self.attempts == 1:
                raise BackendReplyError(
                    "malformed room JSON",
                    usage={"prompt_tokens": 41, "completion_tokens": 10},
                )
            return (
                {"room": "bedroom"},
                {"prompt_tokens": 43, "completion_tokens": 9},
                1.0,
            )

    charged = []
    planner, _ = _make_planner(
        tmp_path, QUESTION_WHERE, on_llm_call=lambda: charged.append(True)
    )
    backend = MalformedOnceBackend()
    planner.orchestrator.backend = backend

    assert planner.infer_room_name(["bed"]) == "bedroom"
    rows = planner.call_log.rows()
    assert [(row["prompt_tokens"], row["completion_tokens"]) for row in rows] == [
        (41, 10),
        (43, 9),
    ]
    assert [row["is_retry"] for row in rows] == [False, True]
    assert charged == [True, True]


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
    # MAPG-10: the user text arrives as two chunks (stable cacheable
    # prefix + volatile suffix) followed by the image part.
    assert len(fake.sent) == 1
    _system, parts, _schema = fake.sent[0]
    assert [p["type"] for p in parts] == ["text", "text", "image_path"]
    assert parts[0].get("cache") is True
    assert parts[1].get("cache") is None


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
    # MAPG-10 parse-once: the orchestrator does NOT run again (the
    # question is episode-constant and the failure history is
    # unchanged), so step 2 is grounding + spatial only.
    _t, target_id, _c, _conf, extra = planner.get_next_action(
        agent_yaw_rad=0.5, agent_pos_hab=pose
    )
    roles = [r["role"] for r in planner.call_log.rows()]
    assert roles == ["orchestrator", "grounding", "grounding", "spatial"]
    assert planner.call_log.total() == 4  # verifier llm_used=False: not a call
    assert all(r["prompt_tokens"] == 111 for r in planner.call_log.rows())
    assert all(r["completion_tokens"] == 22 for r in planner.call_log.rows())
    assert extra["verifier"]["llm_used"] is False
    assert extra["verifier"]["checks"] is not None
    assert extra["metric_parse"]["d0_used_m"] == 3.0
    assert extra["action_type"] in ("goto_object", "lookaround", "goto_frontier", "answer")


def test_orchestrator_reparses_when_history_changes(tmp_path):
    """MAPG-10 parse-once invalidation: a new failure-history entry
    (verifier feedback) must trigger a fresh orchestrator parse so
    prompt rule 5 (choose a different interpretation) still works."""
    planner, _fake = _make_planner(tmp_path, QUESTION_WHERE)
    pose = np.array([1.0, 0.5, -2.0], dtype=np.float32)
    planner.get_next_action(agent_yaw_rad=0.5, agent_pos_hab=pose)
    roles = [r["role"] for r in planner.call_log.rows()]
    assert roles.count("orchestrator") == 1
    planner.blackboard.global_history += "Step 1 FAIL: wrong anchor id.\n"
    planner.get_next_action(agent_yaw_rad=0.5, agent_pos_hab=pose)
    roles = [r["role"] for r in planner.call_log.rows()]
    assert roles.count("orchestrator") == 2


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


def test_planner_applies_model_tiers(tmp_path):
    """MAPG-10: cfg model_tiers overrides the backend model per role;
    null (missing) roles stay on the provider default."""
    cfg = SimpleNamespace(
        agents_impl="unified",
        model_tiers={"orchestrator": "claude-haiku-4-5",
                     "verifier": "claude-haiku-4-5"},
    )
    planner = _TestPlanner(cfg, StubSgSim(), QUESTION_WHERE, out_path=str(tmp_path))
    assert planner.orchestrator.backend.model_name == "claude-haiku-4-5"
    assert planner.verifier.backend.model_name == "claude-haiku-4-5"
    assert planner.grounder.backend.model_name == "claude-opus-4-6"
    assert planner.spatial.backend.model_name == "claude-opus-4-6"
    assert planner.qa.backend.model_name == "claude-opus-4-6"


def test_planner_serializes_compact_when_graph_available(tmp_path):
    """MAPG-10: with a netx-protocol graph on the sim, the blackboard
    scene graph text is the compact line format; the historical JSON
    string is only the fallback for graphless stubs."""
    from src.agents.cost_estimate import SimpleGraph

    class GraphSgSim(StubSgSim):
        def __init__(self):
            super().__init__()
            g = SimpleGraph()
            g.add_node("room_0", name="living room", layer=4,
                       position=[0.0, 0.0, 0.0])
            for o in OBJECTS:
                g.add_node(o["id"], name=o["name"], layer=2,
                           position=list(o["position"]),
                           bbox_extents=list(o["size"]))
                g.add_edge("room_0", o["id"], type="room-to-object")
            g.add_node("agent_0", name="agent", layer=2, timestamp=0.0,
                       position=[1.0, 0.5, -2.0])
            self.filtered_netx_graph = g
            self.curr_agent_id = "agent_0"

    planner, _fake = _make_planner(tmp_path, QUESTION_MCQ, choices=CHOICES)
    planner.sg_sim = GraphSgSim()
    planner.get_next_action(
        agent_yaw_rad=0.5, agent_pos_hab=np.array([1.0, 0.5, -2.0],
                                                  dtype=np.float32)
    )
    sg = planner.blackboard.scene_graph_str
    assert sg.startswith("ROOM room_0 living room")
    assert "OBJ object_1 tv (2.00, 0.50, -3.00) size=(0.90, 0.60, 0.20) room=room_0" in sg
    assert sg.endswith("AGENT agent_0 (1.00, 0.50, -2.00)")
    assert "{" not in sg  # no JSON reached the prompt


def test_planner_legacy_json_mode_preserves_old_text(tmp_path):
    (tmp_path / "current_img_0.png").write_bytes(b"png")
    cfg = SimpleNamespace(agents_impl="unified", sg_serialization="legacy_json")
    planner = _TestPlanner(cfg, StubSgSim(), QUESTION_WHERE, out_path=str(tmp_path))
    assert planner.sg_serialization == "legacy_json"
    assert planner._serialized_scene_graph() == SCENE_GRAPH_STR
