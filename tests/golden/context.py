"""Fixed rendering context for the golden prompt snapshots (MAPG-09).

Everything here is frozen data: any change to these fixtures changes
the rendered prompts and therefore the snapshots, so edit only with
intent and regenerate via ``python tests/golden/regen.py``.

The pose is a plain list (not a numpy array) so its string form does
not depend on numpy's array-printing version.
"""

from src.multi_agent.blackboard import Blackboard

QUESTION_WHERE = "Where is the location 3.0 meters in front of the tv?"
QUESTION_MCQ = "Based on the items on the table, what room am I in?"

CHOICES = [
    "A) Kitchen",
    "B) Bathroom",
    "C) Bedroom (DO NOT SELECT THIS OPTION)",
    "D) Living room",
]

OBJECTS = [
    {
        "id": "object_1",
        "name": "tv",
        "position": [2.0, 0.5, -3.0],
        "size": [0.9, 0.6, 0.2],
    },
    {
        "id": "object_2",
        "name": "sofa",
        "position": [0.0, 0.4, -4.0],
        "size": [1.8, 0.8, 0.9],
    },
]

FRONTIERS = [
    {
        "id": "frontier_7",
        "name": "frontier",
        "position": [4.0, 0.5, -1.0],
        "size": [0.5, 0.5, 0.5],
    }
]

SCENE_GRAPH_STR = (
    '{"nodes": [{"id": "object_1", "name": "tv"}, '
    '{"id": "object_2", "name": "sofa"}, {"id": "room_0", "name": "living room"}]}'
)

AGENT_STATE = "Agent is in room_0 (living room)."

GLOBAL_HISTORY = "Step 2 FAIL: Verifier rejected object_9 for anchor 'tv'.\n"

LEDGER = [
    {
        "agent": "Grounding",
        "type": "MatchObjects",
        "status": "PASS",
        "details": {"anchor": "tv", "matched_object_id": "object_1"},
    }
]

ORCHESTRATOR_OUTPUT = {
    "reasoning": "The query asks for a point 3.0 meters along the tv's intrinsic front.",
    "target_entity": "location",
    "anchors": [{"label": "tv", "modifiers": "large", "metric": "3.0 meters"}],
    "composition_logic": "intrinsic_front",
    "requires_logical_reasoning": False,
}

ANCHOR_OBJ = OBJECTS[0]


def _fill(bb: Blackboard, image_path=None) -> Blackboard:
    bb.update_state(
        t=3,
        pose=[1.0, 0.5, -2.0],
        yaw=0.7853981633974483,
        img_path=image_path,
        sg_str=SCENE_GRAPH_STR,
        agent_state=AGENT_STATE,
        objects=[dict(o) for o in OBJECTS],
        frontiers=[dict(f) for f in FRONTIERS],
    )
    bb.global_history = GLOBAL_HISTORY
    bb.event_ledger = [dict(e) for e in LEDGER]
    return bb


def make_where_blackboard(image_path=None) -> Blackboard:
    return _fill(Blackboard(question=QUESTION_WHERE, mode="where"), image_path)


def make_mcq_blackboard(image_path=None) -> Blackboard:
    bb = _fill(Blackboard(question=QUESTION_MCQ, mode="eqa"), image_path)
    bb.choices = list(CHOICES)
    return bb


def render_all():
    """(role, system, user) for every role under the fixed context."""
    from src.agents.prompts import grounding, orchestrator, qa, spatial, verifier

    where_bb = make_where_blackboard()
    mcq_bb = make_mcq_blackboard()
    out = []
    out.append(("orchestrator",) + orchestrator.render(where_bb))
    out.append(("grounding",) + grounding.render(where_bb, ORCHESTRATOR_OUTPUT))
    out.append(("spatial",) + spatial.render(where_bb, ANCHOR_OBJ))
    out.append(("verifier",) + verifier.render(where_bb))
    out.append(("qa",) + qa.render(mcq_bb))
    return out
