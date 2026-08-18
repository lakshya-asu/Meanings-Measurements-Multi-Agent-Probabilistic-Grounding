"""Compact scene-graph serialization (MAPG-10).

Covers: determinism, the stable-prefix ordering contract (rooms and
objects first, byte-stable under frontier/agent churn), current-pose-
only agent emission, round-trip parsing, the legacy_json byte format,
the mode resolver, and the measured compression ratio the report
cites.
"""

import pytest

from src.agents.cost_estimate import (
    SimpleGraph,
    build_synthetic_graph,
    legacy_node_link_json,
    measure_serialization,
)
from src.agents.serialization import (
    parse_compact,
    resolve_serialization_mode,
    serialize_candidates,
    serialize_frontiers,
    serialize_scene_graph,
    split_stable_prefix,
)


def small_graph():
    g = SimpleGraph()
    g.add_node("room_0", name="living room", layer=4, position=[0.0, 0.0, 0.0])
    g.add_node(
        "object_1",
        name="tv",
        layer=2,
        label=12,
        position=[2.0049, 0.5, -3.0],
        bbox_extents=[0.9, 0.6, 0.2],
    )
    g.add_node("object_2", name="sofa", layer=2, label=7,
               position=[0.0, 0.4, -4.0])
    g.add_node("frontier_0", name="frontier", layer=2,
               position=[4.0, 0.5, -1.0])
    g.add_node("agent_0", name="agent", layer=2, timestamp=0.0,
               position=[9.0, 9.0, 9.0])
    g.add_node("agent_1", name="agent", layer=2, timestamp=1.0,
               position=[1.0, 0.5, -2.0])
    g.add_edge("room_0", "object_1", type="room-to-object")
    g.add_edge("room_0", "object_2", type="room-to-object")
    g.add_edge("frontier_0", "object_2", type="frontier-to-object")
    return g


# ----------------------------------------------------------------------
# Format
# ----------------------------------------------------------------------

def test_compact_format_lines():
    text = serialize_scene_graph(small_graph(), current_agent_id="agent_1")
    assert text.split("\n") == [
        "ROOM room_0 living room",
        "OBJ object_1 tv (2.00, 0.50, -3.00) size=(0.90, 0.60, 0.20) room=room_0",
        "OBJ object_2 sofa (0.00, 0.40, -4.00) room=room_0",
        "FRONTIER frontier_0 (4.00, 0.50, -1.00) near=object_2",
        "AGENT agent_1 (1.00, 0.50, -2.00)",
    ]


def test_historical_agent_nodes_are_dropped():
    text = serialize_scene_graph(small_graph(), current_agent_id="agent_1")
    assert "agent_0" not in text
    assert text.count("AGENT ") == 1


def test_no_current_agent_id_emits_no_agent_line():
    text = serialize_scene_graph(small_graph(), current_agent_id=None)
    assert "AGENT" not in text


def test_determinism_under_insertion_order():
    g1 = small_graph()
    g2 = SimpleGraph()
    # Same content, different insertion order.
    src = small_graph()
    for nid in reversed(list(src.nodes)):
        g2.add_node(nid, **src.nodes[nid])
    for s, t, attrs in src.edges():
        g2.add_edge(s, t, **attrs)
    a = serialize_scene_graph(g1, current_agent_id="agent_1")
    b = serialize_scene_graph(g2, current_agent_id="agent_1")
    assert a == b


def test_numeric_id_sort_appends_new_objects():
    g = small_graph()
    g.add_node("object_10", name="lamp", layer=2, position=[1.0, 1.0, 1.0])
    text = serialize_scene_graph(g, current_agent_id="agent_1")
    lines = text.split("\n")
    # object_10 sorts after object_2 (numeric suffix), so discovery
    # appends to the stable block instead of reshuffling it.
    assert lines.index("OBJ object_10 lamp (1.00, 1.00, 1.00)") > lines.index(
        "OBJ object_2 sofa (0.00, 0.40, -4.00) room=room_0"
    )


# ----------------------------------------------------------------------
# Stable-prefix ordering contract
# ----------------------------------------------------------------------

def test_stable_prefix_precedes_volatile_suffix():
    text = serialize_scene_graph(small_graph(), current_agent_id="agent_1")
    lines = text.split("\n")
    kinds = [line.split(" ", 1)[0] for line in lines]
    first_volatile = min(
        i for i, k in enumerate(kinds) if k in ("FRONTIER", "AGENT")
    )
    assert all(k in ("ROOM", "OBJ") for k in kinds[:first_volatile])
    assert all(k in ("FRONTIER", "AGENT") for k in kinds[first_volatile:])


def test_stable_prefix_bytes_survive_frontier_and_agent_churn():
    g = small_graph()
    before = serialize_scene_graph(g, current_agent_id="agent_1")
    stable_before, _ = split_stable_prefix(before)
    # A step later: frontiers regenerate, the agent moves, a pose node
    # accumulates.
    g.add_node("frontier_1", name="frontier", layer=2,
               position=[5.0, 0.1, 0.4])
    g.add_node("agent_2", name="agent", layer=2, timestamp=2.0,
               position=[1.5, 0.5, -2.5])
    after = serialize_scene_graph(g, current_agent_id="agent_2")
    stable_after, volatile_after = split_stable_prefix(after)
    assert stable_after == stable_before
    assert after.startswith(stable_before)
    assert "FRONTIER frontier_1" in volatile_after
    assert volatile_after.endswith("AGENT agent_2 (1.50, 0.50, -2.50)")


def test_split_stable_prefix_concatenation_is_identity():
    text = serialize_scene_graph(small_graph(), current_agent_id="agent_1")
    stable, volatile = split_stable_prefix(text)
    assert stable + volatile == text


# ----------------------------------------------------------------------
# Round trip
# ----------------------------------------------------------------------

def test_round_trip_recovers_nodes():
    g = small_graph()
    parsed = parse_compact(serialize_scene_graph(g, current_agent_id="agent_1"))
    assert parsed["rooms"] == [{"id": "room_0", "name": "living room"}]
    objs = {o["id"]: o for o in parsed["objects"]}
    assert objs["object_1"]["name"] == "tv"
    assert objs["object_1"]["position"] == [2.0, 0.5, -3.0]  # 2 dp
    assert objs["object_1"]["size"] == [0.9, 0.6, 0.2]
    assert objs["object_1"]["room"] == "room_0"
    assert objs["object_2"]["room"] == "room_0"
    assert "size" not in objs["object_2"]
    assert parsed["frontiers"] == [
        {"id": "frontier_0", "position": [4.0, 0.5, -1.0], "near": ["object_2"]}
    ]
    assert parsed["agents"] == [
        {"id": "agent_1", "position": [1.0, 0.5, -2.0]}
    ]


def test_round_trip_on_synthetic_graph_is_lossless_for_ids():
    graph, agent_id = build_synthetic_graph()
    parsed = parse_compact(
        serialize_scene_graph(graph, current_agent_id=agent_id)
    )
    expect_objects = {n for n in graph.nodes if n.startswith("object_")}
    assert {o["id"] for o in parsed["objects"]} == expect_objects
    assert {r["id"] for r in parsed["rooms"]} == {
        n for n in graph.nodes if n.startswith("room_")
    }
    assert [a["id"] for a in parsed["agents"]] == [agent_id]


# ----------------------------------------------------------------------
# legacy_json mode
# ----------------------------------------------------------------------

def test_legacy_json_matches_networkx_byte_for_byte():
    nx = pytest.importorskip("networkx")
    src = small_graph()
    g = nx.DiGraph()
    for nid in src.nodes:
        g.add_node(nid, **src.nodes[nid])
    for s, t, attrs in src.edges():
        g.add_edge(s, t, **attrs)
    import json

    # The historical bytes use the "links" edge key (container nx
    # 3.4.2); newer networkx defaults to "edges", so pin it here the
    # same way serialize_scene_graph does.
    try:
        expected = json.dumps(nx.node_link_data(g, edges="links"))
    except TypeError:
        expected = json.dumps(nx.node_link_data(g))

    assert serialize_scene_graph(g, mode="legacy_json") == expected
    # The SimpleGraph replica of the legacy format matches too, so the
    # cost-estimate measurement measures the real legacy byte format.
    assert legacy_node_link_json(src) == expected


# ----------------------------------------------------------------------
# Mode resolver
# ----------------------------------------------------------------------

def test_resolve_serialization_mode():
    assert resolve_serialization_mode(None) == ("compact", None)
    assert resolve_serialization_mode("compact") == ("compact", None)
    assert resolve_serialization_mode("LEGACY_JSON") == ("legacy_json", None)
    mode, warn = resolve_serialization_mode("weird")
    assert mode == "compact"
    assert "weird" in warn


# ----------------------------------------------------------------------
# Candidate / frontier lists
# ----------------------------------------------------------------------

def test_serialize_candidates_and_frontiers():
    objs = [
        {"id": "object_1", "name": "tv", "position": [2.0, 0.5, -3.0]},
        {"id": "object_2", "name": "sofa", "position": [0.0, 0.4049, -4.0]},
        "not-a-dict",
    ]
    assert serialize_candidates(objs) == (
        "object_1 tv (2.00, 0.50, -3.00)\nobject_2 sofa (0.00, 0.40, -4.00)"
    )
    assert serialize_candidates([]) == "(none)"
    fr = [{"id": "frontier_7", "position": [4.0, 0.5, -1.0]}]
    assert serialize_frontiers(fr) == "frontier_7 (4.00, 0.50, -1.00)"
    assert serialize_frontiers([]) == "(none)"


# ----------------------------------------------------------------------
# Measured compression (the number the MAPG-10 report cites)
# ----------------------------------------------------------------------

def test_compression_ratio_on_synthetic_60_node_graph():
    sizes = measure_serialization()
    # Clean 60-node graph: the format change alone.
    assert sizes["clean"]["ratio"] >= 3.0
    # Mid-episode shape (+150 stale agent pose nodes): the dominant
    # savings term per method-scenegraph-grounding.md section 5.
    assert sizes["accumulated"]["ratio"] >= 5.0
    # The compact text of the accumulated graph is identical to the
    # clean one except the agent line (stale poses never serialized),
    # so its size does not grow with episode length.
    assert (
        abs(sizes["accumulated"]["compact_chars"] - sizes["clean"]["compact_chars"])
        < 10
    )
