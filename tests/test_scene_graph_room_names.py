import networkx as nx

from src.scene_graph.scene_graph_sim import SceneGraphSim


def _room_stub(callback=None):
    sim = SceneGraphSim.__new__(SceneGraphSim)
    sim.room_name_infer_fn = callback
    sim._room_name_cache = {}
    sim._room_id_name_cache = {}
    sim._room_name_warning_emitted = False
    return sim


def test_room_name_callback_is_cached_by_visible_objects():
    calls = []
    sim = _room_stub(lambda names: calls.append(names) or "Living Room")

    assert sim._infer_room_name(["Sofa", "TV"]) == "living room"
    assert sim._infer_room_name(["TV", "Sofa"]) == "living room"
    assert calls == [("sofa", "tv")]


def test_room_name_without_accounted_callback_never_calls_provider():
    sim = _room_stub()
    assert sim._infer_room_name(["bed"]) == "unknown_room"
    assert sim._infer_room_name_for_id("room_0", ["bed"]) == "unknown_room"
    assert sim._room_name_cache == {}
    assert sim._room_id_name_cache == {}


def test_room_id_is_classified_only_once_as_graph_grows():
    calls = []
    sim = _room_stub(lambda names: calls.append(names) or "living room")

    assert sim._infer_room_name_for_id("room_0", ["sofa"]) == "living room"
    assert sim._infer_room_name_for_id("room_0", ["sofa", "television"]) == "living room"
    assert calls == [("sofa",)]


def _graph_stub(graph, object_ids):
    sim = _room_stub()
    sim.filtered_netx_graph = graph
    sim._object_node_ids = object_ids
    sim._room_ids = ["room_0"]
    sim._room_names = []
    return sim


def test_room_objects_are_read_after_region_nodes_are_removed():
    graph = nx.DiGraph()
    graph.add_node("room_0", name="unknown_room")
    graph.add_node("object_1", name="sofa")
    graph.add_node("object_2", name="television")
    graph.add_node("agent_0", name="agent")
    graph.add_edges_from([
        ("room_0", "object_1"),
        ("room_0", "object_2"),
        ("room_0", "agent_0"),
    ])
    calls = []
    sim = _graph_stub(graph, ["object_1", "object_2"])

    sim.classify_rooms_once(lambda names: calls.append(names) or "living room")

    assert calls == [("sofa", "television")]
    assert graph.nodes["room_0"]["name"] == "living room"


def test_room_objects_are_read_through_region_nodes():
    graph = nx.DiGraph()
    graph.add_node("room_0", name="unknown_room")
    graph.add_node("region_0", name="region")
    graph.add_node("object_1", name="bed")
    graph.add_edge("room_0", "region_0")
    graph.add_edge("region_0", "object_1")
    calls = []
    sim = _graph_stub(graph, ["object_1"])

    sim.classify_rooms_once(lambda names: calls.append(names) or "bedroom")

    assert calls == [("bed",)]
    assert graph.nodes["room_0"]["name"] == "bedroom"
