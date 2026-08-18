"""Compact scene-graph serialization for prompt text (MAPG-10).

The legacy serialization is ``json.dumps(nx.node_link_data(G))`` with
full-precision float reprs, per-edge records that duplicate node names,
and every historical agent-pose node Hydra ever appended (dozens to
hundreds mid-episode). Measured cost: 8k-25k tokens per step on the
spatial/QA prompts (research/method-scenegraph-grounding.md section 5).

The compact format is line-oriented, one entity per line, coordinates
rounded to 2 dp (centimeter precision; the bench annotation noise floor
is larger):

    ROOM room_0 living room
    OBJ object_1 tv (2.00, 0.50, -3.00) size=(0.90, 0.60, 0.20) room=room_0
    FRONTIER frontier_7 (4.00, 0.50, -1.00) near=object_1,object_2
    AGENT agent_3 (1.00, 0.50, -2.00)

Field rules:

- ``size=`` is the node's real AABB extents (``bbox_extents``,
  MAPG-01); omitted when the node carries no box.
- ``room=`` encodes containment (direct room->object edges, or
  room->region->object when regions are present) instead of edge
  records; omitted when no containing room is found.
- ``near=`` keeps the frontier-to-object adjacency as a comma-joined
  id list.
- ``layer``, ``label`` ints, and ``source_name``/``target_name`` edge
  fields are dropped: they never carried task information the id and
  name do not.
- Only the CURRENT agent pose is emitted. Historical agent-pose nodes
  (Hydra appends one per keyframe) stay in the DSG but never reach the
  prompt; they were the dominant term in serialization cost.

STABLE-PREFIX ORDERING CONTRACT (load-bearing for prompt caching,
MAPG-10 part 3): the output is ordered so that the stable part comes
first and the volatile part comes last, byte-stably:

    1. ROOM lines, sorted by node id            (stable)
    2. OBJ lines, sorted by node id             (stable, append-mostly)
    3. FRONTIER lines, sorted by node id        (volatile, per step)
    4. one AGENT line (current pose only)       (volatile, per step)

Sorting is by (id prefix, numeric suffix), so as Hydra discovers new
objects (increasing ids) their lines APPEND to the stable block rather
than reshuffling it, and a provider's byte-prefix cache keeps hitting
the unchanged prefix across steps. Position refinements to an existing
node mutate its line in place; the prefix up to that node still hits.
Callers that split the prompt into cached blocks must put the ROOM+OBJ
block (plus the system text) in the cached prefix and everything from
the first FRONTIER line onward in the volatile suffix;
``split_stable_prefix`` implements exactly that split.

cfg key: ``sg_serialization`` = "compact" (default) | "legacy_json".
Stdlib + networkx only; deterministic; never raises on odd graphs.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, Iterable, List, Optional, Tuple

MODES = ("compact", "legacy_json")
DEFAULT_MODE = "compact"

_ID_RE = re.compile(r"^(?P<prefix>[a-zA-Z]+)_?(?P<num>\d+)?$")


def resolve_serialization_mode(value: Any) -> Tuple[str, Optional[str]]:
    """(mode, warning) from a cfg value; unknown values warn and
    fall back to the default, same contract as the other resolvers."""
    if value is None:
        return DEFAULT_MODE, None
    mode = str(value).lower().strip()
    if mode in MODES:
        return mode, None
    return DEFAULT_MODE, (
        f"Unknown sg_serialization={value!r}; using '{DEFAULT_MODE}'. "
        f"Valid: {MODES}."
    )


def _sort_key(node_id: str) -> Tuple[str, int, str]:
    m = _ID_RE.match(str(node_id))
    if m and m.group("num") is not None:
        return (m.group("prefix"), int(m.group("num")), str(node_id))
    return (str(node_id), -1, str(node_id))


def _fmt_pos(pos: Any) -> Optional[str]:
    try:
        vals = [float(v) for v in list(pos)[:3]]
        if len(vals) < 3:
            return None
        return "({:.2f}, {:.2f}, {:.2f})".format(*vals)
    except (TypeError, ValueError):
        return None


def _node_kind(node_id: str, attrs: Dict[str, Any]) -> str:
    nid = str(node_id).lower()
    for kind in ("object", "room", "frontier", "agent", "region", "building"):
        if nid.startswith(kind):
            return kind
    if str(attrs.get("name", "")).lower() == "frontier":
        return "frontier"
    if "timestamp" in attrs:
        return "agent"
    return "other"


def _room_of(graph: Any, node_id: str) -> Optional[str]:
    """Containing room id via room->object or room->region->object."""
    try:
        for pred in graph.predecessors(node_id):
            p = str(pred)
            if p.lower().startswith("room"):
                return p
        for pred in graph.predecessors(node_id):
            if str(pred).lower().startswith("region"):
                for grand in graph.predecessors(pred):
                    if str(grand).lower().startswith("room"):
                        return str(grand)
    except Exception:
        pass
    return None


def _frontier_neighbors(graph: Any, node_id: str) -> List[str]:
    try:
        near = [
            str(s)
            for s in graph.successors(node_id)
            if str(s).lower().startswith("object")
        ]
        return sorted(near, key=_sort_key)
    except Exception:
        return []


def serialize_scene_graph(
    graph: Any,
    mode: str = DEFAULT_MODE,
    current_agent_id: Optional[str] = None,
) -> str:
    """Serialize a filtered networkx scene graph for prompt text.

    ``mode='legacy_json'`` reproduces the historical
    ``json.dumps(nx.node_link_data(G))`` byte for byte. ``'compact'``
    emits the line format documented above, stable prefix first.
    """
    if mode == "legacy_json":
        import networkx as nx

        try:
            # Pin the "links" edge key: networkx 3.6 changed the
            # default to "edges", which would silently change the
            # legacy bytes. The container (nx 3.4.2) produced "links".
            return json.dumps(nx.node_link_data(graph, edges="links"))
        except TypeError:
            # Older networkx without the edges kwarg: "links" default.
            return json.dumps(nx.node_link_data(graph))

    rooms: List[Tuple[str, Dict[str, Any]]] = []
    objects: List[Tuple[str, Dict[str, Any]]] = []
    frontiers: List[Tuple[str, Dict[str, Any]]] = []
    agent_line: Optional[str] = None

    for node_id in graph.nodes:
        attrs = dict(graph.nodes[node_id])
        kind = _node_kind(node_id, attrs)
        if kind == "room":
            rooms.append((str(node_id), attrs))
        elif kind == "object":
            objects.append((str(node_id), attrs))
        elif kind == "frontier":
            frontiers.append((str(node_id), attrs))
        elif kind == "agent":
            # Current pose only; historical agent nodes are dropped.
            if current_agent_id is not None and str(node_id) == str(current_agent_id):
                pos = _fmt_pos(attrs.get("position"))
                agent_line = f"AGENT {node_id} {pos}" if pos else f"AGENT {node_id}"
        # region/building/other nodes carry no prompt value: dropped.

    lines: List[str] = []

    # ---- stable prefix: rooms then objects, sorted by id -------------
    for rid, attrs in sorted(rooms, key=lambda kv: _sort_key(kv[0])):
        name = str(attrs.get("name", "room"))
        lines.append(f"ROOM {rid} {name}")

    for oid, attrs in sorted(objects, key=lambda kv: _sort_key(kv[0])):
        name = str(attrs.get("name", "object"))
        parts = [f"OBJ {oid} {name}"]
        pos = _fmt_pos(attrs.get("position"))
        if pos:
            parts.append(pos)
        extents = attrs.get("bbox_extents")
        if extents is not None:
            size = _fmt_pos(extents)
            if size:
                parts.append(f"size={size}")
        room = _room_of(graph, oid)
        if room:
            parts.append(f"room={room}")
        lines.append(" ".join(parts))

    # ---- volatile suffix: frontiers then the current agent ----------
    for fid, attrs in sorted(frontiers, key=lambda kv: _sort_key(kv[0])):
        parts = [f"FRONTIER {fid}"]
        pos = _fmt_pos(attrs.get("position"))
        if pos:
            parts.append(pos)
        near = _frontier_neighbors(graph, fid)
        if near:
            parts.append("near=" + ",".join(near))
        lines.append(" ".join(parts))

    if agent_line:
        lines.append(agent_line)

    return "\n".join(lines)


def split_stable_prefix(text: str) -> Tuple[str, str]:
    """Split compact text into (stable_prefix, volatile_suffix).

    The stable prefix is the ROOM+OBJ block; the volatile suffix starts
    at the first FRONTIER or AGENT line (see the ordering contract in
    the module docstring). Concatenating the two halves reproduces the
    input byte for byte.
    """
    lines = text.split("\n")
    for i, line in enumerate(lines):
        if line.startswith("FRONTIER ") or line.startswith("AGENT "):
            stable = "\n".join(lines[:i])
            volatile = "\n".join(lines[i:])
            if stable:
                return stable + "\n", volatile
            return "", text
    return text, ""


def parse_compact(text: str) -> Dict[str, List[Dict[str, Any]]]:
    """Parse compact text back into node dicts (round-trip check).

    Returns {"rooms": [...], "objects": [...], "frontiers": [...],
    "agents": [...]} with ids, names, 2 dp positions, sizes, room and
    near fields where present. Inverse of ``serialize_scene_graph`` up
    to the documented lossy drops (layer, label, edge names, history).
    """
    out: Dict[str, List[Dict[str, Any]]] = {
        "rooms": [],
        "objects": [],
        "frontiers": [],
        "agents": [],
    }

    def _pos_of(chunk: str) -> Optional[List[float]]:
        m = re.search(r"\(([-\d.]+), ([-\d.]+), ([-\d.]+)\)", chunk)
        return [float(m.group(i)) for i in (1, 2, 3)] if m else None

    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        if line.startswith("ROOM "):
            _, rid, name = line.split(" ", 2)
            out["rooms"].append({"id": rid, "name": name})
        elif line.startswith("OBJ "):
            m = re.match(
                r"^OBJ (\S+) (.*?) \(([-\d.]+), ([-\d.]+), ([-\d.]+)\)"
                r"(?: size=\(([-\d.]+), ([-\d.]+), ([-\d.]+)\))?"
                r"(?: room=(\S+))?$",
                line,
            )
            if not m:
                m2 = re.match(r"^OBJ (\S+) (.*)$", line)
                if m2:
                    out["objects"].append({"id": m2.group(1), "name": m2.group(2)})
                continue
            obj: Dict[str, Any] = {
                "id": m.group(1),
                "name": m.group(2),
                "position": [float(m.group(i)) for i in (3, 4, 5)],
            }
            if m.group(6) is not None:
                obj["size"] = [float(m.group(i)) for i in (6, 7, 8)]
            if m.group(9) is not None:
                obj["room"] = m.group(9)
            out["objects"].append(obj)
        elif line.startswith("FRONTIER "):
            fields = line.split(" ")
            fr: Dict[str, Any] = {"id": fields[1]}
            pos = _pos_of(line)
            if pos:
                fr["position"] = pos
            m = re.search(r"near=(\S+)", line)
            if m:
                fr["near"] = m.group(1).split(",")
            out["frontiers"].append(fr)
        elif line.startswith("AGENT "):
            fields = line.split(" ")
            ag: Dict[str, Any] = {"id": fields[1]}
            pos = _pos_of(line)
            if pos:
                ag["position"] = pos
            out["agents"].append(ag)
    return out


# ---------------------------------------------------------------------------
# Candidate/frontier lists for prompt text (replaces json indent=2 dumps)
# ---------------------------------------------------------------------------

def serialize_candidates(objects: Iterable[Any]) -> str:
    """Compact candidate lines: ``id name (x, y, z)`` one per line.

    Replaces the legacy ``json.dumps([...], indent=2)`` object dumps in
    the grounding/QA prompts (~9 lines and ~45-55 tokens per object,
    of which 30-40 percent was whitespace). Input order is preserved:
    the planner's candidate order is part of the prompt contract.
    """
    lines = []
    for o in objects:
        if not isinstance(o, dict):
            continue
        oid = str(o.get("id", ""))
        name = str(o.get("name", "")).strip()
        pos = _fmt_pos(o.get("position") if o.get("position") is not None else o.get("pos"))
        lines.append(" ".join(p for p in (oid, name, pos) if p))
    return "\n".join(lines) if lines else "(none)"


def serialize_frontiers(frontiers: Iterable[Any]) -> str:
    """Compact frontier lines: ``id (x, y, z)`` one per line."""
    lines = []
    for f in frontiers:
        if not isinstance(f, dict):
            continue
        fid = str(f.get("id", ""))
        pos = _fmt_pos(f.get("position") if f.get("position") is not None else f.get("pos"))
        lines.append(" ".join(p for p in (fid, pos) if p))
    return "\n".join(lines) if lines else "(none)"
