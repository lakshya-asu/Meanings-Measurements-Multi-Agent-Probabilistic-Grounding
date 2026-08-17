import rerun as rr
import rerun.blueprint as rrb
import numpy as np


def _short_float(value, digits=2):
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "n/a"


def simplify_step_transcript(ledger, final_decision):
    """Build a compact, structured transcript without model reasoning text."""
    role_lines = {name: [] for name in ("orchestrator", "grounding", "spatial", "verifier")}
    for event in ledger or []:
        agent = str(event.get("agent", "")).strip().lower()
        if agent not in role_lines:
            continue
        details = event.get("details") if isinstance(event.get("details"), dict) else {}
        lines = [f"{event.get('type', 'Event')} | {event.get('status', 'INFO')}"]

        if agent == "orchestrator":
            lines.append(f"Target: {details.get('target_entity', 'n/a')}")
            lines.append(f"Relation: {details.get('composition_logic', 'n/a')}")
            anchors = []
            for anchor in details.get("anchors", []) or []:
                label = str(anchor.get("label", "object")).strip()
                modifier = str(anchor.get("modifiers", "")).strip()
                metric = str(anchor.get("metric", "")).strip()
                anchors.append(" ".join(part for part in (modifier, label, metric) if part))
            lines.append("Anchors: " + (", ".join(anchors) if anchors else "none"))
        elif agent == "grounding":
            grounded = []
            for anchor in details.get("grounded_anchors", []) or []:
                grounded.append(
                    f"{anchor.get('anchor_label', 'object')} -> "
                    f"{anchor.get('matched_object_id', 'NONE')} "
                    f"({_short_float(anchor.get('confidence'))})"
                )
            lines.append("Matches: " + (", ".join(grounded) if grounded else "none"))
            lines.append(f"Explore: {bool(details.get('needs_exploration', False))}")
        elif agent == "spatial":
            lines.extend(
                [
                    f"Theta: {_short_float(details.get('theta'))} rad",
                    f"Phi: {_short_float(details.get('phi'))} rad",
                    f"Kappa: {_short_float(details.get('kappa'))}",
                    f"Frontier: {details.get('target_frontier_id', 'NONE')}",
                ]
            )
        elif agent == "verifier":
            lines.append(f"Verdict: {details.get('status', event.get('status', 'n/a'))}")
            for check_name, check in (details.get("checks", {}) or {}).items():
                if check_name == "all_ok" or not isinstance(check, dict):
                    continue
                state = "SKIP" if check.get("skipped") else "PASS" if check.get("ok") else "FAIL"
                lines.append(f"{check_name}: {state}")

        role_lines[agent].append("\n".join(lines))

    role_summaries = {
        role: "\n\n".join(parts) if parts else "No event this step."
        for role, parts in role_lines.items()
    }
    decision = final_decision or {}
    decision_lines = [
        f"Action: {decision.get('action_type', 'n/a')}",
        f"Target: {decision.get('chosen_id', 'n/a')}",
        f"Confidence: {_short_float(decision.get('confidence'))}",
    ]
    target = decision.get("target_location")
    if isinstance(target, (list, tuple)) and len(target) >= 3:
        decision_lines.append("Point: " + ", ".join(_short_float(v) for v in target[:3]))
    pdf = decision.get("pdf_params") if isinstance(decision.get("pdf_params"), dict) else {}
    if pdf:
        decision_lines.append(f"Navmesh masked: {bool(pdf.get('density_masked', False))}")
        if pdf.get("mass_in_tau_ball") is not None:
            decision_lines.append(f"Mass within 1 m: {_short_float(pdf.get('mass_in_tau_ball'), 3)}")
    return {"roles": role_summaries, "decision": "\n".join(decision_lines)}


class RRLogger:
    def __init__(self, output_path):
        # Initialize Rerun and specify the .rrd file for logging
        full_output_path = output_path / "test_logger.rrd"

        self._timeline = "vlm_plan_logging"
        rr.init(self._timeline)
        rr.save(full_output_path)

        self.primary_camera_entity = "world/camera"

        # Keep the simulation large and central. Compact task and agent views
        # live on the sides so they never cover the 3D scene.
        left_panel = rrb.Vertical(
            rrb.TextDocumentView(
                name="Question and pose", origin="/ui/question", contents=["/ui/question"]
            ),
            rrb.TextDocumentView(
                name="Orchestrator",
                origin="/ui/transcripts/orchestrator",
                contents=["/ui/transcripts/orchestrator"],
            ),
            rrb.TextDocumentView(
                name="Grounding",
                origin="/ui/transcripts/grounding",
                contents=["/ui/transcripts/grounding"],
            ),
            row_shares=[0.2, 0.4, 0.4],
            name="Task and grounding",
        )

        camera_tabs = rrb.Tabs(
            rrb.Spatial2DView(
                name="RGB",
                origin=self.primary_camera_entity,
                contents=["$origin/rgb", "/world/annotations/**"],
            ),
            rrb.Spatial2DView(
                name="Semantic Labels",
                origin=self.primary_camera_entity,
                contents=["$origin/semantic", "/world/annotations/**"],
            ),
            rrb.Spatial2DView(
                name="Instance Labels",
                origin=self.primary_camera_entity,
                contents=["$origin/instance", "/world/annotations/**"],
            ),
            rrb.Spatial2DView(
                name="Depth",
                origin=self.primary_camera_entity,
                contents=["$origin/depth", "/world/annotations/**"],
            ),
            active_tab=0,
            name="Camera views",
        )

        right_panel = rrb.Vertical(
            rrb.TextDocumentView(
                name="Decision", origin="/ui/decision", contents=["/ui/decision"]
            ),
            rrb.TextDocumentView(
                name="Spatial",
                origin="/ui/transcripts/spatial",
                contents=["/ui/transcripts/spatial"],
            ),
            rrb.TextDocumentView(
                name="Verifier",
                origin="/ui/transcripts/verifier",
                contents=["/ui/transcripts/verifier"],
            ),
            camera_tabs,
            row_shares=[0.18, 0.2, 0.27, 0.35],
            name="Decision and camera",
        )

        blueprint = rrb.Horizontal(
            left_panel,
            rrb.Spatial3DView(
                name="Full simulation", origin="/world", contents=["/world/**"]
            ),
            right_panel,
            column_shares=[0.24, 0.52, 0.24],
        )

        rr.send_blueprint(blueprint)

        self._node_color_map = {
            'object': [225,225,0],
            'frontier': [255,0,0],
            'frontier_selected': [255,255,0],
            'region': [0,0,0],
            'room': [255,0,255],
            'building': [0,255,255],
            'agent': [0,0,255],
        }

        self._node_offset = {
            'object': 0.,
            'frontier': 0.,
            'frontier_selected': 0.,
            'region': 1.,
            'room': 2.,
            'building': 3.,
            'agent': 0.,
        }

        self._edge_color_map = {
            'building-to-room': [225,0,0],
            'room-to-region': [0,255,0],
            'region-to-object': [0,0,255],
            'object-to-region': [0,0,255],
            'region-to-frontier': [255,255,255],
            'region-to-region': [0,0,0],
            'region-to-agent': [255,255,0],
            'frontier-to-object': [255,255,0],
            'room-to-room': [0,0,0],
        }

        self._transcript_history = {
            role: [] for role in ("orchestrator", "grounding", "spatial", "verifier")
        }

        self.reset()

    def reset(self):
        self._t = 0
        self._dt = 0.1
        rr.set_time(self._timeline, duration=self._t)

    def log_mesh_data(self, mesh_vertices, mesh_colors, mesh_triangles):
        
        rr.log(
            "world/mesh",
            rr.Mesh3D(
                vertex_positions=mesh_vertices,
                vertex_colors=mesh_colors,
                triangle_indices=mesh_triangles,
            ),
            static=False,
        )
    
    def log_agent_data(self, agent_positions):
        rr.log(f"world/robot_traj", rr.LineStrips3D(agent_positions, colors=[0, 0, 255]))
        rr.log(f"world/robot_pos", rr.Points3D(agent_positions[-1], colors=[0, 0, 255], radii=0.11))

    def log_traj_data(self, agent_positions):
        rr.log("world/desired_traj", rr.LineStrips3D(agent_positions, colors=[0, 255, 255]))
    
    def log_agent_tf(self, pos, quat):
        translation = np.asarray([pos[0], pos[1], pos[2]])
        quat_mod = np.asarray([quat[1], quat[2], quat[3], quat[0]])
        agent_from_world = rr.Transform3D(
            translation=translation, rotation=rr.Quaternion(xyzw=quat_mod), from_parent=False
        )
        rr.log(f"world/agent_tf", agent_from_world)
        
    def log_camera_tf(self, pos, quat, cam_entity=None):
        translation = np.asarray([pos[0], pos[1], pos[2]])
        quat_mod = np.asarray([quat[1], quat[2], quat[3], quat[0]])
        camera_from_world = rr.Transform3D(
            translation=translation, rotation=rr.Quaternion(xyzw=quat_mod), from_parent=False
        )
        if cam_entity is None:
            cam_entity = self.primary_camera_entity
        rr.log(f"{cam_entity}", camera_from_world)

    def log_target_poses(self, target_poses):
        rr.log("world/target_poses", rr.Points3D(target_poses, colors=[0,255,0], radii=0.11))
    
    def log_nodes_paths(self, nodes_paths):
        rr.log(f"world/desired_node_path", rr.Points3D(nodes_paths, colors=[255, 192, 203], radii=0.11)) # pink
        rr.log("world/desired_node_path_edges", rr.LineStrips3D(nodes_paths, colors=[255, 192, 203])) # pink

    def log_text_data(self, text):
        rr.log(
            "ui/decision",
            rr.TextDocument(
                text,
                media_type=rr.MediaType.TEXT,
            ),
        )

    def log_step_transcript(self, question, step_num, agent_pose, ledger, final_decision):
        summary = simplify_step_transcript(ledger, final_decision)
        pose_text = ", ".join(_short_float(value) for value in agent_pose[:3])
        rr.log(
            "ui/question",
            rr.TextDocument(
                f"{question}\n\nStep: {step_num}\nAgent pose: {pose_text}",
                media_type=rr.MediaType.TEXT,
            ),
        )
        for role, text in summary["roles"].items():
            if text == "No event this step.":
                continue
            self._transcript_history[role].append(f"Step {step_num}\n{text}")
            rr.log(
                f"ui/transcripts/{role}",
                rr.TextDocument(
                    "\n\n".join(self._transcript_history[role]),
                    media_type=rr.MediaType.TEXT,
                ),
            )
        rr.log(
            "ui/decision",
            rr.TextDocument(
                f"Step {step_num}\n{summary['decision']}",
                media_type=rr.MediaType.TEXT,
            ),
        )
        return summary

    def log_navmesh_data(self, navmesh):
        # log the frontier nodes with color red
        rr.log(
            f"world/navmesh_nodes",
            rr.Points3D(navmesh, colors=[255,255,255], radii=0.11)
        )

    def log_frontier_data(self, frontier_node_positions):
        # log the frontier nodes with color red
        rr.log(
            "world/frontier_nodes",
            rr.Points3D(frontier_node_positions, colors=[255,0,0], radii=0.08)
        )

    def log_selected_frontier_data(self, frontier_node_positions):
        # log the frontier nodes with color red
        rr.log(
            "world/selected_frontier_nodes",
            rr.Points3D(frontier_node_positions, colors=[255, 255, 0], radii=0.08)
        )
    
    def log_place_data(self, place_node_positions):
        # log the place nodes with color red
        rr.log(
            "world/place_nodes",
            rr.Points3D(place_node_positions, colors=[255,255,255], radii=0.08)
        )

    def log_inplane_place_data(self, place_node_positions):
        # log the place nodes with color red
        rr.log(
            "world/inplane_place_nodes",
            rr.Points3D(place_node_positions, colors=[244, 5, 244], radii=0.09)
        )

    def log_bb_data(self, bb_info):
        rr.log(
            "/world/annotations/bb",
            rr.Boxes3D(
                half_sizes=bb_info['bb_half_sizes'],
                centers=bb_info['bb_centroids'],
                labels=bb_info['bb_labels'],
                colors=bb_info['bb_colors']
            ),
            rr.InstancePoses3D(mat3x3=bb_info['bb_mat3x3']),
            static=False,
        )

    def log_img_data(self, rgb, labels):
        # log the camera transform, rgb image, and depth image
        # rr.log("world/agent", rr.Transform3D(transform=camera_from_world))
        # rr.log("world/agent", rr.Pinhole(image_from_camera=intrinsic, resolution=[w, h]))
        rr.log(f"{self.primary_camera_entity}/rgb", rr.Image(rgb).compress(jpeg_quality=95))
        rr.log(f"{self.primary_camera_entity}/semantic", rr.SegmentationImage(labels))
        # rr.log(f"{self.primary_camera_entity}/semantic", rr.Image(data.colormap(data.labels)).compress(jpeg_quality=95))

    def log_rosbag_img_data(self, data):
        # log the camera transform, rgb image, and depth image
        # rr.log("world/agent", rr.Transform3D(transform=camera_from_world))
        # rr.log("world/agent", rr.Pinhole(image_from_camera=intrinsic, resolution=[w, h]))
        rr.log(f"{self.primary_camera_entity}/rgb", rr.Image(np.transpose(data.rgb, axes=(1, 0, 2))).compress(jpeg_quality=95))
        rr.log(f"{self.primary_camera_entity}/depth", rr.DepthImage(data.depth.T, meter=1.0))
        rr.log(f"{self.primary_camera_entity}/semantic", rr.SegmentationImage(data.semantic_image[:, :, 0].T))

    def log_2d_frontier_data(self, unoccupied, unexplored, tsdf):
        rr.log(f"{self.primary_camera_entity}/unoccupied", rr.Image(unoccupied).compress(jpeg_quality=95))
        rr.log(f"{self.primary_camera_entity}/unexplored", rr.Image(unexplored).compress(jpeg_quality=95))
        rr.log(f"{self.primary_camera_entity}/tsdf", rr.Image(tsdf).compress(jpeg_quality=95))

    def log_3d_frontier_data(self, unoccupied_reachable_normal, frontiers_normal, frontiers_unoccupied):
        rr.log(f"world/tsdf_unoccupied", rr.Points3D(unoccupied_reachable_normal, colors=[255, 0, 0], radii=0.06))
        rr.log(f"world/tsdf_frontiers", rr.Points3D(frontiers_normal, colors=[255, 255, 255], radii=0.08))
        rr.log(f"world/tsdf_explored", rr.Points3D(frontiers_unoccupied, colors=[200, 180, 150], radii=0.08))
    
    def log_hydra_graph(
            self, 
            is_node=True, 
            node_type='object', 
            nodeid=None, 
            edgeid=None, 
            edge_type='room-to-place', 
            node_pos_source=None, 
            node_pos_target=None):

        if is_node:
            node_pos_source[2] += self._node_offset[node_type]
            rr.log(
                f"world/hydra_graph/nodes/{node_type}/{nodeid}",
                rr.Points3D(node_pos_source, colors=self._node_color_map[node_type], radii=0.09)
            )
        else: # edge
            source_type = edge_type.split('-to-')[0]
            target_type = edge_type.split('-to-')[1]

            node_pos_source[2] += self._node_offset[source_type]
            node_pos_target[2] += self._node_offset[target_type]
            rr.log(f"world/hydra_graph/edges/{edge_type}/{edgeid}", rr.Arrows3D(
                origins=node_pos_source,  # Base position of the arrow
                vectors=(node_pos_target-node_pos_source),  # Direction and length of the arrow
                colors=self._edge_color_map[edge_type]
            ))

    def step(self):
        self._t += self._dt
        rr.set_time(self._timeline, duration=self._t)

    def log_clear(self, namespace):
        rr.log(namespace, rr.Clear(recursive=True))
