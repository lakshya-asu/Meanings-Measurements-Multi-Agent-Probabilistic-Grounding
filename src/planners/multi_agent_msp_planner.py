import numpy as np
import json
import click
from pathlib import Path
from typing import Optional, Dict, Any

from src.utils.data_utils import get_latest_image

# Import your existing MSP Engine
from src.planners.vlm_planner_msp_debug import MSPEngineSmart, _parse_q_dist

# Import the new Multi-Agent components
from src.multi_agent.blackboard import Blackboard
from src.multi_agent.agents.orchestrator_agent import OrchestratorAgent
from src.multi_agent.agents.grounding_agent import GroundingAgent
from src.multi_agent.agents.spatial_agent import SpatialAgent
from src.multi_agent.agents.verifier_agent import VerifierAgent
from src.multi_agent.agents.logical_agent import LogicalAgent
from src.multi_agent.agents.qa_agent import QaAgent
from src.multi_agent.agent_setup import AgentFactory

def _write_json(path: Path, obj: Any) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(obj, f, indent=2)
    except Exception as e:
        print(f"[MSP] Failed to write json {path}: {e}")

class MultiAgentMSPPlanner:
    def __init__(self, cfg, sg_sim, question, out_path=".", answer_mode="where", **kwargs):
        self.cfg = cfg
        self.sg_sim = sg_sim
        self.out_path = Path(out_path)
        
        click.secho(f"\n{'='*40}\nINITIALIZING MULTI-AGENT PLANNER\n{'='*40}", fg="magenta", bold=True)
        click.secho(f"Question: {question}", fg="magenta")
        click.secho(f"Mode: {answer_mode}", fg="magenta")
        
        self.blackboard = Blackboard(question=question, mode=answer_mode)
        
        # Determine providers
        providers = kwargs.get("agent_providers", {})
        o_prov = providers.get("orchestrator", "claude")
        g_prov = providers.get("grounding", "claude")
        s_prov = providers.get("spatial", "claude")
        v_prov = providers.get("verifier", "claude")
        l_prov = providers.get("logical", "claude")
        q_prov = providers.get("qa", "claude")
        
        click.secho(f"Providers: Orch={o_prov}, Ground={g_prov}, Spatial={s_prov}, Verif={v_prov}, Logic={l_prov}, QA={q_prov}", fg="yellow")
        
        # Initialize Agents dynamically
        self.orchestrator = AgentFactory.create_agent("orchestrator", provider=o_prov)
        self.grounder = AgentFactory.create_agent("grounding", provider=g_prov)
        self.spatial = AgentFactory.create_agent("spatial", provider=s_prov)
        self.verifier = AgentFactory.create_agent("verifier", provider=v_prov)
        self.logical = AgentFactory.create_agent("logical", provider=l_prov)
        self.qa = AgentFactory.create_agent("qa", provider=q_prov)
        
        if "choices" in kwargs:
            self.blackboard.choices = kwargs["choices"]
        
        # Keep your existing robust math engine
        self.msp_engine = MSPEngineSmart(
            sigma_s_factor=float(getattr(self.cfg, "sigma_s_factor", 0.5)),
            sigma_m_factor=float(getattr(self.cfg, "sigma_m_factor", 0.3)),
            kappa_factor=float(getattr(self.cfg, "kappa_factor", 10.0)),
        )
        
        # New configurable top_k parameter for returning best objects
        self.top_k = int(getattr(self.cfg, "top_k_objects", 2))
        
        # Persistent context
        self.locked_anchor_id = None

    def _get_room_for_node(self, node_id: str) -> Optional[str]:
        """Traverse the Habitat scene graph hierarchy (Node -> Region -> Room)."""
        try:
            graph = getattr(self.sg_sim, "filtered_netx_graph", None)
            if not graph or not graph.has_node(node_id):
                return None
            
            # Walk up the edges to find the parent region or room
            for neighbor in graph.neighbors(node_id):
                neighbor_str = str(neighbor).lower()
                if "room" in neighbor_str:
                    return str(neighbor)
                if "region" in neighbor_str:
                    for room_candidate in graph.neighbors(neighbor):
                        if "room" in str(room_candidate).lower():
                            return str(room_candidate)
        except Exception:
            pass
        return None

    def _get_scene_data(self):
        from src.envs.utils import pos_normal_to_habitat
        objects, frontiers = [], []
        for oid, name in zip(self.sg_sim.object_node_ids, self.sg_sim.object_node_names):
            pos_norm = self.sg_sim.get_position_from_id(oid)
            if pos_norm is not None:
                pos_hab = np.asarray(pos_normal_to_habitat(np.asarray(pos_norm, dtype=np.float32)), dtype=np.float32)
                objects.append({"id": str(oid), "name": str(name).lower(), "position": pos_hab.tolist(), "size": [0.5, 0.5, 0.5]})
        
        for fid in getattr(self.sg_sim, "frontier_node_ids", []) or []:
            pos_norm = self.sg_sim.get_position_from_id(fid)
            if pos_norm is not None:
                pos_hab = np.asarray(pos_normal_to_habitat(np.asarray(pos_norm, dtype=np.float32)), dtype=np.float32)
                frontiers.append({"id": str(fid), "name": "frontier", "position": pos_hab.tolist(), "size": [0.5, 0.5, 0.5]})
        return objects, frontiers

    def get_next_action(self, agent_yaw_rad: float = 0.0, agent_pos_hab: Optional[np.ndarray] = None):
        if agent_pos_hab is None:
            agent_pos_hab = np.array([0, 0, 0], dtype=np.float32)
            
        objects, frontiers = self._get_scene_data()
        img_path = get_latest_image(self.out_path)
        if img_path:
            img_path = str(img_path)
            
        agent_state_str = self.sg_sim.get_current_semantic_state_str()
        
        # --- Step Header Logging ---
        step_num = self.blackboard.step_t + 1
        click.secho(f"\n{'='*20} MULTI-AGENT STEP {step_num} {'='*20}", fg="magenta", bold=True)
        click.secho(f"[Env] Pose: {agent_pos_hab.tolist()} | Yaw: {agent_yaw_rad:.3f} rad", fg="white")
        click.secho(f"[Env] Semantic State: {agent_state_str}", fg="white")
        click.secho(f"[Env] Found {len(objects)} Objects, {len(frontiers)} Frontiers", fg="white")
        click.secho(f"[Scene Graph]\n{self.sg_sim.scene_graph_str}", fg="blue")
        click.secho("-" * 60, fg="white")
        
        # 1. Update Blackboard
        self.blackboard.update_state(
            t=step_num,
            pose=agent_pos_hab,
            yaw=agent_yaw_rad,
            img_path=img_path,
            sg_str=self.sg_sim.scene_graph_str,
            agent_state=agent_state_str,
            objects=objects,
            frontiers=frontiers
        )
        
        def finalize_step(target_pose, target_id, is_conf, conf, extra):
            """Helper to log trace and print final decision before returning."""
            click.secho(f"\n[DECISION] Action: {extra.get('action_type')} | Target ID: {extra.get('chosen_id')} | Conf: {conf:.2f}", fg="yellow", bold=True)
            if extra.get("thought"):
                click.secho(f"[DECISION] Thought: {extra.get('thought')}", fg="yellow")
            
            trace_dump = {
                "t": step_num,
                "agent_pose": agent_pos_hab.tolist(),
                "agent_yaw": agent_yaw_rad,
                "ledger": self.blackboard.event_ledger,
                "final_decision": extra
            }
            _write_json(self.out_path / f"trace_step_{step_num:03d}.json", trace_dump)
            return target_pose, target_id, is_conf, conf, extra

        # =====================================================================
        # MCQ FAST PATH OVERRIDE
        # =====================================================================
        if self.blackboard.choices:
            click.secho(f"[Planner] Multiple Choice Query detected. Executing QA Fast Path.", fg="cyan")
            qa_out = self.qa.process(self.blackboard)
            if qa_out.get("ok", False):
                action_type = qa_out.get("action_type", "lookaround")
                chosen_id = qa_out.get("chosen_id", "NONE")
                ans = qa_out.get("answer", "")
                conf_val = qa_out.get("confidence", 0.0)
                
                # Resolve target pose
                target_pose = None
                if action_type in ["goto_object", "goto_frontier"] and chosen_id != "NONE":
                    target_pose = self.sg_sim.get_position_from_id(chosen_id)
                
                # If action is answer, we don't set a target
                if action_type == "answer":
                    target_id = ans
                    is_conf = (conf_val >= float(getattr(self.cfg, "pre_answer_conf_thresh", 0.8)))
                else:
                    target_id = chosen_id
                    is_conf = False
                    
                extra = {
                    "action_type": action_type,
                    "chosen_id": target_id,
                    "thought": qa_out.get("reasoning", "")
                }
                
                def fallback_step():
                    fid = str(frontiers[0]["id"]) if frontiers else ""
                    fallback_action = "goto_frontier" if fid else "lookaround"
                    return finalize_step(self.sg_sim.get_position_from_id(fid) if fid else None, fid, False, 0.0, {"action_type": fallback_action, "chosen_id": fid, "thought": "QA Fast Path failed geometry. Fallback exploring."})
                
                if action_type in ["goto_object", "goto_frontier"] and target_pose is None:
                    return fallback_step()
                    
                return finalize_step(target_pose, target_id, is_conf, conf_val, extra)
            else:
                click.secho(f"[Planner] QA Fast Path crashed. Proceeding with standard fallback.", fg="red")
        
        # 2. Agent 1: Orchestrate
        orch_out = self.orchestrator.process(self.blackboard)
        
        # 3. Agent 2: Ground
        ground_out = self.grounder.process(self.blackboard, orch_out)
        


        if ground_out.get("needs_exploration", False) or not ground_out.get("grounded_anchors"):
            anchor_in_view = False
        else:
            anchor_in_view = ground_out["grounded_anchors"][0]["matched_object_id"] != "NONE"

        if not hasattr(self, "steps_since_anchor_seen"):
            self.steps_since_anchor_seen = 0

        if not self.locked_anchor_id:
            if not anchor_in_view:
                fid = str(frontiers[0]["id"]) if frontiers else ""
                action = "goto_frontier" if fid else "lookaround"
                return finalize_step(self.sg_sim.get_position_from_id(fid) if fid else None, fid, False, 0.0, {"action_type": action, "chosen_id": fid, "thought": "Missing anchors. Exploring."})

            # Check primary anchor
            primary_anchor_id = ground_out["grounded_anchors"][0]["matched_object_id"]
            if primary_anchor_id == "NONE":
                 fid = str(frontiers[0]["id"]) if frontiers else ""
                 return finalize_step(self.sg_sim.get_position_from_id(fid) if fid else None, fid, False, 0.0, {"action_type": "goto_frontier" if fid else "lookaround", "chosen_id": fid, "thought": "Anchor is NONE. Exploring."})
            
            # Lock the anchor once found
            self.locked_anchor_id = primary_anchor_id
            self.locked_anchor_pos = self.sg_sim.get_position_from_id(self.locked_anchor_id)
            self.steps_since_anchor_seen = 0
            click.secho(f"[Anchor Locked] ID: {self.locked_anchor_id}", fg="green", bold=True)
            # After locking, we navigate near it to verify
            return finalize_step(self.sg_sim.get_position_from_id(self.locked_anchor_id), self.locked_anchor_id, False, 0.0, {"action_type": "goto_object", "chosen_id": self.locked_anchor_id, "thought": "Anchor locked. Navigating to anchor to cross-reference visual with scene graph."})
            
        else:
            primary_anchor_id = self.locked_anchor_id
            if anchor_in_view and ground_out["grounded_anchors"][0]["matched_object_id"] == self.locked_anchor_id:
                self.steps_since_anchor_seen = 0
            else:
                self.steps_since_anchor_seen += 1
                
            if self.steps_since_anchor_seen > 5:
                # We haven't seen the anchor in 5 steps. Navigate back to it.
                self.steps_since_anchor_seen = 0
                return finalize_step(self.locked_anchor_pos, primary_anchor_id, False, 0.0, {"action_type": "goto_object", "chosen_id": primary_anchor_id, "thought": "Lost sight of anchor for 5 steps. Navigating back to its last known position."})
            
        primary_anchor_obj = next((o for o in objects if o["id"] == primary_anchor_id), objects[0])

        # 4. Agent 3: Spatial Geometry
        spatial_out = self.spatial.process(self.blackboard, primary_anchor_obj)
        if not spatial_out.get("ok", False):
            return finalize_step(self.sg_sim.get_position_from_id(primary_anchor_id), primary_anchor_id, False, 0.0, {"action_type": "goto_object", "chosen_id": primary_anchor_id, "thought": "Spatial failed (likely occluded). Moving closer to object."})

        # 5. Agent 4: Verify
        verification = self.verifier.process(self.blackboard)
        if verification.get("status") == "FAIL":
            self.blackboard.global_history += f"Step {step_num} FAIL: {verification.get('feedback')}\n"
            fid = str(frontiers[0]["id"]) if frontiers else ""
            return finalize_step(self.sg_sim.get_position_from_id(fid) if fid else None, fid, False, 0.0, {"action_type": "lookaround", "chosen_id": "", "thought": f"Verifier rejected logic: {verification.get('feedback')}"})

        # (QA Agent call has been moved to the MCQ Fast Path above)

        # =====================================================================
        # 6. Run MSP Math (Probabilistic Scoring & Point Estimation)
        # =====================================================================
        dist_m = _parse_q_dist(self.blackboard.question)
        anchor_pos = np.asarray(primary_anchor_obj["position"], dtype=np.float32)
        
        msp_objects, msp_frontiers = self.msp_engine.score_candidates(
            objects=objects,
            frontiers=frontiers,
            anchor_pos_hab=anchor_pos,
            anchor_size=primary_anchor_obj.get("size", [0.5, 0.5, 0.5]),
            kernel_params=spatial_out,
            question_dist=dist_m,
            planar=True,
            flatten_semantic=bool(getattr(self.cfg, "flatten_semantic", False))
        )

        point_estimate = self.msp_engine.estimate_point_from_pdf(
            anchor_pos_hab=anchor_pos,
            kernel_params=spatial_out,
            question_dist=dist_m,
            anchor_size=primary_anchor_obj.get("size", [0.5, 0.5, 0.5]),
            planar=True,
            use_map=True
        )
        point_xyz = np.asarray(point_estimate["xyz_chosen_hab"], dtype=np.float32)

        # Extract the continuous PDF parameters from the shared debug trace 
        # (All candidates share the same anchor-centric pdf params)
        extracted_pdf_params = {}
        if msp_objects:
            extracted_pdf_params = msp_objects[0].get("_msp_debug", {}).get("metric_semantic_params", {})
            predicates = msp_objects[0].get("_msp_debug", {}).get("predicate_params", {})
            extracted_pdf_params.update(predicates)

        # Build output structure with PDF, point, and top K objects
        
        # --- NEW: Top-Down 2D Matplotlib Heatmap Export ---
        if extracted_pdf_params:
            try:
                import matplotlib.pyplot as plt
                from src.msp.pdf import combined_logpdf as _combined_logpdf
                
                # Create a 10x10m grid centered around the anchor object
                grid_res = 0.2
                x_min, x_max = anchor_pos[0] - 5.0, anchor_pos[0] + 5.0
                z_min, z_max = anchor_pos[2] - 5.0, anchor_pos[2] + 5.0
                
                xx, zz = np.meshgrid(
                    np.arange(x_min, x_max, grid_res),
                    np.arange(z_min, z_max, grid_res)
                )
                yy = np.full_like(xx, anchor_pos[1]) # Keep Y (elevation) constant
                
                # Flatten the grid for logpdf evaluation
                grid_logps = _combined_logpdf(xx.ravel(), yy.ravel(), zz.ravel(), extracted_pdf_params)
                grid_logps = grid_logps.reshape(xx.shape)
                
                # Plot the heat map
                fig, ax = plt.subplots(figsize=(8, 8))
                
                # We use origin='lower' to match array indexing with Cartesian Y-up (which is Z visually in Habitat topdown)
                c = ax.pcolormesh(xx, zz, np.exp(grid_logps), shading='auto', cmap='viridis', alpha=0.8)
                plt.colorbar(c, ax=ax, label='Probability Density')
                
                ax.plot(anchor_pos[0], anchor_pos[2], 'r*', markersize=15, label='Anchor Object')
                ax.plot(agent_pos_hab[0], agent_pos_hab[2], 'b^', markersize=12, label='Agent')
                ax.plot(point_xyz[0], point_xyz[2], 'gx', markersize=12, label='MSP Target Guess')
                
                ax.set_title(f"Step {step_num} Point-Estimation Heatmap")
                ax.set_xlabel("X (World)")
                ax.set_ylabel("Z (World) [Top-down Depth]")
                ax.legend()
                ax.grid(True, linestyle='--', alpha=0.5)
                
                plt.tight_layout()
                heatmap_file = self.out_path / f"heatmap_step_{step_num:03d}.png"
                plt.savefig(heatmap_file, dpi=150)
                plt.close(fig)
                click.secho(f"[MSP] Exported 2D heatmap to {heatmap_file.name}", fg="cyan")
            except Exception as e:
                click.secho(f"[MSP] Failed to generate 2D heatmap: {e}", fg="red")

        top_k_objects = []
        top_k_objects = []
        for obj in msp_objects[:self.top_k]:
             top_k_objects.append({
                 "id": obj["id"],
                 "name": obj.get("name", ""),
                 "position": obj["position"],
                 "confidence": float(np.exp(obj.get("msp_score", -100.0))) # Convert logpdf roughly to score
             })

        # =====================================================================
        # 7. Final Action Decision (Topological Bounding + PDF Ranking)
        # =====================================================================
        anchor_room_id = self._get_room_for_node(primary_anchor_id)
        
        same_room_objects = []
        same_room_frontiers = []
        
        for obj in msp_objects:
            if obj["id"] == primary_anchor_id:
                continue
            obj_room_id = self._get_room_for_node(obj["id"])
            if anchor_room_id and obj_room_id:
                if anchor_room_id == obj_room_id:
                    same_room_objects.append(obj)
            elif np.linalg.norm(np.array(obj["position"]) - anchor_pos) < 6.0:
                same_room_objects.append(obj)

        for f in msp_frontiers:
            f_room_id = self._get_room_for_node(f["id"])
            if anchor_room_id and f_room_id:
                if anchor_room_id == f_room_id:
                    same_room_frontiers.append(f)
            elif np.linalg.norm(np.array(f["position"]) - anchor_pos) < 6.0:
                same_room_frontiers.append(f)

        room_fully_explored = (len(same_room_frontiers) == 0)
        global_map_exhausted = (len(frontiers) == 0)
        
        # Format the comprehensive output
        def build_answer(action_type, chosen_id, conf_score, target_hab, thought):
             return finalize_step(target_hab, chosen_id, True if conf_score > 0.9 else False, conf_score, {
                 "action_type": action_type,
                 "chosen_id": chosen_id,
                 "confidence": conf_score,
                 "thought": thought,
                 "pdf_params": extracted_pdf_params,
                 "target_location": point_xyz.tolist(),
                 "top_k_objects": top_k_objects
             })
             
        is_location_target = any(w in orch_out.get("target_entity", "").lower() for w in ["location", "region", "point", "place", "area"])

        # LOGIC A: If the target is explicitly a location, and we calculated the geometry successfully, go there directly!
        if is_location_target and self.locked_anchor_id:
             return build_answer("goto_object", "POINT_GUESS", 0.95, point_xyz, "Anchor is locked and spatial geometry is calculated. Navigating directly to the requested continuous location.")

        # LOGIC B: If the room is fully explored, trust the PDF rankings.
        if room_fully_explored or global_map_exhausted:
            if same_room_objects:
                best_obj = same_room_objects[0]
                conf_score = 0.95 
                thought = f"Room {anchor_room_id} fully explored. Selected highest PDF-scoring object."
                return build_answer("answer", best_obj["id"], conf_score, None, thought)
            else:
                return build_answer("goto_object", "POINT_GUESS", 0.0, point_xyz, "Room explored but target missing. Navigating to raw PDF median target location.")

        # LOGIC C: Room is NOT fully explored, but we found a fantastic match early.
        if same_room_objects and float(same_room_objects[0].get("msp_score", -100.0)) > -1.5:
             best_obj = same_room_objects[0]
             return build_answer("answer", best_obj["id"], 0.95, None, f"Extremely high PDF match found.")

        # LOGIC D: Keep exploring the remaining frontiers in this room.
        if same_room_frontiers:
            fid = same_room_frontiers[0]["id"]
            return finalize_step(self.sg_sim.get_position_from_id(fid), fid, False, 0.0, {
                "action_type": "goto_frontier", 
                "chosen_id": fid, 
                "thought": f"Exploring remaining frontiers inside the anchor's room ({anchor_room_id})."
            })
            
        # Fallback
        fid = str(frontiers[0]["id"]) if frontiers else ""
        return finalize_step(self.sg_sim.get_position_from_id(fid) if fid else None, fid, False, 0.0, {
            "action_type": "lookaround" if not fid else "goto_frontier", 
            "chosen_id": fid, 
            "thought": "Fallback exploration triggered."
        })
