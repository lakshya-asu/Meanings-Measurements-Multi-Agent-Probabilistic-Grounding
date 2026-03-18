
import json
from pathlib import Path
from datetime import datetime

import click
import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

import habitat_sim
from habitat_sim.utils.common import (
    quat_from_coeffs,
    quat_from_two_vectors,  # optional; kept for future yaw needs
)

import hydra_python

from src.planners.vlm_planner_gemini_spatial import VLMPlannerEQAGeminiSpatial

from src.envs.habitat import run
from src.envs.habitat_interface import HabitatInterface
from src.logging.rr_logger import RRLogger
from src.logging.utils import log_experiment_status, should_skip_experiment
from src.occupancy_mapping.geom import get_cam_intr, get_scene_bnds
from src.occupancy_mapping.tsdf import TSDFPlanner
from src.scene_graph.scene_graph_sim import SceneGraphSim
from src.utils.data_utils import get_traj_len_from_poses
from src.utils.hydra_utils import initialize_hydra_pipeline
from src.envs.utils import pos_habitat_to_normal, pos_normal_to_habitat


# ---------------- I/O ----------------
def load_spatial_data(json_path: str):
    with open(json_path, "r") as f:
        return json.load(f)


# -------------- Warm-up & SG init --------------
def _warmup_like_eqa(habitat_data: HabitatInterface, pipeline, out_dir, rr_logger, tsdf_planner, sg_sim):
    """Use the same warmup pattern as run_vlm_planner_eqa_habitat.py."""
    agent = habitat_data._sim.get_agent(0)
    init_pts_hab = np.array(agent.get_state().position, dtype=np.float32)
    init_angle = float(habitat_data.get_heading_angle())
    tilt = float(getattr(habitat_data.cfg, "camera_tilt_deg", -30.0))
    poses = habitat_data.get_init_poses_eqa(init_pts_hab, init_angle, tilt)
    try:
        run(
            pipeline, habitat_data, poses, output_path=out_dir,
            rr_logger=rr_logger, tsdf_planner=tsdf_planner, sg_sim=sg_sim,
            save_image=True, segmenter=None,
        )
    except Exception as e:
        click.secho(f"[warmup] non-fatal render error: {e}", fg="yellow")


def _ensure_scenegraph_initialized(habitat_data: HabitatInterface, pipeline, out_dir, rr_logger, tsdf_planner, sg_sim):
    """If SG didn’t init during warmup, force tiny extra ticks using the same pose API."""
    def _needs_init():
        return (
            not hasattr(sg_sim, "filtered_netx_graph")
            or sg_sim.filtered_netx_graph is None
            or not hasattr(sg_sim, "curr_agent_id")
        )

    if not _needs_init():
        return

    click.secho("[init] SG not ready after warm-up; forcing extra render ticks…", fg="yellow")
    for _ in range(2):
        agent = habitat_data._sim.get_agent(0)
        init_pts_hab = np.array(agent.get_state().position, dtype=np.float32)
        init_angle = float(habitat_data.get_heading_angle())
        tilt = float(getattr(habitat_data.cfg, "camera_tilt_deg", -30.0))
        poses = habitat_data.get_init_poses_eqa(init_pts_hab, init_angle, tilt)
        try:
            run(
                pipeline, habitat_data,
                poses[:2] if len(poses) > 2 else poses,
                output_path=out_dir, rr_logger=rr_logger,
                tsdf_planner=tsdf_planner, sg_sim=sg_sim,
                save_image=True, segmenter=None,
            )
        except Exception as e:
            click.secho(f"[init] non-fatal render error: {e}", fg="yellow")
        if not _needs_init():
            break


# -------------- Spawn/teleport helpers --------------
def _snap_or_convert_to_habitat(pf, xyz, assume_normal=False):
    """
    Try xyz as Habitat coords; if not navigable (or assume_normal),
    convert from 'normal' -> Habitat then snap to navmesh.
    """
    pt = np.array(xyz, dtype=np.float32)
    if not assume_normal and pf.is_navigable(pt):
        return pf.snap_point(pt)
    try:
        pt_hab = pos_normal_to_habitat(pt)
        return pf.snap_point(pt_hab)
    except Exception:
        # Last resort: random navigable point so we don't crash
        return pf.get_random_navigable_point()


def _find_room_id_for_object(sg_sim, object_label_base: str):
    """
    From the SG, find a room node connected to the first object whose name matches `object_label_base`.
    Returns the room_id or None.
    """
    G = getattr(sg_sim, "filtered_netx_graph", None)
    if G is None:
        return None

    obj_ids = list(getattr(sg_sim, "object_node_ids", []) or [])
    obj_names = list(getattr(sg_sim, "object_node_names", []) or [])
    for oid, nm in zip(obj_ids, obj_names):
        if str(nm).strip().lower() == object_label_base:
            # look for an adjacent room node
            for nb in G.neighbors(oid):
                nd = G.nodes[nb]
                ntype = (nd.get("node_type") or nd.get("type") or "").lower()
                if "room" in ntype:
                    return nb
            break
    return None


def _teleport_to_room_center(habitat_data: HabitatInterface, sg_sim, room_id: str):
    """
    Teleport the agent to the room center (from SG), snapped to navmesh.
    Keep current rotation to avoid 'invalid swizzle' issues.
    Returns the new position (hab) or None on failure.
    """
    try:
        room_pos = sg_sim.get_position_from_id(room_id)  # likely 'normal' coords
    except Exception:
        return None

    pf = habitat_data.pathfinder
    room_hab = _snap_or_convert_to_habitat(pf, room_pos, assume_normal=False)

    st = habitat_sim.AgentState()
    st.position = np.asarray(room_hab, dtype=np.float32)
    st.rotation = habitat_data._sim.get_agent(0).get_state().rotation  # preserve rotation
    try:
        habitat_data._sim.get_agent(0).set_state(st)
        return room_hab
    except Exception:
        return None


# --------- Name/ID & distance helpers (scene-graph based) ----------
def _base_label(s: str) -> str:
    # normalize "toilet_56" -> "toilet"
    return str(s).split("_")[0].strip().lower()

def _sg_find_ids_by_label(sg_sim, base_label: str):
    """Return list of internal object_ids in the SG whose name equals base_label."""
    ids = []
    names = list(getattr(sg_sim, "object_node_names", []) or [])
    oids  = list(getattr(sg_sim, "object_node_ids", []) or [])
    for oid, nm in zip(oids, names):
        if str(nm).strip().lower() == base_label:
            ids.append(oid)
    return ids

def _sg_get_pos(sg_sim, node_id: str):
    """Return np.float32 position for a node_id from the SG, or None."""
    try:
        p = sg_sim.get_position_from_id(node_id)
        return np.array(p, dtype=np.float32)
    except Exception:
        return None

def _measure_declared_to_ref_distance(sg_sim, declared_label: str, ref_label: str):
    """
    Find nearest distance (meters) between any object with name=declared_label
    and any object with name=ref_label in the current scene graph.
    Returns (distance_m, declared_obj_id, ref_obj_id) or (None, None, None).
    """
    A_ids = _sg_find_ids_by_label(sg_sim, _base_label(declared_label))
    B_ids = _sg_find_ids_by_label(sg_sim, _base_label(ref_label))
    best = (None, None, None)

    dmin = np.inf
    for ida in A_ids:
        pa = _sg_get_pos(sg_sim, ida)
        if pa is None:
            continue
        for idb in B_ids:
            pb = _sg_get_pos(sg_sim, idb)
            if pb is None:
                continue
            d = float(np.linalg.norm(pa - pb))
            if d < dmin:
                dmin = d
                best = (dmin, ida, idb)

    if not np.isfinite(dmin):
        return (None, None, None)
    return best


# -------------- Metrics helper --------------
def _shortest_path_len_pf(habitat_data: HabitatInterface, start_pos_hab, goal_pos_hab) -> float:
    """Compute shortest-path length with Habitat pathfinder."""
    pf = habitat_data.pathfinder
    sp = habitat_sim.nav.ShortestPath()
    sp.requested_start = np.array(start_pos_hab, dtype=np.float32)
    sp.requested_end = np.array(goal_pos_hab, dtype=np.float32)
    if not pf.find_path(sp) or len(sp.points) < 2:
        return 0.0
    pts = np.asarray(sp.points, dtype=np.float32)
    segs = np.linalg.norm(pts[1:] - pts[:-1], axis=1)
    return float(segs.sum())


# ---------------- Main ----------------
def main(cfg):
    tasks_data = load_spatial_data(cfg.data.spatial_data_path)

    base_output_root = Path(__file__).resolve().parent.parent / cfg.output_path
    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = base_output_root / run_stamp
    output_path.mkdir(parents=True, exist_ok=True)

    results_filename = output_path / f"{cfg.results_filename}.json"
    device = f"cuda:{cfg.gpu}" if torch.cuda.is_available() else "cpu"

    for task_index, task_data in enumerate(tqdm(tasks_data, desc="Running Tasks")):
        scene_dir = task_data.get("scene_dir", task_data.get("scene"))
        if scene_dir is None:
            click.secho(f"[skip] Task {task_index}: missing scene info.", fg="yellow")
            continue
        scene_id = task_data.get("scene_id", scene_dir.split("-", 1)[-1])
        scene_glb_path = f"{cfg.data.scene_data_path}/{scene_dir}/{scene_id}.basis.glb"

        question = str(task_data["question"])
        ground_truth_target = task_data["ground_truth_target"]
        experiment_id = f'{task_index}_{scene_dir}_{ground_truth_target["name"]}'

        if should_skip_experiment(experiment_id, filename=results_filename):
            click.secho(f"Skipping finished task: {experiment_id}", fg="yellow")
            continue

        # per-task scope with defensive cleanup
        task_output_path: Path = hydra_python.resolve_output_path(output_path / experiment_id)
        habitat_data = None
        pipeline = None
        try:
            click.secho(f"Executing task: {experiment_id}", fg="green")
            click.secho(f"Question: {question}", fg="cyan")

            # --- Habitat + pipeline ---
            habitat_data = HabitatInterface(scene_glb_path, cfg=cfg.habitat, device=device)
            pipeline = initialize_hydra_pipeline(cfg.hydra, habitat_data, task_output_path)
            rr_logger = RRLogger(task_output_path)

            # ---------- Phase 0: dataset spawn (always) ----------
            init_pos = np.array(task_data["initial_pose"]["position"], dtype=np.float32)
            init_rot = np.array(task_data["initial_pose"]["rotation"], dtype=np.float32)
            state = habitat_sim.AgentState()
            state.position = init_pos
            state.rotation = quat_from_coeffs(init_rot)
            habitat_data._sim.get_agent(0).set_state(state)
            click.secho("[spawn] dataset initial_pose (for SG boot)", fg="cyan")

            # ---------- TSDF & SG boot ----------
            floor_height = float(init_pos[1])
            tsdf_bnds, _ = get_scene_bnds(habitat_data.pathfinder, floor_height)
            cam_intr = get_cam_intr(cfg.habitat.hfov, cfg.habitat.img_height, cfg.habitat.img_width)
            tsdf_planner = TSDFPlanner(
                cfg=cfg.frontier_mapping,
                vol_bnds=tsdf_bnds,
                cam_intr=cam_intr,
                floor_height_offset=0,
                pts_init=init_pos,
                rr_logger=rr_logger,
            )

            sg_sim = SceneGraphSim(
                cfg,
                task_output_path,
                pipeline,
                rr_logger,
                device=device,
                clean_ques_ans=question,
                enrich_object_labels=" ",
            )

            # First warm-up so the SG has rooms/objects
            _warmup_like_eqa(habitat_data, pipeline, task_output_path, rr_logger, tsdf_planner, sg_sim)
            _ensure_scenegraph_initialized(habitat_data, pipeline, task_output_path, rr_logger, tsdf_planner, sg_sim)

            # ---------- Phase 1: teleport to the reference object's ROOM ----------
            ref_name = (task_data.get("reference_object") or {}).get("name", "")
            ref_label_base = _base_label(ref_name) if ref_name else ""
            moved_to_room = False
            if ref_label_base:
                try:
                    room_id = _find_room_id_for_object(sg_sim, ref_label_base)
                    if room_id:
                        new_pos = _teleport_to_room_center(habitat_data, sg_sim, room_id)
                        if new_pos is not None:
                            moved_to_room = True
                            init_pos = np.array(new_pos, dtype=np.float32)  # reset SPL baseline to room center
                            click.secho(
                                f"[spawn-room] teleported to {room_id} @ {np.array2string(init_pos, precision=3)}",
                                fg="cyan"
                            )
                except Exception as e:
                    click.secho(f"[spawn-room] failed to find/teleport to reference room: {e}", fg="yellow")

            # Small second warm-up from room center to get a clean local map & image
            if moved_to_room:
                _warmup_like_eqa(habitat_data, pipeline, task_output_path, rr_logger, tsdf_planner, sg_sim)
                _ensure_scenegraph_initialized(habitat_data, pipeline, task_output_path, rr_logger, tsdf_planner, sg_sim)

            # If still not ready, skip this task gracefully to avoid crashes
            if (
                not hasattr(sg_sim, "filtered_netx_graph") or sg_sim.filtered_netx_graph is None
                or not hasattr(sg_sim, "curr_agent_id")
            ):
                click.secho("[skip] SceneGraph not initialized; skipping task.", fg="yellow")
                log_experiment_status(
                    experiment_id, False,
                    metrics={"error": "scene_graph_not_initialized"},
                    filename=results_filename
                )
                continue

            # ---------- Planner ----------
            if "gemini" in cfg.vlm.name.lower():
                vlm_planner = VLMPlannerEQAGeminiSpatial(
                    cfg.vlm,
                    sg_sim,
                    question,
                    ground_truth_target,
                    task_output_path,
                    reference_object=task_data.get("reference_object"),
                    candidate_targets=task_data.get("candidate_targets"),
                )
            else:
                raise NotImplementedError(f"Planner {cfg.vlm.name} not implemented for spatial task.")

            # ---------- Navigation loop ----------
            num_steps = 20
            traj_length = 0.0
            target_declaration = {}
            succ = False

            for step_count in range(num_steps):
                target_pose, target_id, is_confident, confidence_level, target_declaration = (
                    vlm_planner.get_next_action()
                )

                if is_confident or (step_count == num_steps - 1):
                    click.secho(f"Termination condition met. Confident: {is_confident}", fg="blue")
                    break

                if target_pose is None:
                    click.secho("VLM did not provide a valid navigation target.", fg="red")
                    continue

                current_heading = habitat_data.get_heading_angle()
                agent = habitat_data._sim.get_agent(0)
                current_pos_hab = np.array(agent.get_state().position, dtype=np.float32)

                # path in habitat frame
                target_hab = pos_normal_to_habitat(np.array(target_pose, dtype=np.float32))
                target_hab[1] = current_pos_hab[1]

                spath = habitat_sim.nav.ShortestPath()
                spath.requested_start = current_pos_hab
                spath.requested_end = target_hab
                found = habitat_data.pathfinder.find_path(spath)
                if not found or not spath.points:
                    click.secho("Cannot find navigable path. Continuing..", fg="red")
                    continue

                desired_path_norm = pos_habitat_to_normal(np.array(spath.points)[:-1])
                rr_logger.log_traj_data(desired_path_norm)
                rr_logger.log_target_poses(target_pose)

                poses = habitat_data.get_trajectory_from_path_habitat_frame(
                    target_pose, desired_path_norm, current_heading, cfg.habitat.camera_tilt_deg
                )
                if poses is None:
                    click.secho("Cannot build trajectory from path. Continuing..", fg="red")
                    continue

                click.secho(f"Executing trajectory at step {step_count}", fg="yellow")
                run(
                    pipeline, habitat_data, poses, output_path=task_output_path,
                    rr_logger=rr_logger, tsdf_planner=tsdf_planner, sg_sim=sg_sim,
                    save_image=cfg.vlm.use_image, segmenter=None,
                )
                traj_length += get_traj_len_from_poses(poses)

            # ----- Metrics & measured SG distance to the reference -----
            declared_id = target_declaration.get("declared_target_object_id", "N/A")
            ground_truth_id = ground_truth_target["id"]
            succ = (str(declared_id) == str(ground_truth_id))

            if succ:
                click.secho(f"SUCCESS! VLM correctly identified {ground_truth_target['name']}", fg="green")
            else:
                click.secho(f"FAILURE. VLM chose {declared_id}, expected {ground_truth_id}", fg="red")

            agent_final_pos = habitat_data._sim.get_agent(0).get_state().position
            gt_pos_hab = np.array(ground_truth_target["position"], dtype=np.float32)
            loc_err = float(np.linalg.norm(agent_final_pos - gt_pos_hab))

            shortest_path_len = _shortest_path_len_pf(habitat_data, init_pos, gt_pos_hab)
            spl = (shortest_path_len / max(traj_length, shortest_path_len)) if shortest_path_len > 0 else 0.0
            if not succ:
                spl = 0.0

            # NEW: measured distance between declared label and the reference object's label
            ref_label = (task_data.get("reference_object") or {}).get("name", "reference")
            declared_to_ref_dist, declared_obj_internal, ref_obj_internal = (None, None, None)
            if declared_id not in (None, "N/A"):
                try:
                    declared_to_ref_dist, declared_obj_internal, ref_obj_internal = _measure_declared_to_ref_distance(
                        sg_sim, declared_id, ref_label
                    )
                    if declared_to_ref_dist is not None:
                        click.secho(
                            f"[metric] measured distance {_base_label(declared_id)} ↔ {_base_label(ref_label)} = "
                            f"{declared_to_ref_dist:.2f} m  (nodes: {declared_obj_internal} ↔ {ref_obj_internal})",
                            fg="magenta",
                        )
                    else:
                        click.secho("[metric] could not measure declared↔reference distance from SG.", fg="magenta")
                except Exception as e:
                    click.secho(f"[metric] distance calc error: {e}", fg="magenta")

            metrics = {
                "success": succ,
                "localization_error": loc_err,
                "spl": float(spl),
                "vlm_steps": vlm_planner.t,
                "total_steps": step_count + 1,
                "confidence_level": float(target_declaration.get("confidence_level", 0.0)),
                "traj_length": float(traj_length),
                "shortest_path_len": float(shortest_path_len),
                "declared_id": declared_id,
                "ground_truth_id": ground_truth_id,
                # NEW metrics:
                "declared_to_reference_distance": float(declared_to_ref_dist) if declared_to_ref_dist is not None else None,
                "declared_internal_node": declared_obj_internal,
                "reference_internal_node": ref_obj_internal,
            }
            log_experiment_status(experiment_id, succ, metrics=metrics, filename=results_filename)

        finally:
            # Robust teardown (prevents AttributeError + segfault)
            try:
                if habitat_data is not None and hasattr(habitat_data, "_sim"):
                    habitat_data._sim.close(destroy=True)
            except Exception:
                pass
            try:
                if pipeline is not None:
                    pipeline.save()
            except Exception:
                pass


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-cf", "--cfg_file", help="cfg file name (without .yaml)",
        default="spatial_vqa", type=str, required=False,
    )
    args = parser.parse_args()
    config_path = Path(__file__).resolve().parent.parent / "cfg" / f"{args.cfg_file}.yaml"
    cfg = OmegaConf.load(config_path)
    OmegaConf.resolve(cfg)
    main(cfg)
