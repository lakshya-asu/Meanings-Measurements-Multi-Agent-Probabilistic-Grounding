#!/usr/bin/env python3
# /home/artemis/project/graph_eqa_swagat/src/scripts/run_multi_agent_benchmark.py
# Multi-Agent MSP runner + Weights & Biases experiment tracking

from tqdm import tqdm
from omegaconf import OmegaConf
import click
import os
from pathlib import Path
import numpy as np
import torch
import csv
import json
import time

from src.logging.utils import should_skip_experiment, log_experiment_status
from src.envs.utils import pos_habitat_to_normal, pos_normal_to_habitat
from src.occupancy_mapping.geom import get_scene_bnds, get_cam_intr
from src.envs.habitat import run
from src.logging.rr_logger import RRLogger
from src.occupancy_mapping.tsdf import TSDFPlanner
from src.utils.data_utils import (
    get_traj_len_from_poses,
    get_latest_image,          
)
from src.utils.hydra_utils import initialize_hydra_pipeline
from src.scene_graph.scene_graph_sim import SceneGraphSim
from src.envs.habitat_interface import HabitatInterface

# W&B helpers
from src.utils.wandb_utils import init_wandb_run, log_metrics, log_table
import wandb
import habitat_sim
import hydra_python

# --- NEW: Multi-Agent Fat Planner ---
from src.planners.multi_agent_fat_planner import MultiAgentFatPlanner

SEM_LIST = "/datasets/hm3d/train/train-semantic-annots-files.json"
with open(SEM_LIST) as f:
    _semantic_ok = set()
    for p in json.load(f):
        base = os.path.basename(p).split(".")[0]
        _semantic_ok.add(base)

def scene_has_semantics(scene_id: str) -> bool:
    return scene_id in _semantic_ok

def load_init_poses_csv(init_pose_path: str):
    out = {}
    with open(init_pose_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            scene_floor = row["scene_floor"]
            out[scene_floor] = {
                "init_pts": np.array([float(row["init_x"]), float(row["init_y"]), float(row["init_z"])], dtype=np.float32),
                "init_angle": float(row["init_angle"]),
            }
    return out

def load_questions_msp_csv(qpath: str):
    data = []
    with open(qpath, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append(row)
    return data

def _warmup_like_eqa(pipeline, habitat_data, rr_logger, tsdf_planner, sg_sim, output_path, segmenter=None, save_image=True, num_yaws=8):
    agent = habitat_data._sim.get_agent(0)
    pos = agent.get_state().position
    yaw0_deg = np.degrees(float(habitat_data.get_heading_angle()))
    for k in range(num_yaws):
        yaw_deg = yaw0_deg + (360.0 * k / num_yaws)
        poses = habitat_data.get_init_poses_eqa(np.array(pos), float(yaw_deg), 0.0)
        try:
            run(pipeline, habitat_data, poses, output_path, rr_logger, tsdf_planner, sg_sim, save_image, segmenter)
        except Exception as e:
            click.secho(f"[warmup] non-fatal render error: {e}", fg="yellow")

def _ensure_scenegraph_initialized(pipeline, habitat_data, rr_logger, tsdf_planner, sg_sim, output_path, segmenter=None):
    need = (not hasattr(sg_sim, "filtered_netx_graph") or sg_sim.filtered_netx_graph is None or not hasattr(sg_sim, "curr_agent_id"))
    if not need:
        return
    click.secho("[init] SG not ready after warm-up; forcing one more tick…", fg="yellow")
    agent = habitat_data._sim.get_agent(0)
    pos = np.array(agent.get_state().position, dtype=np.float32)
    yaw_deg = np.degrees(float(habitat_data.get_heading_angle()))
    poses = habitat_data.get_init_poses_eqa(pos, float(yaw_deg), 0.0)
    try:
        run(pipeline, habitat_data, poses[:2] if len(poses) > 2 else poses, output_path=output_path, rr_logger=rr_logger, tsdf_planner=tsdf_planner, sg_sim=sg_sim, save_image=True, segmenter=segmenter)
    except Exception as e:
        click.secho(f"[init] non-fatal render error: {e}", fg="yellow")

def _nav_and_run_to_hab_target(pipeline, habitat_data, rr_logger, tsdf_planner, sg_sim, out_dir: Path, target_hab: np.ndarray, segmenter=None, save_image=True):
    agent_st = habitat_data._sim.get_agent(0).get_state()
    start = np.array(agent_st.position, dtype=np.float32)
    end = np.asarray(target_hab, dtype=np.float32).copy()
    end[1] = start[1]  
    spath = habitat_sim.nav.ShortestPath()
    spath.requested_start = start
    spath.requested_end = end
    found = habitat_data.pathfinder.find_path(spath)
    if not found or not spath.points:
        return False, 0.0
    desired_path_norm = pos_habitat_to_normal(np.array(spath.points)[:-1])
    try:
        rr_logger.log_traj_data(desired_path_norm)
    except Exception:
        pass
    poses = habitat_data.get_trajectory_from_path_habitat_frame(pos_habitat_to_normal([spath.requested_end])[0], desired_path_norm, habitat_data.get_heading_angle(), 0.0)
    if poses is None:
        return False, 0.0
    run(pipeline, habitat_data, poses, output_path=out_dir, rr_logger=rr_logger, tsdf_planner=tsdf_planner, sg_sim=sg_sim, save_image=save_image, segmenter=segmenter)
    return True, float(get_traj_len_from_poses(poses))

def _get_num_steps(cfg) -> int:
    for root in ["planner", "vlm"]:
        node = getattr(cfg, root, None)
        if node is None:
            continue
        for k in ["num_steps", "max_steps", "steps"]:
            try:
                v = node.get(k) if hasattr(node, "get") else getattr(node, k, None)
                if v is not None:
                    return int(v)
            except Exception:
                pass
    return 30

def _cfg_get(node, key: str, default):
    try:
        if node is None: return default
        if hasattr(node, "get"): return node.get(key, default)
        return getattr(node, key, default)
    except Exception:
        return default

def main(cfg, dataset_type: str = "spatial", skip: int = 0, max_steps: int = 25):
    click.secho(f"[mode] MULTI-AGENT VLM runner | Dataset: {dataset_type}", fg="cyan")

    import os
    from pathlib import Path
    # Dynamically find the repo root (3 levels up from this script: scripts -> spatial_experiment -> root)
    workspace_root = str(Path(__file__).resolve().parent.parent.parent)
    
    if dataset_type == "grapheqa":
        # Override paths explicitly using the config choice fields or fixed explore-eqa paths
        rel_qpath = getattr(cfg.data, "question_data_path_choice", "/datasets/explore-eqa/questions.csv").lstrip('/')
        rel_initpath = getattr(cfg.data, "init_pose_data_path_choice", "/datasets/explore-eqa/scene_init_poses.csv").lstrip('/')
        
        cfg.data.question_data_path = os.path.join(workspace_root, rel_qpath)
        cfg.data.init_pose_data_path = os.path.join(workspace_root, rel_initpath)
        
        # Also fix semantic_annot_data_path root for filtering
        rel_sempath = getattr(cfg.data, "semantic_annot_data_path", "/datasets/hm3d/train").lstrip('/')
        cfg.data.semantic_annot_data_path = os.path.join(workspace_root, rel_sempath)
        
        from src.utils.data_utils import load_eqa_data
        print(f"[debug] calling load_eqa_data with {cfg.data.question_data_path} ...")
        questions_data_raw, init_pose_data = load_eqa_data(cfg.data)
        print(f"[debug] loaded {len(questions_data_raw)} raw questions")
        
        # Format the loaded data into the exact schema expected below
        questions_data = []
        for q in questions_data_raw:
            # Map 'question_formatted' -> 'msp_question' and extract the raw 'answer'
            q_mapped = dict(q)
            q_mapped["msp_question"] = q.get("question_formatted", q.get("question", ""))
            q_mapped["answer"] = q.get("answer", "")
            questions_data.append(q_mapped)
        print(f"[debug] mapped {len(questions_data)} grapheqa questions")
    else:
        qpath = cfg.data.question_data_path
        questions_data = load_questions_msp_csv(qpath)
        init_pose_data = load_init_poses_csv(cfg.data.init_pose_data_path)

    output_path = Path(__file__).resolve().parent.parent / cfg.output_path
    os.makedirs(str(output_path), exist_ok=True)
    results_filename = output_path / f"{cfg.results_filename}_multi_agent.json"
    device = f"cuda:{cfg.gpu}" if torch.cuda.is_available() else "cpu"

    segmenter = None if cfg.data.use_semantic_data else hydra_python.detection.detic_segmenter.DeticSegmenter(cfg)

    num_steps = min(max_steps, _get_num_steps(cfg))
    anchor_warmup_steps = int(_cfg_get(cfg.vlm, "anchor_warmup_steps", 10))
    anchor_warmup_lookaround_every = int(_cfg_get(cfg.vlm, "anchor_warmup_lookaround_every", 2))
    room_warmup_lookaround = bool(_cfg_get(cfg.vlm, "room_warmup_lookaround", True))
    pre_answer_approach = bool(_cfg_get(cfg.vlm, "pre_answer_approach", True))
    pre_answer_conf_thresh = float(_cfg_get(cfg.vlm, "pre_answer_conf_thresh", 0.85))
    pre_answer_max_extra_steps = int(_cfg_get(cfg.vlm, "pre_answer_max_extra_steps", 1))
    pre_answer_lookaround_yaws = int(_cfg_get(cfg.vlm, "pre_answer_lookaround_yaws", 8))

    run_name = getattr(cfg, "exp_name", "multi_agent_run")
    wandb_run = init_wandb_run(cfg, run_name=run_name)

    per_episode_metrics = []
    total_success = 0
    total_traj_length = 0.0
    episode_traces_rows = []

    try:
        for question_ind in tqdm(range(len(questions_data))):
            if question_ind < skip:
                continue
            q = questions_data[question_ind]
            scene, floor = q["scene"], str(q["floor"])
            scene_id = scene[6:]
            experiment_id = f"{question_ind}_{scene}_{floor}"

            if should_skip_experiment(experiment_id, filename=results_filename):
                continue

            question_path = hydra_python.resolve_output_path(output_path / experiment_id)
            if cfg.data.use_semantic_data and not scene_has_semantics(scene_id):
                continue

            habitat_data = HabitatInterface(f"{cfg.data.scene_data_path}/{scene}/{scene_id}.basis.glb", cfg=cfg.habitat, device=device)
            pipeline = initialize_hydra_pipeline(cfg.hydra, habitat_data, question_path)
            rr_logger = RRLogger(question_path)

            init_pts = init_pose_data[f"{scene}_{floor}"]["init_pts"]
            tsdf_bnds, _ = get_scene_bnds(habitat_data.pathfinder, pos_habitat_to_normal(init_pts)[-1])
            cam_intr = get_cam_intr(cfg.habitat.hfov, cfg.habitat.img_height, cfg.habitat.img_width)

            tsdf_planner = TSDFPlanner(cfg.frontier_mapping, tsdf_bnds, cam_intr, 0, pos_habitat_to_normal(init_pts), rr_logger)

            sg_sim = SceneGraphSim(cfg, question_path, pipeline, rr_logger, device=device, clean_ques_ans=" ", enrich_object_labels=" ")
            sg_sim.enrich_rooms = False  # Disable room enrichment for fat planner ablation

            poses = habitat_data.get_init_poses_eqa(init_pts, init_pose_data[f"{scene}_{floor}"]["init_angle"], 0.0)
            run(pipeline, habitat_data, poses, output_path=question_path, rr_logger=rr_logger, tsdf_planner=tsdf_planner, sg_sim=sg_sim, save_image=cfg.vlm.use_image, segmenter=segmenter)

            _warmup_like_eqa(pipeline=pipeline, habitat_data=habitat_data, rr_logger=rr_logger, tsdf_planner=tsdf_planner, sg_sim=sg_sim, output_path=question_path, segmenter=segmenter, save_image=cfg.vlm.use_image, num_yaws=pre_answer_lookaround_yaws)
            _ensure_scenegraph_initialized(pipeline=pipeline, habitat_data=habitat_data, rr_logger=rr_logger, tsdf_planner=tsdf_planner, sg_sim=sg_sim, output_path=question_path, segmenter=segmenter)

            anchor_label = q.get("anchor_label", None) or q.get("primary_object", None)
            anchor_center_hab = None
            try:
                if q.get("anchor_center_x", None) is not None:
                    anchor_center_hab = np.array([float(q["anchor_center_x"]), float(q["anchor_center_y"]), float(q["anchor_center_z"])], dtype=np.float32)
            except Exception:
                pass

            # Initialize Fat Planner
            vlm_planner = MultiAgentFatPlanner(
                cfg.vlm,
                sg_sim,
                q["msp_question"],
                out_path=str(question_path),
                anchor_label=anchor_label,
                anchor_center_hab=anchor_center_hab,
                anchor_front_yaw_world=float(q["ann_yaw_rad"]) if q.get("ann_yaw_rad", None) not in [None, ""] else None,
                choices=q.get("choices", []),
                agent_providers={"qa": "claude"}
            )

            succ = False
            traj_length = 0.0
            final_pred = None
            ep_vlm_calls = 0
            ep_steps = 0
            ep_t0 = time.time()
            anchor_found = False

            # Warmup loop: keep searching until Grounding Agent confirms anchors
            for w in range(anchor_warmup_steps):
                agent_st = habitat_data._sim.get_agent(0).get_state()
                target_pose, target_id, is_conf, conf, extra = vlm_planner.get_next_action(
                    agent_yaw_rad=float(habitat_data.get_heading_angle()),
                    agent_pos_hab=np.array(agent_st.position),
                )
                ep_steps += 1
                ep_vlm_calls += 1 # Only 1 Fat Agent runs per step

                action_type = extra.get("action_type", "goto_frontier")
                
                # Check if grounding was successful (not returning goto_frontier/lookaround fallback)
                if action_type not in ["goto_frontier", "lookaround"]:
                    anchor_found = True
                    click.secho(f"[warmup] Multi-Agent confirmed anchor at warmup step {w}", fg="green")
                    break

                if (w % max(1, anchor_warmup_lookaround_every)) == 0:
                    _warmup_like_eqa(pipeline=pipeline, habitat_data=habitat_data, rr_logger=rr_logger, tsdf_planner=tsdf_planner, sg_sim=sg_sim, output_path=question_path, segmenter=segmenter, save_image=cfg.vlm.use_image, num_yaws=pre_answer_lookaround_yaws)

                if action_type == "lookaround" or target_pose is None:
                    continue

                agent_st2 = habitat_data._sim.get_agent(0).get_state()
                end_hab = pos_normal_to_habitat(target_pose) if len(target_pose) == 3 else target_pose
                end_hab = np.asarray(end_hab, dtype=np.float32)
                end_hab[1] = agent_st2.position[1]

                ok, dlen = _nav_and_run_to_hab_target(pipeline, habitat_data, rr_logger, tsdf_planner, sg_sim, question_path, end_hab, segmenter, cfg.vlm.use_image)
                if ok: traj_length += dlen

            if anchor_found and room_warmup_lookaround:
                click.secho("[warmup] anchor found -> performing close-range lookaround to enrich objects", fg="cyan")
                _warmup_like_eqa(pipeline=pipeline, habitat_data=habitat_data, rr_logger=rr_logger, tsdf_planner=tsdf_planner, sg_sim=sg_sim, output_path=question_path, segmenter=segmenter, save_image=cfg.vlm.use_image, num_yaws=pre_answer_lookaround_yaws)

            # MAIN EPISODE LOOP
            for step in range(num_steps):
                agent_st = habitat_data._sim.get_agent(0).get_state()
                target_pose, target_id, is_conf, conf, extra = vlm_planner.get_next_action(
                    agent_yaw_rad=float(habitat_data.get_heading_angle()),
                    agent_pos_hab=np.array(agent_st.position),
                )
                ep_steps += 1
                ep_vlm_calls += 4

                action_type = extra.get("action_type", "goto_frontier")
                chosen_id = str(extra.get("chosen_id", "")).strip()

                if pre_answer_approach and action_type == "answer" and float(conf) >= pre_answer_conf_thresh:
                    did_confirm = False
                    for _k in range(pre_answer_max_extra_steps):
                        target_hab = None
                        if chosen_id == "POINT_GUESS":
                            pg = extra.get("target_xyz_hab", None)
                            if isinstance(pg, list) and len(pg) == 3: target_hab = np.asarray(pg, dtype=np.float32)
                        elif chosen_id:
                            try:
                                tgt_norm = sg_sim.get_position_from_id(chosen_id)
                                if tgt_norm is not None and len(tgt_norm) == 3:
                                    target_hab = np.asarray(pos_normal_to_habitat(np.asarray(tgt_norm, dtype=np.float32)), dtype=np.float32)
                            except Exception:
                                target_hab = None

                        if target_hab is not None:
                            ok, dlen = _nav_and_run_to_hab_target(pipeline, habitat_data, rr_logger, tsdf_planner, sg_sim, question_path, target_hab, segmenter, cfg.vlm.use_image)
                            if ok: traj_length += dlen

                            _warmup_like_eqa(pipeline=pipeline, habitat_data=habitat_data, rr_logger=rr_logger, tsdf_planner=tsdf_planner, sg_sim=sg_sim, output_path=question_path, segmenter=segmenter, save_image=cfg.vlm.use_image, num_yaws=pre_answer_lookaround_yaws)

                            agent_st3 = habitat_data._sim.get_agent(0).get_state()
                            target_pose2, target_id2, is_conf2, conf2, extra2 = vlm_planner.get_next_action(
                                agent_yaw_rad=float(habitat_data.get_heading_angle()),
                                agent_pos_hab=np.array(agent_st3.position),
                            )
                            ep_steps += 1
                            ep_vlm_calls += 4

                            if extra2.get("action_type") == "answer":
                                extra = extra2
                                conf = float(extra2.get("confidence", conf2))
                                is_conf = bool(is_conf2) or (conf >= 0.90)
                                action_type = "answer"
                                did_confirm = True
                                break

                    if did_confirm:
                        final_pred = extra
                        succ = True
                        rr_logger.log_text_data(f"FINAL ANSWER (confirmed): {final_pred}")
                        break

                if is_conf or (conf > 0.9 and action_type == "answer"):
                    final_pred = extra
                    succ = True
                    rr_logger.log_text_data(f"FINAL ANSWER: {final_pred}")
                    break

                if action_type == "lookaround":
                    _warmup_like_eqa(pipeline=pipeline, habitat_data=habitat_data, rr_logger=rr_logger, tsdf_planner=tsdf_planner, sg_sim=sg_sim, output_path=question_path, segmenter=segmenter, save_image=cfg.vlm.use_image, num_yaws=pre_answer_lookaround_yaws)
                    continue

                if target_pose is not None:
                    path = habitat_sim.nav.ShortestPath()
                    path.requested_start = agent_st.position
                    end_hab = pos_normal_to_habitat(target_pose) if len(target_pose) == 3 else target_pose
                    end_hab = np.asarray(end_hab, dtype=np.float32)
                    end_hab[1] = agent_st.position[1]
                    path.requested_end = end_hab

                    if habitat_data.pathfinder.find_path(path):
                        desired_path = pos_habitat_to_normal(np.array(path.points)[:-1])
                        rr_logger.log_traj_data(desired_path)
                        poses = habitat_data.get_trajectory_from_path_habitat_frame(pos_habitat_to_normal([path.requested_end])[0], desired_path, habitat_data.get_heading_angle(), 0.0)
                        run(pipeline, habitat_data, poses, output_path=question_path, rr_logger=rr_logger, tsdf_planner=tsdf_planner, sg_sim=sg_sim, save_image=cfg.vlm.use_image, segmenter=segmenter)
                        traj_length += get_traj_len_from_poses(poses)
                    else:
                        click.secho(f"Pathfinding failed to {action_type} {target_id}", fg="red")

            ep_time = time.time() - ep_t0
            final_conf = float(final_pred.get("confidence", 0.0)) if isinstance(final_pred, dict) else 0.0

            per_episode_metrics.append({
                "question_index": question_ind,
                "experiment_id": experiment_id,
                "scene": scene,
                "floor": floor,
                "scene_floor": f"{scene}_{floor}",
                "msp_question": q.get("msp_question", ""),
                "mode": "multi_agent",
                "success": bool(succ),
                "num_steps": int(ep_steps),
                "vlm_calls": int(ep_vlm_calls),
                "traj_length": float(traj_length),
                "episode_time_sec": float(ep_time),
                "final_confidence": float(final_conf),
                "anchor_found": bool(anchor_found),
            })

            total_traj_length += traj_length
            if succ: total_success += 1

            log_metrics({"episode_success": 1.0 if succ else 0.0, "episode_num_steps": ep_steps, "episode_vlm_calls": ep_vlm_calls, "episode_traj_length": float(traj_length), "episode_time_sec": float(ep_time), "episode_final_confidence": float(final_conf), "episode_anchor_found": 1.0 if anchor_found else 0.0}, step=question_ind)
            log_experiment_status(experiment_id, succ, metrics={"traj_length": float(traj_length), "final_pred": final_pred, "mode": "multi_agent", "anchor_found": bool(anchor_found)}, filename=results_filename)

            # Retrieve Multi-Agent Blackboard History
            blackboard_history = vlm_planner.blackboard.global_history if hasattr(vlm_planner, "blackboard") else ""
            
            wandb_image = None
            try:
                final_img_path = get_latest_image(question_path)
                if final_img_path is not None:
                    wandb_image = wandb.Image(str(final_img_path), caption=f"{experiment_id}")
            except Exception:
                pass

            episode_traces_rows.append([
                question_ind, experiment_id, scene, floor, q.get("msp_question", ""), q.get("answer", ""), bool(succ), float(final_conf), blackboard_history, wandb_image,
            ])

            wandb.log({"episode_trace/question_index": question_ind, "episode_trace/experiment_id": experiment_id, "episode_trace/success": 1.0 if succ else 0.0, "episode_trace/final_confidence": float(final_conf), "episode_trace/blackboard_log": blackboard_history, "episode_trace/image": wandb_image}, step=question_ind)

            habitat_data._sim.close(destroy=True)
            pipeline.save()

        if per_episode_metrics:
            import numpy as _np
            num_episodes = len(per_episode_metrics)
            success_rate = total_success / float(num_episodes)
            per_episode_path = output_path / "per_episode_results.json"
            with open(per_episode_path, "w") as f: json.dump(per_episode_metrics, f, indent=2)

            log_metrics({"success_rate": success_rate, "num_episodes": num_episodes, "avg_turns": float(_np.mean([e["num_steps"] for e in per_episode_metrics])), "avg_vlm_calls": float(_np.mean([e["vlm_calls"] for e in per_episode_metrics])), "avg_traj_length": float(_np.mean([e["traj_length"] for e in per_episode_metrics])), "avg_episode_time_sec": float(_np.mean([e["episode_time_sec"] for e in per_episode_metrics])), "avg_final_confidence": float(_np.mean([e["final_confidence"] for e in per_episode_metrics])), "avg_anchor_found": float(_np.mean([1.0 if e.get("anchor_found", False) else 0.0 for e in per_episode_metrics])), "total_traj_length": float(total_traj_length)})
            log_table("per_episode", per_episode_metrics)

            if episode_traces_rows:
                traces_table = wandb.Table(columns=["question_index", "experiment_id", "scene", "floor", "msp_question", "ground_truth", "success", "final_confidence", "blackboard_log", "image"], data=episode_traces_rows)
                wandb.log({"episode_traces": traces_table})

    finally:
        try: wandb.finish()
        except Exception: pass

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-cf", "--cfg_file", required=True)
    parser.add_argument("--dataset", type=str, choices=["spatial", "grapheqa"], default=None, 
                        help="Which dataset format to load. Prompted interactively if left blank.")
    parser.add_argument("--skip", type=int, default=0, help="Number of queries to skip before starting.")
    parser.add_argument("--max_steps", type=int, default=10, help="Limit number of steps per agent episode.")
    args = parser.parse_args()
    
    dataset = args.dataset
    if dataset is None:
        dataset = click.prompt("Which dataset format to run?", type=click.Choice(["spatial", "grapheqa"]), default="spatial")

    cfg = OmegaConf.load(Path(__file__).resolve().parent.parent / "cfg" / f"{args.cfg_file}.yaml")
    OmegaConf.resolve(cfg)
    main(cfg, dataset_type=dataset, skip=args.skip, max_steps=args.max_steps)