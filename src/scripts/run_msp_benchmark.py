# MSP SMART VLM runner + Weights & Biases experiment tracking

from tqdm import tqdm
from omegaconf import OmegaConf
import click
import os
from pathlib import Path
import numpy as np
import torch
import csv
import json
import time  # for episode timing

from src.logging.utils import should_skip_experiment, log_experiment_status
from src.envs.utils import pos_habitat_to_normal, pos_normal_to_habitat
from src.occupancy_mapping.geom import get_scene_bnds, get_cam_intr
from src.envs.habitat import run
from src.logging.rr_logger import RRLogger
from src.occupancy_mapping.tsdf import TSDFPlanner
from src.utils.data_utils import (
    get_traj_len_from_poses,
    get_latest_image,          # NEW: for final VLM image logging
)
from src.utils.hydra_utils import initialize_hydra_pipeline
from src.scene_graph.scene_graph_sim import SceneGraphSim
from src.envs.habitat_interface import HabitatInterface

# W&B helpers
from src.utils.wandb_utils import init_wandb_run, log_metrics, log_table
import wandb

import habitat_sim
import hydra_python

# --- Smart MSP Planner ---
from src.planners.vlm_planner_msp import VLMPlannerMSP_Smart


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
                "init_pts": np.array(
                    [float(row["init_x"]), float(row["init_y"]), float(row["init_z"])],
                    dtype=np.float32,
                ),
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


def run_lookaround(
    pipeline,
    habitat_data,
    rr_logger,
    tsdf_planner,
    sg_sim,
    output_path,
    segmenter=None,
    save_image=True,
    num_yaws=8,
):
    agent = habitat_data._sim.get_agent(0)
    pos = agent.get_state().position
    yaw0_deg = np.degrees(float(habitat_data.get_heading_angle()))
    for k in range(num_yaws):
        yaw_deg = yaw0_deg + (360.0 * k / num_yaws)
        poses = habitat_data.get_init_poses_eqa(np.array(pos), float(yaw_deg), 0.0)
        run(
            pipeline,
            habitat_data,
            poses,
            output_path,
            rr_logger,
            tsdf_planner,
            sg_sim,
            save_image,
            segmenter,
        )


def _get_num_steps(cfg) -> int:
    """
    Accept:
      - cfg.planner.num_steps / max_steps / steps
      - cfg.vlm.num_steps / max_steps / steps   (common in your setup)
    Default 30.
    """
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


def main(cfg):
    qpath = cfg.data.question_data_path
    click.secho("[mode] MSP SMART VLM runner", fg="cyan")

    questions_data = load_questions_msp_csv(qpath)
    init_pose_data = load_init_poses_csv(cfg.data.init_pose_data_path)

    output_path = Path(__file__).resolve().parent.parent / cfg.output_path
    os.makedirs(str(output_path), exist_ok=True)
    results_filename = output_path / f"{cfg.results_filename}.json"
    device = f"cuda:{cfg.gpu}" if torch.cuda.is_available() else "cpu"

    # Segmenter init (Detic or GT)
    segmenter = None if cfg.data.use_semantic_data else hydra_python.detection.detic_segmenter.DeticSegmenter(cfg)

    num_steps = _get_num_steps(cfg)
    click.secho(f"[runner] num_steps={num_steps}", fg="yellow")

    # ----------------------------------------------------------------------
    # Initialize W&B run once for the whole benchmark
    # ----------------------------------------------------------------------
    run_name = getattr(cfg, "exp_name", None)
    wandb_run = init_wandb_run(cfg, run_name=run_name)

    # Collect per-episode metrics so we can compute aggregates at the end
    per_episode_metrics = []
    total_success = 0
    total_traj_length = 0.0

    # NEW: table rows for reasoning + final image per episode
    episode_traces_rows = []

    try:
        for question_ind in tqdm(range(len(questions_data))):
            q = questions_data[question_ind]
            scene, floor = q["scene"], str(q["floor"])
            scene_id = scene[6:]
            experiment_id = f"{question_ind}_{scene}_{floor}"

            if should_skip_experiment(experiment_id, filename=results_filename):
                continue

            question_path = hydra_python.resolve_output_path(output_path / experiment_id)
            if cfg.data.use_semantic_data and not scene_has_semantics(scene_id):
                continue

            habitat_data = HabitatInterface(
                f"{cfg.data.scene_data_path}/{scene}/{scene_id}.basis.glb",
                cfg=cfg.habitat,
                device=device,
            )
            pipeline = initialize_hydra_pipeline(cfg.hydra, habitat_data, question_path)
            rr_logger = RRLogger(question_path)

            # Init TSDF
            init_pts = init_pose_data[f"{scene}_{floor}"]["init_pts"]
            tsdf_bnds, _ = get_scene_bnds(
                habitat_data.pathfinder,
                pos_habitat_to_normal(init_pts)[-1],
            )
            cam_intr = get_cam_intr(
                cfg.habitat.hfov,
                cfg.habitat.img_height,
                cfg.habitat.img_width,
            )

            tsdf_planner = TSDFPlanner(
                cfg.frontier_mapping,
                tsdf_bnds,
                cam_intr,
                0,
                pos_habitat_to_normal(init_pts),
                rr_logger,
            )

            sg_sim = SceneGraphSim(
                cfg,
                question_path,
                pipeline,
                rr_logger,
                device=device,
                clean_ques_ans=" ",
                enrich_object_labels=" ",
            )

            poses = habitat_data.get_init_poses_eqa(
                init_pts,
                init_pose_data[f"{scene}_{floor}"]["init_angle"],
                0.0,
            )
            run(
                pipeline,
                habitat_data,
                poses,
                output_path=question_path,
                rr_logger=rr_logger,
                tsdf_planner=tsdf_planner,
                sg_sim=sg_sim,
                save_image=cfg.vlm.use_image,
                segmenter=segmenter,
            )

            # --- Anchor disambiguation inputs from CSV ---
            # CSV fields: anchor_label, anchor_center_x/y/z, ann_yaw_rad
            anchor_label = q.get("anchor_label", None) or q.get("primary_object", None)
            anchor_center_hab = None
            try:
                if q.get("anchor_center_x", None) is not None:
                    anchor_center_hab = np.array(
                        [
                            float(q["anchor_center_x"]),
                            float(q["anchor_center_y"]),
                            float(q["anchor_center_z"]),
                        ],
                        dtype=np.float32,
                    )
            except Exception:
                anchor_center_hab = None

            vlm_planner = VLMPlannerMSP_Smart(
                cfg.vlm,
                sg_sim,
                q["msp_question"],
                out_path=str(question_path),
                anchor_label=anchor_label,
                anchor_center_hab=anchor_center_hab,
                anchor_front_yaw_world=float(q["ann_yaw_rad"])
                if q.get("ann_yaw_rad", None) not in [None, ""]
                else None,
            )

            # Episode metrics
            succ = False
            traj_length = 0.0
            final_pred = None
            ep_vlm_calls = 0
            ep_steps = 0
            ep_t0 = time.time()

            for step in range(num_steps):
                ep_steps += 1
                ep_vlm_calls += 1

                agent_st = habitat_data._sim.get_agent(0).get_state()

                target_pose, target_id, is_conf, conf, extra = vlm_planner.get_next_action(
                    agent_yaw_rad=float(habitat_data.get_heading_angle()),
                    agent_pos_hab=np.array(agent_st.position),
                )

                # "answer" is decided inside the planner; if it claims confident, stop.
                if is_conf or (conf > 0.9 and extra.get("action_type") == "answer"):
                    final_pred = extra
                    succ = True
                    rr_logger.log_text_data(f"FINAL ANSWER: {final_pred}")
                    break

                action_type = extra.get("action_type", "goto_frontier")

                if action_type == "lookaround":
                    run_lookaround(
                        pipeline,
                        habitat_data,
                        rr_logger,
                        tsdf_planner,
                        sg_sim,
                        question_path,
                        segmenter,
                    )
                    continue

                if target_pose is not None:
                    path = habitat_sim.nav.ShortestPath()
                    path.requested_start = agent_st.position

                    # sg_sim.get_position_from_id returns NORMAL frame
                    # convert to habitat frame for pathfinding:
                    end_hab = (
                        pos_normal_to_habitat(target_pose)
                        if len(target_pose) == 3
                        else target_pose
                    )
                    end_hab = np.asarray(end_hab, dtype=np.float32)
                    end_hab[1] = agent_st.position[1]  # keep height

                    path.requested_end = end_hab

                    if habitat_data.pathfinder.find_path(path):
                        desired_path = pos_habitat_to_normal(np.array(path.points)[:-1])
                        rr_logger.log_traj_data(desired_path)

                        poses = habitat_data.get_trajectory_from_path_habitat_frame(
                            pos_habitat_to_normal([path.requested_end])[0],
                            desired_path,
                            habitat_data.get_heading_angle(),
                            0.0,
                        )

                        run(
                            pipeline,
                            habitat_data,
                            poses,
                            output_path=question_path,
                            rr_logger=rr_logger,
                            tsdf_planner=tsdf_planner,
                            sg_sim=sg_sim,
                            save_image=cfg.vlm.use_image,
                            segmenter=segmenter,
                        )
                        traj_length += get_traj_len_from_poses(poses)
                    else:
                        click.secho(
                            f"Pathfinding failed to {action_type} {target_id}",
                            fg="red",
                        )

            ep_time = time.time() - ep_t0
            final_conf = 0.0
            if isinstance(final_pred, dict):
                try:
                    final_conf = float(final_pred.get("confidence", 0.0))
                except Exception:
                    final_conf = 0.0

            # ------------------------------------------------------------------
            # Per-episode metrics (for research-grade tracking)
            # ------------------------------------------------------------------
            ep_record = {
                "question_index": question_ind,
                "experiment_id": experiment_id,
                "scene": scene,
                "floor": floor,
                "scene_floor": f"{scene}_{floor}",
                "msp_question": q.get("msp_question", ""),
                "mode": "msp_smart",
                "success": bool(succ),
                "num_steps": int(ep_steps),
                "vlm_calls": int(ep_vlm_calls),
                "traj_length": float(traj_length),
                "episode_time_sec": float(ep_time),
                "final_confidence": float(final_conf),
            }
            per_episode_metrics.append(ep_record)

            total_traj_length += traj_length
            if succ:
                total_success += 1

            # Stream per-episode stats to W&B as the run progresses
            log_metrics(
                {
                    "episode_success": 1.0 if succ else 0.0,
                    "episode_num_steps": ep_steps,
                    "episode_vlm_calls": ep_vlm_calls,
                    "episode_traj_length": float(traj_length),
                    "episode_time_sec": float(ep_time),
                    "episode_final_confidence": float(final_conf),
                },
                step=question_ind,
            )

            # Existing JSON log (kept as-is)
            log_experiment_status(
                experiment_id,
                succ,
                metrics={
                    "traj_length": float(traj_length),
                    "final_pred": final_pred,
                    "mode": "msp_smart",
                },
                filename=results_filename,
            )

            # ----------------------------------------------------------
            # NEW: Grab last trace + final image for this episode
            # ----------------------------------------------------------
            kernel_reasoning = ""
            selector_thought = ""
            try:
                trace_files = sorted(question_path.glob("trace_step_*.json"))
                if trace_files:
                    with open(trace_files[-1], "r") as tf:
                        last_trace = json.load(tf)
                    kernel_reasoning = last_trace.get("kernel", {}).get("reasoning", "")
                    selector_thought = last_trace.get("guardrails", {}).get("thought", "")
            except Exception as e:
                kernel_reasoning = f"[trace_load_error] {e}"
                selector_thought = ""

            wandb_image = None
            try:
                final_img_path = get_latest_image(question_path)
                if final_img_path is not None:
                    wandb_image = wandb.Image(
                        str(final_img_path),
                        caption=f"{experiment_id}",
                    )
            except Exception:
                wandb_image = None

            # Collect row for episode_traces table
            episode_traces_rows.append([
                question_ind,
                experiment_id,
                scene,
                floor,
                q.get("msp_question", ""),
                bool(succ),
                float(final_conf),
                kernel_reasoning,
                selector_thought,
                wandb_image,
            ])

            # Also log a one-off summary for this episode (reasoning + image)
            wandb.log(
                {
                    "episode_trace/question_index": question_ind,
                    "episode_trace/experiment_id": experiment_id,
                    "episode_trace/success": 1.0 if succ else 0.0,
                    "episode_trace/final_confidence": float(final_conf),
                    "episode_trace/kernel_reasoning": kernel_reasoning,
                    "episode_trace/selector_thought": selector_thought,
                    "episode_trace/image": wandb_image,
                },
                step=question_ind,
            )

            habitat_data._sim.close(destroy=True)
            pipeline.save()

        # ----------------------------------------------------------------------
        # After all episodes: aggregate metrics + per-episode table + traces
        # ----------------------------------------------------------------------
        if per_episode_metrics:
            import numpy as _np

            num_episodes = len(per_episode_metrics)
            success_rate = total_success / float(num_episodes)

            avg_turns = float(
                _np.mean([e["num_steps"] for e in per_episode_metrics])
            )
            avg_vlm_calls = float(
                _np.mean([e["vlm_calls"] for e in per_episode_metrics])
            )
            avg_traj = float(
                _np.mean([e["traj_length"] for e in per_episode_metrics])
            )
            avg_ep_time = float(
                _np.mean([e["episode_time_sec"] for e in per_episode_metrics])
            )
            avg_conf = float(
                _np.mean([e["final_confidence"] for e in per_episode_metrics])
            )

            # Save to disk for offline analysis
            per_episode_path = output_path / "per_episode_results.json"
            with open(per_episode_path, "w") as f:
                json.dump(per_episode_metrics, f, indent=2)

            # Log aggregate metrics to W&B
            log_metrics(
                {
                    "success_rate": success_rate,
                    "num_episodes": num_episodes,
                    "avg_turns": avg_turns,
                    "avg_vlm_calls": avg_vlm_calls,
                    "avg_traj_length": avg_traj,
                    "avg_episode_time_sec": avg_ep_time,
                    "avg_final_confidence": avg_conf,
                    "total_traj_length": float(total_traj_length),
                }
            )

            # Log per-episode table to W&B (metrics only)
            log_table("per_episode", per_episode_metrics)

            # ------------------------------------------------------
            # NEW: W&B Table with reasoning + final image per episode
            # ------------------------------------------------------
            if episode_traces_rows:
                traces_table = wandb.Table(
                    columns=[
                        "question_index",
                        "experiment_id",
                        "scene",
                        "floor",
                        "msp_question",
                        "success",
                        "final_confidence",
                        "kernel_reasoning",
                        "selector_thought",
                        "image",
                    ],
                    data=episode_traces_rows,
                )
                wandb.log({"episode_traces": traces_table})

            # ------------------------------------------------------
            # NEW: Log raw logs/traces as a W&B artifact
            # ------------------------------------------------------
            try:
                artifact = wandb.Artifact(
                    name=f"msp_run_{getattr(cfg, 'exp_name', 'run')}",
                    type="msp-run-logs",
                )
                # Add core trace/log files
                for p in output_path.glob("**/vlm_calls.jsonl"):
                    artifact.add_file(str(p))
                for p in output_path.glob("**/trace_step_*.json"):
                    artifact.add_file(str(p))
                for p in output_path.glob("**/llm_outputs_smart*.txt"):
                    artifact.add_file(str(p))
                # Add per-episode summary JSON
                artifact.add_file(str(per_episode_path))
                wandb.log_artifact(artifact)
            except Exception as e:
                click.secho(f"[WARN] Failed to log artifact: {e}", fg="yellow")

    finally:
        # Ensure the W&B run closes even if something crashes
        try:
            wandb.finish()
        except Exception:
            pass


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-cf", "--cfg_file", required=True)
    args = parser.parse_args()
    cfg = OmegaConf.load(
        Path(__file__).resolve().parent.parent / "cfg" / f"{args.cfg_file}.yaml"
    )
    OmegaConf.resolve(cfg)
    main(cfg)