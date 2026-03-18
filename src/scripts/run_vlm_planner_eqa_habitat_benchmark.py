from tqdm import tqdm
from omegaconf import OmegaConf
import click
import os, time
from pathlib import Path
import numpy as np
import torch
import csv

from src.logging.utils import should_skip_experiment, log_experiment_status
from src.envs.utils import pos_habitat_to_normal, pos_normal_to_habitat
from src.occupancy_mapping.geom import get_scene_bnds, get_cam_intr
from src.envs.habitat import run
from src.logging.rr_logger import RRLogger
from src.occupancy_mapping.tsdf import TSDFPlanner
from src.utils.data_utils import load_eqa_data, get_instruction_from_eqa_data, get_traj_len_from_poses
from src.utils.hydra_utils import initialize_hydra_pipeline

from src.scene_graph.scene_graph_sim import SceneGraphSim
from src.planners.vlm_planner_benchmark_gemini import VLMPlannerEQAGemini #VLMPlannerEQAGPT
from src.envs.habitat_interface import HabitatInterface

import habitat_sim
import hydra_python
import json


# -----------------------------
# Semantics availability check
# -----------------------------
SEM_LIST = "/datasets/hm3d/train/train-semantic-annots-files.json"

with open(SEM_LIST) as f:
    _semantic_ok = set()
    for p in json.load(f):
        base = os.path.basename(p).split(".")[0]  # <SCENE> (e.g., HkseAnWCgqk)
        _semantic_ok.add(base)

def scene_has_semantics(scene_id: str) -> bool:
    return scene_id in _semantic_ok
## labels 
def dump_sg_objects(sg_sim, tag="", max_items=200):
    try:
        objs = list(zip(sg_sim.object_node_ids, sg_sim.object_node_names))
    except Exception as e:
        print(f"[SG][{tag}] cannot read object lists: {e}")
        return

    rooms = []
    try:
        rooms = list(zip(sg_sim.room_node_ids, sg_sim.room_node_names))
    except Exception:
        pass

    print("\n" + "=" * 90)
    print(f"[SG DUMP] {tag}")
    if rooms:
        print(f"[SG] rooms ({len(rooms)}): " + ", ".join([f"{rid}:{rname}" for rid, rname in rooms[:30]]))
        if len(rooms) > 30:
            print(f"[SG] ... {len(rooms)-30} more rooms")

    print(f"[SG] objects total: {len(objs)}")
    for i, (oid, oname) in enumerate(objs[:max_items]):
        print(f"  - {oid}: {oname}")
    if len(objs) > max_items:
        print(f"[SG] ... {len(objs)-max_items} more objects")
    print("=" * 90 + "\n")

# -----------------------------
# Minimal CSV loaders for MSP
# -----------------------------
def load_init_poses_csv(init_pose_path: str):
    """
    Supports both:
      - scene_init_poses.csv style (scene_floor, init_x, init_y, init_z, init_angle)
      - scene_init_poses_semantic_only.csv style (same columns)
    Returns dict: scene_floor -> {"init_pts": np.array([x,y,z]), "init_angle": float}
    """
    out = {}
    with open(init_pose_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            scene_floor = row["scene_floor"]
            x = float(row["init_x"])
            y = float(row["init_y"])
            z = float(row["init_z"])
            ang = float(row["init_angle"])
            out[scene_floor] = {"init_pts": np.array([x, y, z], dtype=np.float32), "init_angle": ang}
    return out


def load_questions_msp_csv(qpath: str):
    """
    Reads questions_msp_sample_1.csv
    Expected columns: scene,floor,source_question,primary_object,distance_m,predicate,msp_question,ann_ok
    Returns list of dicts.
    """
    data = []
    with open(qpath, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append(row)
    return data


def is_msp_csv(qpath: str) -> bool:
    with open(qpath, "r") as f:
        header = f.readline().strip().split(",")
    return "msp_question" in header


def main(cfg):
    # Detect MSP vs EQA based on question CSV columns
    qpath = cfg.data.question_data_path
    if is_msp_csv(qpath):
        click.secho("[mode] Detected MSP question CSV (msp_question column found).", fg="cyan")
        questions_data = load_questions_msp_csv(qpath)
        init_pose_data = load_init_poses_csv(cfg.data.init_pose_data_path)
        is_msp = True
    else:
        click.secho("[mode] Detected EQA question CSV.", fg="cyan")
        questions_data, init_pose_data = load_eqa_data(cfg.data)
        is_msp = False

    output_path = Path(__file__).resolve().parent.parent / cfg.output_path
    os.makedirs(str(output_path), exist_ok=True)
    results_filename = output_path / f"{cfg.results_filename}.json"
    device = f"cuda:{cfg.gpu}" if torch.cuda.is_available() else "cpu"

    eqa_enrich_labels = OmegaConf.load(Path(__file__).resolve().parent.parent / cfg.data.explore_eqa_dataset_enrich_labels)

    if not cfg.data.use_semantic_data:
        from hydra_python.detection.detic_segmenter import DeticSegmenter
        segmenter = DeticSegmenter(cfg)
    else:
        segmenter = None

    successes = 0

    for question_ind in tqdm(range(len(questions_data))):

        if is_msp:
            q = questions_data[question_ind]
            scene = q["scene"]
            floor = str(q["floor"])
            scene_floor = f"{scene}_{floor}"

            # MSP question text
            msp_question = q["msp_question"]
            primary_object = q.get("primary_object", None)

            # We’re not doing GT correctness here; “answer” is None
            answer = None
            experiment_id = f"{question_ind}_{scene}_{floor}"

        else:
            question_data = questions_data[question_ind]
            scene = question_data["scene"]
            floor = question_data["floor"]
            scene_floor = scene + "_" + floor
            answer = question_data["answer"]
            experiment_id = f"{question_ind}_{scene}_{floor}"

        if should_skip_experiment(experiment_id, filename=results_filename):
            click.secho(f"Skipping==Index: {question_ind} Scene: {scene} Floor: {floor}=======", fg="yellow")
            continue
        else:
            click.secho(f"Executing=========Index: {question_ind} Scene: {scene} Floor: {floor}=======", fg="green")

        raw_question_path = output_path / experiment_id
        if raw_question_path.exists():
            click.secho(f"[resume] Reusing existing folder: {raw_question_path}", fg="yellow")
            question_path = raw_question_path
        else:
            question_path = hydra_python.resolve_output_path(raw_question_path)

        scene_name = f"{cfg.data.scene_data_path}/{scene}/{scene[6:]}.basis.glb"
        scene_id = scene[6:]

        if cfg.data.use_semantic_data and not scene_has_semantics(scene_id):
            click.secho(f"[skip] {scene} has no semantics; skipping.", fg="yellow")
            continue

        # Build VLM question + init Habitat + pipeline
        habitat_data = HabitatInterface(scene_name, cfg=cfg.habitat, device=device)
        pipeline = initialize_hydra_pipeline(cfg.hydra, habitat_data, question_path)
        rr_logger = RRLogger(question_path)

        # Extract initial pose
        init_pts = init_pose_data[scene_floor]["init_pts"]
        init_angle = init_pose_data[scene_floor]["init_angle"]

        # TSDF setup
        pts_normal = pos_habitat_to_normal(init_pts)
        floor_height = pts_normal[-1]
        tsdf_bnds, scene_size = get_scene_bnds(habitat_data.pathfinder, floor_height)
        cam_intr = get_cam_intr(cfg.habitat.hfov, cfg.habitat.img_height, cfg.habitat.img_width)

        tsdf_planner = TSDFPlanner(
            cfg=cfg.frontier_mapping,
            vol_bnds=tsdf_bnds,
            cam_intr=cam_intr,
            floor_height_offset=0,
            pts_init=pts_normal,
            rr_logger=rr_logger,
        )

        # Enrich labels (same as before)
        if not is_msp:
            question_data = questions_data[question_ind]
            if f"{question_ind}_{scene}" in eqa_enrich_labels:
                label = eqa_enrich_labels[f"{question_ind}_{scene}"]["labels"]
            else:
                label = " "
            vlm_question, clean_ques_ans, choices, vlm_pred_candidates = get_instruction_from_eqa_data(question_data)
        else:
            # Minimal: use msp_question directly
            clean_ques_ans = " "
            choices, vlm_pred_candidates = [], []
            label = " "
            # Keep the question text as-is; we only append anchor id for logging stability (no geometry hint)
            if primary_object:
                vlm_question = f"{msp_question} (Anchor object id: {primary_object})"
            else:
                vlm_question = msp_question

        sg_sim = SceneGraphSim(
            cfg,
            question_path,
            pipeline,
            rr_logger,
            device=device,
            clean_ques_ans=clean_ques_ans,
            enrich_object_labels=label,
        )

        # Init view
        poses = habitat_data.get_init_poses_eqa(init_pts, init_angle, cfg.habitat.camera_tilt_deg)
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
        dump_sg_objects(sg_sim, tag=f"init_view q={question_ind} scene={scene_floor}")

        # Instantiate planner (Gemini)
        if "gpt" in cfg.vlm.name.lower():
            # (Optional) You can later extend GPT planner similarly if you want MSP there too
            vlm_planner = VLMPlannerEQAGPT(cfg.vlm, sg_sim, vlm_question, vlm_pred_candidates, choices, answer, question_path)
        elif "gemini" in cfg.vlm.name.lower():
            if is_msp:
                # uses cfg.vlm.answer_mode: "msp_object" or "msp_point"
                vlm_planner = VLMPlannerEQAGemini(
                    cfg.vlm,
                    sg_sim,
                    vlm_question,
                    pred_candidates=None,
                    choices=None,
                    answer=None,
                    output_path=question_path,
                    anchor_object_id=primary_object,
                )
            else:
                vlm_planner = VLMPlannerEQAGemini(cfg.vlm, sg_sim, vlm_question, vlm_pred_candidates, choices, answer, question_path)
        else:
            raise NotImplementedError("VLM planner not implemented.")

        click.secho(f"Index:{question_ind} Scene: {scene} Floor: {floor}", fg="green")
        click.secho(f"Question:\n{vlm_planner._question}\n", fg="green")

        num_steps = 3
        succ = False
        planning_steps = 0
        traj_length = 0.0

        final_pred = None

        for cnt_step in range(num_steps):
            start = time.time()
            target_pose, target_id, is_confident, confidence_level, answer_output = vlm_planner.get_next_action()
            click.secho(
                f"VLM planning time for overall step {cnt_step} and vlm step {planning_steps} is {time.time()-start}",
                fg="green",
            )

            # Stop condition: confident
            if is_confident or (confidence_level > 0.9):
                final_pred = answer_output
                succ = True  # For MSP, succ means "answered confidently"; for EQA, succ means correct below

                if not is_msp:
                    # original correctness logic
                    answer_token = answer_output.get("answer") if isinstance(answer_output, dict) else answer_output
                    succ = (answer == answer_token)
                    if succ:
                        successes += 1
                        result = f"Success at vlm step{planning_steps} for {question_ind}:{scene_floor}"
                        click.secho(result, fg="blue")
                        click.secho(f"VLM Planner answer: {answer_token}, Correct answer: {answer}", fg="blue")
                    else:
                        result = f"Failure at vlm step {planning_steps} for {question_ind}:{scene_floor}"
                        click.secho(result, fg="red")
                        click.secho(f"VLM Planner answer: {answer_token}, Correct answer: {answer}", fg="red")
                else:
                    # MSP logging
                    mode = getattr(cfg.vlm, "answer_mode", "msp_object")
                    if mode == "msp_object":
                        sel_id = final_pred.get("selected_object_id", None)
                        sel_name = final_pred.get("selected_object_name", None)
                        sel_center = None
                        if sel_id:
                            try:
                                sel_center = sg_sim.get_position_from_id(sel_id)
                                sel_center = [float(sel_center[0]), float(sel_center[1]), float(sel_center[2])]
                            except Exception:
                                sel_center = None
                        click.secho(f"[MSP] Confident answer (OBJECT): {sel_id} ({sel_name}) center={sel_center}", fg="blue")
                    elif mode == "msp_point":
                        xyz = final_pred.get("target_point_xyz", None)
                        click.secho(f"[MSP] Confident answer (POINT): xyz={xyz}", fg="blue")
                    else:
                        click.secho(f"[MSP] Confident answer (unknown mode): {final_pred}", fg="blue")

                rr_logger.log_text_data(vlm_planner.full_plan + "\n" + f"FINAL: {final_pred}")
                break

            # Otherwise: execute navigation step
            if target_pose is not None:
                current_heading = habitat_data.get_heading_angle()

                agent = habitat_data._sim.get_agent(0)
                current_pos = agent.get_state().position

                frontier_habitat = pos_normal_to_habitat(target_pose)
                frontier_habitat[1] = current_pos[1]

                path = habitat_sim.nav.ShortestPath()
                path.requested_start = current_pos
                path.requested_end = frontier_habitat

                found_path = habitat_data.pathfinder.find_path(path)

                if found_path:
                    desired_path = pos_habitat_to_normal(np.array(path.points)[:-1])
                    rr_logger.log_traj_data(desired_path)
                    rr_logger.log_target_poses(target_pose)
                else:
                    click.secho(f"Cannot find navigable path at {cnt_step}. Continuing..", fg="red")
                    continue

                poses = habitat_data.get_trajectory_from_path_habitat_frame(
                    target_pose, desired_path, current_heading, cfg.habitat.camera_tilt_deg
                )
                if poses is not None:
                    click.secho(f"Executing trajectory at overall step {cnt_step} and vlm step {planning_steps}", fg="yellow")
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
                    dump_sg_objects(sg_sim, tag=f"init_view q={question_ind} scene={scene_floor}")
                    rr_logger.log_text_data(vlm_planner.full_plan)
                    planning_steps += 1
                else:
                    click.secho(f"Cannot find trajectory from navigable path at {cnt_step}. Continuing..", fg="red")
                    continue
            else:
                click.secho(f"VLM planner failed at overall step {cnt_step}. Continuing...", fg="red")

        metrics = {
            "vlm_steps": planning_steps,
            "overall_steps": cnt_step,
            "is_confident": bool(final_pred is not None),
            "confidence_level": float(final_pred.get("confidence_level", 0.0)) if isinstance(final_pred, dict) else 0.0,
            "traj_length": traj_length,
            "final_pred": final_pred,
            "mode": "msp" if is_msp else "eqa",
            "answer_mode": getattr(cfg.vlm, "answer_mode", "eqa"),
        }

        log_experiment_status(experiment_id, succ, metrics=metrics, filename=results_filename)
        habitat_data._sim.close(destroy=True)
        pipeline.save()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-cf", "--cfg_file", help="cfg file name", default="", type=str, required=True)
    args = parser.parse_args()

    config_path = Path(__file__).resolve().parent.parent / "cfg" / f"{args.cfg_file}.yaml"
    cfg = OmegaConf.load(config_path)

    OmegaConf.resolve(cfg)
    main(cfg)
