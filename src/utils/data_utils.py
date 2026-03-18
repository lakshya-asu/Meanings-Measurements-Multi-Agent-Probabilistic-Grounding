import csv, os, ast
import numpy as np

def load_eqa_data(cfg):
    # Load dataset
    with open(cfg.question_data_path) as f:
        questions_data = [
            {k: v for k, v in row.items()}
            for row in csv.DictReader(f, skipinitialspace=True)
        ]
    
    # Filter to include only scenes with semantic annotations
    semantic_scenes = [f for f in os.listdir(cfg.semantic_annot_data_path) if os.path.isdir(os.path.join(cfg.semantic_annot_data_path, f))]

    filtered_question_data = []
    if cfg.use_semantic_data:
        for data in questions_data:
            if data['scene'] in semantic_scenes:
                filtered_question_data.append(data)
    else:
        for data in questions_data:
            if data['scene'] not in semantic_scenes:
                filtered_question_data.append(data)

    with open(cfg.init_pose_data_path) as f:
        init_pose_data = {}
        for row in csv.DictReader(f, skipinitialspace=True):
            init_pose_data[row["scene_floor"]] = {
                "init_pts": [
                    float(row["init_x"]),
                    float(row["init_y"]),
                    float(row["init_z"]),
                ],
                "init_angle": float(row["init_angle"]),
            }
    print(f"Loaded {len(filtered_question_data)} questions.")
    return filtered_question_data, init_pose_data


def get_instruction_from_eqa_data(question_data):
    question = question_data["question"]
    # self.choices = [c.split("'")[1] for c in question_data["choices"].split("',")]
    clean_ques_ans = question_data["question"]
    choices = ast.literal_eval(question_data["choices"])
    # Re-format the question to follow LLaMA style
    vlm_question = question
    vlm_pred_candidates = ["A", "B", "C", "D"]
    for token, choice in zip(vlm_pred_candidates, choices):
        vlm_question += "\n" + token + "." + " " + choice
        if ("do not choose" not in choice.lower()) and (choice.lower() not in ['yes', 'no']):
            clean_ques_ans += "  " + token + "." + " " + choice
    return vlm_question, clean_ques_ans, choices, vlm_pred_candidates

def get_traj_len_from_poses(poses):
    pts = np.array([pt[1] for pt in poses])
    deltas = np.diff(pts, axis=0)
    segment_lengths = np.linalg.norm(deltas, axis=1)
    return np.sum(segment_lengths)

def get_latest_image(output_folder):
    png_files = [file for file in os.listdir(output_folder) if file.startswith('current_img_')]
    indices = [int(f.split('current_img_')[1].split('.png')[0]) for f in png_files]
    return output_folder / f"current_img_{np.max(indices)}.png"
