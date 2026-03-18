import sys
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def _plot_logical_distribution(objects, obj_probs, out_path, step_num):
    if not obj_probs or not objects: return
    prob_map = {item["object_id"]: item["probability"] for item in obj_probs}
    xs, zs, colors = [], [], []
    best_prob, best_pos = -1.0, None
    for obj in objects:
        pos = obj.get("position", [0, 0, 0])
        p = prob_map.get(obj["id"], 0.0)
        xs.append(pos[0])
        zs.append(pos[2])
        colors.append(p)
        if p > best_prob:
            best_prob = p
            best_pos = (pos[0], pos[2])

    if not xs: return

    plt.figure(figsize=(8, 8))
    scatter = plt.scatter(xs, zs, c=colors, cmap="Reds", s=(np.array(colors) * 300) + 20, alpha=0.8, edgecolors="grey")
    if best_pos and best_prob > 0.0:
        bx, bz = best_pos
        box_size = 0.5
        rect = plt.Rectangle((bx - box_size/2, bz - box_size/2), box_size, box_size, 
                             fill=False, edgecolor='blue', linewidth=3, label="Top Logical Pick")
        plt.gca().add_patch(rect)
    plt.colorbar(scatter, label="Logical Reasoning Probability")
    plt.xlabel("x (habitat)")
    plt.ylabel("z (habitat)")
    plt.title(f"LogicalAgent Confidence Heatmap (Step {step_num})")
    if best_pos: plt.legend()
    plt.tight_layout()
    plt.savefig(out_path / f"heatmap_step_{step_num:03d}.png", dpi=150)
    plt.close()

objects = [
    {"id": "obj1", "position": [1.0, 0.0, 1.0]},
    {"id": "obj2", "position": [-2.0, 0.0, 3.0]},
    {"id": "obj3", "position": [0.5, 0.0, -1.5]}
]

probs = [
    {"object_id": "obj1", "probability": 0.1},
    {"object_id": "obj2", "probability": 0.8},
    {"object_id": "obj3", "probability": 0.1}
]

out = Path("/home/flux/graph_eqa_swagat/test_output")
out.mkdir(exist_ok=True)
_plot_logical_distribution(objects, probs, out, 1)
print("Heatmap generated.")
