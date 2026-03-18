import numpy as np
import json
import click
from pathlib import Path
from typing import Optional, Dict, Any

from src.utils.data_utils import get_latest_image

from src.multi_agent.blackboard import Blackboard
from src.ablations.ablation2.single_vlm_agent import SingleVLMAgent

def _write_json(path: Path, obj: Any) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(obj, f, indent=2)
    except Exception as e:
        print(f"[Ablation2] Failed to write json {path}: {e}")

class PlannerAblation2:
    def __init__(self, cfg, sg_sim, question, out_path=".", answer_mode="where", **kwargs):
        self.cfg = cfg
        self.sg_sim = sg_sim
        self.out_path = Path(out_path)
        
        click.secho(f"\n{'='*40}\nINITIALIZING ABLATION 2 PLANNER (SINGLE VLM)\n{'='*40}", fg="magenta", bold=True)
        click.secho(f"Question: {question}", fg="cyan")
        
        self.blackboard = Blackboard(question=question, mode=answer_mode)
        
        # Only one agent
        self.vlm_agent = SingleVLMAgent()
        
        if "choices" in kwargs:
            self.blackboard.choices = kwargs["choices"]

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
        
        step_num = self.blackboard.step_t + 1
        click.secho(f"\n{'='*20} ABLATION 2 (SINGLE VLM) STEP {step_num} {'='*20}", fg="cyan", bold=True)
        click.secho(f"[Env] Pose: {agent_pos_hab.tolist()} | Yaw: {agent_yaw_rad:.3f} rad", fg="white")
        click.secho(f"[Env] Semantic State: {agent_state_str}", fg="white")
        click.secho(f"[Env] Found {len(objects)} Objects, {len(frontiers)} Frontiers", fg="white")
        click.secho("-" * 60, fg="white")
        
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
        
        out = self.vlm_agent.process(self.blackboard)
        
        action_type = out.get("action_type", "lookaround")
        chosen_id = out.get("chosen_id", "")
        answer = out.get("answer", "")
        conf = out.get("confidence", 0.0)
        xyz_target = out.get("xyz_target", [])
        thought = out.get("reasoning", "")
        
        # Resolve target pose
        target_pose = None
        if action_type in ["goto_object", "goto_frontier"]:
            target_pose = self.sg_sim.get_position_from_id(chosen_id)
        elif action_type == "goto_xyz" and len(xyz_target) == 3:
            target_pose = xyz_target
            
        # Hard fallback
        if action_type in ["goto_object", "goto_frontier", "goto_xyz"] and target_pose is None:
            click.secho(f"[Ablation2] Target {chosen_id} or xyz {xyz_target} invalid. Falling back to lookaround.", fg="red")
            action_type = "lookaround"
            
        is_conf = False
        if action_type == "answer" and conf >= float(getattr(self.cfg, "pre_answer_conf_thresh", 0.8)):
             is_conf = True

        extra = {
            "action_type": action_type,
            "chosen_id": answer if action_type == "answer" else chosen_id,
            "thought": thought,
            "confidence": conf
        }

        click.secho(f"\n[DECISION] Action: {action_type} | Target: {extra['chosen_id']} | Conf: {conf:.2f}", fg="yellow", bold=True)
        if thought:
             click.secho(f"[DECISION] Thought: {thought}", fg="yellow")
             
        trace_dump = {
            "t": step_num,
            "agent_pose": agent_pos_hab.tolist(),
            "agent_yaw": agent_yaw_rad,
            "ledger": self.blackboard.event_ledger,
            "final_decision": extra
        }
        _write_json(self.out_path / f"trace_step_{step_num:03d}.json", trace_dump)
        
        # Return format expected by run_ablation2.py
        # target_pose, target_id, is_conf, conf, extra
        if action_type == "answer":
             return None, answer, is_conf, conf, extra
        return target_pose, chosen_id, is_conf, conf, extra
