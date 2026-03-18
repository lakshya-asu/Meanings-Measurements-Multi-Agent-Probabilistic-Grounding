import numpy as np
import json
import click
from pathlib import Path
from typing import Optional, Dict, Any

from src.utils.data_utils import get_latest_image
from src.multi_agent.blackboard import Blackboard
from src.multi_agent.agent_setup import AgentFactory

def _write_json(path: Path, obj: Any) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(obj, f, indent=2)
    except Exception as e:
        print(f"[MSP] Failed to write json {path}: {e}")

class MultiAgentFatPlanner:
    def __init__(self, cfg, sg_sim, question, out_path=".", answer_mode="where", **kwargs):
        self.cfg = cfg
        self.sg_sim = sg_sim
        self.out_path = Path(out_path)
        
        click.secho(f"\n{'='*40}\nINITIALIZING FAT PLANNER\n{'='*40}", fg="magenta", bold=True)
        click.secho(f"Question: {question}", fg="magenta")
        click.secho(f"Mode: {answer_mode}", fg="magenta")
        
        self.blackboard = Blackboard(question=question, mode=answer_mode)
        
        providers = kwargs.get("agent_providers", {})
        q_prov = providers.get("qa", "claude")
        click.secho(f"Fat Agent Provider: QA={q_prov}", fg="yellow")
        
        self.qa = AgentFactory.create_agent("qa", provider=q_prov)
        
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
        click.secho(f"\n{'='*20} FAT AGENT STEP {step_num} {'='*20}", fg="magenta", bold=True)
        click.secho(f"[Env] Found {len(objects)} Objects, {len(frontiers)} Frontiers", fg="white")
        
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

        qa_out = self.qa.process(self.blackboard)
        if qa_out.get("ok", False):
            action_type = qa_out.get("action_type", "lookaround")
            chosen_id = qa_out.get("chosen_id", "NONE")
            ans = qa_out.get("answer", "")
            conf_val = qa_out.get("confidence", 0.0)
            
            target_pose = None
            if action_type in ["goto_object", "goto_frontier"] and chosen_id != "NONE":
                target_pose = self.sg_sim.get_position_from_id(chosen_id)
            
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
            click.secho(f"[Planner] QA Fast Path crashed. Fallback.", fg="red")
            fid = str(frontiers[0]["id"]) if frontiers else ""
            fallback_action = "goto_frontier" if fid else "lookaround"
            return finalize_step(self.sg_sim.get_position_from_id(fid) if fid else None, fid, False, 0.0, {"action_type": fallback_action, "chosen_id": fid, "thought": "Crash fallback exploring."})
