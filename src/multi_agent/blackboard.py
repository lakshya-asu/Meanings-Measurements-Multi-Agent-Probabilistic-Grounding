import click
import json
from typing import List, Dict, Any, Optional
import numpy as np

class Blackboard:
    def __init__(self, question: str, mode: str):
        self.question = question
        self.mode = mode  # "where", "which", or "eqa"
        self.choices: List[str] = []
        
        # Step context
        self.step_t = 0
        self.agent_pose_hab: Optional[np.ndarray] = None
        self.agent_yaw_rad: float = 0.0
        self.current_image_path: Optional[str] = None
        self.scene_graph_str: str = ""
        self.agent_semantic_state: str = ""
        
        # Data populated by the pipeline
        self.available_objects: List[Dict[str, Any]] = []
        self.available_frontiers: List[Dict[str, Any]] = []
        
        # The chronological ledger for the CURRENT step
        self.event_ledger: List[Dict[str, Any]] = []
        
        # The persistent history across ALL steps
        self.global_history: str = ""

    def update_state(self, t: int, pose: np.ndarray, yaw: float, img_path: str, sg_str: str, agent_state: str, objects: List, frontiers: List):
        self.step_t = t
        self.agent_pose_hab = pose
        self.agent_yaw_rad = yaw
        self.current_image_path = img_path
        self.scene_graph_str = sg_str
        self.agent_semantic_state = agent_state
        self.available_objects = objects
        self.available_frontiers = frontiers
        self.event_ledger = [] # Clear the ledger for the new physical step

    def append_event(self, agent_name: str, event_type: str, details: Any, status: str = "INFO"):
        """status can be INFO, PASS, or FAIL"""
        entry = {
            "agent": agent_name,
            "type": event_type,
            "status": status,
            "details": details
        }
        self.event_ledger.append(entry)
        
        # --- NEW: Verbose, Color-Coded Terminal Logging ---
        color = "green" if status == "PASS" else "red" if status == "FAIL" else "cyan"
        click.secho(f"\n>>> [{agent_name} | {event_type} | {status}]", fg=color, bold=True)
        
        if isinstance(details, dict) or isinstance(details, list):
            click.secho(json.dumps(details, indent=2), fg=color)
        else:
            click.secho(str(details), fg=color)
        click.secho("-" * 60, fg="white")
        
    def get_ledger_str(self) -> str:
        if not self.event_ledger:
            return "No events yet in this step."
        
        lines = []
        for e in self.event_ledger:
            lines.append(f"[{e['status']}] {e['agent']} ({e['type']}): {e['details']}")
        return "\n".join(lines)