# src/utils/wandb_utils.py
from __future__ import annotations
import os
from typing import Any, Dict, Optional

import wandb
from git import Repo  # pip install GitPython if you want this

def _get_git_info(root_path: str = ".") -> Dict[str, Any]:
    try:
        repo = Repo(root_path, search_parent_directories=True)
        return {
            "git_commit": str(repo.head.commit.hexsha),
            "git_is_dirty": repo.is_dirty(),
            "git_branch": repo.active_branch.name if not repo.head.is_detached else "DETACHED",
        }
    except Exception:
        return {}

def init_wandb_run(
    cfg: Any,
    run_name: Optional[str] = None,
    project: str = "spatial-msp-vlm-planner",
    entity: Optional[str] = None,
) -> wandb.sdk.wandb_run.Run:
    """
    Initialize a W&B run given a Hydra cfg.
    """
    # Flatten a bit: cfg -> dict
    cfg_dict = {}
    try:
        # Hydra OmegaConf
        from omegaconf import OmegaConf
        cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    except Exception:
        # Fallback: naive __dict__
        cfg_dict = dict(cfg.__dict__)

    git_info = _get_git_info()

    run = wandb.init(
        project=project,
        entity=entity,
        name=run_name or cfg_dict.get("exp_name", None),
        config={**cfg_dict, **git_info},
        settings=wandb.Settings(start_method="thread"),
    )
    return run

def log_metrics(metrics: Dict[str, Any], step: Optional[int] = None):
    """
    Simple wrapper so you can call log_metrics({...}) anywhere.
    """
    try:
        wandb.log(metrics, step=step)
    except Exception as e:
        print(f"[W&B] log_metrics failed: {e}")

def log_table(name: str, data: Any):
    """
    If `data` is a list of dicts, convert to W&B Table.
    """
    try:
        if isinstance(data, list) and data and isinstance(data[0], dict):
            columns = list(data[0].keys())
            rows = [[row.get(c, None) for c in columns] for row in data]
            table = wandb.Table(columns=columns, data=rows)
            wandb.log({name: table})
        else:
            # Fallback: just log as json
            wandb.log({name: data})
    except Exception as e:
        print(f"[W&B] log_table failed: {e}")