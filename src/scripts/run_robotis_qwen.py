import sys
import os
import time
from pathlib import Path
import click
import numpy as np
import hydra_python as hydra
from omegaconf import OmegaConf

from stretch.core.parameters import Parameters
from src.real_world.robotis_hydra_agent import RobotisHydraAgent
from src.planners.multi_agent_fat_planner import MultiAgentFatPlanner
from src.scene_graph.scene_graph_sim import SceneGraphSim
from stretch.core.robot import AbstractRobotClient
from stretch.core.interfaces import Observations

class MockPipeline:
    def __init__(self, ds_graph):
        self.graph = ds_graph

class DummyRobotisClient(AbstractRobotClient):
    """
    A minimal implementation of a Robot client required passing the Pyre checks
    and serving as placeholder for the real Robotis AI Worker API hook.
    """
    def __init__(self):
        self.running = True
        self._rerun = None
        self._last_fail = False

    def get_base_pose(self):
        return np.array([0.0, 0.0, 0.0])

    def move_base_to(self, target, relative=False, blocking=True, timeout=10.0, verbose=False):
        print(f"[Robotis] Moving to {target}")
        return True

    def get_observation(self):
        return Observations(
            rgb=np.zeros((480, 640, 3), dtype=np.uint8),
            depth=np.zeros((480, 640), dtype=np.float32),
            camera_K=np.eye(3),
            camera_pose=np.eye(4),
            lidar_timestamp=time.time()
        )
    
    def in_manipulation_mode(self): return False
    def switch_to_manipulation_mode(self): pass
    def switch_to_navigation_mode(self): pass
    def move_to_nav_posture(self): pass
    def get_robot_model(self):
        class DummyModel:
            def get_footprint(self): return [[-0.2, -0.2], [0.2, -0.2], [0.2, 0.2], [-0.2, 0.2]]
        return DummyModel()
    def start(self): return True
    def stop(self): self.running = False
    def get_pose_graph(self): return []
    def execute_trajectory(self, pts, **kwargs): return True
    def wait_for_waypoint(self, pt, **kwargs): return True
    def last_motion_failed(self): return self._last_fail

@click.command()
@click.option("--question", default="Where is the object?", help="Current query.")
@click.option("--scene_graph_path", default="src/zed2i/backend/dsg.json", help="Path to offline dsg.json.")
@click.option("--max_steps", default=10, type=int)
@click.option("--vlm_provider", default="qwen", help="Which VLM to use for single agent QA.")
def run_robotis_offline_agent(question, scene_graph_path, max_steps, vlm_provider):
    print(f"Loading prerecorded graph from {scene_graph_path}...")
    ds_graph = hydra.DynamicSceneGraph.load(scene_graph_path)
    pipeline_mock = MockPipeline(ds_graph)

    # Initialize Dummy Robotis hook
    robot_client = DummyRobotisClient()

    # Base params matching stretch API
    parameters = Parameters(
        encoder="clip",
        voxel_size=0.05,
        use_scene_graph=True,
        trajectory_pos_err_threshold=0.2,
        trajectory_rot_err_threshold=0.75,
        tts_engine="dummy",
        hydra_update_freq=5.0,
        task={"command": question},
        agent={"sweep_head_on_update": False, "use_realtime_updates": False},
        motion_planner={
            "step_size": 0.1, "rotation_step_size": 0.1,
            "frontier": {
                "dilate_frontier_size": 2, "dilate_obstacle_size": 2,
                "default_expand_frontier_size": 5, "min_dist": 0.5, "step_dist": 0.5,
                "min_points_for_clustering": 5, "num_clusters": 5, "cluster_threshold": 0.5,
                "contract_traversible_size": 1
            },
            "goals": {"manipulation_radius": 0.5},
            "shortcut_plans": False, "simplify_plans": False,
        },
        data={"initial_state": {"head": [0.0, 0.0]}}
    )

    out_path = Path("outputs/run_robotis_offline")
    out_path.mkdir(parents=True, exist_ok=True)

    class DummyCfg:
        class scene_graph_sim:
            enrich_rooms = False
            save_image = False
            include_regions = True
            no_scene_graph = False
            enrich_frontiers = True
            class key_frame_selection:
                choose_final_image = False
                use_clip_for_images = False
                use_siglip_for_images = False
        class vlm:
            pass

    # Initialize the SceneGraphSim with the mock pipeline
    sg_sim = SceneGraphSim(
        cfg=DummyCfg(), 
        output_path=out_path, 
        pipeline=pipeline_mock, 
        device="cpu",
        clean_ques_ans=question,
        enrich_object_labels="object"
    )
    # Recreate the graph networkx representation
    sg_sim._build_sg_from_hydra_graph()
    
    # Initialize our modified Robotis agent
    agent = RobotisHydraAgent(
        robot=robot_client,
        parameters=parameters,
        sg_sim=sg_sim,
        output_path=out_path,
        enable_realtime_updates=False
    )
    
    agent.start(can_move=False, verbose=True)

    click.secho(f"Initializing single QA agent planner (Provider: {vlm_provider})", fg="cyan")
    
    vlm_planner = MultiAgentFatPlanner(
        cfg=DummyCfg().vlm,
        sg_sim=sg_sim,
        question=question,
        out_path=str(out_path),
        agent_providers={"qa": vlm_provider}
    )

    # Agent executes the plan on the prerecorded SG and simulated robot actions
    agent.run_eqa_vlm_planner(
        vlm_planner=vlm_planner,
        sg_sim=sg_sim,
        manual_wait=False,
        max_planning_steps=max_steps,
        go_home_at_end=False
    )

if __name__ == "__main__":
    run_robotis_offline_agent()
