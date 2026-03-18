import os
import numpy as np
from tqdm import tqdm
from PIL import Image
from scipy.spatial.transform import Rotation as R
from src.envs.utils import hydra_get_mesh, get_cam_pose_tsdf, pos_habitat_to_normal

def hydra_output_callback(pipeline, visualizer):
    """Show graph."""
    if visualizer:
        visualizer.update_graph(pipeline.graph)

def _take_step(pipeline, data, pose, labels, image_viz, is_eqa=False, segmenter=None):
    timestamp, world_t_body, q_wxyz = pose
    q_xyzw = np.roll(q_wxyz, -1) #changing to xyzw format

    world_T_body = np.eye(4)
    world_T_body[:3, 3] = world_t_body
    world_T_body[:3, :3] = R.from_quat(q_xyzw).as_matrix()
    data.set_pose(timestamp, world_T_body, is_eqa=is_eqa)


    if data.rgb is not None:
        labels = segmenter(data.rgb) if segmenter else data.labels

    if is_eqa:
        pose_cam = get_cam_pose_tsdf(data.get_depth_sensor_state())
        world_t_body = pose_cam[:3, 3]
        q_xyzw = R.from_matrix(pose_cam[:3, :3]).as_quat()
        q_wxyz = np.roll(q_xyzw, 1)

    pipeline.step(timestamp, world_t_body, q_wxyz, data.depth, labels, data.rgb)

def run(
    pipeline,
    habitat_data,
    pose_source,
    segmenter=None,
    step_callback=hydra_output_callback,
    output_path=None,
    rr_logger=None,
    sg_sim=None,
    tsdf_planner=None,
    save_image=False
):

    agent_positions, agent_quats_wxyz = [], []
    imgs_rgb, imgs_depth, extrinsics = [], [], []

    for pose in tqdm(pose_source, desc='Executing traj'):
        pipeline.graph.save(output_path / "dsg.json", False)
        pipeline.graph.save_filtered(output_path / "filtered_dsg.json", False)
        
        if habitat_data.rgb is not None:
            labels = segmenter(habitat_data.rgb) if segmenter else habitat_data.labels
        else:
            labels = np.zeros((640, 480)).astype(int)
            
        _take_step(pipeline, habitat_data, pose, labels, image_viz=None, is_eqa=True, segmenter=segmenter)
        imgs_rgb.append(habitat_data.rgb)
        imgs_depth.append(habitat_data.depth)

        agent_pos, agent_quat_wxyz = habitat_data.get_state(is_eqa=True)
        agent_positions.append(agent_pos)
        agent_quats_wxyz.append(agent_quat_wxyz)
        camera_pos, camera_quat_wxyz = habitat_data.get_camera_pos(is_eqa=True)
        mesh_vertices, mesh_colors, mesh_triangles = hydra_get_mesh(pipeline)

        cam_pose_tsdf = get_cam_pose_tsdf(habitat_data.get_depth_sensor_state())
        extrinsics.append(cam_pose_tsdf)
        pts_normal = pos_habitat_to_normal(pose[1])

        if tsdf_planner:
            tsdf_planner.update(
                habitat_data.rgb,
                habitat_data.depth,
                pts_normal,
                cam_pose_tsdf,
            )
            frontier_nodes = tsdf_planner.frontier_to_sample_normal
            
        if rr_logger:
            rr_logger.log_mesh_data(mesh_vertices, mesh_colors, mesh_triangles)
            rr_logger.log_agent_data(agent_positions)
            rr_logger.log_agent_tf(agent_pos, agent_quat_wxyz)
            rr_logger.log_camera_tf(camera_pos, camera_quat_wxyz)
            rr_logger.log_img_data(habitat_data.rgb, labels)
            rr_logger.step()

        if step_callback:
            step_callback(pipeline, None)

    if save_image:
        curr_img = Image.fromarray(habitat_data.rgb)
        curr_img.save(output_path / "current_img.png")

    if sg_sim:
        # Should be done after saving default image cos this update overwrites it
        sg_sim.update(
            imgs_rgb=imgs_rgb, 
            imgs_depth=imgs_depth, 
            intrinsics=habitat_data.intrinsics, 
            extrinsics=extrinsics, 
            frontier_nodes=frontier_nodes)
    
