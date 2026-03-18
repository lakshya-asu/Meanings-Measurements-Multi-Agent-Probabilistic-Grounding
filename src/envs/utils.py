import numpy as np
import quaternion

def hydra_get_mesh(pipeline):
    vertices = pipeline.graph.mesh.get_vertices()
    faces = pipeline.graph.mesh.get_faces()

    mesh_vertices = vertices[:3, :].T
    mesh_triangles = faces.T
    mesh_colors = vertices[3:, :].T

    return mesh_vertices, mesh_colors, mesh_triangles

def get_cam_pose_tsdf(depth_sensor_state):
    # Update camera info
    quaternion_0 = depth_sensor_state.rotation
    translation_0 = depth_sensor_state.position
    cam_pose = np.eye(4)
    cam_pose[:3, :3] = quaternion.as_rotation_matrix(quaternion_0)
    cam_pose[:3, 3] = translation_0
    cam_pose_normal = pose_habitat_to_normal(cam_pose)
    cam_pose_tsdf = pose_normal_to_tsdf(cam_pose_normal)
    return cam_pose_tsdf

def pose_habitat_to_normal(pose):
    # T_normal_cam = T_normal_habitat * T_habitat_cam
    return np.dot(
        np.array([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]), pose
    )

def pose_normal_to_tsdf(pose):
    return np.dot(
        pose, np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    )

def pos_habitat_to_normal(pts):
    # -90 deg around x-axis
    return np.dot(pts, np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]]))

def pos_normal_to_habitat(pts):
    # +90 deg around x-axis
    return np.dot(pts, np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]]))