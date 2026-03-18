# Modified by 2024 Allen Ren, Princeton University
# Copyright (c) 2018 Andy Zeng
# Source: https://github.com/andyzeng/tsdf-fusion-python/blob/master/fusion.py
# BSD 2-Clause License

# Copyright (c) 2019, Princeton University
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import numpy as np
from numba import njit, prange
import random
import logging
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage
from skimage import measure
from sklearn.cluster import DBSCAN
from scipy.ndimage import gaussian_filter, gaussian_filter1d
import rerun as rr

from .geom import (
    points_in_circle,
    find_normal,
    close_operation,
    rigid_transform,
    run_dijkstra,
    fps,
)


class TSDFPlanner:
    """Volumetric TSDF Fusion of RGB-D Images. No GPU mode.

    Add frontier-based exploration and semantic map.
    """

    def __init__(
        self,
        cfg,
        vol_bnds,
        cam_intr=None,
        floor_height_offset=0,
        pts_init=None,
        rr_logger=None,
    ):
        """Constructor.
        Args:
          vol_bnds (ndarray): An ndarray of shape (3, 2). Specifies the
            xyz bounds (min/max) in meters.
          voxel_size (float): The volume discretization in meters.
        """
        vol_bnds = np.asarray(vol_bnds)
        init_clearance = cfg.init_clearance * 2
        assert vol_bnds.shape == (3, 2), "[!] `vol_bnds` should be of shape (3, 2)."
        assert (vol_bnds[:, 0] < vol_bnds[:, 1]).all()

        # Define voxel volume parameters
        self._cfg = cfg
        self._vol_bnds = vol_bnds
        self._cam_intr = cam_intr
        self._voxel_size = float(cfg.tsdf_grid_size)
        self._trunc_margin = 5 * self._voxel_size  # truncation on SDF
        self._color_const = 256 * 256
        self._img_width = cfg.img_width
        self._img_height = cfg.img_height
        self._height_offset = cfg.height_offset
        self._range_height = cfg.range_height
        self._margin_h=int(cfg.margin_h_ratio * cfg.img_height)
        self._margin_w=int(cfg.margin_w_ratio * cfg.img_width)
        self._frontier_min_neighbors = cfg.visual_prompt.frontier_min_neighbors
        self._frontier_max_neighbors = cfg.visual_prompt.frontier_max_neighbors
        self._point_min_dist = cfg.visual_prompt.point_min_dist
        self._point_max_dist = cfg.visual_prompt.point_max_dist

        self._rr_logger = rr_logger

        # Adjust volume bounds and ensure C-order contiguous
        self._vol_dim = (
            np.ceil((self._vol_bnds[:, 1] - self._vol_bnds[:, 0]) / self._voxel_size)
            .copy(order="C")
            .astype(int)
        )
        self._vol_bnds[:, 1] = self._vol_bnds[:, 0] + self._vol_dim * self._voxel_size
        self._vol_origin = self._vol_bnds[:, 0].copy(order="C").astype(np.float32)

        # Initialize pointers to voxel volume in CPU memory
        # Assume all unobserved regions are occupied
        self._tsdf_vol_cpu = -np.ones(self._vol_dim).astype(np.float32)
        # for computing the cumulative moving average of observations per voxel
        self._weight_vol_cpu = np.zeros(self._vol_dim).astype(np.float32)
        self._color_vol_cpu = np.zeros(self._vol_dim).astype(np.float32)

        # Semantic value
        self._val_vol_cpu = np.zeros(self._vol_dim).astype(np.float32)
        self._weight_val_vol_cpu = np.zeros(self._vol_dim[:2]).astype(np.float32)

        # Explored or not
        self._explore_vol_cpu = np.zeros(self._vol_dim).astype(np.float32)

        # Get voxel grid coordinates
        xv, yv, zv = np.meshgrid(
            range(self._vol_dim[0]),
            range(self._vol_dim[1]),
            range(self._vol_dim[2]),
            indexing="ij",
        )
        self.vox_coords = (
            np.concatenate(
                [xv.reshape(1, -1), yv.reshape(1, -1), zv.reshape(1, -1)], axis=0
            )
            .astype(int)
            .T
        )

        # pre-compute
        self.cam_pts_pre = TSDFPlanner.vox2world(
            self._vol_origin, self.vox_coords, self._voxel_size
        )

        # Find the minimum height voxel
        self.min_height_voxel = int(floor_height_offset / self._voxel_size)
        self._height_voxel = int(self._height_offset / self._voxel_size) + self.min_height_voxel

        # For masking the area around initial pose to be unoccupied
        coords_init = self.world2vox(pts_init)
        self.init_points = points_in_circle(
            coords_init[0],
            coords_init[1],
            int(init_clearance / self._voxel_size),
            self._vol_dim[:2],
        )

        self.target_point = None

    @staticmethod
    @njit(parallel=True)
    def vox2world(vol_origin, vox_coords, vox_size):
        """Convert voxel grid coordinates to world coordinates."""
        vol_origin = vol_origin.astype(np.float32)
        vox_coords = vox_coords.astype(np.float32)
        cam_pts = np.empty_like(vox_coords, dtype=np.float32)
        for i in prange(vox_coords.shape[0]):
            for j in range(3):
                cam_pts[i, j] = vol_origin[j] + (vox_size * vox_coords[i, j])
        return cam_pts

    @staticmethod
    @njit(parallel=True)
    def cam2pix(cam_pts, intr):
        """Convert camera coordinates to pixel coordinates."""
        intr = intr.astype(np.float32)
        fx, fy = intr[0, 0], intr[1, 1]
        cx, cy = intr[0, 2], intr[1, 2]
        pix = np.empty((cam_pts.shape[0], 2), dtype=np.int64)
        for i in prange(cam_pts.shape[0]):
            pix[i, 0] = int(np.round((cam_pts[i, 0] * fx / cam_pts[i, 2]) + cx))
            pix[i, 1] = int(np.round((cam_pts[i, 1] * fy / cam_pts[i, 2]) + cy))
        return pix

    def pix2cam(self, pix, intr):
        """Convert pixel coordinates to camera coordinates."""
        intr = intr.astype(np.float32)
        fx, fy = intr[0, 0], intr[1, 1]
        cx, cy = intr[0, 2], intr[1, 2]
        cam_pts = np.empty((pix.shape[0], 3), dtype=np.float32)
        for i in range(cam_pts.shape[0]):
            cam_pts[i, 2] = 1
            cam_pts[i, 0] = (pix[i, 0] - cx) / fx * cam_pts[i, 2]
            cam_pts[i, 1] = (pix[i, 1] - cy) / fy * cam_pts[i, 2]
        return cam_pts

    def world2vox(self, pts):
        pts = pts - self._vol_origin
        coords = np.round(pts / self._voxel_size).astype(int)
        coords = np.clip(coords, 0, self._vol_dim - 1)
        return coords

    @staticmethod
    @njit(parallel=True)
    def integrate_tsdf(tsdf_vol, dist, w_old, obs_weight):
        """Integrate the TSDF volume."""
        tsdf_vol_int = np.empty_like(tsdf_vol, dtype=np.float32)
        w_new = np.empty_like(w_old, dtype=np.float32)
        for i in prange(len(tsdf_vol)):
            w_new[i] = w_old[i] + obs_weight
            tsdf_vol_int[i] = (w_old[i] * tsdf_vol[i] + obs_weight * dist[i]) / w_new[i]
        return tsdf_vol_int, w_new

    def integrate_sem(
        self,
        sem_pix,
        radius=1.0,  # meter
        obs_weight=1.0,
    ):
        """Add semantic value to the 2D map by marking a circle of specified radius"""
        assert len(self.candidates) == len(sem_pix)
        for p_ind, p in enumerate(self.candidates):
            radius_vox = int(radius / self._voxel_size)
            pts = points_in_circle(p[0], p[1], radius_vox, self._vol_dim[:2])
            for pt in pts:
                w_old = self._weight_val_vol_cpu[pt[0], pt[1]].copy()
                self._weight_val_vol_cpu[pt[0], pt[1]] += obs_weight
                self._val_vol_cpu[pt[0], pt[1]] = (
                    w_old * self._val_vol_cpu[pt[0], pt[1]]
                    + obs_weight * sem_pix[p_ind]
                ) / self._weight_val_vol_cpu[pt[0], pt[1]]

    def integrate(
        self,
        color_im,
        depth_im,
        cam_intr,
        cam_pose,
        sem_im=None,
        w_new=None,
        obs_weight=1.0,
        margin_h=240,  # from top
        margin_w=120,  # each side
    ):
        """Integrate an RGB-D frame into the TSDF volume.
        Args:
          color_im (ndarray): An RGB image of shape (H, W, 3).
          depth_im (ndarray): A depth image of shape (H, W).
          cam_intr (ndarray): The camera intrinsics matrix of shape (3, 3).
          cam_pose (ndarray): The camera pose (i.e. extrinsics) of shape (4, 4).
          sem_im (ndarray): An semantic image of shape (H, W).
          obs_weight (float): The weight to assign for the current observation. A higher
            value
          margin_h (int): The margin from the top of the image to exclude when integrating explored
          margin_w (int): The margin from the sides of the image to exclude when integrating explored
        """
        im_h, im_w = depth_im.shape

        # Fold RGB color image into a single channel image
        color_im = color_im.astype(np.float32)
        color_im = np.floor(
            color_im[..., 2] * self._color_const
            + color_im[..., 1] * 256
            + color_im[..., 0]
        )

        # Convert voxel grid coordinates to pixel coordinates
        cam_pts = rigid_transform(self.cam_pts_pre, np.linalg.inv(cam_pose))
        pix_z = cam_pts[:, 2]
        pix = TSDFPlanner.cam2pix(cam_pts, cam_intr)
        pix_x, pix_y = pix[:, 0], pix[:, 1]

        # Eliminate pixels outside view frustum
        valid_pix = np.logical_and(
            pix_x >= 0,
            np.logical_and(
                pix_x < im_w,
                np.logical_and(pix_y >= 0, np.logical_and(pix_y < im_h, pix_z > 0)),
            ),
        )
        depth_val = np.zeros(pix_x.shape)
        depth_val[valid_pix] = depth_im[pix_y[valid_pix], pix_x[valid_pix]]

        # narrow view
        valid_pix_narrow = np.logical_and(
            pix_x >= margin_w,
            np.logical_and(
                pix_x < (im_w - margin_w),
                np.logical_and(
                    pix_y >= margin_h,
                    np.logical_and(pix_y < im_h, pix_z > 0),
                ),
            ),
        )
        depth_val_narrow = np.zeros(pix_x.shape)
        depth_val_narrow[valid_pix_narrow] = depth_im[
            pix_y[valid_pix_narrow], pix_x[valid_pix_narrow]
        ]

        # Integrate TSDF
        depth_diff = depth_val - pix_z
        valid_pts = np.logical_and(depth_val > 0, depth_diff >= -self._trunc_margin)
        dist = np.maximum(-1, np.minimum(1, depth_diff / self._trunc_margin))
        valid_vox_x = self.vox_coords[valid_pts, 0]
        valid_vox_y = self.vox_coords[valid_pts, 1]
        valid_vox_z = self.vox_coords[valid_pts, 2]
        w_old = self._weight_vol_cpu[valid_vox_x, valid_vox_y, valid_vox_z]

        

        depth_diff_narrow = depth_val_narrow - pix_z
        valid_pts_narrow = np.logical_and(
            depth_val_narrow > 0, depth_diff_narrow >= -self._trunc_margin
        )
        valid_vox_x_narrow = self.vox_coords[valid_pts_narrow, 0]
        valid_vox_y_narrow = self.vox_coords[valid_pts_narrow, 1]
        valid_vox_z_narrow = self.vox_coords[valid_pts_narrow, 2]
        
        
        if w_new is None:
            tsdf_vals = self._tsdf_vol_cpu[valid_vox_x, valid_vox_y, valid_vox_z]
            valid_dist = dist[valid_pts]
            tsdf_vol_new, w_new = TSDFPlanner.integrate_tsdf(
                tsdf_vals, valid_dist, w_old, obs_weight
            )
            self._weight_vol_cpu[valid_vox_x, valid_vox_y, valid_vox_z] = w_new
            self._tsdf_vol_cpu[valid_vox_x, valid_vox_y, valid_vox_z] = tsdf_vol_new

            # Mark explored
            self._explore_vol_cpu[
                valid_vox_x_narrow, valid_vox_y_narrow, valid_vox_z_narrow
            ] = 1
        
            # Integrate color
            old_color = self._color_vol_cpu[valid_vox_x, valid_vox_y, valid_vox_z]
            old_b = np.floor(old_color / self._color_const)
            old_g = np.floor((old_color - old_b * self._color_const) / 256)
            old_r = old_color - old_b * self._color_const - old_g * 256
            new_color = color_im[pix_y[valid_pts], pix_x[valid_pts]]
            new_b = np.floor(new_color / self._color_const)
            new_g = np.floor((new_color - new_b * self._color_const) / 256)
            new_r = new_color - new_b * self._color_const - new_g * 256
            new_b = np.minimum(
                255.0, np.round((w_old * old_b + obs_weight * new_b) / w_new)
            )
            new_g = np.minimum(
                255.0, np.round((w_old * old_g + obs_weight * new_g) / w_new)
            )
            new_r = np.minimum(
                255.0, np.round((w_old * old_r + obs_weight * new_r) / w_new)
            )
            self._color_vol_cpu[valid_vox_x, valid_vox_y, valid_vox_z] = (
                new_b * self._color_const + new_g * 256 + new_r
            )

        # Integrate semantics if specified
        if sem_im is not None:
            old_sem = self._val_vol_cpu[valid_vox_x, valid_vox_y, valid_vox_z]
            new_sem = sem_im[pix_y[valid_pts], pix_x[valid_pts]]
            new_sem = (w_old * old_sem + obs_weight * new_sem) / w_new
            self._val_vol_cpu[valid_vox_x, valid_vox_y, valid_vox_z] = new_sem
        return w_new

    def get_volume(self):
        return self._tsdf_vol_cpu, self._color_vol_cpu

    def get_point_cloud(self):
        """Extract a point cloud from the voxel volume."""
        tsdf_vol, color_vol = self.get_volume()

        # Marching cubes
        # verts = measure.marching_cubes(tsdf_vol, level=0, method='lewiner')[0]
        # See: https://github.com/andyzeng/tsdf-fusion-python/issues/24
        verts = measure.marching_cubes(
            tsdf_vol, mask=np.logical_and(tsdf_vol > -0.5, tsdf_vol < 0.5), level=0
        )[0]
        verts_ind = np.round(verts).astype(int)
        verts = verts * self._voxel_size + self._vol_origin

        # Get vertex colors
        rgb_vals = color_vol[verts_ind[:, 0], verts_ind[:, 1], verts_ind[:, 2]]
        colors_b = np.floor(rgb_vals / self._color_const)
        colors_g = np.floor((rgb_vals - colors_b * self._color_const) / 256)
        colors_r = rgb_vals - colors_b * self._color_const - colors_g * 256
        colors = np.floor(np.asarray([colors_r, colors_g, colors_b])).T
        colors = colors.astype(np.uint8)

        pc = np.hstack([verts, colors])
        return pc

    def get_mesh(self):
        """Compute a mesh from the voxel volume using marching cubes."""
        tsdf_vol, color_vol = self.get_volume()

        # Marching cubes
        # verts, faces, norms, vals = measure.marching_cubes(tsdf_vol, level=0, method='lewiner')
        # See: https://github.com/andyzeng/tsdf-fusion-python/issues/24
        verts, faces, norms, vals = measure.marching_cubes(
            tsdf_vol, mask=np.logical_and(tsdf_vol > -0.5, tsdf_vol < 0.5), level=0
        )
        verts_ind = np.round(verts).astype(int)
        verts = verts * self._voxel_size + self._vol_origin

        # Get vertex colors
        rgb_vals = color_vol[verts_ind[:, 0], verts_ind[:, 1], verts_ind[:, 2]]
        colors_b = np.floor(rgb_vals / self._color_const)
        colors_g = np.floor((rgb_vals - colors_b * self._color_const) / 256)
        colors_r = rgb_vals - colors_b * self._color_const - colors_g * 256
        colors = np.floor(np.asarray([colors_r, colors_g, colors_b])).T
        colors = colors.astype(np.uint8)
        return verts, faces, norms, colors

    ############# For building semantic map and exploration #############

    def find_prompt_points_within_view(
        self,
        pts,
        im_w,
        im_h,
        cam_intr,
        cam_pose,
        height=0.4,
        cluster_threshold=1.0,
        num_prompt_points=3,
        num_max_unoccupied=300,
        min_points_for_clustering=3,
        point_min_dist=2,
        point_max_dist=10,
        cam_offset=0.5,
        frontier_min_neighbors=3,
        frontier_max_neighbors=4,
        **kwargs,
    ):
        """Find locations within view, which will then be prompted with VLM to get their semantic values.
        Locations include:
            (1) frontiers within view
            (2) empty locations that are sufficiently far from the current point within view
        """
        cur_point = self.world2vox(pts)
        island, unoccupied = self.get_island_around_pts(pts, height=height)
        unexplored = (np.sum(self._explore_vol_cpu, axis=-1) == 0).astype(int)
        for point in self.init_points:
            unexplored[point[0], point[1]] = 0
        occupied = np.logical_not(unoccupied).astype(int)
        cam_pose = cam_pose @ np.array(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, cam_offset],
                [0, 0, 0, 1],
            ]
        )
        mask = self.get_current_view_mask(
            cam_intr, cam_pose, im_w, im_h, slack=0, margin_h=100, margin_w=30
        )

        ############## Get unoccupied reachable points in view ##############

        # Mask the unoccupied region to be only the current view
        unoccupied_in_view = np.multiply(unoccupied, mask)
        # unoccupied_reachable_in_view2 = np.argwhere((island) & (unoccupied_in_view))

        unoccupied_reachable = np.argwhere((island) & (unoccupied))

        unoccupied_reachable_in_view = np.argwhere((island) & (unoccupied_in_view))

        # Subsample - weigh closer points more
        if len(unoccupied_reachable_in_view) > 0:
            subsample_inds = np.random.choice(
                range(len(unoccupied_reachable_in_view)),
                min(num_max_unoccupied, len(unoccupied_reachable_in_view)),
                replace=False,
            )
            unoccupied_reachable_in_view = unoccupied_reachable_in_view[subsample_inds]
        else:
            unoccupied_reachable_in_view = np.empty((0, 2))

        # Check unoccupied between point and current point - skip if any occupied
        unoccupied_reachable_in_view_new = np.empty((0, 2))
        for point in unoccupied_reachable_in_view:
            if not self.check_occupied_between(point, cur_point, occupied, threshold=1):
                unoccupied_reachable_in_view_new = np.concatenate(
                    (unoccupied_reachable_in_view_new, [point]), axis=0
                )
        unoccupied_reachable_in_view = unoccupied_reachable_in_view_new

        # Only keep points within desired range
        if len(unoccupied_reachable_in_view) > 0:
            dist_all = np.sqrt(
                (unoccupied_reachable_in_view[:, 0] - cur_point[0]) ** 2
                + (unoccupied_reachable_in_view[:, 1] - cur_point[1]) ** 2
            )
            unoccupied_reachable_in_view = unoccupied_reachable_in_view[
                (dist_all > point_min_dist / self._voxel_size)
                & (dist_all < point_max_dist / self._voxel_size)
            ]
            dist_all = dist_all[
                (dist_all > point_min_dist / self._voxel_size)
                & (dist_all < point_max_dist / self._voxel_size)
            ]

        ################## Get frontiers in view ##################

        # Get unexplored region - mark points around init points to be explored
        kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
        unexplored_neighbors = ndimage.convolve(
            unexplored, kernel, mode="constant", cval=0.0
        )
        unoccupied_neighbors = ndimage.convolve(
            unoccupied.astype(int), kernel, mode="constant", cval=0.0
        )
        frontiers_in_view = np.empty((0, 2))

        ################## Combine points, cluster ##################
        candidates_pre_cluster = np.concatenate(
            [frontiers_in_view, unoccupied_reachable_in_view], axis=0
        )
        frontiers = np.argwhere(
            island
            & (unexplored_neighbors >= frontier_min_neighbors)
            & (unexplored_neighbors <= frontier_max_neighbors)
        )
        frontiers_unoccupied = np.argwhere(
            island
            & (unoccupied_neighbors >= frontier_min_neighbors)
            & (unoccupied_neighbors <= frontier_max_neighbors)
        )

        height_voxel = int(height / self._voxel_size) + self.min_height_voxel
        frontier_obstacle = np.argwhere(
            island
            & np.logical_and(
            self._tsdf_vol_cpu[:, :, height_voxel] < 0.9, self._tsdf_vol_cpu[:, :, height_voxel] > 0, self._tsdf_vol_cpu[:, :, 0] < 0
        ))
        
        # Convert back to world coordinates
        unoccupied_reachable_normal = unoccupied_reachable * self._voxel_size + self._vol_origin[:2]
        frontiers_normal = frontiers * self._voxel_size + self._vol_origin[:2]
        frontiers_unoccupied_normal = frontier_obstacle * self._voxel_size + self._vol_origin[:2]
        self._rr_logger.log_2d_frontier_data(unoccupied*255, unexplored*255, self._tsdf_vol_cpu[:, :, height_voxel]*255, cur_point, candidates_pre_cluster, unexplored_neighbors, frontiers)
        
        unoccupied_reachable_normal = np.concatenate([unoccupied_reachable_normal, np.full((unoccupied_reachable_normal.shape[0],1), pts[2]+height)],1)
        frontiers_normal = np.concatenate([frontiers_normal, np.full((frontiers_normal.shape[0],1), pts[2]+height)],1)
        frontiers_unoccupied_normal = np.concatenate([frontiers_unoccupied_normal, np.full((frontiers_unoccupied_normal.shape[0],1), pts[2]+height)],1)

        self._rr_logger.log_3d_frontier_data(unoccupied_reachable_normal, frontiers_normal, frontiers_unoccupied_normal)


        # Save global info
        self.cur_point, self.island, self.unexplored = cur_point, island, unexplored
        self.unoccupied, self.occupied = unoccupied, occupied
        self.unexplored_neighbors = unexplored_neighbors

        return unoccupied_reachable_normal, frontiers_normal

    def update(
        self,
        color_im,
        depth_im,
        pts,
        cam_pose,
        obs_weight=1.0,
    ):
        self.integrate(
            color_im=color_im,
            depth_im=depth_im,
            cam_intr=self._cam_intr,
            cam_pose=cam_pose,
            obs_weight=obs_weight,
            margin_h=self._margin_h,
            margin_w=self._margin_w,
        )
        self.cur_pos = pts.copy()
        self.cur_point = self.world2vox(pts)
        island, unoccupied = self.get_island_around_pts_all_heights(pts, height=self._height_offset)
        unexplored = (np.sum(self._explore_vol_cpu, axis=-1) == 0).astype(int)
        for point in self.init_points:
            unexplored[point[0], point[1]] = 0
        occupied = np.logical_not(unoccupied).astype(int)
        explored = np.logical_not(unexplored).astype(int)

        explored_reachable = np.argwhere((island) & (explored))
        explored_reachable_img = np.logical_and(island, explored).astype(int)

        unoccupied_reachable = np.argwhere((island) & (unoccupied))
        unoccupied_reachable_img = np.logical_and(island, unoccupied).astype(int)

        kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
        unexplored_neighbors = ndimage.convolve(
            unexplored, kernel, mode="constant", cval=0.0
        )
        unoccupied_neighbors = ndimage.convolve(
            unoccupied.astype(int), kernel, mode="constant", cval=0.0
        )

        frontiers_unexplored = np.argwhere(
            island
            & (unexplored_neighbors >= self._frontier_min_neighbors)
            & (unexplored_neighbors <= self._frontier_max_neighbors)
        )
        
        
        # Logging 2D images
        # self._rr_logger.log_2d_frontier_data(
        #     unoccupied_reachable_img*255, 
        #     unexplored*255, 
        #     self._tsdf_vol_cpu[:, :, self._height_voxel]*255, 
        # )

        # Convert back to world coordinates
        unoccupied_reachable_normal = unoccupied_reachable * self._voxel_size + self._vol_origin[:2]
        frontiers_unexplored_normal = frontiers_unexplored * self._voxel_size + self._vol_origin[:2]
        explored_reachable_normal = explored_reachable * self._voxel_size + self._vol_origin[:2]
        
        unoccupied_reachable_normal = np.concatenate([unoccupied_reachable_normal, np.full((unoccupied_reachable_normal.shape[0],1), pts[2]+self._height_offset)],1)
        self.frontiers_unexplored_normal = np.concatenate([frontiers_unexplored_normal, np.full((frontiers_unexplored_normal.shape[0],1), pts[2]+self._height_offset)],1)
        explored_reachable_normal = np.concatenate([explored_reachable_normal, np.full((explored_reachable_normal.shape[0],1), pts[2]+self._height_offset)],1)

        # self._rr_logger.log_3d_frontier_data(unoccupied_reachable_normal, frontiers_unexplored_normal, explored_reachable_normal)

        clustered_frontiers = self.cluster_frontiers(frontiers_unexplored)

        dist_all = np.sqrt(
            (clustered_frontiers[:, 0] - self.cur_point[0]) ** 2
            + (clustered_frontiers[:, 1] - self.cur_point[1]) ** 2
        )
        frontiers_within_limit = clustered_frontiers[
            (dist_all > self._point_min_dist / self._voxel_size)
            & (dist_all < self._point_max_dist / self._voxel_size)
        ]

        self.frontiers_to_sample = frontiers_within_limit

        self.explored_reachable_img = explored_reachable_img # For planning

        # For sampling goal
        self.frontier_to_sample_normal = self.frontiers_to_sample * self._voxel_size + self._vol_origin[:2] 
        self.frontier_to_sample_normal = np.concatenate([self.frontier_to_sample_normal, np.full((self.frontier_to_sample_normal.shape[0],1), pts[2]+self._height_offset)],1)
        
        rr.log(f"world/tsdf_unoccupied", rr.Points3D(unoccupied_reachable_normal, colors=[255, 0, 0], radii=0.06))
        rr.log(f"world/tsdf_explored", rr.Points3D(explored_reachable_normal, colors=[200, 180, 150], radii=0.08))
        rr.log(f"world/tsdf_frontiers", rr.Points3D(self.frontiers_unexplored_normal, colors=[255, 255, 255], radii=0.08))
        rr.log(f"world/tsdf_frontiers_clustered", rr.Points3D(self.frontier_to_sample_normal, colors=[0, 0, 0], radii=0.11))
    
    def cluster_frontiers(self, frontiers):
        # cluster, or return none
        if len(frontiers) < self._cfg.visual_prompt.min_points_for_clustering:
            return frontiers
        else:
            clusters = fps(frontiers, min(self._cfg.visual_prompt.num_clusters, len(frontiers)-1))
            # merge clusters if too close to each other
            clusters_new = np.empty((0, 2))
            for cluster in clusters:
                if len(clusters_new) == 0:
                    clusters_new = np.vstack((clusters_new, cluster))
                else:
                    clusters_array = np.array(clusters_new)
                    dist = np.sqrt(np.sum((clusters_array - cluster) ** 2, axis=1))
                    if np.min(dist) > self._cfg.visual_prompt.cluster_threshold / self._voxel_size:
                        clusters_new = np.vstack((clusters_new, cluster))
        return clusters_new
    
    def sample_frontier(self):
        
        sample_idx = np.random.choice(self.frontiers_to_sample.shape[0], size=1, replace=False)[0]
        frontier_sample = self.frontiers_to_sample[sample_idx]
        frontier_normal = frontier_sample * self._voxel_size + self._vol_origin[:2]
        frontier_normal = np.append(frontier_normal, self.cur_pos[2]+self._height_offset)

        path = run_dijkstra(np.logical_not(self.explored_reachable_img.astype(bool)), self.cur_point, frontier_sample)

        path_normal = np.array(path) * self._voxel_size + self._vol_origin[:2]
        # Apply Gaussian smoothing (sigma controls smoothness)
        # smoothed_trajectory_x = gaussian_filter1d(path_normal[:, 0], sigma=1)
        # smoothed_trajectory_y = gaussian_filter1d(path_normal[:, 1], sigma=1)
        # path_normal = np.vstack((smoothed_trajectory_x, smoothed_trajectory_y)).T

        path_normal = np.concatenate([path_normal, np.full((path_normal.shape[0],1), self.cur_pos[2]+self._height_offset)],1)

        # self._rr_logger.log_traj_data(path_normal)
        # self._rr_logger.log_target_poses(frontier_normal)
        return path_normal, frontier_normal

    def path_to_frontier(self, frontier_normal):
        frontier_sample = self.world2vox(frontier_normal)
        frontier_normal = frontier_sample[:2] * self._voxel_size + self._vol_origin[:2]
        frontier_normal = np.append(frontier_normal, self.cur_pos[2]+self._height_offset)

        path = run_dijkstra(np.logical_not(self.explored_reachable_img.astype(bool)), self.cur_point, frontier_sample)

        path_normal = np.array(path) * self._voxel_size + self._vol_origin[:2]
        # Apply Gaussian smoothing (sigma controls smoothness)
        # smoothed_trajectory_x = gaussian_filter1d(path_normal[:, 0], sigma=1)
        # smoothed_trajectory_y = gaussian_filter1d(path_normal[:, 1], sigma=1)
        # path_normal = np.vstack((smoothed_trajectory_x, smoothed_trajectory_y)).T

        path_normal = np.concatenate([path_normal, np.full((path_normal.shape[0],1), self.cur_pos[2]+self._height_offset)],1)

        # self._rr_logger.log_traj_data(path_normal)
        # self._rr_logger.log_target_poses(frontier_normal)
        return path_normal
    
    def find_next_pose(
        self,
        pts,
        angle,
        flag_no_val_weight=False,
        unexplored_T=0.5,
        unoccupied_T=3,
        val_T=0.5,
        val_dir_T=0.5,
        dist_T=10,
        min_dist_from_cur=0.5,
        max_dist_from_cur=3,
        frontier_spacing=1.5,
        frontier_min_neighbors=3,
        frontier_max_neighbors=4,
        max_unexplored_check_frontier=3.0,
        max_unoccupied_check_frontier=1.0,
        max_val_check_frontier=5.0,
        smooth_sigma=5,
        eps=0.5,
        **kwargs,
    ):
        """Determine the next frontier to traverse to with semantic-value-weighted sampling."""
        cur_point = self.world2vox(pts)
        if hasattr(self, "cur_point"):
            island = self.island
            unoccupied, occupied = self.unoccupied, self.occupied
            unexplored, unexplored_neighbors = (
                self.unexplored,
                self.unexplored_neighbors,
            )
        else:
            island, unoccupied = self.get_island_around_pts(pts, height=0.4)
            occupied = np.logical_not(unoccupied).astype(int)
            unexplored = (np.sum(self._explore_vol_cpu, axis=-1) == 0).astype(int)
            for point in self.init_points:
                unexplored[point[0], point[1]] = 0
            kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
            unexplored_neighbors = ndimage.convolve(
                unexplored, kernel, mode="constant", cval=0.0
            )
        self.unexplored_neighbors = unexplored_neighbors
        self.unoccupied = unoccupied

        # get semantic map by taking max over z
        val_vol_2d = np.max(self._val_vol_cpu, axis=2).copy()

        # smoothen the map
        val_vol_2d = gaussian_filter(val_vol_2d, sigma=smooth_sigma)

        # use nonzero_val_vol quantile for frontier normalization
        frontiers = np.argwhere(
            island
            & (unexplored_neighbors >= frontier_min_neighbors)
            & (unexplored_neighbors <= frontier_max_neighbors)
        )
        frontiers_pre_cluster = frontiers.copy()

        # Fit frontiers
        if len(frontiers) > 10:
            db = DBSCAN(eps=eps, min_samples=2).fit(frontiers)
            labels = db.labels_
            # get one point from each cluster
            frontiers_new = np.empty((0, 2))
            for label in np.unique(labels):
                if label == -1:
                    continue
                cluster = frontiers[labels == label]
                # take the one that is closest to mean
                dist = np.sqrt(
                    (cluster[:, 0] - np.mean(cluster[:, 0])) ** 2
                    + (cluster[:, 1] - np.mean(cluster[:, 1])) ** 2
                )
                center = cluster[np.argmin(dist)]
                frontiers_new = np.append(frontiers_new, [center], axis=0)
            frontiers = frontiers_new.astype(int)

        # subsample
        frontiers_weight = np.zeros((len(frontiers)))

        # Commit
        point_type = "current"
        if self.target_point is None:

            # Get weights for frontiers
            frontiers_weight = np.empty((0))
            frontiers_new = np.empty((0, 2))
            # start_time = time.time(
            for point in frontiers:

                # find normal into unexplored
                normal = self.find_normal_into_space(point, unexplored, unexplored)

                # Then check how much unoccupied in that direction
                max_pixel_check = int(max_unoccupied_check_frontier / self._voxel_size)
                dir_pts = np.round(
                    point + np.arange(max_pixel_check)[:, np.newaxis] * normal
                ).astype(int)
                dir_pts = self.clip_2d_array(dir_pts)
                unoccupied_rate = (
                    np.sum(unoccupied[dir_pts[:, 0], dir_pts[:, 1]] == 1)
                    / max_pixel_check
                )

                # Check the radio of unexplored in the direction, until hits obstacle
                max_pixel_check = int(max_unexplored_check_frontier / self._voxel_size)
                dir_pts = np.round(
                    point + np.arange(max_pixel_check)[:, np.newaxis] * normal
                ).astype(int)
                dir_pts = self.clip_2d_array(dir_pts)
                unexplored_rate = (
                    np.sum(unexplored[dir_pts[:, 0], dir_pts[:, 1]] == 1)
                    / max_pixel_check
                )

                # Check value in the direction
                max_pixel_check = int(max_val_check_frontier / self._voxel_size)
                dir_pts = np.round(
                    point + np.arange(max_pixel_check)[:, np.newaxis] * normal
                ).astype(int)
                dir_pts = self.clip_2d_array(dir_pts)
                val_vol_2d_dir = val_vol_2d[dir_pts[:, 0], dir_pts[:, 1]]
                # keep non zero value only
                val_vol_2d_dir = val_vol_2d_dir[val_vol_2d_dir > 0]
                if len(val_vol_2d_dir) == 0:
                    val = 0
                else:
                    val = np.mean(val_vol_2d_dir)

                # Get weight - unexplored, unoccupied, and value
                weight = np.exp(unexplored_rate / unexplored_T)  # [0-1] before T
                weight *= np.exp(unoccupied_rate / unoccupied_T)  # [0-1] before T
                if not flag_no_val_weight:
                    weight *= np.exp(
                        val_vol_2d[point[0], point[1]] / val_T
                    )  # [0-1] before T
                    weight *= np.exp(val / val_dir_T)  # [0-1] before T

                # Check distance to current point - make weight very small if too close and aligned
                dist = (
                    np.sqrt(
                        (cur_point[0] - point[0]) ** 2 + (cur_point[1] - point[1]) ** 2
                    )
                    * self._voxel_size
                )
                pts_angle = np.arctan2(normal[1], normal[0]) - np.pi / 2
                weight *= np.exp(-dist / dist_T)
                if (
                    dist < min_dist_from_cur / self._voxel_size
                    and np.abs(angle - pts_angle) < np.pi / 6
                ):
                    weight *= 1e-3

                # Save weight
                frontiers_weight = np.append(frontiers_weight, weight)
                frontiers_new = np.concatenate((frontiers_new, [point]), axis=0)
            frontiers = frontiers_new.astype(int)
            logging.info(f"Number of frontiers for next pose: {len(frontiers)}")

            # raise frontier value if there is frontier
            if len(frontiers) > 0:
                logging.info(
                    f"Mean and std of frontier weight: {np.mean(frontiers_weight):.3f},"
                    f" {np.std(frontiers_weight):.3f}"
                )
                point_type = "frontier"

                # take best point until it satisfies condition
                max_try = 50
                cnt_try = 0
                while 1:
                    cnt_try += 1
                    if cnt_try > max_try:
                        point_type = "current"
                        break
                    frontiers_weight_red = frontiers_weight / np.mean(
                        frontiers_weight
                    )  # prevent overflowing
                    frontier_ind = np.random.choice(
                        range(len(frontiers)),
                        p=frontiers_weight_red / np.sum(frontiers_weight_red),
                    )
                    logging.info(f"weight: {frontiers_weight[frontier_ind]:.3f}")
                    max_point = frontiers[frontier_ind]

                    # find the direction into unexplored
                    direction = self.find_normal_into_space(
                        max_point, unexplored, unexplored
                    )

                    # Move back in the opposite direction of the normal by spacing, so the robot can see the frontier
                    # there is a chance that the point is outside the free space
                    next_point = np.array(max_point, dtype=float)
                    max_backtrack = int(frontier_spacing / self._voxel_size)
                    min_backtrack = 2
                    num_backtrack = 0
                    while 1:
                        next_point -= direction
                        num_backtrack += 1
                        if num_backtrack >= max_backtrack:
                            break

                        # break if close to boundary
                        if not self.check_within_bnds(next_point):
                            break

                        # break if occupied
                        if (
                            occupied[int(next_point[0]), int(next_point[1])]
                            or not island[int(next_point[0]), int(next_point[1])]
                        ):
                            next_point += 2 * direction
                            break
                    next_point = np.round(next_point).astype(int)
                    if (
                        num_backtrack >= min_backtrack
                        and self.check_within_bnds(next_point)
                        and island[int(next_point[0]), int(next_point[1])]
                    ):
                        break  # stop searching

            # no patch used
            if point_type == "current":
                logging.info("No patches, return current point and random direction")
                max_point = cur_point[:2]
                next_point = cur_point[:2]
                direction = np.random.rand(2) - 0.5
                direction = direction / np.linalg.norm(direction)
        else:
            point_type = "commit"
            next_point = self.target_point.copy()
            direction = self.target_direction.copy()
            max_point = self.max_point.copy()
        logging.info(f"Next pose type: {point_type}")

        # Check if the point is beyond the max dist. Note that not using dist from dijkstra for saving time, then not taking account into obstacles when calculating distance
        dist = np.sqrt(
            (next_point[0] - cur_point[0]) ** 2 + (next_point[1] - cur_point[1]) ** 2
        )
        if dist > max_dist_from_cur / self._voxel_size:
            self.target_point = next_point.copy()
            self.target_direction = direction.copy()
            self.max_point = max_point.copy()

            island_free = np.logical_not(island)  # 0 for free
            path = run_dijkstra(island_free, cur_point, next_point)
            max_num = min(int(max_dist_from_cur / self._voxel_size), len(path) - 1)
            next_point = np.array(path[max_num])
            direction = max_point - next_point  # direction to the max point
            direction = direction / np.linalg.norm(direction)
            logging.info(
                f"Current {cur_point[:2]}, target {self.target_point}, move to"
                f" {next_point}"
            )
        if dist <= max_dist_from_cur / self._voxel_size or max_num == len(path) - 1:
            self.target_point = None
            self.target_direction = None
            self.max_point = None

        # Plot
        fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(20, 18))
        ax1.imshow(unoccupied)
        ax1.scatter(max_point[1], max_point[0], c="r", s=30, label="max")
        ax1.scatter(cur_point[1], cur_point[0], c="b", s=30, label="current")
        ax1.scatter(next_point[1], next_point[0], c="g", s=30, label="actual")
        ax1.set_title("Unoccupied")
        ax2.imshow(island)
        ax2.set_title("Island")
        ax3.imshow(unexplored_neighbors)
        for point in frontiers_pre_cluster:
            ax3.scatter(point[1], point[0], color="white", s=20, alpha=1)
        ax3.set_title("Unexplored neighbors")
        im = ax4.imshow(val_vol_2d)
        for point in frontiers:
            ax4.scatter(point[1], point[0], color="white", s=20, alpha=1)
        fig.colorbar(im, orientation="vertical", ax=ax4, fraction=0.046, pad=0.04)
        ax4.scatter(max_point[1], max_point[0], c="r", s=30, label="max")
        ax4.scatter(cur_point[1], cur_point[0], c="b", s=30, label="current")
        ax4.scatter(next_point[1], next_point[0], c="g", s=30, label="actual")
        ax4.quiver(
            next_point[1],
            next_point[0],
            direction[1],
            direction[0],
            color="r",
            scale=5,
            angles="xy",
            alpha=0.2,
        )
        ax4.set_title("Current sem values")
        im = ax5.imshow(island)
        ax5.set_title("Path on island")
        frontier_weights = np.zeros_like(val_vol_2d)
        for point, weight in zip(frontiers, frontiers_weight):
            frontier_weights[point[0], point[1]] = weight
        im = ax6.imshow(frontier_weights)
        fig.colorbar(im, orientation="vertical", ax=ax6, fraction=0.046, pad=0.04)
        ax6.scatter(max_point[1], max_point[0], c="r", s=20, label="max")
        ax6.set_title("Frontier weights")

        # Convert back to world coordinates
        next_point_normal = next_point * self._voxel_size + self._vol_origin[:2]

        # Find the yaw angle again
        next_yaw = np.arctan2(direction[1], direction[0]) - np.pi / 2
        return next_point_normal, next_yaw, next_point, fig

    def get_island_around_pts(self, pts, fill_dim=0.4, height=0.4):
        """Find the empty space around the point (x,y,z) in the world frame"""
        # Convert to voxel coordinates
        cur_point = self.world2vox(pts)

        # Check if the height voxel is occupied
        height_voxel = int(height / self._voxel_size) + self.min_height_voxel
        unoccupied = np.logical_and(
            self._tsdf_vol_cpu[:, :, height_voxel] > 0, self._tsdf_vol_cpu[:, :, 0] < 0
        )  # check there is ground below

        # Set initial pose to be free
        for point in self.init_points:
            unoccupied[point[0], point[1]] = 1

        # filter small islands smaller than size 2x2 and fill in gap of size 2
        fill_size = int(fill_dim / self._voxel_size)
        structuring_element_close = np.ones((fill_size, fill_size)).astype(bool)
        unoccupied = close_operation(unoccupied, structuring_element_close)

        # Find the connected component closest to the current location is, if the current location is not free
        # this is a heuristic to determine reachable space, although not perfect
        islands = measure.label(unoccupied, connectivity=1)
        if unoccupied[cur_point[0], cur_point[1]] == 1:
            islands_ind = islands[cur_point[0], cur_point[1]]  # use current one
        else:
            # find the closest one - tbh, this should not happen, but it happens when the robot cannot see the space immediately in front of it because of camera height and fov
            y, x = np.ogrid[: unoccupied.shape[0], : unoccupied.shape[1]]
            dist_all = np.sqrt((x - cur_point[1]) ** 2 + (y - cur_point[0]) ** 2)
            dist_all[islands == islands[cur_point[0], cur_point[1]]] = np.inf
            island_coords = np.unravel_index(np.argmin(dist_all), dist_all.shape)
            islands_ind = islands[island_coords[0], island_coords[1]]
        island = islands == islands_ind
        return island, unoccupied

    def get_island_around_pts_all_heights(self, pts, fill_dim=0.6, height=0.4):
        """Find the empty space around the point (x,y,z) in the world frame"""
        # Convert to voxel coordinates
        cur_point = self.world2vox(pts)

        # Check if the height voxel is occupied
        height_voxel = int(height / self._voxel_size) + self.min_height_voxel
        
        min_height_voxel = max(int((height+self._range_height[0]) / self._voxel_size) + self.min_height_voxel, 3)
        max_height_voxel = int((height+self._range_height[1])/self._voxel_size) + self.min_height_voxel

        unoccupied = np.logical_and(
            np.min(self._tsdf_vol_cpu[:,:,min_height_voxel:max_height_voxel], axis=2) > 0, self._tsdf_vol_cpu[:, :, 0] < 0
        )  # check there is ground below

        # Set initial pose to be free
        for point in self.init_points:
            unoccupied[point[0], point[1]] = 1

        # filter small islands smaller than size 2x2 and fill in gap of size 2
        fill_size = int(fill_dim / self._voxel_size)
        structuring_element_close = np.ones((fill_size, fill_size)).astype(bool)
        unoccupied = close_operation(unoccupied, structuring_element_close)

        # Find the connected component closest to the current location is, if the current location is not free
        # this is a heuristic to determine reachable space, although not perfect
        islands = measure.label(unoccupied, connectivity=1)
        if unoccupied[cur_point[0], cur_point[1]] == 1:
            islands_ind = islands[cur_point[0], cur_point[1]]  # use current one
        else:
            # find the closest one - tbh, this should not happen, but it happens when the robot cannot see the space immediately in front of it because of camera height and fov
            y, x = np.ogrid[: unoccupied.shape[0], : unoccupied.shape[1]]
            dist_all = np.sqrt((x - cur_point[1]) ** 2 + (y - cur_point[0]) ** 2)
            dist_all[islands == islands[cur_point[0], cur_point[1]]] = np.inf
            island_coords = np.unravel_index(np.argmin(dist_all), dist_all.shape)
            islands_ind = islands[island_coords[0], island_coords[1]]
        island = islands == islands_ind
        return island, unoccupied

    def get_current_view_mask(
        self,
        cam_intr,
        cam_pose,
        im_w,
        im_h,
        slack=0,
        margin_h=0,
        margin_w=0,
    ):
        cam_pts = rigid_transform(self.cam_pts_pre, np.linalg.inv(cam_pose))
        pix_z = cam_pts[:, 2]
        pix = TSDFPlanner.cam2pix(cam_pts, cam_intr)
        pix_x, pix_y = pix[:, 0], pix[:, 1]
        valid_pix = np.logical_and(
            pix_x >= -slack + margin_w,
            np.logical_and(
                pix_x < (im_w + slack - margin_w),
                np.logical_and(
                    pix_y >= -slack + margin_h,
                    np.logical_and(pix_y < im_h + slack, pix_z > 0),
                ),
            ),
        )
        # make a 2D mask where valid pix is 1 and 0 otherwise
        valid_pix = valid_pix.reshape(self._vol_dim).astype(int)
        mask = np.max(valid_pix, axis=2)  # take the max over height (z)
        return mask

    def check_occupied_between(self, p1, p2, occupied, threshold):
        direction = np.array([p2[0] - p1[0], p2[1] - p1[1]]).astype(float)
        num_points = int(np.linalg.norm(direction))
        dir_norm = direction / np.linalg.norm(direction)
        points_between = (
            p1[:2] + dir_norm * np.arange(num_points + 1)[:, np.newaxis]
        ).astype(int)
        points_occupied = np.sum(occupied[points_between[:, 0], points_between[:, 1]])
        return points_occupied > threshold

    def check_within_bnds(self, pts, slack=0):
        return not (
            pts[0] <= slack
            or pts[0] >= self._vol_dim[0] - slack
            or pts[1] <= slack
            or pts[1] >= self._vol_dim[1] - slack
        )

    def clip_2d_array(self, array):
        return array[
            (array[:, 0] >= 0)
            & (array[:, 0] < self._vol_dim[0])
            & (array[:, 1] >= 0)
            & (array[:, 1] < self._vol_dim[1])
        ]

    def find_normal_into_space(self, point, island, space, num_check=10):
        """Find the normal direction into the space"""
        normal = find_normal(
            island.astype(int), point[0], point[1]
        )  # but normal is ambiguous, so need to find which direction is unoccupied
        dir_1 = (point + np.arange(num_check)[:, np.newaxis] * normal).astype(int)
        dir_2 = (point - np.arange(num_check)[:, np.newaxis] * normal).astype(int)
        dir_1 = self.clip_2d_array(dir_1)
        dir_2 = self.clip_2d_array(dir_2)
        dir_1_occupied = np.sum(space[dir_1[:, 0], dir_1[:, 1]])
        dir_2_occupied = np.sum(space[dir_2[:, 0], dir_2[:, 1]])
        direction = normal
        if dir_1_occupied < dir_2_occupied:
            direction *= -1
        elif dir_1_occupied == dir_2_occupied:  # randomly choose one
            if random.random() < 0.5:
                direction *= -1
        return direction
