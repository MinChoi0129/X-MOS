# Adapted from 4DMOS (https://github.com/PRBonn/4DMOS), MIT License, see LICENSE
import numpy as np
from kiss_icp.config import KISSConfig
from kiss_icp.kiss_icp import KissICP

from core.config import DataConfig, OdometryConfig


def parse_config(config_data: DataConfig, config_odometry: OdometryConfig):
    kiss_config = KISSConfig()
    kiss_config.data.deskew = config_data.deskew
    kiss_config.data.max_range = config_data.max_range
    kiss_config.data.min_range = config_data.min_range
    kiss_config.mapping.voxel_size = config_odometry.voxel_size
    kiss_config.mapping.max_points_per_voxel = config_odometry.max_points_per_voxel
    kiss_config.adaptive_threshold.initial_threshold = config_odometry.initial_threshold
    kiss_config.adaptive_threshold.min_motion_th = config_odometry.min_motion_th
    return kiss_config


class Odometry(KissICP):
    def __init__(
        self,
        config_data: DataConfig,
        config_odometry: OdometryConfig,
    ):
        if config_data.deskew:
            raise NotImplementedError("Deskewing is not supported, the scans have no per-point timestamps")
        kiss_config = parse_config(config_data, config_odometry)
        super().__init__(kiss_config)

    def register_points(self, points, timestamps, scan_index):
        # Apply motion compensation
        points_prep = self.preprocessor.preprocess(points, timestamps, self.last_delta)

        # Voxelize
        source, points_downsample = self.voxelize(points_prep)

        # Get motion prediction and adaptive_threshold
        sigma = self.adaptive_threshold.get_threshold()

        # Compute initial_guess for ICP
        initial_guess = self.last_pose @ self.last_delta

        new_pose = self.registration.align_points_to_map(
            points=source,
            voxel_map=self.local_map,
            initial_guess=initial_guess,
            max_correspondance_distance=3 * sigma,
            kernel=sigma / 3,
        )

        # Compute the difference between the prediction and the actual estimate
        model_deviation = np.linalg.inv(initial_guess) @ new_pose

        # Update step: threshold, local map, delta, and the last pose
        self.adaptive_threshold.update_model_deviation(model_deviation)
        self.local_map.update(points_downsample, new_pose)
        self.last_delta = np.linalg.inv(self.last_pose) @ new_pose
        self.last_pose = new_pose

        return self.transform(points, self.last_pose)

    def transform(self, points, pose):
        points_hom = np.hstack((points, np.ones((len(points), 1))))
        points = (pose @ points_hom.T).T[:, :3]
        return points

    def current_pose(self):
        return self.last_pose

    def current_location(self):
        return self.last_pose[:3, 3]
