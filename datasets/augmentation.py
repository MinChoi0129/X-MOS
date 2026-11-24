# Adapted from 4DMOS (https://github.com/PRBonn/4DMOS), MIT License, see LICENSE
import numpy as np
import torch


def rotate_point_cloud(points):
    """Randomly rotate the point clouds to augment the dataset"""
    rotation_angle = torch.rand(1) * 2 * torch.pi
    cosval = torch.cos(rotation_angle)
    sinval = torch.sin(rotation_angle)
    rotation_matrix = torch.Tensor([[cosval, -sinval, 0], [sinval, cosval, 0], [0, 0, 1]])
    points = points @ rotation_matrix.type_as(points)
    return points


def rotate_perturbation_point_cloud(points, angle_sigma=0.2, angle_clip=0.5):
    """Randomly perturb the point clouds by small rotations"""
    angles = torch.clip(angle_sigma * torch.randn(3), -angle_clip, angle_clip)
    Rx = torch.Tensor(
        [
            [1, 0, 0],
            [0, torch.cos(angles[0]), -torch.sin(angles[0])],
            [0, torch.sin(angles[0]), torch.cos(angles[0])],
        ]
    )
    Ry = torch.Tensor(
        [
            [torch.cos(angles[1]), 0, torch.sin(angles[1])],
            [0, 1, 0],
            [-torch.sin(angles[1]), 0, torch.cos(angles[1])],
        ]
    )
    Rz = torch.Tensor(
        [
            [torch.cos(angles[2]), -torch.sin(angles[2]), 0],
            [torch.sin(angles[2]), torch.cos(angles[2]), 0],
            [0, 0, 1],
        ]
    )
    rotation_matrix = Rz @ Ry @ Rx
    points = points @ rotation_matrix.type_as(points)
    return points


def random_flip_point_cloud(points):
    """Randomly flip the point cloud. Flip is per points."""
    if torch.rand(1).item() > 0.5:
        points = torch.multiply(points, torch.tensor([-1, 1, 1]).type_as(points))
    if torch.rand(1).item() > 0.5:
        points = torch.multiply(points, torch.tensor([1, -1, 1]).type_as(points))
    return points


def random_scale_point_cloud(points, scale_low=0.8, scale_high=1.2):
    """Randomly scale the points."""
    scales = (scale_low - scale_high) * torch.rand(1, 3) + scale_high
    points = torch.multiply(points, scales.type_as(points))
    return points


def random_crop_mask(batch, crop_mean=15.0, crop_sigma=2.0):
    """Mask of the points [N, 6] = [batch_idx, x, y, z, t, label] inside a randomly sized x-y box
    that is centered at a randomly sampled point."""
    sample_point = batch[np.random.choice(range(len(batch))), 1:4]
    crop_x = np.random.normal(crop_mean, crop_sigma)
    crop_y = np.random.normal(crop_mean, crop_sigma)

    dist = torch.abs(batch[:, 1:4] - sample_point).reshape(-1, 3)
    mask = dist[:, 0] < crop_x
    mask = torch.logical_and(mask, dist[:, 1] < crop_y)
    return mask


def random_subsample_mask(num_points, max_dropout_ratio=0.5):
    """Mask that keeps each point with a random probability in [max_dropout_ratio, 1]."""
    dropout = (1 - max_dropout_ratio) * torch.rand(1) + max_dropout_ratio
    return torch.rand(num_points) < dropout
