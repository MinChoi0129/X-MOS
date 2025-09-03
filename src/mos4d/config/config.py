# MIT License
#
# Copyright (c) 2022 Ignacio Vizzo, Tiziano Guadagnino, Benedikt Mersch, Cyrill
# Stachniss.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
from pathlib import Path
from typing import List

from pydantic import BaseModel, Field


class DataConfig(BaseModel):
    deskew: bool = False
    max_range: float = 100.0
    min_range: float = 3.0


class OdometryConfig(BaseModel):
    voxel_size: float = 0.5
    max_points_per_voxel: int = 20
    initial_threshold: float = 2.0
    min_motion_th: float = 0.1


class MOSConfig(BaseModel):
    voxel_size_mos: float = 0.1
    delay_mos: int = 10
    prior: float = 0.25
    max_range_mos: float = 50.0
    min_range_mos: float = 0.0


class TrainingConfig(BaseModel):
    id: str = "experiment_id"
    train: List[str] = Field(
        default_factory=lambda: [
            "00",
            "01",
            "02",
            "03",
            "04",
            "05",
            "06",
            "07",
            "09",
            "10",
        ]
    )

    #######################################################
    val: List[str] = Field(default_factory=lambda: ["08"])
    batch_size: int = 16
    # batch_size: int = 1  # Debugging
    accumulate_grad_batches: int = 1
    max_epochs: int = 100
    lr: float = 0.0001
    lr_epoch: int = 1
    lr_decay: float = 0.99
    weight_decay: float = 0.0001
    num_workers: int = 4
    #######################################################
    lambda_ce: float = 0.3
    lambda_kd: float = 0.7
    temperature: float = 3.0
    #######################################################
    # Distillation chain control
    # Example: [128, 64, 32, 16] and stage_index=0 -> 128->64, 1 -> 64->32, 2 -> 32->16
    distill_chain: List[int] = [128, 64, 32, 16]
    distill_stage_index: int = 2
    # Beam-down behavior
    beam_within_band_keep: float = 0.5
    beam_kmeans_iters: int = 10
    #######################################################
    keep_training: bool = True
    keep_training_weights: Path = (
        "/home/work/4DMOS/models/4DMOS/helimos_pseudo_ouster_for_velodyne/version_1/checkpoints/helimos_pseudo_ouster_for_velodyne_epoch=056_val_moving_iou=0.541.ckpt"
    )
    #######################################################
    is_distill_mode: bool = True
    O_teacher_weights: Path = (
        "/home/work/4DMOS/models/4DMOS/helimos_pseudo_ouster_for_velodyne/version_1/checkpoints/helimos_pseudo_ouster_for_velodyne_epoch=056_val_moving_iou=0.541.ckpt"
    )
    A_teacher_weights: Path = ""
    V_teacher_weights: Path = ""
    L_teacher_weights: Path = ""
    #######################################################
