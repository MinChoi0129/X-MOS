# Adapted from 4DMOS (https://github.com/PRBonn/4DMOS), MIT License, see LICENSE
from pathlib import Path
from typing import List, Optional

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class DataConfig(BaseModel):
    # Not supported (the scans have no per-point timestamps), kept because it is part of the cache key
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
    val: List[str] = Field(default_factory=lambda: ["08"])
    batch_size: int = 16
    accumulate_grad_batches: int = 1
    max_epochs: int = 100
    lr: float = 0.0001
    lr_epoch: int = 1
    lr_decay: float = 0.99
    weight_decay: float = 0.0001
    num_workers: int = 4

    # Weight transfer: initialize the network with the weights of a previous stage
    keep_training: bool = False
    keep_training_weights: Optional[Path] = None

    # Loss: L_total = lambda_ce * L_MOS + lambda_kd * L_KD (L_KD only in distill mode)
    lambda_ce: float = 0.3
    lambda_kd: float = 0.7
    temperature: float = 3.0

    # Knowledge distillation from the teachers that have weights configured.
    # kd_mode: "sensor_aware" (X-MOS), "attention" or "average" (ablations)
    is_distill_mode: bool = False
    kd_mode: str = "sensor_aware"
    O_teacher_weights: Optional[Path] = None
    A_teacher_weights: Optional[Path] = None
    V_teacher_weights: Optional[Path] = None
    L_teacher_weights: Optional[Path] = None

    # Multi-level adaptive teacher generation: train on pseudo low-beam data.
    # With distill_chain [128, 64, 32, 16], distill_stage_index 0 is 128 -> 64, 1 is 64 -> 32 and
    # 2 is 32 -> 16. The teacher sees the higher and the student the lower resolution of the stage.
    pseudo_beam_down: bool = False
    distill_chain: List[int] = [128, 64, 32, 16]
    distill_stage_index: int = 2
    beam_within_band_keep: float = 0.5
    beam_kmeans_iters: int = 10


class MOS4DConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="mos4d_")
    out_dir: str = "results"
    data: DataConfig = DataConfig()
    odometry: OdometryConfig = OdometryConfig()
    mos: MOSConfig = MOSConfig()
    training: TrainingConfig = TrainingConfig()


def load_config(config_file: Optional[Path]) -> MOS4DConfig:
    """Loads the configuration from an optional yaml file, everything else keeps its default."""
    data = None
    if config_file is not None:
        with open(config_file) as cfg_file:
            data = yaml.safe_load(cfg_file)
    return MOS4DConfig(**(data or {}))


def write_config(config: MOS4DConfig, filename: str):
    with open(filename, "w") as outfile:
        yaml.dump(config.model_dump(), outfile, default_flow_style=False)
