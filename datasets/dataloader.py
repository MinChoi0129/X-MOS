# Adapted from 4DMOS (https://github.com/PRBonn/4DMOS), MIT License, see LICENSE
import os
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset

from core.config import DataConfig, MOS4DConfig, OdometryConfig
from datasets import dataset_factory, sequence_dataloaders
from core.odometry import Odometry
from datasets.sensors import DEFAULT_SENSOR_INDEX, SENSOR_NAME_TO_INDEX
from core.cache import get_cache, memoize

# Name under which the samples are cached. It is the name the method had in 4DMOS, so that caches
# that were created before the code was restructured stay valid.
CACHE_KEY = "mos4d.datasets.mos4d_dataset.MOS4DDataset.get_past_point_clouds"


def collate_fn(batch):
    """Stacks the samples to a tensor [N, 6] = [batch_idx, x, y, z, t, label] and returns it
    together with the sensor index [N] of each point (the privileged information of X-MOS)."""
    tensor_batch, sensor_batch = None, None
    for i, (past_point_clouds, sensor_name) in enumerate(batch):
        ones = torch.ones(len(past_point_clouds), 1).type_as(past_point_clouds)
        tensor = torch.hstack([i * ones, past_point_clouds])
        tensor_batch = tensor if tensor_batch is None else torch.vstack([tensor_batch, tensor])

        sensor_index = SENSOR_NAME_TO_INDEX.get(sensor_name, DEFAULT_SENSOR_INDEX)
        sensor_tensor = torch.full((len(past_point_clouds),), sensor_index, dtype=torch.long)
        sensor_batch = sensor_tensor if sensor_batch is None else torch.cat([sensor_batch, sensor_tensor])

    return tensor_batch, sensor_batch


class MOS4DDataModule(LightningDataModule):
    """Training and validation set for Pytorch Lightning"""

    def __init__(self, dataloader: str, data_dir: Path, config: MOS4DConfig, cache_dir: Path):
        super(MOS4DDataModule, self).__init__()
        self.dataloader = dataloader
        self.data_dir = data_dir
        self.config = config
        self.cache_dir = cache_dir
        if self.cache_dir == None:
            print("No cache specified, therefore disabling shuffle during training!")
        self.shuffle = True if self.cache_dir is not None else False

        assert dataloader in sequence_dataloaders()

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        train_set = MOS4DDataset(self.dataloader, self.data_dir, self.config, self.config.training.train, self.cache_dir)
        val_set = MOS4DDataset(self.dataloader, self.data_dir, self.config, self.config.training.val, self.cache_dir)

        self.train_loader = DataLoader(
            dataset=train_set,
            batch_size=self.config.training.batch_size,
            collate_fn=collate_fn,
            shuffle=self.shuffle,
            num_workers=self.config.training.num_workers,
            pin_memory=True,
            persistent_workers=True,
            drop_last=False,
            timeout=0,
        )

        self.train_iter = iter(self.train_loader)

        self.valid_loader = DataLoader(
            dataset=val_set,
            batch_size=1,
            collate_fn=collate_fn,
            shuffle=False,
            num_workers=self.config.training.num_workers,
            pin_memory=True,
            persistent_workers=True,
            drop_last=False,
            timeout=0,
        )

        self.valid_iter = iter(self.valid_loader)

        print("Loaded {:d} training and {:d} validation samples.".format(len(train_set), len(val_set)))

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.valid_loader


class MOS4DDataset(Dataset):
    """Caches and returns scan and local maps for multiple sequences"""

    def __init__(
        self,
        dataloader: str,
        data_dir: Path,
        config: MOS4DConfig,
        sequences: list,
        cache_dir: Path,
    ):
        self.config = config
        self.sequences = sequences
        self._print = False

        # Cache
        if cache_dir is not None:
            directory = os.path.join(cache_dir, dataloader)
            self.use_cache = True
            self.cache = get_cache(directory=directory)
            print("Using cache at ", directory)
        else:
            self.use_cache = False
            self.cache = get_cache(directory=os.path.join(data_dir, "cache"))

        # Create datasets and map a sample index to the sequence and scan index
        self.datasets = {}
        self.idx_mapper = {}
        idx = 0
        for sequence in self.sequences:
            self.datasets[sequence] = dataset_factory(
                dataloader=dataloader,
                data_dir=data_dir,
                sequence=sequence,
            )
            for sample_idx in range(len(self.datasets[sequence])):
                self.idx_mapper[idx] = (sequence, sample_idx)
                idx += 1

        self.sequence = None
        self.odometry = Odometry(self.config.data, self.config.odometry)
        self.poses = []

    def __len__(self):
        return len(self.idx_mapper.keys())

    def __getitem__(self, idx):
        sequence, scan_index = self.idx_mapper[idx]
        batch = self.get_past_point_clouds(
            sequence,
            scan_index,
            self.config.mos.delay_mos,
            dict(self.config.data),
            dict(self.config.odometry),
        )
        return batch, sequence.split("/")[0]  # past point clouds, sensor name (e.g. "Ouster")

    @memoize(name=CACHE_KEY)
    def get_past_point_clouds(
        self,
        sequence: int,
        scan_index: int,
        n_past_scans: int,
        data_config_dict: Dict,
        odometry_config_dict: Dict,
    ):
        """Returns up to n_past_scans in local frame and labels."""
        if not self._print:
            print("*****Caching now*****")
            self._print = True

        scan_points, timestamps, scan_labels = self.datasets[sequence][scan_index]

        # Only consider valid points
        valid_mask = scan_labels != -1
        scan_points = scan_points[valid_mask]
        scan_labels = scan_labels[valid_mask]

        if self.sequence != sequence or len(scan_points) == 0:
            data_config = DataConfig().model_validate(data_config_dict)
            odometry_config = OdometryConfig().model_validate(odometry_config_dict)

            self.sequence = sequence
            self.odometry = Odometry(data_config, odometry_config)
            self.poses = []

        # Register
        self.odometry.register_points(scan_points, timestamps, scan_index)
        self.poses.append(self.odometry.last_pose)

        # Collect past point clouds
        list_past_points = []
        n_poses_available = len(self.poses)
        for index in range(0, min(n_poses_available, n_past_scans)):
            past_points, _, past_labels = self.datasets[sequence][scan_index - index]

            valid_mask = past_labels != -1
            past_labels = past_labels[valid_mask]
            past_points = past_points[valid_mask]

            past_pose = self.poses[-(index + 1)]
            past_points_transformed = self.odometry.transform(past_points, np.linalg.inv(self.odometry.last_pose) @ past_pose)

            list_past_points.append(
                np.hstack(
                    [
                        past_points_transformed,
                        (scan_index - index) * np.ones((len(past_labels), 1)),
                        past_labels.reshape(-1, 1),
                    ]
                )
            )
        return torch.tensor(np.vstack(list_past_points)).to(torch.float32).reshape(-1, 5)
