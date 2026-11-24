# Adapted from 4DMOS (https://github.com/PRBonn/4DMOS), MIT License, see LICENSE
import glob
import os

import numpy as np
from kiss_icp.datasets.kitti import KITTIOdometryDataset


class SemanticKITTIDataset(KITTIOdometryDataset):
    def __init__(self, data_dir, sequence: str, *_, **__):
        self.sequence_id = sequence.zfill(2)
        self.kitti_sequence_dir = os.path.join(data_dir, "sequences", self.sequence_id)
        self.velodyne_dir = os.path.join(self.kitti_sequence_dir, "velodyne/")

        self.scan_files = sorted(glob.glob(self.velodyne_dir + "*.bin"))
        self.calibration = self.read_calib_file(os.path.join(self.kitti_sequence_dir, "calib.txt"))

        # Load GT Poses (if available)
        self.poses_fn = os.path.join(data_dir, f"poses/{self.sequence_id}.txt")
        if os.path.exists(self.poses_fn):
            self.gt_poses = self.load_poses(self.poses_fn)

        # Add correction for KITTI datasets, can be easilty removed if unwanted
        from kiss_icp.pybind import kiss_icp_pybind

        self.correct_kitti_scan = lambda frame: np.asarray(
            kiss_icp_pybind._correct_kitti_scan(kiss_icp_pybind._Vector3dVector(frame))
        )

        self.label_dir = os.path.join(self.kitti_sequence_dir, "labels/")
        self.label_files = sorted(glob.glob(self.label_dir + "*.label"))

    def __getitem__(self, idx):
        points, timestamps = self.scans(idx)
        labels = (
            self.read_labels(self.label_files[idx])
            if self.label_files
            else np.full((len(points), 1), -1, dtype=np.int32)
        )
        return points, timestamps, labels

    def read_labels(self, filename):
        """Load moving object labels from .label file"""
        orig_labels = np.fromfile(filename, dtype=np.int32).reshape((-1))
        orig_labels = orig_labels & 0xFFFF  # Mask semantics in lower half
        labels = np.zeros_like(orig_labels)
        labels[orig_labels <= 1] = -1  # Unlabeled (0), outlier (1)
        labels[orig_labels > 250] = 1  # Moving
        labels = labels.astype(dtype=np.int32).reshape(-1)
        return labels
