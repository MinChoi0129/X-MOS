# Adapted from 4DMOS (https://github.com/PRBonn/4DMOS), MIT License, see LICENSE
import numpy as np

from datasets.kitti import SemanticKITTIDataset


class SemanticApolloDataset(SemanticKITTIDataset):
    def __init__(self, data_dir, sequence: str, *_, **__):
        super().__init__(data_dir=data_dir, sequence=sequence, *_, **__)
        self.correct_kitti_scan = lambda frame: frame

    def get_frames_timestamps(self) -> np.ndarray:
        return np.arange(0, self.__len__(), 1.0)

    def read_labels(self, filename):
        """Load moving object labels from .label file"""
        orig_labels = np.fromfile(filename, dtype=np.int32).reshape((-1))
        orig_labels = orig_labels & 0xFFFF  # Mask semantics in lower half
        labels = np.zeros_like(orig_labels)
        labels[orig_labels > 250] = 1  # Moving
        labels = labels.astype(dtype=np.int32).reshape(-1)
        return labels
