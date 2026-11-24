# Adapted from 4DMOS (https://github.com/PRBonn/4DMOS), MIT License, see LICENSE
import os
from pathlib import Path

import numpy as np
from kiss_icp.metrics import absolute_trajectory_error, sequence_error
from kiss_icp.tools.pipeline_results import PipelineResults

from core.metrics import get_f1, get_iou, get_precision, get_recall, get_stats


class MOSPipelineResults(PipelineResults):
    def __init__(self) -> None:
        super().__init__()

    def eval_odometry(self, gt_poses, poses):
        avg_tra, avg_rot = sequence_error(gt_poses, poses)
        ate_rot, ate_trans = absolute_trajectory_error(gt_poses, poses)
        self.append(desc="Average Translation Error", units="%", value=avg_tra)
        self.append(desc="Average Rotational Error", units="deg/m", value=avg_rot)
        self.append(desc="Absoulte Trajectory Error (ATE)", units="m", value=ate_trans)
        self.append(desc="Absoulte Rotational Error (ARE)\n", units="rad", value=ate_rot)

    def eval_mos(self, confusion_matrix, desc=""):
        iou = get_iou(confusion_matrix)
        tp, fp, fn = get_stats(confusion_matrix)
        recall = get_recall(confusion_matrix)
        precision = get_precision(confusion_matrix)
        f1 = get_f1(confusion_matrix)
        self.append(desc=desc, units="", value="")
        self.append(desc="Moving IoU", units="%", value=iou[1].item() * 100)
        self.append(desc="Moving Recall", units="%", value=recall[1].item() * 100)
        self.append(desc="Moving Precision", units="%", value=precision[1].item() * 100)
        self.append(desc="Moving F1", units="%", value=f1[1].item() * 100)
        self.append(desc="Moving TP", units="points", value=int(tp[1].item()))
        self.append(desc="Moving FP", units="points", value=int(fp[1].item()))
        self.append(desc="Moving FN\n", units="points", value=int(fn[1].item()))

    def eval_fps(self, times, desc="Average Frequency"):
        def _get_fps(times):
            total_time_s = sum(times) * 1e-9
            return float(len(times) / total_time_s)

        avg_fps_mos = int(np.floor(_get_fps(times)))
        self.append(
            desc=f"{desc}",
            units="Hz",
            value=avg_fps_mos,
            trunc=True,
        )


class StubWriter:
    def write(self, *_, **__):
        pass


class KITTIWriter:
    """Writes the predictions as SemanticKITTI .label files (9: static, 251: moving)."""

    def write(self, pred_labels: np.ndarray, filename: str):
        os.makedirs(Path(filename).parent, exist_ok=True)
        kitti_labels = np.copy(pred_labels)
        kitti_labels[pred_labels == 0] = 9
        kitti_labels[pred_labels == 1] = 251
        kitti_labels = kitti_labels.reshape(-1).astype(np.int32)
        kitti_labels.tofile(filename)
