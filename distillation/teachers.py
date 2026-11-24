"""Frozen sensor-specific expert teachers."""

from pathlib import Path
from typing import Dict

from networks.mos4d import MOS4DNet
from datasets.sensors import TEACHER_KEYS
from core.checkpoint import load_mos_state_dict


def load_teacher(weights: Path, voxel_size: float) -> MOS4DNet:
    teacher = MOS4DNet(voxel_size)
    teacher.load_state_dict(load_mos_state_dict(weights))
    teacher.cuda().eval()
    for param in teacher.parameters():
        param.requires_grad = False
    return teacher


def load_teachers(training_config, voxel_size: float) -> Dict[str, MOS4DNet]:
    """Loads every teacher that has weights configured (`<key>_teacher_weights`), keyed by sensor.

    The teachers are returned in a plain dict on purpose: they must not be registered as
    submodules of the training module, otherwise they would end up in the student checkpoint.
    """
    teachers = {}
    for key in TEACHER_KEYS:
        weights = getattr(training_config, f"{key}_teacher_weights")
        if weights:
            teachers[key] = load_teacher(weights, voxel_size)
            print(f"**Teacher {key} loaded: {weights}**")
    return teachers
