"""Loading of MOS4DNet weights from the checkpoints written by the training module."""

import sys
import types
from pathlib import Path
from typing import Dict

import torch

from core import config as _config

# Keys of a training checkpoint that do not belong to the MOS network itself
_NON_NETWORK_PREFIXES = ("_teacher.", "q_projection", "k_projection")


def _register_legacy_modules():
    """The checkpoints store the config objects as hyperparameters. Checkpoints written before the
    code was restructured reference them as `mos4d.config.config.*`, so keep that name importable."""
    for name in ("mos4d", "mos4d.config"):
        sys.modules.setdefault(name, types.ModuleType(name))
    for name in ("mos4d.config.config", "mos4d.config.parser"):
        sys.modules.setdefault(name, _config)


def load_mos_state_dict(weights: Path) -> Dict[str, torch.Tensor]:
    """Returns the state dict of a checkpoint such that it can be loaded into a MOS4DNet."""
    # The checkpoints store the config objects as hyperparameters, so they are not weights-only
    _register_legacy_modules()
    state_dict = torch.load(weights, weights_only=False)["state_dict"]
    state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}
    state_dict = {k.replace("mos.", ""): v for k, v in state_dict.items()}
    state_dict = {k: v for k, v in state_dict.items() if "MOSLoss" not in k}
    state_dict = {k: v for k, v in state_dict.items() if not k.startswith(_NON_NETWORK_PREFIXES)}
    return state_dict
