#!/usr/bin/env python3
"""Runs a short, seeded training with the mos4d package found in <src> and dumps everything that
has to be identical between two versions of the code (given as the directory that is put on the
Python path: the root of this repository, or the src directory of the reference): the loss of every step, the validation
metrics and the final weights.

MinkowskiEngine is not deterministic on the GPU, so by default everything runs on the CPU, where
two runs are bit-identical. Set DEVICE=gpu to run on the GPU instead.

Usage: run_case.py <src> <config.yaml> <data> <cache> <out dir> [--precache]
"""
import os
import sys

src, config_file, data, cache, out = sys.argv[1:6]
sys.path.insert(0, src)

from pathlib import Path

import torch
from pytorch_lightning import Callback, Trainer

try:  # layout of this repository
    from core.config import load_config
    from core.seed import set_seed
    from core.training_module import TrainingModule
    from datasets.dataloader import MOS4DDataModule, MOS4DDataset, collate_fn
except ModuleNotFoundError:  # layout of the reference, the package of 4DMOS
    from mos4d.config import load_config
    from mos4d.datasets.mos4d_dataset import MOS4DDataModule, MOS4DDataset, collate_fn
    from mos4d.training_module import TrainingModule
    from mos4d.utils.seed import set_seed

DEVICE = os.environ.get("DEVICE", "cpu")
if DEVICE == "cpu":
    # The training module moves its networks to the GPU by hand, keep them where they are
    torch.nn.Module.cuda = lambda self, *args, **kwargs: self

module_file = Path(sys.modules[TrainingModule.__module__].__file__).resolve()
assert module_file.is_relative_to(Path(src).resolve()), module_file
out = Path(out)
out.mkdir(parents=True, exist_ok=True)
cfg = load_config(Path(config_file))

if "--precache" in sys.argv:
    # The cache has to be filled sequentially, the odometry needs the scans in order
    sequences = cfg.training.train + cfg.training.val
    loader = torch.utils.data.DataLoader(
        MOS4DDataset("helimos", data, cfg, sequences, cache), batch_size=1, collate_fn=collate_fn
    )
    with torch.no_grad():
        for _ in loader:
            pass
    sys.exit(0)


class Recorder(Callback):
    def __init__(self):
        self.train_losses, self.val_losses, self.val_metrics = [], [], []

    def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx):
        loss = outputs.get("loss") if isinstance(outputs, dict) else outputs  # skipped steps have no loss
        self.train_losses.append(None if loss is None else loss.detach().double().item())

    def on_validation_batch_end(self, trainer, module, outputs, batch, batch_idx, dataloader_idx=0):
        self.val_losses.append(None if outputs is None else outputs.detach().double().item())

    def on_validation_epoch_end(self, trainer, module):
        self.val_metrics.append({k: v.item() for k, v in trainer.callback_metrics.items() if k.startswith("val_")})


set_seed(66)
model = TrainingModule(cfg)
dataset = MOS4DDataModule(dataloader="helimos", data_dir=data, config=cfg, cache_dir=cache)
recorder = Recorder()
torch.set_float32_matmul_precision("high")
trainer = Trainer(
    accelerator=DEVICE,
    devices=1,
    logger=False,
    enable_checkpointing=False,
    enable_progress_bar=False,
    max_epochs=cfg.training.max_epochs,
    callbacks=[recorder],
)
trainer.fit(model, dataset)
trainer.save_checkpoint(out / "last.ckpt")

torch.save(
    {
        "train_losses": recorder.train_losses,
        "val_losses": recorder.val_losses,
        "val_metrics": recorder.val_metrics,
        "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
    },
    out / "result.pt",
)
steps = [l for l in recorder.train_losses if l is not None]
print(f"[{out}] {len(steps)}/{len(recorder.train_losses)} train steps, first loss {steps[0]:.6f}, last loss {steps[-1]:.6f}")
print(f"[{out}] validation: {recorder.val_metrics[-1]}")
