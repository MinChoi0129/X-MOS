import argparse
import os
from pathlib import Path

import torch
from pytorch_lightning import Trainer
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

from core.config import load_config
from core.seed import set_seed
from core.training_module import TrainingModule
from datasets.dataloader import MOS4DDataModule


def get_args():
    parser = argparse.ArgumentParser("X-MOS Training")
    parser.add_argument("--config", type=Path, required=True, help="Config of the training stage, see config/")
    parser.add_argument("--data", type=Path, required=True, help="Root of the dataset")
    parser.add_argument("--cache", type=Path, required=True, help="Directory of the cache, see XMOS_precache.py")
    parser.add_argument("--dataloader", type=str, default="helimos")
    parser.add_argument("--log_dir", type=str, default="logs")
    parser.add_argument("--validate_only", action="store_true", help="Only validate the weights of keep_training_weights")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    set_seed(66)

    cfg = load_config(args.config)
    model = TrainingModule(cfg)
    dataset = MOS4DDataModule(dataloader=args.dataloader, data_dir=args.data, config=cfg, cache_dir=args.cache)

    # Checkpoints are written to <log_dir>/<training.id>/version_<n>/checkpoints
    os.makedirs(args.log_dir, exist_ok=True)
    tb_logger = pl_loggers.TensorBoardLogger(args.log_dir, name=cfg.training.id, default_hp_metric=False)
    checkpoint_saver = ModelCheckpoint(
        monitor="val_moving_iou",
        filename=cfg.training.id + "_{epoch:03d}_{val_moving_iou:.3f}",
        mode="max",
        save_last=True,
    )

    torch.set_float32_matmul_precision("high")
    trainer = Trainer(
        accelerator="gpu",
        devices=1,
        logger=tb_logger,
        max_epochs=cfg.training.max_epochs,
        callbacks=[LearningRateMonitor(logging_interval="step"), checkpoint_saver],
    )

    if args.validate_only:
        trainer.validate(model, dataset)
    else:
        trainer.fit(model, dataset)
