import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from core.config import load_config
from datasets.dataloader import MOS4DDataset, collate_fn


def get_args():
    parser = argparse.ArgumentParser("X-MOS Cache Generation")
    parser.add_argument("--config", type=Path, required=True, help="Config with the train and val sequences to cache")
    parser.add_argument("--data", type=Path, required=True, help="Root of the dataset")
    parser.add_argument("--cache", type=Path, required=True, help="Directory where the cache is created")
    parser.add_argument("--dataloader", type=str, default="helimos")
    parser.add_argument("--sequences", type=str, nargs="+", default=None, help="Cache these sequences instead")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    cfg = load_config(args.config)
    sequences = args.sequences if args.sequences is not None else cfg.training.train + cfg.training.val

    # The odometry aligns each scan to the previous ones, so the scans have to be visited in order
    data_iterable = DataLoader(
        MOS4DDataset(args.dataloader, args.data, cfg, sequences, args.cache),
        batch_size=1,
        collate_fn=collate_fn,
        shuffle=False,
        num_workers=0,
    )
    with torch.no_grad():
        for _ in tqdm(data_iterable, desc="Caching data", unit=" items", dynamic_ncols=True):
            pass
