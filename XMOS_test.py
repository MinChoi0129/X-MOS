import argparse
from pathlib import Path

from core.pipeline import MOS4DPipeline
from datasets import dataset_factory


def get_args():
    parser = argparse.ArgumentParser("X-MOS Evaluation with the receding horizon strategy of 4DMOS")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True, help="Root of the dataset")
    parser.add_argument("--sequence", type=str, required=True, help='e.g. "Velodyne/test.txt" (HeLiMOS) or "08" (KITTI)')
    parser.add_argument("--dataloader", type=str, default="helimos", help="helimos, kitti or apollo")
    parser.add_argument("--config", type=Path, default=None, help="[Optional] e.g. config/test/full_range.yaml")
    parser.add_argument("--save_kitti", action="store_true", help="Save the predictions as SemanticKITTI .label files")
    parser.add_argument("--n_scans", type=int, default=-1, help="Number of scans to process, default is all of them")
    parser.add_argument("--jump", type=int, default=0, help="Index of the first scan")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    MOS4DPipeline(
        dataset=dataset_factory(dataloader=args.dataloader, data_dir=args.data, sequence=args.sequence),
        weights=args.checkpoint,
        config=args.config,
        save_kitti=args.save_kitti,
        n_scans=args.n_scans,
        jump=args.jump,
    ).run().print()
