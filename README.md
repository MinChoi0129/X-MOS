# X-MOS

**A Heterogeneous Cross-LiDAR Generalization Framework for Moving Object Segmentation**

Official implementation of X-MOS (ICRA 2026). Sensor-specific expert teachers are distilled into a single student, using the sensor type as privileged information, so that one model works across heterogeneous LiDAR sensors. Built on [4DMOS](https://github.com/PRBonn/4DMOS).

---

## Setup

```bash
docker build -f docker/Dockerfile -t xmos .
docker run --gpus all -it --shm-size 16g -v $(pwd):/workspace/X-MOS -v /path/to/HeLiMOS:/data/HeLiMOS xmos
```

The image contains PyTorch 2.7 (CUDA 12.8) and a patched [MinkowskiEngine](https://github.com/NVIDIA/MinkowskiEngine). Without docker, install both yourself and run `pip install -r requirements.txt`.

## Dataset

[HeLiMOS](https://sites.google.com/view/helimos) in its original layout: `{Ouster, Velodyne, Aeva, Avia}/{velodyne/, labels/, calib.txt}` and `train.txt`, `val.txt`, `test.txt` in the root.

Set `HELIMOS_PATH` and `CACHE_PATH` in the scripts, then cache the aligned scans once:

```bash
bash scripts/precache.sh
```

## Training

```bash
bash scripts/train.sh
```

X-MOS is trained in stages, one run per `CONFIG` (set at the top of `scripts/train.sh`):

| # | Stage | `CONFIG` |
|---|---|---|
| 1 | Easy-to-learn teachers (O, A, L) | `config/baseline/{ouster, aeva, avia}.yaml` |
| 2 | Multi-level adaptation, 128 → 64 → 32 → 16 beams | `config/xmos/{1_mla_128to64, 2_mla_64to32, 3_mla_32to16}.yaml` |
| 3 | Velodyne teacher (V) | `config/xmos/4_velodyne_teacher.yaml` |
| 4 | Sensor-aware student | `config/xmos/5_student.yaml` |

Checkpoints are saved to `logs/<training.id>/`. Each stage loads the weights of the previous ones from the paths in its config, by default `checkpoints/<training.id>.ckpt`.

In stage 2, teacher and student are roles within one step: the student of a step is the teacher of the next, so from the second step on both see pseudo low-beam data.

`config/baseline/` and `config/ablation/` reproduce Table I and Table II of the paper.

## Evaluation

```bash
bash scripts/val.sh              # validation split
bash scripts/test.sh             # test split of every HeLiMOS sensor
bash scripts/test_zero_shot.sh   # SemanticKITTI, Sipailou Campus, Apollo
```

## Citation

If you find our work useful in your research, please consider citing our paper:

```bibtex
@inproceedings{lee2026xmos,
  author={Lee, Minjae and Ha, Ilhwan and Choi, Sang-Min and Kim, Gun-Woo and Lee, Suwon},
  booktitle={2026 IEEE International Conference on Robotics and Automation (ICRA)},
  title={X-MOS: A Heterogeneous Cross-LiDAR Generalization Framework for Moving Object Segmentation},
  year={2026},
  pages={4818-4825},
  doi={10.1109/ICRA57385.2026.11696227}}
```
