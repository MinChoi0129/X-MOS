#!/bin/bash
# X-MOS Evaluation - test split of every HeLiMOS sensor (receding horizon strategy)

CHECKPOINT="checkpoints/helimos_xmos_student.ckpt"
HELIMOS_PATH="/path/to/HeLiMOS"
SENSORS="Ouster Velodyne Aeva Avia"
GPU=0

# ============================================================
export CUDA_VISIBLE_DEVICES=${GPU}
export OMP_NUM_THREADS=8

for SENSOR in ${SENSORS}; do
    python XMOS_test.py \
        --checkpoint ${CHECKPOINT} \
        --data ${HELIMOS_PATH} \
        --dataloader helimos \
        --sequence ${SENSOR}/test.txt
done
