#!/bin/bash
# X-MOS Training - one call per stage, see the table in the README for the order

CONFIG="config/xmos/5_student.yaml"
HELIMOS_PATH="/path/to/HeLiMOS"
CACHE_PATH="/path/to/HeLiMOS_cache"
GPU=0

# ============================================================
export CUDA_VISIBLE_DEVICES=${GPU}
export OMP_NUM_THREADS=10

python XMOS_train.py \
    --config ${CONFIG} \
    --data ${HELIMOS_PATH} \
    --cache ${CACHE_PATH} \
    "$@"
