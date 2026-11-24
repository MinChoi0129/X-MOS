#!/bin/bash
# X-MOS Zero-Shot Evaluation - unseen datasets in the SemanticKITTI layout, no fine-tuning

CHECKPOINT="checkpoints/helimos_xmos_student.ckpt"
DATA_PATH="/path/to/SemanticKITTI/dataset"   # contains sequences/<id>/{velodyne,labels,calib.txt}
DATALOADER="kitti"                           # "kitti" (SemanticKITTI, Sipailou Campus) or "apollo"
SEQUENCES="08"
GPU=0

# ============================================================
export CUDA_VISIBLE_DEVICES=${GPU}
export OMP_NUM_THREADS=8

for SEQUENCE in ${SEQUENCES}; do
    python XMOS_test.py \
        --checkpoint ${CHECKPOINT} \
        --data ${DATA_PATH} \
        --dataloader ${DATALOADER} \
        --sequence ${SEQUENCE} \
        --config config/test/full_range.yaml
done
