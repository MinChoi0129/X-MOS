#!/bin/bash
# X-MOS Cache Generation - aligns the past scans of every sample with KISS-ICP and stores them

CONFIG="config/baseline/all.yaml"   # caches the train and val split of all four sensors
HELIMOS_PATH="/path/to/HeLiMOS"
CACHE_PATH="/path/to/HeLiMOS_cache"

# ============================================================
python XMOS_precache.py \
    --config ${CONFIG} \
    --data ${HELIMOS_PATH} \
    --cache ${CACHE_PATH}
