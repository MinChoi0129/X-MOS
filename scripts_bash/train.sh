#!/bin/bash

# MODE=only_test_aeva # config/helimos 참조
HELIMOS_PATH=/home/ssd_4tb/minjae/HeLiMOS
CACHE_PATH=/home/ssd_4tb/minjae/HeLiMOS_cache

export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=8


MODE=velodyne # config/helimos 참조
python3 scripts/train.py ${HELIMOS_PATH} helimos ${CACHE_PATH} --config config/helimos/${MODE}_training.yaml