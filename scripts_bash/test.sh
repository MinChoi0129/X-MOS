#!/bin/bash


export OMP_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=1

HELIMOS_PATH=/home/ssd_4tb/minjae/HeLiMOS

###########################################################################################################################

CKPT_FILEPATH=/home/work/4DMOS/models/4DMOS/helimos_all/case031/checkpoints/helimos_all_epoch=099_val_moving_iou=0.849.ckpt

TEST_TARGET=Velodyne # Velodyne, Ouster, Aeva, Avia
mos4d_pipeline ${CKPT_FILEPATH} ${HELIMOS_PATH} --dataloader helimos -s ${TEST_TARGET}/test.txt

TEST_TARGET=Aeva # Velodyne, Ouster, Aeva, Avia
mos4d_pipeline ${CKPT_FILEPATH} ${HELIMOS_PATH} --dataloader helimos -s ${TEST_TARGET}/test.txt

TEST_TARGET=Avia # Velodyne, Ouster, Aeva, Avia
mos4d_pipeline ${CKPT_FILEPATH} ${HELIMOS_PATH} --dataloader helimos -s ${TEST_TARGET}/test.txt

TEST_TARGET=Ouster # Velodyne, Ouster, Aeva, Avia
mos4d_pipeline ${CKPT_FILEPATH} ${HELIMOS_PATH} --dataloader helimos -s ${TEST_TARGET}/test.txt
