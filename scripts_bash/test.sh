#!/bin/bash


export OMP_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=1

HELIMOS_PATH=/home/ssd_4tb/minjae/HeLiMOS

# ###########################################################################################################################

CKPT_FILEPATH=/home/work/4DMOS/models/4DMOS/helimos_velodyne/version_0/checkpoints/helimos_velodyne_epoch=093_val_moving_iou=0.657.ckpt

TEST_TARGET=Velodyne # Velodyne, Ouster, Aeva, Avia
mos4d_pipeline ${CKPT_FILEPATH} ${HELIMOS_PATH} --dataloader helimos -s ${TEST_TARGET}/test.txt

TEST_TARGET=Aeva # Velodyne, Ouster, Aeva, Avia
mos4d_pipeline ${CKPT_FILEPATH} ${HELIMOS_PATH} --dataloader helimos -s ${TEST_TARGET}/test.tx

TEST_TARGET=Avia # Velodyne, Ouster, Aeva, Avia
mos4d_pipeline ${CKPT_FILEPATH} ${HELIMOS_PATH} --dataloader helimos -s ${TEST_TARGET}/test.txt

TEST_TARGET=Ouster # Velodyne, Ouster, Aeva, Avia
mos4d_pipeline ${CKPT_FILEPATH} ${HELIMOS_PATH} --dataloader helimos -s ${TEST_TARGET}/test.txt

