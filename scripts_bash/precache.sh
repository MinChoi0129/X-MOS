#!/bin/bash

MODE=all # all, hard, omni, solid
HELIMOS_PATH=/home/ssd_4tb/minjae/HeLiMOS
CACHE_PATH=/home/ssd_4tb/minjae/HeLiMOS_cache

python3 scripts/precache.py ${HELIMOS_PATH} helimos ${CACHE_PATH} --config config/helimos/${MODE}_training.yaml
