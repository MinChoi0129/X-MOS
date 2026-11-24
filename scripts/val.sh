#!/bin/bash
# X-MOS Validation - validation split of the config, with the weights of keep_training_weights

CONFIG="config/xmos/5_student.yaml"

# ============================================================
bash "$(dirname "$0")/train.sh" --config ${CONFIG} --validate_only
