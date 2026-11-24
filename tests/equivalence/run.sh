#!/bin/bash
# Checks that the code of this checkout trains exactly like a reference commit: same loss in every
# step, same validation metrics and same final weights for a seeded run on synthetic data.
#
# The training runs on the CPU, where two runs of the same code are bit-identical. MinkowskiEngine
# is not deterministic on the GPU: with DEVICE=gpu even the reference differs from itself, so the
# differences are only printed. The inference pipeline always needs a GPU.
#
# Usage (inside the docker container, from the root of the repository, needs the .git directory):
#   tests/equivalence/run.sh [reference commit, default: 4126a13] [work dir, default: /tmp/xmos_eq]
set -e

REF=${1:-4126a13}
WORK=${2:-/tmp/xmos_eq}
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)
NEW=$REPO
OLD=$WORK/ref/src
DATA=$WORK/data
CACHE=$WORK/cache
# CPU usage is PARALLEL runs at a time x OMP_NUM_THREADS threads each. Both runs of a comparison
# need the same number of threads, otherwise the sums of the CPU kernels are rounded differently.
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
PARALLEL=${PARALLEL:-4}
export DEVICE=${DEVICE:-cpu}
# The reference code loads checkpoints with a plain torch.load, PyTorch >= 2.6 needs this for it
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

rm -rf $WORK && mkdir -p $WORK/ref $WORK/cfg
python $HERE/make_synthetic_helimos.py $DATA

# Reference trees: the commit as it is, and with the ablation losses switched on by hand in the
# way the reference code was meant to be used (commenting lines in and out)
git -C $REPO archive $REF src | tar -x -C $WORK/ref
# The reference imports its C++ extension, which is only needed for deskewing (switched off)
mkdir -p $OLD/mos4d/pybind && touch $OLD/mos4d/pybind/mos4d_pybind.py
for MODE in average attention; do
    cp -r $WORK/ref $WORK/ref_$MODE
    python - $WORK/ref_$MODE/src/mos4d/training_module.py $MODE <<'PY'
import sys
path, mode = sys.argv[1:]
s = open(path).read()
active = "            kd_loss = self.sensor_aware_kd_loss(teacher_outputs, student_logits_for_kd, sequence)"
call = {
    "average": "self.average_based_kd_loss(teacher_outputs, student_logits_for_kd)",
    "attention": "self.attention_based_kd_loss(teacher_outputs, student_logits_for_kd, student_bottleneck)",
}[mode]
assert active in s and f"# kd_loss = {call}" in s
s = s.replace(active, f"            kd_loss = {call}")
if mode == "attention":
    for line in ("bottleneck_dim = 64", "self.attention_dim = 64", "self.q_projection = ", "self.k_projection = "):
        assert f"            # {line}" in s
        s = s.replace(f"            # {line}", f"            {line}")
    # The reference turns to NaN as soon as a blended teacher probability underflows to zero, the
    # gradient through the target is then -inf. Keep the target in log space, like the new code.
    target = "        soft_teacher = F.softmax(blended_logits / self._temperature, dim=-1)\n"
    loss = "        total_kd_loss = self._distill_loss(soft_student, soft_teacher) * (self._temperature**2)\n"
    head, tail = s.split(target)
    assert loss in tail
    tail = tail.replace(loss, loss.replace("self._distill_loss(soft_student, soft_teacher)", 'F.kl_div(soft_student, soft_teacher, reduction="batchmean", log_target=True)'), 1)
    s = head + target.replace("F.softmax", "F.log_softmax") + tail
open(path, "w").write(s)
PY
done

# The reference never switches the pseudo low-beam mode on and crashes if it is switched on by hand,
# because the logits of the teacher (denser input) are not reduced to the points of the student.
# This tree has the mode switched on plus that one indexing, to compare the rest of the stage.
cp -r $WORK/ref $WORK/ref_mla
python - $WORK/ref_mla/src/mos4d/training_module.py <<'PY'
import sys
path = sys.argv[1]
s = open(path).read()
logits = 't_logits = teacher_rtn["logits"][:, 1:]  # (N, 2)'
assert s.count("is_pseudo_mode=False") == 2 and logits in s
s = s.replace("is_pseudo_mode=False", "is_pseudo_mode=True")
s = s.replace(logits, 't_logits = teacher_rtn["logits"][:, 1:][kept_indices]')
open(path, "w").write(s)
PY

ALL='["Avia/train.txt", "Aeva/train.txt", "Velodyne/train.txt", "Ouster/train.txt"]'
ALL_VAL='["Avia/val.txt", "Aeva/val.txt", "Velodyne/val.txt", "Ouster/val.txt"]'
COMMON="  batch_size: 2
  num_workers: 2
  max_epochs: 2"
NO_INIT="  keep_training: False
  is_distill_mode: False"
STUDENT="  keep_training: True
  keep_training_weights: $WORK/ref.supervised_ouster/last.ckpt
  is_distill_mode: True
  O_teacher_weights: $WORK/ref.supervised_ouster/last.ckpt
  V_teacher_weights: $WORK/ref.supervised/last.ckpt
  A_teacher_weights: $WORK/ref.supervised/last.ckpt
  L_teacher_weights: $WORK/ref.supervised_ouster/last.ckpt"

cfg() { printf "training:\n  id: $1\n  train: $2\n  val: $3\n$COMMON\n$4\n" > $WORK/cfg/$1.yaml; }
cfg supervised "$ALL" "$ALL_VAL" "$NO_INIT"
cfg supervised_ouster '["Ouster/train.txt"]' '["Ouster/val.txt"]' "$NO_INIT
  lr: 0.001"
cfg student_sensor_aware "$ALL" "$ALL_VAL" "$STUDENT
  kd_mode: sensor_aware"
cfg student_average "$ALL" "$ALL_VAL" "$STUDENT
  kd_mode: average"
cfg student_attention "$ALL" "$ALL_VAL" "$STUDENT
  kd_mode: attention"
cfg student_random_init "$ALL" "$ALL_VAL" "$(echo "$STUDENT" | sed 's/keep_training: True/keep_training: False/')
  kd_mode: sensor_aware"
for STAGE in 0 1 2; do
    cfg mla_stage$STAGE '["Ouster/train.txt"]' '["Ouster/val.txt"]' "  keep_training: True
  keep_training_weights: $WORK/ref.supervised_ouster/last.ckpt
  is_distill_mode: True
  O_teacher_weights: $WORK/ref.supervised_ouster/last.ckpt
  pseudo_beam_down: True
  distill_stage_index: $STAGE"
done

run() { # <name of the run> <src> <case>
    if ! python $HERE/run_case.py $2 $WORK/cfg/$3.yaml $DATA $CACHE $WORK/$1.$3 > $WORK/$1.$3.log 2>&1; then
        tail -n 20 $WORK/$1.$3.log
        exit 1
    fi
    grep "^\[$WORK" $WORK/$1.$3.log
    if grep -q "Caching now" $WORK/$1.$3.log; then
        echo "$1.$3 did not find its samples in the cache"; exit 1
    fi
}
FAILED=0
check() { # <description> <result a> <result b>
    printf "%-46s" "$1"
    python $HERE/compare.py $WORK/$2/result.pt $WORK/$3/result.pt || FAILED=1
}

echo "== Filling the cache"
python $HERE/run_case.py $OLD $WORK/cfg/supervised.yaml $DATA $CACHE $WORK/precache --precache > /dev/null

start() { # <name of the run> <src> <case>: runs in the background, at most PARALLEL at a time
    while [ $(jobs -pr | wc -l) -ge $PARALLEL ]; do wait -n || exit 1; done
    run $1 $2 $3 &
}
both() { # <case> [reference tree]: the reference and the new code
    start ref ${2:-$OLD} $1
    start new $NEW $1
}
finish() { # waits for all runs that were started
    for JOB in $(jobs -p); do wait $JOB || exit 1; done
}
echo "== Supervised training"
both supervised
both supervised_ouster
start ref_again $OLD supervised_ouster
finish
echo "== Distillation of the student and multi-level adaptation"
for STAGE in 0 1 2; do
    both mla_stage$STAGE $WORK/ref_mla/src
done
both student_sensor_aware
both student_random_init
both student_average $WORK/ref_average/src
both student_attention $WORK/ref_attention/src
finish

echo
echo "== Inference pipeline on the Velodyne test split"
CKPT=$WORK/ref.supervised/last.ckpt
mkdir -p $WORK/pipeline_ref $WORK/pipeline_new
(cd $WORK/pipeline_ref && PYTHONPATH=$OLD python -c "from mos4d.cli import app; app()" \
    $CKPT $DATA --dataloader helimos -s Velodyne/test.txt 2>&1 | grep -A1 "^IoU" > iou.txt) || true
(cd $WORK/pipeline_new && PYTHONPATH=$NEW python $NEW/XMOS_test.py \
    --checkpoint $CKPT --data $DATA --sequence Velodyne/test.txt 2>&1 | grep -A1 "^IoU" > iou.txt) || true
printf "%-46s" "pipeline: reference vs new"
if cmp -s $WORK/pipeline_ref/iou.txt $WORK/pipeline_new/iou.txt && [ -s $WORK/pipeline_new/iou.txt ]; then
    echo "$(tr '\n' ' ' < $WORK/pipeline_new/iou.txt)-> IDENTICAL"
else
    echo "DIFFERENT"; FAILED=1
fi

echo
echo "== Results"
check "determinism: reference vs reference" ref_again.supervised_ouster ref.supervised_ouster
check "supervised (4 sensors): reference vs new" ref.supervised new.supervised
check "supervised (Ouster): reference vs new" ref.supervised_ouster new.supervised_ouster
check "student, sensor-aware KD: reference vs new" ref.student_sensor_aware new.student_sensor_aware
check "student, random init: reference vs new" ref.student_random_init new.student_random_init
check "student, average KD: reference vs new" ref.student_average new.student_average
check "student, attention KD: reference vs new" ref.student_attention new.student_attention
for STAGE in 0 1 2; do
    check "MLA stage $STAGE: patched reference vs new" ref.mla_stage$STAGE new.mla_stage$STAGE
done
if [ $DEVICE != cpu ]; then
    echo "DEVICE=$DEVICE: the runs are not deterministic, compare the differences with the determinism rows"
    exit 0
fi
exit $FAILED
