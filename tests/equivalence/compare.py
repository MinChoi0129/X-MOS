#!/usr/bin/env python3
"""Compares two results of run_case.py. Usage: compare.py <a/result.pt> <b/result.pt>"""
import math
import sys

import torch

a, b = (torch.load(f, weights_only=False) for f in sys.argv[1:3])


def max_diff(xs, ys):
    assert len(xs) == len(ys), f"different number of steps: {len(xs)} != {len(ys)}"
    diff = 0.0
    for x, y in zip(xs, ys):
        assert (x is None) == (y is None), "a step was skipped in only one of the runs"
        if x is not None and not (math.isnan(x) and math.isnan(y)):
            diff = max(diff, abs(x - y))
    return diff


def weight_diff(x, y):
    x, y = x.double(), y.double()
    both_nan = x.isnan() & y.isnan()
    diff = (x - y).abs()[~both_nan]
    return 0.0 if diff.numel() == 0 else diff.max().item()  # NaN in only one of them stays NaN


train = max_diff(a["train_losses"], b["train_losses"])
val = max_diff(a["val_losses"], b["val_losses"])
# The first validation (sanity check) has no metrics yet
assert [m.keys() for m in a["val_metrics"]] == [m.keys() for m in b["val_metrics"]], "different metrics"
metrics = max(max_diff([m[k]], [n[k]]) for m, n in zip(a["val_metrics"], b["val_metrics"]) for k in m)
assert a["state_dict"].keys() == b["state_dict"].keys(), "different keys in the state dict"
weights = max(weight_diff(a["state_dict"][k], b["state_dict"][k]) for k in a["state_dict"])
identical = train == val == metrics == weights == 0.0
finite = all(l is None or math.isfinite(l) for l in a["train_losses"])
print(
    f"steps: {len(a['train_losses'])} train / {len(a['val_losses'])} val | max abs diff: "
    f"train loss {train:.3e}, val loss {val:.3e}, val metrics {metrics:.3e}, weights {weights:.3e} "
    f"-> {'IDENTICAL' if identical else 'DIFFERENT'}{'' if finite else ' (the losses diverge to NaN in both runs)'}"
)
sys.exit(0 if identical else 1)
