#!/usr/bin/env bash
# 2026-10-09. Recipe throughout: 500 steps, LR 2e-5, from ft_cpt_silver.
#   Step 1 -- the seeds an ensemble needs (seed 42 of each split already exists:
#             exp/both_n500_lr2e-5, exp/both_n500_lr2e-5_s43, final_n500_lr2e-5).
#   Step 3 -- mask_time_prob, untouched until now (0.05), seed 42 on
#             sharedtask_both so it pairs with exp/both_n500_lr2e-5.
# Each run evaluates once, at its last step, on its own split's dev.
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=.venv/bin/python
M=data/manifests

run() {  # out manifest [extra args]
  local out=$1 man=$2; shift 2
  if [[ -f $out/dev_metrics.json ]]; then echo "skip $out"; return; fi
  echo "=== $out  ($(date +%H:%M))"
  $PY shared-task/03_train.py --init checkpoints/hf/ft_cpt_silver --out "$out" \
      --manifest_dir "$man" --max_steps 500 --eval_steps 500 --lr 2e-5 "$@" \
      > logs/exp_$(basename "$out").log 2>&1 || echo "FAILED $out"
}

# Step 1
run checkpoints/hf/exp/both_n500_lr2e-5_s44  $M/sharedtask_both --seed 44
run checkpoints/hf/final_n500_lr2e-5_s43     $M/sharedtask      --seed 43
run checkpoints/hf/final_n500_lr2e-5_s44     $M/sharedtask      --seed 44
# Step 3
run checkpoints/hf/exp/both_n500_lr2e-5_mask0.2 $M/sharedtask_both --seed 42 --mask_time_prob 0.2
run checkpoints/hf/exp/both_n500_lr2e-5_mask0.5 $M/sharedtask_both --seed 42 --mask_time_prob 0.5
echo "=== all done ($(date +%H:%M))"
