#!/usr/bin/env bash
# 1. The submission model: full speaker split, the winner of
#    2026-10-08_steps_and_memorisation.sh (500 steps at LR 2e-5).
# 2. Refinement on the speaker+text-disjoint split: confirm the winner with a
#    second seed, and try longer at the lower LRs.
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=.venv/bin/python
M=data/manifests

run() {  # out manifest steps [extra args]
  local out=$1 man=$2 steps=$3; shift 3
  if [[ -f $out/dev_metrics.json ]]; then echo "skip $out"; return; fi
  echo "=== $out  ($(date +%H:%M))"
  $PY shared-task/03_train.py --init checkpoints/hf/ft_cpt_silver --out "$out" \
      --manifest_dir "$man" --max_steps "$steps" --eval_steps "$steps" "$@" \
      > logs/exp_$(basename "$out").log 2>&1 || echo "FAILED $out"
}

run checkpoints/hf/final_n500_lr2e-5        $M/sharedtask      500  --lr 2e-5
run checkpoints/hf/exp/both_n500_lr2e-5_s43 $M/sharedtask_both 500  --lr 2e-5 --seed 43
run checkpoints/hf/exp/both_n1000_lr2e-5    $M/sharedtask_both 1000 --lr 2e-5
run checkpoints/hf/exp/both_n1000_lr1e-5    $M/sharedtask_both 1000 --lr 1e-5
echo "=== all done ($(date +%H:%M))"
