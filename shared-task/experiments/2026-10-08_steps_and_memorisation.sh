#!/usr/bin/env bash
# Two questions, answered on the speaker+text-disjoint split (HANDOFF finding G):
#   1. How many fine-tuning steps from the baseline? (sweep, each run ends its
#      own LR schedule at max_steps, so every point is a finished model)
#   2. How much does hearing a dev sentence from OTHER speakers help?
#        A    = sharedtask_both               (dev sentences never in train)
#        B    = sharedtask_both_devtext       (+ train speakers reading them, +3.65 h)
#        B_eq = sharedtask_both_devtext_eq    (same as B, minus 3.64 h of random A
#                                              clips: same hours as A)
#      Same dev in all three. A vs B_eq isolates memorisation from data volume.
# Manifests: 01_build_manifests.py --split_by both --dev_frac 0.10 --heldout_frac 0.10
#            [--train_on_dev_sentences]; B_eq built from B as described in HANDOFF.
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=.venv/bin/python
M=data/manifests

run() {  # name manifest steps [extra args]
  local name=$1 man=$2 steps=$3; shift 3
  local out=checkpoints/hf/exp/$name
  if [[ -f $out/dev_metrics.json ]]; then echo "skip $name"; return; fi
  echo "=== $name  ($(date +%H:%M))"
  $PY shared-task/03_train.py --init checkpoints/hf/ft_cpt_silver --out "$out" \
      --manifest_dir "$man" --max_steps "$steps" --eval_steps "$steps" "$@" \
      > logs/exp_$name.log 2>&1 || echo "FAILED $name (see logs/exp_$name.log)"
}

for n in 100 150 200 250 300 500; do run both_n$n $M/sharedtask_both $n; done
run both_n500_lr2e-5 $M/sharedtask_both 500 --lr 2e-5

run both_n250_s43   $M/sharedtask_both            250 --seed 43
run devtext_n250    $M/sharedtask_both_devtext    250
run deveq_n250      $M/sharedtask_both_devtext_eq 250
run deveq_n250_s43  $M/sharedtask_both_devtext_eq 250 --seed 43
echo "=== all done ($(date +%H:%M))"
