#!/usr/bin/env bash
# 2026-10-09, steps 2-3: every system on DEV only. The heldout is NOT touched
# here -- it is measured once, after the decisions, by 2026-10-09_eval_heldout.sh.
# Run only with the GPU free (HANDOFF finding K).
#
# Decoding: "v3" = checkpoints/lm/submission_v3.json (scripted greedy,
# spontaneous + LM) -- the system as submitted, and what decisions use.
# "greedy" = no LM anywhere, to separate the LM's share from the model's.
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=.venv/bin/python
R=results/2026-10-09
BOTH=data/manifests/sharedtask_both
ST=data/manifests/sharedtask
H=checkpoints/hf
V3="--lm_config checkpoints/lm/submission_v3.json --lm_dir checkpoints/lm"
mkdir -p $R

ev() {  # name manifest decoding models...   (FUSION=prob|logit for ensembles)
  local name=$1 man=$2 dec=$3; shift 3
  local out=$R/dev__${name}__${dec}.json
  if [[ -f $out ]]; then echo "skip $out"; return; fi
  local lm=""; [[ $dec == v3 ]] && lm=$V3
  echo "=== $out  ($(date +%H:%M:%S))"
  $PY shared-task/04_evaluate.py --model "$@" --manifest_dir $man --split dev \
      --ensemble_fusion ${FUSION:-prob} \
      $lm --metrics_json $out > ${out%.json}.log 2>&1 || echo "FAILED $out"
}

BOTH_SEEDS="$H/exp/both_n500_lr2e-5 $H/exp/both_n500_lr2e-5_s43 $H/exp/both_n500_lr2e-5_s44"
ST_SEEDS="$H/final_n500_lr2e-5 $H/final_n500_lr2e-5_s43 $H/final_n500_lr2e-5_s44"

for dec in v3 greedy; do
  # Step 2 on the honest split: each seed, then the ensemble of the three
  ev both_s42 $BOTH $dec $H/exp/both_n500_lr2e-5
  ev both_s43 $BOTH $dec $H/exp/both_n500_lr2e-5_s43
  ev both_s44 $BOTH $dec $H/exp/both_n500_lr2e-5_s44
  # Ensembles: "logit" (mean of logits, as first specified) and "prob" (mean
  # of probabilities) -- see ensemble.py for why the first loses
  FUSION=logit ev both_ens3logit $BOTH $dec $BOTH_SEEDS
  FUSION=prob  ev both_ens3prob  $BOTH $dec $BOTH_SEEDS
  # Step 3: mask_time_prob, seed 42 (pairs with both_s42 at 0.05)
  ev both_s42_mask0.2 $BOTH $dec $H/exp/both_n500_lr2e-5_mask0.2
  ev both_s42_mask0.5 $BOTH $dec $H/exp/both_n500_lr2e-5_mask0.5
  # The submission split's own dev (optimistic for scripted, finding I)
  ev st_s42 $ST $dec $H/final_n500_lr2e-5
  ev st_s43 $ST $dec $H/final_n500_lr2e-5_s43
  ev st_s44 $ST $dec $H/final_n500_lr2e-5_s44
  FUSION=logit ev st_ens3logit $ST $dec $ST_SEEDS
  FUSION=prob  ev st_ens3prob  $ST $dec $ST_SEEDS
done
echo "=== all done ($(date +%H:%M:%S))"
