#!/usr/bin/env bash
# 2026-10-09, step 5: the HELDOUT, measured ONCE, after every decision.
#
# PRE-REGISTRATION (written before this script was first run)
#   Decisions locked by step 4, on dev only:
#     - single model, no ensemble (neither fusion beat the seed dispersion)
#     - mask_time_prob 0.05 (0.2 and 0.5 lost all four numbers)
#     - decoding v3 = checkpoints/lm/submission_v3.json
#   PRIMARY -- "the paper's result": exp/both_n500_lr2e-5 (seed 42) + v3 on the
#     sharedtask_both heldout: speaker- AND text-disjoint, never used to decide.
#   DELIVERABLE: final_n500_lr2e-5 (seed 42) + v3 on the sharedtask heldout.
#     Its scripted half shares every sentence with train (optimistic, finding I).
#   SECONDARY, decide nothing: seeds 43 and 44 of both recipes (seed spread on
#     heldout), and greedy decoding of every model (the LM's effect on heldout).
#   Nothing changes after these numbers are seen. If anything did, the heldout
#   would become a second dev and this measurement would be void.
#
# Run only with the GPU free (HANDOFF finding K). Refuses to overwrite.
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=.venv/bin/python
R=results/2026-10-09
H=checkpoints/hf
V3="--lm_config checkpoints/lm/submission_v3.json --lm_dir checkpoints/lm"
mkdir -p $R

ev() {  # name manifest decoding model
  local name=$1 man=$2 dec=$3 model=$4
  local out=$R/heldout__${name}__${dec}.json
  if [[ -f $out ]]; then echo "REFUSING: $out exists -- the heldout is measured once"; return; fi
  local lm=""; [[ $dec == v3 ]] && lm=$V3
  echo "=== $out  ($(date +%H:%M:%S))"
  $PY shared-task/04_evaluate.py --model $model --manifest_dir $man --split heldout \
      $lm --metrics_json $out > ${out%.json}.log 2>&1 || echo "FAILED $out"
}

for dec in v3 greedy; do
  ev both_s42 data/manifests/sharedtask_both $dec $H/exp/both_n500_lr2e-5       # PRIMARY (v3)
  ev st_s42   data/manifests/sharedtask      $dec $H/final_n500_lr2e-5          # DELIVERABLE (v3)
  ev both_s43 data/manifests/sharedtask_both $dec $H/exp/both_n500_lr2e-5_s43
  ev both_s44 data/manifests/sharedtask_both $dec $H/exp/both_n500_lr2e-5_s44
  ev st_s43   data/manifests/sharedtask      $dec $H/final_n500_lr2e-5_s43
  ev st_s44   data/manifests/sharedtask      $dec $H/final_n500_lr2e-5_s44
done
echo "=== all done ($(date +%H:%M:%S))"
