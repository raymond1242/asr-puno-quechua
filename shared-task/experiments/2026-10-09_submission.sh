#!/usr/bin/env bash
# 2026-10-09, step 6: the submission, with the system locked in step 4 --
# single model final_n500_lr2e-5 (sharedtask, seed 42) + decoding v3.
# Run only with the GPU free (HANDOFF finding K).
set -euo pipefail
cd "$(dirname "$0")/../.."
.venv/bin/python shared-task/05_predict.py \
    --model checkpoints/hf/final_n500_lr2e-5 \
    --audio_dir data/test/qxp_test_dataset/audios \
    --template data/test/qxp_test_dataset/qxp.tsv \
    --lm_config checkpoints/lm/submission_v3.json --lm_dir checkpoints/lm \
    --out submission/final_n500_lr2e-5_v3_2026-10-09/qxp
