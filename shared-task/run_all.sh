#!/usr/bin/env bash
# =============================================================================
# SIMBig 2026 Task 1 — Puno Quechua ASR, end to end
# =============================================================================
# On the GPU machine:
#     git pull && bash shared-task/run_all.sh
#
# Every stage is resumable and skips work that is already done, so re-running
# after a crash, a disconnect or a reboot is safe and cheap.
#
# Stages (run a subset with --from / --only):
#     download   fetch Scripted 27.0 + Spontaneous 5.0 from Mozilla
#     convert    turn the published fairseq checkpoint into a HF model
#     manifests  build train/dev/heldout from the curated reference text
#     train      CTC fine-tuning
#     evaluate   WER and CER for scripted and spontaneous, plus the mean
#     predict    transcribe the test audio and package qxp.zip
# =============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PY="${PY:-python}"
INIT="${INIT:-checkpoints/hf/ft_cpt_silver}"
OUT="${OUT:-checkpoints/hf/ft_curated}"
MANIFESTS="${MANIFESTS:-data/manifests/sharedtask}"
MAX_STEPS="${MAX_STEPS:-8000}"
TEST_DIR="${TEST_DIR:-data/test/qxp_test_dataset}"
EXTRA_TRAIN_ARGS="${EXTRA_TRAIN_ARGS:-}"

STAGES=(download convert manifests train evaluate predict)
FROM=""; ONLY=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --from) FROM="$2"; shift 2 ;;
    --only) ONLY="$2"; shift 2 ;;
    -h|--help) sed -n '2,20p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done

should_run() {
  local stage="$1"
  [[ -n "$ONLY" ]] && { [[ "$ONLY" == "$stage" ]] && return 0 || return 1; }
  if [[ -n "$FROM" ]]; then
    local seen=0
    for s in "${STAGES[@]}"; do
      [[ "$s" == "$FROM" ]] && seen=1
      [[ "$s" == "$stage" ]] && { [[ $seen -eq 1 ]] && return 0 || return 1; }
    done
  fi
  return 0
}

banner() { printf '\n\033[1m=== %s ===\033[0m\n' "$1"; }

# ---------------------------------------------------------------- download --
if should_run download; then
  banner "download"
  if [[ -f data/scripted/qxp/validated.tsv && -f data/spontaneous/ss-corpus-qxp.tsv ]]; then
    echo "corpora already present — skipping"
  else
    [[ -f .env ]] || { echo "No .env. cp shared-task/.env.example .env and add MDC_API_KEY" >&2; exit 1; }
    $PY shared-task/00_download.py
  fi
fi

# ----------------------------------------------------------------- convert --
# The published XLS-R checkpoints are fairseq .pt files, and fairseq 0.12.2 has
# no wheel for modern Python. This converts without installing it.
if should_run convert; then
  banner "convert"
  if [[ -d "$INIT" && -f "$INIT/model.safetensors" ]]; then
    echo "$INIT already exists — skipping"
  else
    mkdir -p checkpoints/ft_cpt_silver
    if [[ ! -f checkpoints/ft_cpt_silver/checkpoint_best.pt ]]; then
      echo "downloading the fairseq checkpoint (~3.6 GB)"
      curl -L --retry 10 --retry-delay 5 -C - \
        -o checkpoints/ft_cpt_silver/checkpoint_best.pt \
        "https://huggingface.co/QuechuaBase/xls-r-cpt-qxp-silver/resolve/main/checkpoint_best.pt"
    fi
    $PY eval/hf/convert_fairseq_w2v2.py \
      --ckpt checkpoints/ft_cpt_silver/checkpoint_best.pt \
      --dict data/manifests/finetune/qxp_v2/dict.ltr.txt \
      --out "$INIT"
  fi
fi

# --------------------------------------------------------------- manifests --
if should_run manifests; then
  banner "manifests"
  $PY shared-task/01_build_manifests.py --out "$MANIFESTS"
fi

# ------------------------------------------------------------------- train --
if should_run train; then
  banner "train"
  # shellcheck disable=SC2086
  $PY shared-task/03_train.py \
    --init "$INIT" --out "$OUT" \
    --manifest_dir "$MANIFESTS" \
    --max_steps "$MAX_STEPS" \
    $EXTRA_TRAIN_ARGS
fi

# ---------------------------------------------------------------- evaluate --
if should_run evaluate; then
  banner "evaluate"
  $PY shared-task/04_evaluate.py \
    --model "$OUT" --manifest_dir "$MANIFESTS" --split dev \
    --save_predictions "$OUT/predictions"
fi

# ----------------------------------------------------------------- predict --
if should_run predict; then
  banner "predict"
  if [[ ! -d "$TEST_DIR/audios" ]]; then
    echo "No test audio at $TEST_DIR/audios — skipping"
  else
    $PY shared-task/05_predict.py \
      --model "$OUT" \
      --audio_dir "$TEST_DIR/audios" \
      --template "$TEST_DIR/qxp.tsv" \
      --out submission/qxp
    echo
    echo "Submit submission/qxp.zip"
    echo "Check it first: head -3 submission/qxp.tsv"
  fi
fi

banner "done"
