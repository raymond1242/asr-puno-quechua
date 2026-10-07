#!/usr/bin/env python3
"""
Evaluate a HuggingFace transformers ASR model on a TSV manifest.

Runs natively on macOS (MPS/CPU) — no fairseq, no Docker, no GPU required.

Usage:
  python eval/hf/run_hf_asr.py \
      --model QuechuaBase/whisper-base-qxp-finetuned \
      --tsv data/additional_data/additional_data_qxp.tsv \
      --audio_dir data/additional_data/wav \
      --out results/hf/whisper_ood.tsv

TSV must have a path column (audio filename) and a reference-text column.
Metrics use the same normalization as eval/omnilingual/compute_wer.py.
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from jiwer import cer, wer
from scipy.signal import resample_poly
from math import gcd

PUNCT = ["?", "!", "¿", "¡", ".", ","]


def remove_punct(sentence):
    """Same normalization as eval/omnilingual/compute_wer.py."""
    return "".join(x.lower() for x in str(sentence) if x not in PUNCT)


def load_audio(path, target_sr=16000):
    audio, sr = sf.read(str(path), always_2d=True)
    audio = audio.mean(axis=1)
    if sr != target_sr:
        g = gcd(sr, target_sr)
        audio = resample_poly(audio, target_sr // g, sr // g)
    return audio.astype(np.float32)


def pick_device(requested):
    if requested:
        return requested
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="HF model id or local path")
    p.add_argument("--tsv", required=True)
    p.add_argument("--audio_dir", default=None, help="Prefix for relative audio paths")
    p.add_argument("--path_col", default=None, help="Column with audio filename (auto-detected)")
    p.add_argument("--ref_col", default=None, help="Column with reference text (auto-detected)")
    p.add_argument("--out", required=True)
    p.add_argument("--device", default=None)
    p.add_argument("--language", default="es", help="Whisper decoder language prefix")
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()

    from transformers import AutoConfig, AutoProcessor

    device = pick_device(args.device)
    print(f"Device: {device}")

    df = pd.read_csv(args.tsv, sep="\t")

    path_col = args.path_col or next(
        c for c in ["path", "audio_file", "file", "filename"] if c in df.columns
    )
    ref_col = args.ref_col or next(
        c for c in ["transcript_for_asr", "sentence", "transcription", "text"] if c in df.columns
    )
    print(f"Columns: path={path_col!r} ref={ref_col!r}  ({len(df)} rows)")

    df = df.dropna(subset=[path_col, ref_col])
    if args.limit:
        df = df.head(args.limit)

    config = AutoConfig.from_pretrained(args.model)
    is_whisper = config.model_type == "whisper"
    processor = AutoProcessor.from_pretrained(args.model)

    if is_whisper:
        from transformers import AutoModelForSpeechSeq2Seq as ModelCls
    else:
        from transformers import AutoModelForCTC as ModelCls

    model = ModelCls.from_pretrained(args.model).to(device).eval()
    print(f"Model: {args.model}  ({config.model_type}, "
          f"{sum(p.numel() for p in model.parameters())/1e6:.0f}M params)")

    audio_dir = Path(args.audio_dir) if args.audio_dir else None
    rows, total_audio_s = [], 0.0
    t0 = time.time()

    for i, row in enumerate(df.itertuples(index=False), 1):
        fname = getattr(row, path_col)
        apath = audio_dir / fname if audio_dir else Path(fname)
        try:
            audio = load_audio(apath)
            total_audio_s += len(audio) / 16000
            inputs = processor(
                audio, sampling_rate=16000, return_tensors="pt",
                **({"return_attention_mask": True} if not is_whisper else {}),
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                if is_whisper:
                    ids = model.generate(
                        inputs["input_features"],
                        language=args.language,
                        task="transcribe",
                        max_new_tokens=220,
                    )
                    text = processor.batch_decode(ids, skip_special_tokens=True)[0]
                else:
                    logits = model(**inputs).logits
                    text = processor.batch_decode(logits.argmax(-1).cpu().numpy())[0]
        except Exception as e:  # noqa: BLE001
            print(f"  WARNING {apath}: {e}")
            text = "__ERROR__"

        rows.append({"path": fname, "gold": getattr(row, ref_col), "transcription": text})
        if i % 20 == 0 or i == len(df):
            print(f"  {i}/{len(df)}  ({time.time()-t0:.0f}s)")

    out = pd.DataFrame(rows)
    out = out[out["transcription"] != "__ERROR__"].copy()
    out["gold_norm"] = out["gold"].apply(remove_punct)
    out["hyp_norm"] = out["transcription"].apply(remove_punct)
    out = out[out["gold_norm"].str.strip() != ""]

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, sep="\t", index=False)

    w = wer(out["gold_norm"].tolist(), out["hyp_norm"].tolist())
    c = cer(out["gold_norm"].tolist(), out["hyp_norm"].tolist())
    elapsed = time.time() - t0

    print()
    print(f"  model        : {args.model}")
    print(f"  utterances   : {len(out)}")
    print(f"  audio        : {total_audio_s/60:.1f} min")
    print(f"  wall clock   : {elapsed/60:.1f} min  ({total_audio_s/max(elapsed,1e-9):.1f}x realtime)")
    print(f"  WER          : {w*100:.2f}%")
    print(f"  CER          : {c*100:.2f}%")
    print(f"  saved        : {args.out}")


if __name__ == "__main__":
    main()
