#!/usr/bin/env python3
"""
Score a model the way SIMBig 2026 Task 1 will score it.

The organisers rank on a Mean Ranking over four numbers — WER and CER for
scripted and for spontaneous speech — so a model that wins one domain and
loses the other does not win. This prints all four and their mean, and refuses
to report a single headline figure.

Text is normalised the way the test README specifies: Unicode NFC, lowercase,
punctuation removed, apostrophe and ñ kept.

Usage:
  python shared-task/04_evaluate.py --model checkpoints/hf/ft_curated

  # a different split, e.g. the text-disjoint dev from IMPROVEMENTS.md item 1
  python shared-task/04_evaluate.py --model ... \
      --manifest_dir data/manifests/sharedtask_textdisjoint

  # the held-out sets, once, at the very end
  python shared-task/04_evaluate.py --model ... --split heldout
"""

import argparse
import json
import sys
import time
import unicodedata
from math import gcd
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from scipy.signal import resample_poly

ROOT = Path(__file__).resolve().parent.parent

# The test README: "Unicode NFC, lowercase, punctuation removed.
# The apostrophe ' and ñ are kept."
PUNCT = set("?!¿¡.,;:\"“”«»()[]{}…—–-")


def score_normalize(text):
    text = unicodedata.normalize("NFC", str(text)).lower()
    return " ".join("".join(" " if c in PUNCT else c for c in text).split())


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
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def decode_ctc_batch(model, processor, audios, device):
    """Decode a padded CTC batch, trimming each clip to its own length.

    Decoding the full padded width is a silent corruption: the model emits
    symbols over the zero padding and they land at the end of the shorter
    clips' transcripts. On a 12-clip sample that turned 0.99% CER into 20.06%.
    The true frame count per clip comes from the attention mask, run through
    the conv frontend's downsampling.
    """
    inputs = processor(audios, sampling_rate=16000, return_tensors="pt",
                       padding=True, return_attention_mask=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        logits = model(**inputs).logits

    ids = logits.argmax(-1).cpu().numpy()
    if "attention_mask" in inputs:
        frames = model._get_feat_extract_output_lengths(
            inputs["attention_mask"].sum(-1)
        ).cpu().numpy()
    else:
        frames = [ids.shape[1]] * len(ids)
    return [processor.decode(ids[i, :int(frames[i])]) for i in range(len(ids))]


def transcribe(model, processor, paths, device, is_whisper, batch_size, language,
               decoder=None):
    out = []
    step = 1 if is_whisper else batch_size
    for i in range(0, len(paths), step):
        audios = [load_audio(p) for p in paths[i:i + step]]
        if decoder is not None:
            from ctc_lm import ctc_logprobs
            out += [decoder.decode(lp)
                    for lp in ctc_logprobs(model, processor, audios, device)]
        elif is_whisper:
            with torch.no_grad():
                feats = processor(audios[0], sampling_rate=16000,
                                  return_tensors="pt")["input_features"].to(device)
                ids = model.generate(feats, language=language, task="transcribe",
                                     max_new_tokens=440)
            out += processor.batch_decode(ids, skip_special_tokens=True)
        else:
            out += decode_ctc_batch(model, processor, audios, device)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--manifest_dir", default=str(ROOT / "data/manifests/sharedtask"))
    p.add_argument("--split", default="dev", choices=["dev", "heldout"])
    p.add_argument("--device", default=None)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--language", default="es")
    p.add_argument("--limit", type=int, default=None, help="For a quick check")
    p.add_argument("--save_predictions", default=None,
                   help="Directory to write per-utterance hypotheses")
    p.add_argument("--lm_config", default=None,
                   help="Beam search with a character LM: the .best.json that "
                        "07_tune_lm.py writes (LM, alpha, beta per domain)")
    p.add_argument("--lm_dir", default=None,
                   help="Where the LMs named in --lm_config live. "
                        "Default: the config's own directory")
    args = p.parse_args()

    from transformers import AutoConfig, AutoProcessor
    import jiwer

    man = Path(args.manifest_dir)
    domains = {}
    for domain in ("scripted", "spontaneous"):
        tsv = man / f"{args.split}_{domain}.tsv"
        if tsv.exists():
            domains[domain] = pd.read_csv(tsv, sep="\t", quoting=3)
    if not domains:
        sys.exit(f"No {args.split}_*.tsv manifests in {man}. Run 01_build_manifests.py.")

    device = pick_device(args.device)
    config = AutoConfig.from_pretrained(args.model)
    is_whisper = config.model_type == "whisper"
    processor = AutoProcessor.from_pretrained(args.model)
    if is_whisper:
        from transformers import AutoModelForSpeechSeq2Seq as ModelCls
    else:
        from transformers import AutoModelForCTC as ModelCls
    model = ModelCls.from_pretrained(args.model).to(device).eval()

    decoders = {}
    if args.lm_config:
        if is_whisper:
            sys.exit("--lm_config applies to CTC models only")
        from ctc_lm import decoders_from_config
        vocab = processor.tokenizer.convert_ids_to_tokens(list(range(len(processor.tokenizer))))
        decoders = decoders_from_config(args.lm_config, vocab, args.lm_dir)

    print(f"Model      : {args.model} ({config.model_type}) on {device}")
    print(f"Manifests  : {man}  split={args.split}")
    for domain in domains:
        print(f"Decoding   : {domain:<12} "
              + ("beam search + LM" if domain in decoders else "greedy"))
    print()

    results, t0 = {}, time.time()
    for domain, df in domains.items():
        if args.limit:
            df = df.head(args.limit)
        paths = df.path.tolist()
        hyps = transcribe(model, processor, paths, device, is_whisper,
                          args.batch_size, args.language, decoders.get(domain))

        refs = [score_normalize(t) for t in df.text]
        hyps_n = [score_normalize(t) for t in hyps]
        pairs = [(r, h) for r, h in zip(refs, hyps_n) if r.strip()]
        refs_k, hyps_k = [r for r, _ in pairs], [h for _, h in pairs]

        results[domain] = {
            "clips": len(pairs),
            "hours": round(float(df.duration_s.sum() / 3600), 3),
            "wer": jiwer.wer(refs_k, hyps_k) * 100,
            "cer": jiwer.cer(refs_k, hyps_k) * 100,
        }

        if args.save_predictions:
            d = Path(args.save_predictions)
            d.mkdir(parents=True, exist_ok=True)
            pd.DataFrame({"path": df.path, "ref": refs, "hyp": hyps_n}).to_csv(
                d / f"{args.split}_{domain}.tsv", sep="\t", index=False, quoting=3)

    # ---- report ----------------------------------------------------------
    print(f"{'domain':<14}{'clips':>7}{'hours':>8}{'WER %':>9}{'CER %':>9}")
    print("-" * 47)
    for domain, r in results.items():
        print(f"{domain:<14}{r['clips']:>7,}{r['hours']:>8.2f}"
              f"{r['wer']:>9.2f}{r['cer']:>9.2f}")

    if len(results) > 1:
        mw = sum(r["wer"] for r in results.values()) / len(results)
        mc = sum(r["cer"] for r in results.values()) / len(results)
        print("-" * 47)
        print(f"{'mean':<14}{'':>7}{'':>8}{mw:>9.2f}{mc:>9.2f}")
        print("\nThe ranking is a Mean Ranking over these four numbers, so both "
              "domains\ncount equally -- read the mean, not the scripted column.")
        results["mean"] = {"wer": mw, "cer": mc}
    else:
        print(f"\nOnly {list(results)[0]} was scored; the ranking needs both domains.")

    print(f"\n{time.time()-t0:.0f}s total")

    out = Path(args.model) / f"metrics_{args.split}{'_lm' if decoders else ''}.json"
    try:
        out.write_text(json.dumps(results, indent=2))
        print(f"Saved {out}")
    except OSError:
        pass  # a hub id, not a local directory


if __name__ == "__main__":
    main()
