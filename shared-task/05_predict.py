#!/usr/bin/env python3
"""
Transcribe the released test audio and package the SIMBig 2026 Task 1 submission.

The organisers' format:
  a zip holding one TSV per language, e.g. qxp.tsv, with two columns —
  audio file name, then predicted transcription. No header is specified,
  so we write none; --header adds one if they later ask for it.

Usage:
  python shared-task/05_predict.py \
      --model checkpoints/hf/ft_curated \
      --audio_dir data/test/scripted data/test/spontaneous \
      --out submission/qxp

Writes submission/qxp.tsv and submission/qxp.zip.

The audio file name column carries the basename exactly as delivered
(including its extension), since that is what the organisers match against.
"""

import argparse
import sys
import time
import zipfile
from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly

AUDIO_EXT = {".wav", ".mp3", ".flac", ".m4a", ".ogg", ".opus"}


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


def collect_audio(dirs, template=None):
    """List the audio to transcribe.

    With --template, the organisers' own qxp.tsv drives the list and the row
    order, so the submission matches their file set exactly: every file they
    expect gets a row, in their order, and a file we cannot find is reported
    rather than silently dropped.
    """
    found = {}
    for d in dirs:
        d = Path(d)
        if not d.exists():
            sys.exit(f"Audio directory not found: {d}")
        for p in sorted(d.rglob("*")):
            if p.suffix.lower() not in AUDIO_EXT:
                continue
            if p.name in found:
                sys.exit(f"Duplicate audio file name across directories: {p.name}\n"
                         f"  {found[p.name]}\n  {p}")
            found[p.name] = p

    if template:
        names = [line.split("\t")[0].strip()
                 for line in Path(template).read_text(encoding="utf-8").splitlines()
                 if line.strip()]
        missing = [n for n in names if n not in found]
        if missing:
            sys.exit(f"{len(missing)} file(s) listed in {template} are not in the audio "
                     f"directories, e.g. {missing[:5]}")
        extra = len(found) - len(names)
        if extra:
            print(f"  note: {extra} audio file(s) present but not listed in the template "
                  f"-- following the template")
        return [found[n] for n in names]

    paths = list(found.values())
    if not paths:
        sys.exit(f"No audio files found under: {', '.join(map(str, dirs))}")

    return paths


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="HF model directory or hub id")
    p.add_argument("--audio_dir", required=True, nargs="+",
                   help="One or more directories of test audio (searched recursively)")
    p.add_argument("--template", default=None,
                   help="The organisers' qxp.tsv: drives the file list and row order")
    p.add_argument("--out", default="submission/qxp",
                   help="Output stem; writes <stem>.tsv and <stem>.zip")
    p.add_argument("--device", default=None)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--language", default="es", help="Whisper decoder language prefix")
    p.add_argument("--header", action="store_true",
                   help="Write a header row (the spec does not ask for one)")
    p.add_argument("--no_zip", action="store_true")
    p.add_argument("--lm_config", default=None,
                   help="Beam search with a character LM: the .best.json that "
                        "07_tune_lm.py writes (LM, alpha, beta per domain)")
    p.add_argument("--lm_dir", default=None,
                   help="Where the LMs named in --lm_config live. "
                        "Default: the config's own directory")
    p.add_argument("--metadata", default=None,
                   help="qxp_test_dataset.tsv: its `type` column picks each clip's "
                        "LM. Default: next to --template")
    args = p.parse_args()

    from transformers import AutoConfig, AutoProcessor

    device = pick_device(args.device)
    config = AutoConfig.from_pretrained(args.model)
    is_whisper = config.model_type == "whisper"
    processor = AutoProcessor.from_pretrained(args.model)

    if is_whisper:
        from transformers import AutoModelForSpeechSeq2Seq as ModelCls
    else:
        from transformers import AutoModelForCTC as ModelCls
    model = ModelCls.from_pretrained(args.model).to(device).eval()

    paths = collect_audio(args.audio_dir, args.template)
    print(f"Model : {args.model} ({config.model_type}) on {device}")
    print(f"Audio : {len(paths)} files from {', '.join(map(str, args.audio_dir))}")

    # Beam search: each clip decodes with its own domain's LM and weights. A
    # clip with no known domain is an error, never a silent fall back to greedy.
    decoders, domain_of = {}, {}
    if args.lm_config:
        if is_whisper:
            sys.exit("--lm_config applies to CTC models only")
        import pandas as pd
        from ctc_lm import ctc_logprobs, decoders_from_config
        meta = Path(args.metadata) if args.metadata else (
            Path(args.template).parent / "qxp_test_dataset.tsv" if args.template else None)
        if meta is None or not meta.exists():
            sys.exit("--lm_config needs --metadata (qxp_test_dataset.tsv) for each clip's type")
        domain_of = dict(pd.read_csv(meta, sep="\t", quoting=3)[["file_name", "type"]].values)
        vocab = processor.tokenizer.convert_ids_to_tokens(list(range(len(processor.tokenizer))))
        decoders = decoders_from_config(args.lm_config, vocab, args.lm_dir)
        unknown = [p.name for p in paths if domain_of.get(p.name) not in decoders]
        if unknown:
            sys.exit(f"{len(unknown)} clips have no type in {meta} or no LM for "
                     f"their type, e.g. {unknown[:3]}")
        for d, n in pd.Series([domain_of[p.name] for p in paths]).value_counts().items():
            how = "beam search + LM" if decoders[d].has_lm else "no LM (greedy path)"
            print(f"LM    : {d:<12} {n} clips, {how}")

    rows, failures, total_s = [], [], 0.0
    t0 = time.time()

    # CTC batches cleanly with an attention mask; Whisper we keep one at a time
    # so a long clip cannot distort a padded batch.
    step = 1 if is_whisper else args.batch_size
    for i in range(0, len(paths), step):
        chunk = paths[i:i + step]
        audios, kept = [], []
        for path in chunk:
            try:
                a = load_audio(path)
                audios.append(a)
                kept.append(path)
                total_s += len(a) / 16000
            except Exception as e:  # noqa: BLE001
                failures.append((path, str(e)))
                rows.append((path.name, ""))
        if not audios:
            continue

        try:
            with torch.no_grad():
                if is_whisper:
                    inputs = processor(audios[0], sampling_rate=16000, return_tensors="pt")
                    ids = model.generate(
                        inputs["input_features"].to(device),
                        language=args.language, task="transcribe", max_new_tokens=440,
                    )
                    texts = processor.batch_decode(ids, skip_special_tokens=True)
                elif decoders:
                    lps = ctc_logprobs(model, processor, audios, device)
                    texts = [decoders[domain_of[path.name]].decode(lp)
                             for path, lp in zip(kept, lps)]
                else:
                    inputs = processor(
                        audios, sampling_rate=16000, return_tensors="pt",
                        padding=True, return_attention_mask=True,
                    )
                    inputs = {k: v.to(device) for k, v in inputs.items()}
                    logits = model(**inputs).logits
                    # Trim each clip to its own frame count before decoding.
                    # Decoding the full padded width appends whatever the model
                    # emits over the zero padding to the shorter clips, which
                    # measured as 0.99% -> 20.06% CER on a sample.
                    ids = logits.argmax(-1).cpu().numpy()
                    frames = model._get_feat_extract_output_lengths(
                        inputs["attention_mask"].sum(-1)
                    ).cpu().numpy()
                    texts = [processor.decode(ids[j, :int(frames[j])])
                             for j in range(len(ids))]
        except Exception as e:  # noqa: BLE001
            for path in kept:
                failures.append((path, str(e)))
                rows.append((path.name, ""))
            continue

        for path, text in zip(kept, texts):
            rows.append((path.name, " ".join(text.split())))

        n = min(i + step, len(paths))
        if n % 100 < step or n == len(paths):
            print(f"  {n}/{len(paths)}  ({time.time()-t0:.0f}s)")

    # Preserve the delivered order, and guarantee one row per audio file.
    order = {p.name: k for k, p in enumerate(paths)}
    rows.sort(key=lambda r: order[r[0]])
    assert len(rows) == len(paths), f"{len(rows)} rows for {len(paths)} files"

    out_tsv = Path(args.out).with_suffix(".tsv")
    out_tsv.parent.mkdir(parents=True, exist_ok=True)
    # Written by hand rather than through csv: the format is "name<TAB>text",
    # and every csv dialect either quotes the apostrophes that are part of
    # Quechua spelling or escapes something the organisers' parser will not
    # unescape. Our text is already restricted to [a-z ñ ' space], so there is
    # no delimiter to escape — but assert it instead of assuming it.
    with open(out_tsv, "w", newline="", encoding="utf-8") as f:
        if args.header:
            f.write("audio\ttranscription\n")
        for name, text in rows:
            if "\t" in text or "\n" in text or "\r" in text:
                sys.exit(f"Transcription for {name} contains a tab or newline; "
                         f"that would corrupt the submission.")
            f.write(f"{name}\t{text}\n")

    # A file that failed (e.g. CUDA out of memory) has an empty row in the TSV.
    # Never package that: a run that lost 560 of 659 rows to an OOM once
    # produced a valid-looking zip.
    if not args.no_zip and not failures:
        out_zip = Path(args.out).with_suffix(".zip")
        with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.write(out_tsv, arcname=out_tsv.name)

    empty = sum(1 for _, t in rows if not t.strip())
    elapsed = time.time() - t0
    print()
    print(f"  files        : {len(rows)}")
    print(f"  empty output : {empty}" + ("   <-- investigate before submitting" if empty else ""))
    print(f"  audio        : {total_s/60:.1f} min")
    print(f"  wall clock   : {elapsed/60:.1f} min ({total_s/max(elapsed,1e-9):.0f}x realtime)")
    print(f"  tsv          : {out_tsv}")
    if not args.no_zip and not failures:
        print(f"  zip          : {Path(args.out).with_suffix('.zip')}")
    if failures:
        print(f"\n  {len(failures)} file(s) failed:")
        for path, err in failures[:10]:
            print(f"    {path.name}: {err[:200]}")
        sys.exit(f"\n{len(failures)} file(s) failed -- zip NOT written. "
                 f"The TSV has empty rows for them; fix the cause and re-run.")


if __name__ == "__main__":
    main()
