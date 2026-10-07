#!/usr/bin/env python3
"""
Tune LM beam search on dev: which LM, its weight (alpha) and the word bonus
(beta), separately for scripted and spontaneous.

The model runs once per dev clip; its log-probabilities are cached, and the
grid is decoded from the cache on every CPU core. Re-tuning a different LM or
grid costs no GPU time.

Two sanity rows come first and should agree with 04_evaluate.py:
  greedy        argmax decoding of the cached log-probabilities
  beam, no LM   beam search with no LM, which reduces to the same best path

Usage:
  python shared-task/06_build_lm.py --orders 6
  python shared-task/07_tune_lm.py --model checkpoints/hf/ft_curated \
      --lms_scripted scripted_o6 scripted_nodev_o6 \
      --lms_spontaneous spont_gold_o6 spont_all_o6

Choose the LM by the dev number, then read the `scripted_nodev` row next to
it: the gap between the two is how much of the scripted gain is the LM
recalling sentences it was trained on (see 06_build_lm.py).
"""

import argparse
import importlib
import itertools
import json
import multiprocessing as mp
import pickle
import time
from itertools import groupby
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ctc_lm import CTCBeamDecoder, ctc_logprobs

ROOT = Path(__file__).resolve().parent.parent
ev = importlib.import_module("04_evaluate")  # load_audio, score_normalize, pick_device


# --------------------------------------------------------------------------
# Emissions
# --------------------------------------------------------------------------
def dev_logprobs(model, processor, paths, device, batch_size):
    out = []
    for i in range(0, len(paths), batch_size):
        audios = [ev.load_audio(p) for p in paths[i:i + batch_size]]
        out += ctc_logprobs(model, processor, audios, device)
    return out


def load_or_compute(model_dir, man, cache, device, batch_size):
    if cache.exists():
        print(f"Emissions: cached {cache}")
        return pickle.loads(cache.read_bytes())

    from transformers import AutoModelForCTC, AutoProcessor
    device = ev.pick_device(device)
    processor = AutoProcessor.from_pretrained(model_dir)
    model = AutoModelForCTC.from_pretrained(model_dir).to(device).eval()
    vocab = processor.tokenizer.convert_ids_to_tokens(list(range(len(processor.tokenizer))))

    data, t0 = {"vocab": vocab, "model": str(model_dir)}, time.time()
    for domain in ("scripted", "spontaneous"):
        df = pd.read_csv(man / f"dev_{domain}.tsv", sep="\t", quoting=3)
        data[domain] = {
            "refs": [ev.score_normalize(t) for t in df.text],
            "logprobs": dev_logprobs(model, processor, df.path.tolist(), device, batch_size),
        }
    print(f"Emissions: computed on {device} in {time.time()-t0:.0f}s -> {cache}")
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_bytes(pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL))
    del model
    torch.cuda.empty_cache()
    return data


# --------------------------------------------------------------------------
# Decoding, in worker processes
# --------------------------------------------------------------------------
_DATA, _LM_DIR = None, None


def _init(data, lm_dir):
    global _DATA, _LM_DIR
    _DATA, _LM_DIR = data, lm_dir


def greedy(logprobs, vocab):
    skip = {"<pad>", "<s>", "</s>", "<unk>"}
    ids = [i for i, _ in groupby(logprobs.argmax(-1))]
    text = "".join(" " if vocab[i] == "|" else vocab[i] for i in ids if vocab[i] not in skip)
    return " ".join(text.split())


def _decode(job):
    domain, lm, alpha, beta, beam = job
    lps = _DATA[domain]["logprobs"]
    if lm == "greedy":
        hyps = [greedy(lp, _DATA["vocab"]) for lp in lps]
    else:
        dec = CTCBeamDecoder(
            _DATA["vocab"],
            lm_path=None if lm == "none" else _LM_DIR / f"{lm}.arpa",
            lm_weight=alpha, sil_score=beta, beam=beam,
        )
        hyps = [dec.decode(lp) for lp in lps]
    return job, hyps


def score(refs, hyps):
    import jiwer
    pairs = [(r, ev.score_normalize(h)) for r, h in zip(refs, hyps) if r.strip()]
    r, h = [p[0] for p in pairs], [p[1] for p in pairs]
    return jiwer.wer(r, h) * 100, jiwer.cer(r, h) * 100


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--manifest_dir", default=str(ROOT / "data/manifests/sharedtask"))
    p.add_argument("--lm_dir", default=str(ROOT / "checkpoints/lm"))
    p.add_argument("--lms_scripted", nargs="+", default=["scripted_o6", "scripted_nodev_o6", "all_o6"])
    p.add_argument("--lms_spontaneous", nargs="+", default=["spont_gold_o6", "spont_all_o6", "all_o6"])
    p.add_argument("--alphas", type=float, nargs="+", default=[0.25, 0.5, 0.75, 1.0, 1.5, 2.0])
    p.add_argument("--betas", type=float, nargs="+", default=[-1.0, 0.0, 1.0, 2.0])
    p.add_argument("--beam", type=int, default=64)
    p.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 2))
    p.add_argument("--device", default=None)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--cache", default=None,
                   help="Emission cache. Default: checkpoints/emissions/<model>.pkl")
    p.add_argument("--out", default=None, help="CSV of every configuration")
    args = p.parse_args()

    man, lm_dir = Path(args.manifest_dir), Path(args.lm_dir)
    tag = str(Path(args.model)).strip("/").replace("/", "__")
    cache = Path(args.cache) if args.cache else ROOT / "checkpoints/emissions" / f"{tag}.pkl"
    data = load_or_compute(Path(args.model), man, cache, args.device, args.batch_size)

    for lm in set(args.lms_scripted + args.lms_spontaneous):
        if not (lm_dir / f"{lm}.arpa").exists():
            raise SystemExit(f"No {lm_dir / lm}.arpa -- run 06_build_lm.py first")

    jobs = []
    for domain, lms in (("scripted", args.lms_scripted), ("spontaneous", args.lms_spontaneous)):
        jobs += [(domain, "greedy", 0.0, 0.0, 0), (domain, "none", 0.0, 0.0, args.beam)]
        jobs += [(domain, lm, a, b, args.beam)
                 for lm, a, b in itertools.product(lms, args.alphas, args.betas)]

    t0 = time.time()
    rows = []
    ctx = mp.get_context("fork")  # workers inherit the emissions, no copy
    with ctx.Pool(args.workers, initializer=_init, initargs=(data, lm_dir)) as pool:
        for (domain, lm, a, b, beam), hyps in pool.imap_unordered(_decode, jobs):
            wer, cer = score(data[domain]["refs"], hyps)
            rows.append({"domain": domain, "lm": lm, "alpha": a, "beta": b,
                         "beam": beam, "wer": wer, "cer": cer})
    print(f"Decoded {len(jobs)} configurations in {time.time()-t0:.0f}s "
          f"on {args.workers} workers\n")

    df = pd.DataFrame(rows)
    out = Path(args.out) if args.out else lm_dir / f"tune__{tag}.csv"
    df.sort_values(["domain", "wer", "cer"]).to_csv(out, index=False)

    best = {}
    for domain in ("scripted", "spontaneous"):
        d = df[df.domain == domain]
        print(f"{domain:<12}{'lm':<22}{'alpha':>6}{'beta':>6}{'WER %':>8}{'CER %':>8}")
        print("-" * 62)
        for lm in ["greedy", "none"] + [l for l in d.lm.unique() if l not in ("greedy", "none")]:
            r = d[d.lm == lm].sort_values(["wer", "cer"]).iloc[0]
            print(f"{'':<12}{lm:<22}{r.alpha:>6.2f}{r.beta:>6.1f}{r.wer:>8.2f}{r.cer:>8.2f}")
        r = d[~d.lm.isin(["greedy", "none"])].sort_values(["wer", "cer"]).iloc[0]
        best[domain] = r
        print()

    g = df[df.lm == "greedy"].set_index("domain")
    print(f"{'':<14}{'greedy WER':>11}{'CER':>7}{'beam+LM WER':>13}{'CER':>7}")
    for domain in ("scripted", "spontaneous"):
        print(f"{domain:<14}{g.loc[domain, 'wer']:>11.2f}{g.loc[domain, 'cer']:>7.2f}"
              f"{best[domain].wer:>13.2f}{best[domain].cer:>7.2f}"
              f"   ({best[domain].lm}, alpha {best[domain].alpha}, beta {best[domain].beta})")
    print(f"{'mean':<14}{g.wer.mean():>11.2f}{g.cer.mean():>7.2f}"
          f"{(best['scripted'].wer + best['spontaneous'].wer)/2:>13.2f}"
          f"{(best['scripted'].cer + best['spontaneous'].cer)/2:>7.2f}")
    print(f"\nAll configurations -> {out}")
    (out.with_suffix(".best.json")).write_text(json.dumps(
        {d: r.to_dict() for d, r in best.items()}, indent=2, default=float))


if __name__ == "__main__":
    main()
