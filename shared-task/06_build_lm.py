#!/usr/bin/env python3
"""
Build character n-gram LMs for beam search (IMPROVEMENTS.md item 3).

Text comes from train.tsv only. Dev and held-out transcripts never enter an
LM: that would leak, and the dev number would become fiction.

One LM per text source, because the test metadata says which clips are
scripted and which spontaneous, so each domain can get its own LM and weight:

  scripted          distinct scripted sentences in train
  scripted_nodev    the same, minus every sentence that is also in dev
  spont_gold        spontaneous gold + pending transcriptions
  spont_all         spont_gold + the silver auto-transcriptions
  all               scripted + spont_all
  all_nodev         all, minus every sentence that is also in dev

Why the `_nodev` sources. All 1,188 dev sentences also occur in train, read
by other speakers, so a high-order character LM can recall a dev sentence
whole -- and tuned on dev, it learns to. The test's read sentences are NOT
from those 2,067: our greedy hypotheses on the test audio sit as far from
the nearest pool sentence (median normalised edit distance 0.48) as dev
hypotheses do once their own sentence is removed from the pool (0.52), and
nowhere near the 0.00 of dev sentences that are in it. So tune the scripted
LM weight on a `_nodev` LM: that is the condition the test is in.

Usage:
  python shared-task/06_build_lm.py                      # orders 6 8 10 12
  python shared-task/06_build_lm.py --orders 10 --sources scripted spont_all
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import pandas as pd

from ctc_lm import build_arpa, check_arpa, text_to_symbols

ROOT = Path(__file__).resolve().parent.parent

SOURCES = ("scripted", "scripted_nodev", "spont_gold", "spont_all", "all", "all_nodev")


def source_texts(man):
    train = pd.read_csv(man / "train.tsv", sep="\t", quoting=3)
    train["text"] = train.text.astype(str)
    dev = set(pd.read_csv(man / "dev_scripted.tsv", sep="\t", quoting=3).text.astype(str))

    def uniq(s):
        # Scripted sentences are read by ~11 speakers each: that repetition is
        # an artefact of collection, not of the language, so count each once.
        return sorted(set(t for t in s if t.strip()))

    scripted = uniq(train.text[train.source == "scripted"])
    spont = train[train.source == "spontaneous"]
    gold = uniq(spont.text[spont.label_type != "silver"])
    spont_all = uniq(spont.text)
    return {
        "scripted": scripted,
        "scripted_nodev": [t for t in scripted if t not in dev],
        "spont_gold": gold,
        "spont_all": spont_all,
        "all": sorted(set(scripted) | set(spont_all)),
        "all_nodev": sorted((set(scripted) | set(spont_all)) - dev),
    }


def char_perplexity(model, texts):
    """Per-symbol perplexity, word delimiters and </s> included."""
    logp, n = 0.0, 0
    for t in texts:
        syms = " ".join(text_to_symbols(t))
        logp += model.score(syms, bos=True, eos=True)
        n += len(syms.split()) + 1
    return 10 ** (-logp / n)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest_dir", default=str(ROOT / "data/manifests/sharedtask"))
    p.add_argument("--out_dir", default=str(ROOT / "checkpoints/lm"))
    p.add_argument("--orders", type=int, nargs="+", default=[6, 8, 10, 12])
    p.add_argument("--sources", nargs="+", default=list(SOURCES), choices=SOURCES)
    args = p.parse_args()

    import kenlm

    man, out = Path(args.manifest_dir), Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    texts = source_texts(man)
    dev = {d: pd.read_csv(man / f"dev_{d}.tsv", sep="\t", quoting=3).text.astype(str).tolist()
           for d in ("scripted", "spontaneous")}

    print(f"{'source':<16}{'lines':>7}{'chars':>9}")
    for s in args.sources:
        print(f"{s:<16}{len(texts[s]):>7,}{sum(len(t) for t in texts[s]):>9,}")

    print(f"\n{'lm':<22}{'n-grams':>10}{'max |sum-1|':>13}"
          f"{'ppl dev_scr':>13}{'ppl dev_spo':>13}{'secs':>7}")
    stats = {}
    for s in args.sources:
        for order in args.orders:
            name = f"{s}_o{order}"
            path = out / f"{name}.arpa"
            t0 = time.time()
            counts = build_arpa(texts[s], order, path)
            worst = check_arpa(path, texts[s])
            if worst > 1e-3:
                sys.exit(f"{path}: a context's probabilities sum to 1 +- {worst:.2e}. "
                         "The ARPA is malformed; not using it.")
            m = kenlm.Model(str(path))
            ppl = {d: char_perplexity(m, dev[d]) for d in dev}
            stats[name] = {"source": s, "order": order, "lines": len(texts[s]),
                           "ngrams": counts, "max_sum_dev": worst,
                           **{f"ppl_dev_{d}": v for d, v in ppl.items()}}
            print(f"{name:<22}{sum(counts):>10,}{worst:>13.1e}"
                  f"{ppl['scripted']:>13.3f}{ppl['spontaneous']:>13.3f}"
                  f"{time.time()-t0:>7.1f}")

    (out / "stats.json").write_text(json.dumps(stats, indent=2))
    print(f"\nWrote {len(stats)} LMs -> {out}")


if __name__ == "__main__":
    main()
