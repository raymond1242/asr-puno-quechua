#!/usr/bin/env python3
"""
Collect the 2026-10-09 measurements into the tables the decisions use.

Reads results/2026-10-09/{dev,heldout}__<system>__<decoding>.json, written by
2026-10-09_eval_dev.sh and 2026-10-09_eval_heldout.sh. Every row carries the
four numbers the ranking averages over -- scripted WER and CER, spontaneous
WER and CER -- plus the mean WER and mean CER. Never WER alone.

Deltas are signed so that positive = better (lower error) for the system named
first, in both WER and CER.

  python shared-task/experiments/collect_2026-10-09.py
  python shared-task/experiments/collect_2026-10-09.py --markdown
"""

import argparse
import json
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
R = ROOT / "results/2026-10-09"
COLS = ("s_wer", "s_cer", "p_wer", "p_cer", "m_wer", "m_cer")
HEAD = ("scr WER", "scr CER", "spo WER", "spo CER", "mean WER", "mean CER")


def load():
    rows = {}
    for f in sorted(R.glob("*__*__*.json")):
        split, name, dec = f.stem.split("__")
        m = json.loads(f.read_text())
        s, p = m["scripted"], m["spontaneous"]
        rows[(split, name, dec)] = {
            "s_wer": s["wer"], "s_cer": s["cer"], "p_wer": p["wer"], "p_cer": p["cer"],
            "m_wer": (s["wer"] + p["wer"]) / 2, "m_cer": (s["cer"] + p["cer"]) / 2,
            "manifest": Path(m["meta"]["manifest_dir"]).name,
        }
    return rows


def fmt(vals):
    return "".join(f"{v:>10.2f}" for v in vals)


def seed_stats(rows, split, prefix, dec):
    seeds = [rows[k] for k in rows if k[0] == split and k[2] == dec
             and k[1] in (f"{prefix}_s42", f"{prefix}_s43", f"{prefix}_s44")]
    if len(seeds) < 2:
        return None
    out = {}
    for c in COLS:
        xs = [r[c] for r in seeds]
        out[c] = (st.mean(xs), st.stdev(xs), min(xs), max(xs))
    return out, len(seeds)


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--markdown", action="store_true")
    args = a.parse_args()
    rows = load()
    if not rows:
        raise SystemExit(f"No results in {R}")

    for dec in ("v3", "greedy"):
        print(f"\n######## decoding: {dec}")
        print(f"{'split':<8}{'system':<20}{'manifest':<18}" + "".join(f"{h:>10}" for h in HEAD))
        for (split, name, d), r in sorted(rows.items()):
            if d == dec:
                print(f"{split:<8}{name:<20}{r['manifest']:<18}{fmt(r[c] for c in COLS)}")

        for split in ("dev", "heldout"):
            for prefix in ("both", "st"):
                got = seed_stats(rows, split, prefix, dec)
                if not got:
                    continue
                s, n = got
                print(f"\n{split} {prefix}: mean of {n} seeds (sample std; min-max)")
                for c, h in zip(COLS, HEAD):
                    m, sd, lo, hi = s[c]
                    print(f"  {h:<9} {m:6.2f}  std {sd:4.2f}  range {lo:5.2f}-{hi:5.2f} ({hi-lo:4.2f})")
                for fusion in ("logit", "prob"):
                    ens = rows.get((split, f"{prefix}_ens3{fusion}", dec))
                    if not ens:
                        continue
                    print(f"  {fusion}-ensemble vs seed mean (positive = ensemble better):"
                          + "".join(f"  {h} {s[c][0]-ens[c]:+.2f}" for c, h in zip(COLS, HEAD)))
                    best = min((rows[(split, f'{prefix}_s{x}', dec)] for x in (42, 43, 44)
                                if (split, f'{prefix}_s{x}', dec) in rows), key=lambda r: r["m_wer"])
                    print(f"  {fusion}-ensemble vs best seed (by mean WER):"
                          + "".join(f"  {h} {best[c]-ens[c]:+.2f}" for c, h in zip(COLS, HEAD)))

        base = rows.get(("dev", "both_s42", dec))
        for mask in ("0.2", "0.5"):
            r = rows.get(("dev", f"both_s42_mask{mask}", dec))
            if base and r:
                print(f"\nmask {mask} vs mask 0.05, both seed 42 (positive = mask better):"
                      + "".join(f"  {h} {base[c]-r[c]:+.2f}" for c, h in zip(COLS, HEAD)))
            r43 = rows.get(("dev", f"both_s43_mask{mask}", dec))
            b43 = rows.get(("dev", "both_s43", dec))
            if r43 and b43:
                print(f"mask {mask} vs mask 0.05, both seed 43 (positive = mask better):"
                      + "".join(f"  {h} {b43[c]-r43[c]:+.2f}" for c, h in zip(COLS, HEAD)))

    if args.markdown:
        print("\n| split | system | decoding | " + " | ".join(HEAD) + " |")
        print("|---|---|---|" + "---:|" * len(HEAD))
        for (split, name, d), r in sorted(rows.items()):
            print(f"| {split} | {name} | {d} | " + " | ".join(f"{r[c]:.2f}" for c in COLS) + " |")


if __name__ == "__main__":
    main()
