#!/usr/bin/env python3
"""
Build training manifests for SIMBig 2026 Task 1 from the v27 / v5 releases.

Reads
    data/scripted/qxp/{validated.tsv, clip_durations.tsv, clips/}
    data/shared-task/validated_sentences_curated.tsv     <- official reference text
    data/spontaneous/{ss-corpus-qxp.tsv, audios/}
    data/splits/silver_spontaneous/*.tsv                 <- auto transcripts (reused)

Writes data/manifests/sharedtask/
    train.tsv  dev_scripted.tsv  dev_spontaneous.tsv
    heldout_scripted.tsv  heldout_spontaneous.tsv        <- never trained on
    stats.json

Manifest columns: path, text, duration_s, speaker, source, label_type, sentence_id


DECISIONS ENCODED HERE (agreed 2026-10-07) -- change them with the flags, not by
editing, so a run stays reproducible from its command line:

 1. Reference text for scripted speech is `sentence_curated`, never the raw
    `sentence`. The organisers curated it precisely because it is what the
    evaluation will score against; 627 of 2,067 sentences differ in spelling.

 2. Clips longer than --max_s (default 30 s) and shorter than --min_s are
    dropped. Spontaneous speech holds a 28.6-minute outlier that would blow up
    memory on its own. 30 s is the paper's Whisper cut-off; its 20 s wav2vec2
    cut-off costs far too much here, because spontaneous answers average 17.5 s:
      cap   usable audio   dev clips
      20 s      16.8 h        195
      30 s      25.8 h        260
    Batches are built by total duration (see 03_train.py), so long clips cost
    throughput rather than risking an OOM.

 3. Spontaneous clips the organisers marked `split == "test"` (296 clips) are
    held out of training AND of dev. Their own test release is separate audio,
    but the rules say the test must not appear in training "at any point", and
    these are the clips they designated. Not worth the risk.

 4. The scripted split is speaker-disjoint, as the paper's was. Note that the
    sentences necessarily overlap across splits: 77 speakers read the same
    2,067 sentences, so no speaker-disjoint split can also separate text.
    `--split_by sentence` builds a text-disjoint split instead, keeping the
    16 duplicate sentence pairs together, at the cost of sharing speakers.

 5. `prompt` is the question read to the speaker, NOT what they said. Only
    `transcription` is a label. Training on `prompt` would teach the model that
    every 20-second answer transcribes to a 5-word question.
"""

import argparse
import json
import sys
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data/manifests/sharedtask"

# The orthography the organisers curated the scripted reference to:
# a-z, ñ and the apostrophe. No accents, no punctuation, no digits.
ALLOWED = set("abcdefghijklmnopqrstuvwxyzñ' ")

# Apostrophe variants collapse to U+0027. In Puno Quechua the apostrophe is
# part of the letter for ejectives (k' p' q' t' ch'), so three Unicode spellings
# of it would be three different phonemes to a CTC head.
APOSTROPHES = {"’": "'", "‘": "'", "´": "'", "`": "'", "ʼ": "'"}

# Accented vowels appear only in Spanish loanwords inside the spontaneous
# transcriptions; the curated convention writes them bare.
DEACCENT = {"á": "a", "é": "e", "í": "i", "ó": "o", "ú": "u",
            "ü": "u", "à": "a", "è": "e", "ì": "i", "ò": "o", "ù": "u"}


def normalize(text, keep_accents=False):
    """Fold text into the organisers' curated orthography.

    The scripted reference already follows it. The spontaneous transcriptions
    do not: they carry 28 off-convention characters — digits, braces, brackets,
    a stray control character, curly apostrophes and Spanish accents. Left in,
    those become CTC output symbols the model can never usefully predict.

    Verified against data/shared-task/validated_sentences_curated.tsv, which has
    zero hyphens and zero accented vowels in 2,067 sentences: it writes the
    dubitative suffix as `chaycha` (not `chayrachá`) and reduplication as two
    words (`phukuspa phukuspa`), exactly what this produces.

    `keep_accents` exists because the test's own scoring only applies NFC,
    lowercasing and punctuation removal — it does NOT strip accents. The
    scripted reference has none, but the spontaneous reference may keep them
    (0.45% of words in the spontaneous gold carry one). Build both and compare
    on dev rather than guessing; see shared-task/IMPROVEMENTS.md.
    """
    text = unicodedata.normalize("NFC", str(text)).lower()
    allowed = ALLOWED | set(DEACCENT) if keep_accents else ALLOWED
    out = []
    for c in text:
        c = APOSTROPHES.get(c, c)
        if not keep_accents:
            c = DEACCENT.get(c, c)
        out.append(c if c in allowed else " ")
    return " ".join("".join(out).split())


def require(path, hint):
    if not Path(path).exists():
        sys.exit(f"Missing {path}\n  {hint}")
    return path


# --------------------------------------------------------------------------
# Scripted
# --------------------------------------------------------------------------
def load_scripted(args):
    base = ROOT / "data/scripted/qxp"
    require(base / "validated.tsv", "Run shared-task/00_download.py")
    require(ROOT / "data/shared-task/validated_sentences_curated.tsv",
            "git pull -- it ships in the repo")

    v = pd.read_csv(base / "validated.tsv", sep="\t", quoting=3)
    dur = pd.read_csv(base / "clip_durations.tsv", sep="\t", quoting=3)
    dur.columns = ["path", "duration_ms"]
    cur = pd.read_csv(ROOT / "data/shared-task/validated_sentences_curated.tsv",
                      sep="\t", quoting=3)

    df = v.merge(dur, on="path", how="left").merge(cur, on="sentence_id", how="left")

    missing = df.sentence_curated.isna().sum()
    if missing:
        print(f"  WARNING: {missing} clips have no curated text; falling back to `sentence`")
        df["sentence_curated"] = df.sentence_curated.fillna(df.sentence)

    out = pd.DataFrame({
        "path": df.path.map(lambda p: str(base / "clips" / p)),
        "text": df.sentence_curated.map(lambda t: normalize(t, args.keep_accents)),
        "duration_s": df.duration_ms / 1000.0,
        "speaker": df.client_id,
        "source": "scripted",
        "label_type": "gold",
        "sentence_id": df.sentence_id,
    })
    print(f"  scripted: {len(out):,} validated clips, {out.duration_s.sum()/3600:.2f} h, "
          f"{out.speaker.nunique()} speakers, {out.sentence_id.nunique():,} sentences")
    return out


# --------------------------------------------------------------------------
# Spontaneous
# --------------------------------------------------------------------------
def load_spontaneous(args):
    base = ROOT / "data/spontaneous"
    require(base / "ss-corpus-qxp.tsv", "Run shared-task/00_download.py")
    # Default quoting, NOT quoting=3: unlike the Common Voice files, this one is
    # standard-CSV quoted, and 21 transcriptions carry embedded double quotes
    # that QUOTE_NONE would leave doubled and wrapped in the text.
    s = pd.read_csv(base / "ss-corpus-qxp.tsv", sep="\t")

    # `transcription` is the only label column. `prompt` is the question.
    has_text = s.transcription.notna() & (s.transcription.astype(str).str.strip() != "")
    gold = s[has_text].copy()
    gold["label_type"] = gold.votes.fillna(0).astype(int).map(
        lambda v: "gold" if v >= 1 else "pending"
    )

    g = pd.DataFrame({
        "path": gold.audio_file.map(lambda p: str(base / "audios" / p)),
        "text": gold.transcription.map(lambda t: normalize(t, args.keep_accents)),
        "duration_s": gold.duration_ms / 1000.0,
        "speaker": gold.client_id,
        "source": "spontaneous",
        "label_type": gold.label_type,
        "sentence_id": pd.NA,
        "official_split": gold.split.fillna(""),
    })
    n_gold = (g.label_type == "gold").sum()
    print(f"  spontaneous gold: {n_gold:,} validated + "
          f"{(g.label_type=='pending').sum()} pending, {g.duration_s.sum()/3600:.2f} h")

    # ---- silver: reuse the repo's auto transcripts for clips with no gold text
    silver = pd.DataFrame()
    silver_dir = ROOT / "data/splits/silver_spontaneous"
    if not args.no_silver and silver_dir.exists():
        parts = [pd.read_csv(silver_dir / f"{sp}.tsv", sep="\t", quoting=3)
                 for sp in ("train", "dev", "test") if (silver_dir / f"{sp}.tsv").exists()]
        if parts:
            sv = pd.concat(parts, ignore_index=True).drop_duplicates(subset="path")
            gold_files = set(s.loc[has_text, "audio_file"].astype(str))
            known = s.set_index("audio_file")

            sv = sv[~sv.path.astype(str).isin(gold_files)]          # gold wins
            sv = sv[sv.path.astype(str).isin(known.index)]          # must exist in v5
            meta = known.loc[sv.path.astype(str)]

            silver = pd.DataFrame({
                "path": sv.path.map(lambda p: str(base / "audios" / p)).values,
                "text": sv.sentence.map(lambda t: normalize(t, args.keep_accents)).values,
                "duration_s": (meta.duration_ms / 1000.0).values,
                "speaker": meta.client_id.values,
                "source": "spontaneous",
                "label_type": "silver",
                "sentence_id": pd.NA,
                "official_split": meta.split.fillna("").values,
            })
            print(f"  spontaneous silver: {len(silver):,} clips, "
                  f"{silver.duration_s.sum()/3600:.2f} h")

    return pd.concat([g, silver], ignore_index=True) if len(silver) else g


# --------------------------------------------------------------------------
# Splitting
# --------------------------------------------------------------------------
def split_by_speaker(df, dev_frac, heldout_frac, seed):
    """Assign whole speakers to dev/heldout until each reaches its share of audio."""
    by_spk = (df.groupby("speaker").duration_s.sum()
                .sample(frac=1.0, random_state=seed))
    total = by_spk.sum()
    dev, held, acc_d, acc_h = set(), set(), 0.0, 0.0
    for spk, secs in by_spk.items():
        if acc_h < heldout_frac * total:
            held.add(spk); acc_h += secs
        elif acc_d < dev_frac * total:
            dev.add(spk); acc_d += secs
    where = df.speaker.map(lambda s: "heldout" if s in held else "dev" if s in dev else "train")
    return where


def split_by_sentence(df, dev_frac, heldout_frac, seed):
    """Text-disjoint split. Sentences sharing identical text move together, which
    covers the 16 duplicate pairs the organisers flagged."""
    key = df.groupby("text").size().index.to_series().sample(frac=1.0, random_state=seed)
    secs = df.groupby("text").duration_s.sum()
    total = secs.sum()
    dev, held, acc_d, acc_h = set(), set(), 0.0, 0.0
    for text in key:
        if acc_h < heldout_frac * total:
            held.add(text); acc_h += secs[text]
        elif acc_d < dev_frac * total:
            dev.add(text); acc_d += secs[text]
    return df.text.map(lambda t: "heldout" if t in held else "dev" if t in dev else "train")


def split_by_both(df, dev_frac, heldout_frac, seed, dev_text_frac, heldout_text_frac,
                  train_on_dev_sentences=False):
    """Speaker- AND text-disjoint split: the condition the official test is in.

    The test's read sentences are not among the 2,067 we train on (see
    06_build_lm.py), and its speakers are new. A speaker-only dev rewards a
    model for remembering sentences other speakers read in train. Here dev is
    dev speakers reading dev sentences; train is train speakers reading train
    sentences; the crossings are dropped ("unused"). Dev size is roughly
    dev_frac x dev_text_frac of the audio, so both fractions run larger than
    in the single-axis splits.

    train_on_dev_sentences is the memorisation control: train speakers'
    readings of the dev sentences go to train instead of being dropped. Dev is
    unchanged, so the gap between the two builds on the same dev is what
    having heard a sentence from other speakers is worth.
    """
    spk = split_by_speaker(df, dev_frac, heldout_frac, seed)
    txt = split_by_sentence(df, dev_text_frac, heldout_text_frac, seed)
    out = np.where(spk == txt, spk, "unused")
    if train_on_dev_sentences:
        out = np.where((spk == "train") & (txt == "dev"), "train", out)
    return pd.Series(out, index=df.index)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--min_s", type=float, default=0.4)
    p.add_argument("--max_s", type=float, default=30.0,
                   help="Drop longer clips (decision 2)")
    p.add_argument("--dev_frac", type=float, default=0.05)
    p.add_argument("--heldout_frac", type=float, default=0.05)
    p.add_argument("--split_by", choices=["speaker", "sentence", "both"], default="speaker")
    p.add_argument("--dev_text_frac", type=float, default=0.15,
                   help="--split_by both: share of sentences reserved for dev")
    p.add_argument("--heldout_text_frac", type=float, default=0.15,
                   help="--split_by both: share of sentences reserved for heldout")
    p.add_argument("--train_on_dev_sentences", action="store_true",
                   help="--split_by both: memorisation control -- train on the dev "
                        "sentences as read by train speakers (dev itself unchanged)")
    p.add_argument("--keep_accents", action="store_true",
                   help="Keep accented vowels instead of folding them (see normalize())")
    p.add_argument("--no_silver", action="store_true")
    p.add_argument("--no_pending", action="store_true",
                   help="Drop the 148 transcribed-but-unvalidated spontaneous clips")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default=str(OUT))
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stats = {"config": vars(args)}

    print("Loading corpora")
    scripted = load_scripted(args)
    spont = load_spontaneous(args)

    # ---- filters ----------------------------------------------------------
    def apply_filters(df, name):
        n0, h0 = len(df), df.duration_s.sum() / 3600
        df = df[df.text.str.strip() != ""]
        df = df[df.duration_s.between(args.min_s, args.max_s)]
        df = df[df.path.map(lambda q: Path(q).exists())]
        dropped = n0 - len(df)
        print(f"  {name}: kept {len(df):,}/{n0:,} clips "
              f"({df.duration_s.sum()/3600:.2f} h of {h0:.2f} h), dropped {dropped:,}")
        return df.reset_index(drop=True)

    print(f"\nFiltering to {args.min_s}-{args.max_s}s, non-empty text, audio present")
    scripted = apply_filters(scripted, "scripted")
    spont = apply_filters(spont, "spontaneous")

    if args.no_pending:
        spont = spont[spont.label_type != "pending"].reset_index(drop=True)

    # ---- scripted split ---------------------------------------------------
    print(f"\nScripted split (by {args.split_by})")
    if args.split_by == "both":
        scripted["split"] = split_by_both(scripted, args.dev_frac, args.heldout_frac, args.seed,
                                          args.dev_text_frac, args.heldout_text_frac,
                                          args.train_on_dev_sentences)
        unused = scripted.split == "unused"
        print(f"  {unused.sum():,} clips ({scripted.duration_s[unused].sum()/3600:.2f} h) cross "
              f"a speaker and a sentence boundary -- dropped")
    else:
        splitter = split_by_speaker if args.split_by == "speaker" else split_by_sentence
        scripted["split"] = splitter(scripted, args.dev_frac, args.heldout_frac, args.seed)

    # ---- spontaneous split: honour the organisers' own assignment ---------
    print("Spontaneous split (organisers' `split` column; silver -> train)")

    def spont_split(row):
        off = str(row.get("official_split", "") or "")
        if off == "test":
            return "heldout"          # decision 3
        if off == "dev":
            return "dev"
        if row.label_type == "silver":
            return "train"            # silver never evaluates anything
        return "train"

    spont["split"] = spont.apply(spont_split, axis=1)

    # ---- write ------------------------------------------------------------
    cols = ["path", "text", "duration_s", "speaker", "source", "label_type", "sentence_id"]
    train = pd.concat([scripted[scripted.split == "train"],
                       spont[spont.split == "train"]], ignore_index=True)

    writes = {
        "train": train,
        "dev_scripted": scripted[scripted.split == "dev"],
        "dev_spontaneous": spont[spont.split == "dev"],
        "heldout_scripted": scripted[scripted.split == "heldout"],
        "heldout_spontaneous": spont[spont.split == "heldout"],
    }

    print("\nManifests")
    for name, df in writes.items():
        df = df[cols].reset_index(drop=True)
        df.to_csv(out / f"{name}.tsv", sep="\t", index=False, quoting=3)
        by_src = df.groupby("source").duration_s.sum() / 3600 if len(df) else {}
        detail = "  ".join(f"{k} {v:.2f}h" for k, v in dict(by_src).items())
        stats[name] = {
            "clips": int(len(df)),
            "hours": round(float(df.duration_s.sum() / 3600), 3),
            "speakers": int(df.speaker.nunique()) if len(df) else 0,
            "by_label": {k: int(v) for k, v in df.label_type.value_counts().items()},
        }
        print(f"  {name:22s} {len(df):6,} clips  {df.duration_s.sum()/3600:6.2f} h   {detail}")

    # ---- leakage checks ---------------------------------------------------
    print("\nChecks")
    ok = True
    train_paths = set(train.path)
    for name in ("dev_scripted", "dev_spontaneous", "heldout_scripted", "heldout_spontaneous"):
        overlap = train_paths & set(writes[name].path)
        flag = "OK" if not overlap else f"FAIL ({len(overlap)} shared clips)"
        ok &= not overlap
        print(f"  train vs {name:22s} {flag}")

    tr_spk, dv_spk = set(scripted[scripted.split == "train"].speaker), \
                     set(scripted[scripted.split == "dev"].speaker)
    print(f"  scripted speaker overlap train/dev   "
          f"{'OK' if not (tr_spk & dv_spk) else f'{len(tr_spk & dv_spk)} shared'}")

    tr_txt = set(scripted[scripted.split == "train"].text)
    dv_txt = set(scripted[scripted.split == "dev"].text)
    shared = len(tr_txt & dv_txt)
    stats["scripted_sentence_overlap_dev"] = shared
    print(f"  scripted sentence overlap train/dev  {shared}/{len(dv_txt)} dev sentences "
          f"also in train" + ("   (expected with a speaker split)" if args.split_by == "speaker" else ""))

    (out / "stats.json").write_text(json.dumps(stats, indent=2, default=str))
    print(f"\nWrote {out}/  (stats.json has the full breakdown)")
    if not ok:
        sys.exit("Leakage detected -- refusing to call this a clean build.")
    print("Next: python shared-task/03_train.py --init checkpoints/hf/ft_cpt_silver --out checkpoints/hf/ft_curated")


if __name__ == "__main__":
    main()
