# Improvements to try

**Baseline to beat** (`xls-r-cpt-qxp-silver` converted, scored on our dev):
scripted 16.08 WER / 3.77 CER, spontaneous 10.23 WER / 1.39 CER,
mean **13.15 WER / 2.58 CER**.

Ideas parked for after the first end-to-end run works. Each one names what to
measure, so none of them becomes a change we believe in without evidence.

Measure everything on **`dev_scripted` + `dev_spontaneous`**, and report the
mean of the four numbers (scripted WER, scripted CER, spontaneous WER,
spontaneous CER) — that is what the organisers rank on. A change that wins on
scripted and loses on spontaneous is not a win.

---

## 1. A dev split without sentence overlap

**Status:** decided to keep the current speaker-disjoint split, matching the
paper. This is the follow-up experiment, not a replacement.

**The situation.** 22,727 recordings come from only 2,067 sentences, each read
by roughly 11 people. So every sentence in `dev_scripted` also appears in
`train` — 1,188 of 1,188. No speaker-disjoint split can avoid this.

**Why it might matter.** If the model partly memorises the 2,067 sentences
rather than learning to hear, dev WER flatters it, and we would pick
checkpoints and hyperparameters on an inflated number.

**Why it might not.** The official scripted test is read speech, very likely
drawn from the same sentence corpus. If so the overlap is *realistic*, and a
text-disjoint dev would be needlessly pessimistic.

**What to do.** Build both and compare — one command:

```bash
python shared-task/01_build_manifests.py \
    --split_by sentence --out data/manifests/sharedtask_textdisjoint
python shared-task/04_evaluate.py \
    --model checkpoints/hf/ft_curated \
    --manifest_dir data/manifests/sharedtask_textdisjoint
```

**How to read it.** A small gap means little memorisation and the current dev is
trustworthy. A large gap tells us how much of our dev number is memory, and the
text-disjoint split becomes the better one to select checkpoints on.

---

## 1b. Note on dev difficulty

One dev speaker, `e990fbdf` (59 of 1,493 scripted clips), scores **106% WER**.
Their audio is technically fine — normal RMS and peak, no clipping — the model
simply fails on them, and a speaker-disjoint split means they were never seen
in training. The real test has 7 speakers, its scripted half in their twenties
and thirties, so our dev is probably pessimistic. Worth remembering before
reading a dev regression as a real one.

---

## 2. Accents: fold or keep

**The situation.** We fold `á é í ó ú ü` to bare vowels, because the curated
scripted reference contains zero accented characters in 2,067 sentences.

**The risk.** The test README says scoring applies only NFC, lowercasing and
punctuation removal — it does **not** strip accents. The scripted reference has
none, so there we are safe. The spontaneous reference is a different matter:
0.45% of words in the existing spontaneous gold carry an accent (`arí`,
`allintachá`, `aqnatamá`), and nobody curated that half of the test.

**The size of it.** 0.45% of words sounds small, but the baseline's spontaneous
WER is 3.15%. Always missing those words would cost up to 0.45 points absolute —
roughly a 14% relative degradation on half the ranking.

**The counter-argument.** The accent is orthographic, not acoustic. There is no
audible difference between `ari` and `arí`, so a model may not be able to learn
the distinction reliably anyway, and carrying six extra symbols costs vocabulary.

**What to do.** Both manifests already build:

```bash
python shared-task/01_build_manifests.py \
    --keep_accents --out data/manifests/sharedtask_accents
```

Fine-tune on each and compare spontaneous WER. Decide on the number.

---

## 3. Beam search with a subword language model

**The biggest expected win, and the cheapest.**

**Why.** Decoding today is greedy argmax — `05_predict.py` takes the highest
scoring symbol at each 20 ms frame, independently, with no notion of the word
being formed. The error profile says this is exactly where the headroom is: on
the OOD set the best model makes 23.4% substitutions with almost no deletions or
insertions. The acoustics are right; the spelling is one character off.

**Why subword, not word.** 77% of distinct words in this corpus occur once
(the organisers say so in `data/shared-task/README.md`). A word-level LM would
never have seen the correct form. A character or BPE n-gram model learns the
morphology — which suffixes follow which — and generalises to unseen words.

**How.** `pyctcdecode` + a KenLM n-gram trained on the training transcripts.
Trains on CPU in minutes. Tune `alpha` (LM weight) and `beta` (word bonus) on
dev; they matter more than the model order.

**Watch out.** The LM must be trained only on `train.tsv` text. Training it on
dev or test transcripts would leak, and the dev number would become fiction.

---

## 4. Upsample elder speakers

**Found while inspecting the test set, and it is a real mismatch.**

The spontaneous half of the test is **221 clips, all from speakers aged 60+**.
Our spontaneous training data is nothing like that:

| Age | Training hours | Share |
|---|---:|---:|
| twenties | 14.85 | 41.9% |
| thirties | 8.86 | 25.0% |
| fourties | 4.18 | 11.8% |
| fifties | 3.39 | 9.6% |
| **sixties + seventies** | **2.90** | **8.2%** |
| teens | 1.23 | 3.5% |

Half the ranking is decided on a voice profile that makes up 8% of our
spontaneous training audio. Older speakers differ in rate, voice quality and
often in vocabulary.

**What to do.** Add an age-weighted sampler, or simply repeat 60+ clips. The
manifest does not carry `age` yet — add it in `01_build_manifests.py` (it is in
`ss-corpus-qxp.tsv` and `validated.tsv`), then weight.

**Watch out.** Only 2.90 h exists from 60+ speakers, so aggressive upsampling
will overfit those few voices. Try 2x and 3x, measure, and keep SpecAugment on.

The scripted half needs none of this: its test speakers are in their twenties
and thirties, which is 25 of our 30.5 scripted hours.

---

## 5. Redo CPT on the v27 / v5 audio

**The paper's CPT ran on the older release. There is more unlabelled audio now**
— 29.6 h of untranscribed spontaneous speech alone — and CPT needs no
transcripts at all.

**Cost.** This is the expensive one: 10,000 updates over ~65 h of audio.
Budget a full day of GPU.

**Expected.** The paper found CPT worth 42% relative on scripted WER
(2.06% to 1.19%). Redoing it on more in-domain audio should help, but it is a
day of GPU for a second-order gain. Only after 1, 3 and 4.

---

## 6. XLS-R-1B

**Do not confuse this with the paper's checkpoint.** The paper uses XLS-R-**300M**
(315M parameters). `facebook/wav2vec2-xls-r-1b` is a different, larger public
model from Meta — and it has **no Quechua adaptation**, no CPT.

So this trades "three times the parameters" against "loses the language
adaptation that the paper showed is worth 42% relative". It is not obviously a
win, and it means a full training run from scratch.

**Only if** items 1-4 are done, the GPU is idle, and there are days to spare.

---

## 7. Ensembling

Average the logits of two or three fine-tuned models (different seeds, or
accents/no-accents, or 300M and 1B) before decoding. Reliable small gains,
costs nothing but inference time, and inference here runs at 86x realtime.

Worth doing last, as a free half-point, if the submission deadline allows.

---

## Not worth doing

- **Regenerating the silver transcriptions.** `data/splits/silver_spontaneous/`
  already covers 5,782 clips that have no gold text in v5. Re-running
  omniASR-7B would cost hours of GPU to reproduce what we have.
- **Training on `prompt`.** It is the question read to the speaker, not what
  they said. 150 distinct prompts for 7,305 clips.
- **The Common Voice `train/dev/test.tsv`.** 709 / 679 / 679 clips, far too
  small, and they ignore the 22,727 validated clips we actually want.
