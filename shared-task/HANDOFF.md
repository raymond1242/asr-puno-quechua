# Handoff — where this stands and what was learned

Written 2026-10-07, after setting the pipeline up on a Mac. Read this first when
picking the work up on another machine or in a new session.

**Deadline: 18 October 2026** — predictions *and* a 4-8 page LNCS paper, both.
Reserve the last three days for the paper.

---

## State

| | |
|---|---|
| Pipeline | Written and tested end to end on CPU. Pushed to `origin/shared-task`. |
| Data | Scripted 27.0 + Spontaneous 5.0 downloaded, manifests built and verified. |
| Test set | Received: 659 clips. **Not downloadable — copy it across by hand.** |
| Baseline | Measured. Nothing trained yet. |
| Next | Train (~2 h on a GPU), then the subword LM. |

Nothing in `checkpoints/` or `data/` is in git — 4.8 GB of model weights and
several GB of audio. `run_all.sh` rebuilds all of it except the test audio.

---

## The number to beat

`QuechuaBase/xls-r-cpt-qxp-silver`, converted to transformers, on our dev split:

| Domain | WER | CER |
|---|---:|---:|
| scripted | 16.08 | 3.77 |
| spontaneous | 10.23 | 1.39 |
| **mean** | **13.15** | **2.58** |

Reproduce it with:

```bash
python shared-task/04_evaluate.py --model checkpoints/hf/ft_cpt_silver
```

---

## Findings, most useful first

### 1. The curated reference is worth ~8 WER points

Scoring the *same* hypotheses against the raw `sentence` instead of
`sentence_curated` gives **8.13%** rather than **16.08%**. The baseline was
trained on the old spelling and 627 of 2,067 sentences changed. Half the error
is orthography, not acoustics.

This is why the default run fine-tunes on curated text, and it is the single
largest improvement available. Expect most of those 8 points back.

### 2. The ranking is a mean over four numbers

WER and CER, for scripted *and* spontaneous. This rules out the paper's best
scripted model:

| Model | Scripted WER | Spontaneous WER |
|---|---:|---:|
| xls-r + CPT, **V** | **1.19** | 13.6 |
| xls-r + CPT, **V+S** | 2.11 | **3.15** |

V+S is the target. Never optimise scripted alone.

### 3. The spontaneous test is all elders — our training data is not

The 221 spontaneous test clips are **100% speakers aged 60+**. Only 8.2% of our
spontaneous training hours (2.90 h of 35.4 h) come from that age group. Half the
ranking rides on a voice profile we barely train on. See IMPROVEMENTS item 4.

The scripted half is fine: its speakers are in their twenties and thirties,
which is 25 of our 30.5 scripted hours.

### 4. How the test will be scored

From its README: **Unicode NFC, lowercase, punctuation removed. The apostrophe
and ñ are kept.** Note what is *absent*: accents are **not** stripped. The
curated scripted reference has none, but nobody curated the spontaneous
reference, and 0.45% of words in the spontaneous gold carry an accent. Hence
`--keep_accents` on the manifest builder — build both, measure, decide.

### 5. Two file-format traps

`ss-corpus-qxp.tsv` **is** standard-CSV quoted: read it with pandas' default
quoting. Reading it with `quoting=3` leaves 21 transcriptions wrapped in quotes
and with doubled quotes inside. The Common Voice files are the opposite — the
organisers' own instructions say `quoting=3`.

The spontaneous transcriptions carry 28 off-convention characters across 243
rows: digits, braces, brackets, Spanish accents, three Unicode spellings of the
apostrophe, and a control character at lines 515, 759 and 784. Three
apostrophes means one ejective phoneme becomes three CTC symbols.

`normalize()` folds all of it, verified against the curated reference, which
has zero hyphens and zero accents in 2,067 sentences, writes the dubitative as
`chaycha`, and writes reduplication as two words (`phukuspa phukuspa`).

### 6. `prompt` is not a transcription

In `ss-corpus-qxp.tsv`, `prompt` is the question read to the speaker — 150
distinct prompts for 7,305 clips. Only `transcription` is a label, and it is
present for just 1,222 of them. Training on `prompt` would teach the model that
every 20-second answer transcribes to a 5-word question.

### 7. The silver data already exists

`data/splits/silver_spontaneous/` holds auto transcripts for 5,782 clips that
have no gold text in v5 — about 21 usable hours after filtering. Do not spend
GPU hours re-running omniASR to recreate it.

### 8. Our dev is pessimistic

One dev speaker, `e990fbdf` (59 of 1,493 scripted clips), scores **106% WER**.
Their audio is technically fine — normal RMS and peak, no clipping — the model
simply fails on them, and a speaker-disjoint split means they were never in
training. The real test has 7 speakers and no such outlier. Do not read a small
dev regression as real without checking the per-speaker breakdown.

### 9. Sentence overlap in scripted dev cannot be avoided

All 1,188 dev sentences also appear in train. 77 speakers read the same 2,067
sentences, so no speaker-disjoint split separates text too. The paper had the
same. `--split_by sentence` builds the text-disjoint variant for comparison;
see IMPROVEMENTS item 1.

### 10. fairseq cannot be installed, and does not need to be

fairseq 0.12.2 has no wheel for Python 3.11 or ARM. `eval/hf/convert_fairseq_w2v2.py`
unpickles the checkpoint with stub classes and remaps the weights to
`Wav2Vec2ForCTC`. Verified: the converted model scores 27.26 WER on the paper's
out-of-domain set against the paper's 27.4 — the port is faithful.

Two traps it handles, both of which produce silent garbage if missed:

- fairseq nests the conv layer norm one level deeper
  (`conv_layers.{i}.2.1.weight`), because it wraps the LayerNorm between two
  transposes.
- `dict.ltr.txt` in this repo gained the `|` word-delimiter *after* some
  checkpoints were trained. **Verify the vocabulary size per checkpoint**; the
  converter refuses to write a model if a single tensor is missing or extra.

### 11. Hardware reality

- Training the paper's full recipe on an M4 Pro: ~122 h. Not viable — hence the GPU.
- Inference is cheap: 86x realtime on Apple MPS, 18x on CPU. The whole 659-clip
  test takes 3 minutes on a laptop CPU.
- A 30 s clip cap keeps 87% of the spontaneous audio; a 20 s cap keeps 56%.
  No test clip exceeds 30 s.

---

## Next steps, in order

1. **Train.** `bash shared-task/run_all.sh` — the default warm-starts from the
   converted baseline and fine-tunes on curated text. ~2 h. Confirm the ~8
   points come back.
2. **Subword LM + beam search** (IMPROVEMENTS item 3). The error profile is 23%
   substitutions with almost no insertions or deletions — words almost right.
   Cheap, CPU-only, the most promising remaining lever.
3. **Upsample elder speakers** (item 4), for the spontaneous half.
4. **Accents on/off** (item 2) — one training run each, decide on the number.
5. **Submit early.** `05_predict.py` already produces a valid `qxp.zip`. Make
   one from whatever the best model is, well before the 18th, then improve it.

Full list with what to measure: `shared-task/IMPROVEMENTS.md`.

---

## Running it elsewhere

```bash
git clone -b shared-task https://github.com/raymond1242/asr-puno-quechua.git
cd asr-puno-quechua
python -m venv .venv && .venv/bin/pip install -r shared-task/requirements.txt
cp shared-task/.env.example .env          # paste MDC_API_KEY
# copy data/test/qxp_test_dataset/ across by hand — it is not downloadable
bash shared-task/run_all.sh
```

GitHub drops password auth: use `gh auth login`, a Personal Access Token, or SSH.

Remotes: `origin` is the fork (raymond1242), `upstream` is QuechuaBase. Pull
their changes with `git pull upstream main`.

To move a trained model between machines, push it to a private HuggingFace repo
or use `scp` — it is 1.2 GB and must not go into git.
