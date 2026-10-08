# Handoff — where this stands and what was learned

Written 2026-10-07, after setting the pipeline up on a Mac. Read this first when
picking the work up on another machine or in a new session.

**Deadline: 18 October 2026** — predictions *and* a 4-8 page LNCS paper, both.
Reserve the last three days for the paper.

---

## State

| | |
|---|---|
| Pipeline | Runs end to end on the GPU box (RTX 5090, torch 2.11 cu128, transformers 5.19). |
| Data | Scripted 27.0 + Spontaneous 5.0 downloaded, manifests built and verified. |
| Test set | Received: 659 clips. **Not downloadable — copy it across by hand.** |
| Baseline | Reproduced on the GPU box: 13.15 / 2.58, identical to the Mac. |
| Trained | Default run done; **best is step 500** (mean 9.28 WER), it overfits after. |
| LM | Character n-gram + beam search built and tuned: ~0.8 WER per domain, honestly measured. |
| Candidate | `submission/ckpt500_lm_v2/qxp.zip` — step 500 + LM (`submission_v2.json`). Valid format, not yet submitted. |

Update 2026-10-07 (GPU box). Read **Findings from the GPU box** first: two of
them overturn assumptions below (the test's read sentences are new; the
spontaneous dev has no elders).

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

## Findings from the GPU box (2026-10-07), most useful first

### A. The test's read sentences are NOT the 2,067 we train on

The Common Voice v27 sentence pool is exactly the 2,067 sentences, all
recorded, so it looked certain the test reused them. It does not. Using only
our own greedy hypotheses on the test audio (no labels), the distance to the
nearest pool sentence:

| | median edit dist. | within 0.05 |
|---|---:|---:|
| dev, its sentence in the pool | 0.000 | 91% |
| dev, its sentence removed from the pool | 0.516 | 0% |
| **test scripted** | **0.480** | **3.7%** |

The test behaves like "sentence removed". Its sentences also look longer
(8-12 words against a pool mean of 4). Consequences:

- **Our scripted dev is optimistic**, for the acoustic model too: it has
  heard every dev sentence read by ~11 train speakers. Select checkpoints on
  `data/manifests/sharedtask_both` (`01_build_manifests.py --split_by both`):
  speaker- AND text-disjoint, 339 dev clips, 0 shared sentences.
- **A high LM weight is a trap.** Tuned on the speaker-only dev, the grid
  picks alpha 2.0 (scripted 4.54 WER) because the LM recalls whole dev
  sentences; on new sentences that same setting gives 9.83 / 3.56, worse than
  greedy. Tune scripted alpha on a `_nodev` LM.

### B. The spontaneous dev has no elders — and no gold elder data exists

The test's 221 spontaneous clips are all 60+. Our `dev_spontaneous` is 258 of
260 clips from **two speakers in their twenties**. Across the whole corpus,
60+ speakers have 1 gold and 17 pending transcriptions; their 528 clips
(2.06 h, 9 speakers) are all silver. There is no clean way to measure the half
of the ranking that matters most. Gold spontaneous in train is ~2.2 h; the
other 23 h are silver.

### C. The default run overfits after ~500 steps

| step | scripted WER/CER | spontaneous WER/CER | mean WER |
|---|---|---|---:|
| baseline | 16.08 / 3.77 | 10.23 / 1.39 | 13.15 |
| **500** | **8.74 / 2.87** | **9.83 / 1.43** | **9.28** |
| 1000 | 8.91 / 3.01 | 10.93 / 1.52 | 9.92 |
| 2500 | 10.06 / 4.08 | 10.88 / 1.65 | 10.47 |
| 3000 | 8.67 / 3.47 | 10.67 / 1.57 | 9.67 |
| 8000 | 10.31 / 5.03 | 11.00 / 1.57 | 10.66 |

The run kept step 500 as its final model (`checkpoints/hf/ft_curated`).
Training loss keeps falling (0.30 -> 0.15 by step 1500). The orthography gain
arrives in the first ~500 steps: 7.3 of the 7.95 scripted points come back.
After that it memorises; silver labels (90% of spontaneous train) are a
suspect for the spontaneous drift. 8000 steps is far too long for a warm start.

### D. The in-training WER was corrupted — fixed

`03_train.py`'s eval decoded the padded width and decoded the Trainer's -100
padding as a literal `<unk>`. It reported **24.65** mean WER for a model
`04_evaluate.py` scored at **10.95**, and checkpoint selection ran on that
number. Fixed (padded frames forced to blank, -100 mapped to blank); the two
now agree to 0.02. Selection is also now on the **mean** WER of both domains;
it was on scripted alone, which would have kept step 1500 — worse than the
baseline on spontaneous.

### E. LM + beam search: ~0.8 WER per domain, honestly measured

Character n-gram (orders 4-12, modified Kneser-Ney written by `ctc_lm.py`,
verified to sum to one in KenLM) + flashlight's lexicon-free beam search,
which scores the LM on every character. On step 500:

| | scripted WER/CER | spontaneous WER/CER | mean |
|---|---|---|---|
| greedy | 8.72 / 2.87 | 9.83 / 1.42 | 9.28 / 2.15 |
| beam + LM | **7.94** / 2.93 | **9.11** / 1.37 | **8.52** / 2.15 |

Scripted: `scripted_nodev_o6`, alpha 0.625, beta 0 (new-sentence condition).
Spontaneous: `spont_all_o8`, alpha 0.5, beta 1 — silver text helps the LM
(gold-only: 9.44). Adding spontaneous text to the scripted LM hurts (8.24-8.32)
despite lower perplexity. Beam 128 = beam 64. Config: `checkpoints/lm/submission_v1.json`.

Re-tuned on `both_s2000` (finding G), the best weights moved: spontaneous
beta 1 gained 0.72 WER on step 500 and 0.00 there. **Use
`checkpoints/lm/submission_v2.json`: alpha 0.5, beta 0 in both domains**, the
setting with the best worst-case gain across the two models (scripted +0.30
to +0.57 WER, spontaneous +0.14 to +0.41, CER within ±0.08). Optima found on
260-clip or 339-clip devs are noise at the 0.2-point level; alpha >= 1 hurts
everywhere once the sentences are new.

### G. On the speaker+text-disjoint split, the optimum is at <= 250 steps

`checkpoints/hf/both_s2000`: 2000 steps on `sharedtask_both`, eval every 250.

| step | scripted WER/CER | spontaneous WER/CER | mean WER |
|---|---|---|---:|
| **250** | **4.60 / 0.63** | **9.83 / 1.44** | **7.22** |
| 500 | 4.90 / 0.62 | 11.07 / 1.58 | 7.99 |
| 1000 | 5.87 / 0.75 | 10.69 / 1.53 | 8.28 |
| 2000 | 5.42 / 0.70 | 10.02 / 1.41 | 7.72 |

The first eval is the best, so the optimum may be earlier still. Absolute
numbers do NOT compare with the speaker-only dev: this dev is 4 speakers and
1,347 words (one WER point = 13 words) and lacks `e990fbdf`. With the LM
(`checkpoints/lm_both/`, which exclude the dev sentences): scripted 4.60 ->
4.16, spontaneous 9.83 -> 9.35.

Still unmeasured: how much hearing a sentence from other speakers helps the
acoustic model. Clean test: retrain on `sharedtask_both` plus the dropped
crossing clips (train speakers reading dev sentences), compare on the same dev.

### F. Setup traps on the GPU box

- RTX 50xx (sm_120) needs torch **cu128**; the cu124 the requirements used to
  suggest does not run.
- pip now resolves **transformers 5**, which dropped `warmup_ratio` and
  `group_by_length` from `TrainingArguments`. Fixed portably in `03_train.py`.
- No `curl` on the box: `run_all.sh` now fetches the checkpoint with
  `huggingface_hub`.
- The Mozilla archives unpack into `cv-corpus-27.0-.../qxp/` and
  `sps-corpus-5.0-...-qxp/`; `00_download.py` now flattens them.
- KenLM order > 6 needs kenlm and flashlight-text rebuilt with the same max
  order; recipe in `requirements.txt`.
- `run_all.sh --only evaluate` scores `$OUT` (the fine-tuned model). For the
  baseline: `OUT=checkpoints/hf/ft_cpt_silver bash shared-task/run_all.sh --only evaluate`.

---

## Findings from the Mac, most useful first

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

1. **Pin down the training length.** Finding G puts the optimum at <= 250
   steps. Sweep 100-300 on `sharedtask_both` with `--eval_steps 50`, then
   retrain on the full split for that many steps, with `--max_steps` set to
   it so the LR schedule matches. Each 250 steps is ~4 min on the 5090.
2. **Silver or not.** Same short run with `01_build_manifests.py --no_silver`.
   Note the spontaneous dev cannot show the effect on elders (finding B).
3. **Re-tune the LM** on the chosen model: `07_tune_lm.py` caches emissions
   (~45 s GPU), the grid is CPU. For scripted use `_nodev` LMs or LMs built
   on `sharedtask_both` (`06_build_lm.py --manifest_dir ... --out_dir checkpoints/lm_both`).
4. **Elders** (IMPROVEMENTS item 4): upsampling 60+ means upsampling silver.
   Decide whether that is worth it without a way to measure it.
5. **Accents on/off** (item 2).
6. **Submit early.** `submission/ckpt500_lm_v2/qxp.zip` is a valid candidate now.
   Keep each candidate in its own `submission/<name>/qxp.*` — the zip must
   contain a file named exactly `qxp.tsv`.

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
