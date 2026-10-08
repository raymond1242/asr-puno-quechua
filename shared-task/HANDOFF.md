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
| Trained | **`checkpoints/hf/final_n500_lr2e-5`**: 500 steps, LR 2e-5 (findings C, G, H). |
| LM | Character n-gram + beam search. On the final recipe it helps spontaneous only (finding J). |
| Candidate | **`submission/final_n500_lr2e-5_v3/qxp.zip`** — final model, scripted greedy, spontaneous + LM (`submission_v3.json`). Valid, not yet submitted. Dev (speaker-only, optimistic for scripted): 9.23 / 3.01 scripted, 9.06 / 1.30 spontaneous. |

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

### H. Steps and LR (2026-10-08): a broad plateau, 250-1000 steps

`shared-task/experiments/2026-10-08_steps_and_memorisation.sh`. Each run ends
its own LR schedule, eval on the `sharedtask_both` dev only:

| run | scripted WER/CER | spontaneous WER/CER | mean WER/CER |
|---|---|---|---|
| 100 steps | 5.79 / 0.76 | 9.68 / 1.37 | 7.74 / 1.06 |
| 200 steps | 5.64 / 0.73 | 9.47 / 1.35 | 7.56 / 1.04 |
| 250 steps (seeds 42, 43) | 5.20, 4.97 | 9.28, 9.47 | 7.24, 7.22 |
| 300 steps | 5.12 / 0.67 | 9.47 / 1.38 | 7.30 / 1.03 |
| 500 steps | 4.97 / 0.66 | 9.73 / 1.39 | 7.35 / 1.03 |
| 500 steps, LR 2e-5 (seeds 42, 43) | 4.60, 4.97 | 9.32, 9.54 | 6.96, 7.26 |
| 1000 steps, LR 2e-5 | 4.83 / 0.60 | 9.64 / 1.33 | 7.23 / 0.96 |
| 1000 steps, LR 1e-5 | 4.90 / 0.64 | 9.42 / 1.31 | 7.16 / 0.98 |

Every run from 250 to 1000 steps, LR 1e-5 to 5e-5, lands at 7.1-7.3 mean WER:
within seed noise (the two seeds of one recipe differ by 0.30). The first seed
of LR 2e-5 (6.96) looked like a winner; the second (7.26) says it was luck.
What matters is not training 8000 steps. The submission model
`checkpoints/hf/final_n500_lr2e-5` (500 steps, LR 2e-5, full speaker split)
sits on the plateau; no reason to retrain it. Seed noise this size is what
model averaging (IMPROVEMENTS item 7) removes — the cheapest remaining gain. On the (optimistic) speaker-only dev it scores 9.23 / 3.01
scripted, 9.52 / 1.32 spontaneous — scripted lower than step 500 of the default
run (8.74), as expected: that run's higher LR memorises the shared sentences
more, which that dev rewards (finding I).

### I. Hearing a sentence from other speakers is worth ~1 WER point

Same dev (`sharedtask_both`), 250 steps, two seeds where marked:

| train | scripted WER | scripted CER |
|---|---|---|
| A: dev sentences never in train (39.94 h) | 5.20, 4.97 | 0.65, 0.62 |
| B_eq: + train speakers reading them, same 39.94 h | **3.79, 4.23** | **0.48, 0.55** |
| B: + those clips, 43.58 h (one seed) | 4.68 | 0.60 |

The seed ranges of A and B_eq do not overlap: having heard the dev sentences
from other speakers lowers scripted WER by ~1.1 points (~21% relative) and CER
by ~19%. That is the optimism of the speaker-only dev, measured. B_eq (from B,
minus 3.64 h of random A scripted clips) separates it from data volume.
`01_build_manifests.py --train_on_dev_sentences` builds B.

### J. On the final recipe the LM helps spontaneous only

Re-tuned on `both_n500_lr2e-5` (clean LMs in `checkpoints/lm_both/`), and
taking the worst case over the three models measured:

- **Spontaneous**: `spont_all_o8`, alpha 0.5, beta 0 gains on all three
  (+0.41, +0.14, +0.31 WER), CER unchanged.
- **Scripted**: no setting gains measurably (best worst case +0.07 WER; on
  the final recipe everything is within ±0.15, i.e. 2 words). The better the
  acoustic model, the less a 2,000-short-sentence LM has to fix, and the test's
  sentences are longer and from another source.

Hence `checkpoints/lm/submission_v3.json`: scripted greedy (`"lm": null`),
spontaneous with the LM.

### K. `05_predict.py` used to zip a broken submission

Inference started while a training run held 27 GB of the GPU: CUDA OOM on 560
of 659 files. The script wrote empty rows for them and zipped the result, with
only a warning. It now refuses to write the zip when any file fails, and exits
non-zero. Do not run inference next to a training run on this card.

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

Done 2026-10-08: steps and LR (finding H), memorisation (I), the LM on the
final recipe (J). Current submission: `submission/final_n500_lr2e-5_v3/qxp.zip`.

1. **Submit it.** It is valid now (659 rows, none empty, `qxp.tsv` inside).
   Keep each later candidate in its own `submission/<name>/qxp.*`.
2. **Model averaging** (IMPROVEMENTS item 7): seed noise is ~0.3 mean WER, so
   averaging the logits of 3-5 seeds of the final recipe should be a real gain.
   Each seed is ~7 min. Measure on `sharedtask_both` first.
3. **Train on dev too?** The final model could add `dev_scripted`,
   `heldout_scripted` and `dev_spontaneous` (260 gold clips — gold spontaneous
   is scarce). Never `heldout_spontaneous`: those are the organisers' `test`
   clips (decision 3 in `01_build_manifests.py`). Nothing left to select on
   afterwards, so do it last.
4. **Silver or not.** Same short run with `01_build_manifests.py --no_silver`.
   The spontaneous dev cannot show the effect on elders (finding B).
5. **Elders** (IMPROVEMENTS item 4): upsampling 60+ means upsampling silver.
   Decide whether that is worth it without a way to measure it.
6. **Accents on/off** (item 2).
7. **The paper.** Reserve the last three days. Findings A, C, D, I and the
   plateau of H are results in their own right.

Re-tuning the LM on a new model: `07_tune_lm.py` caches emissions (~45 s GPU),
the grid is CPU. For scripted use LMs that exclude the dev sentences
(`checkpoints/lm_both/` with `--manifest_dir data/manifests/sharedtask_both`).
Never run it, or `05_predict.py`, next to a training run (finding K).

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
