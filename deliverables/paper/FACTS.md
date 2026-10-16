# Verified facts for the paper

Every number that appears in `main.tex` must appear here first, with where it
came from. Re-verify before submitting; do not quote a figure that is not on
this sheet.

Legend for **Source**:
- `live` — recomputed from the data in this repository on 2026-10-12
- `S07` — `shared-task/EXPLICACION_SESION_2026-10-07.md`
- `S09` — `shared-task/EXPLICACION_SESION_2026-10-09.md`
- `H` — `shared-task/HANDOFF.md`
- `paper` — Huaman et al., arXiv:2605.28253 (the baseline paper)

---

## Corpora

| Fact | Value | Source |
|---|---|---|
| Scripted v27, validated clips | 22,727 | live |
| Scripted v27, hours | 30.49 | live |
| Scripted v27, speakers | 77 | live |
| Scripted v27, distinct sentences | 2,067 | live |
| Readings per sentence (mean) | ~11 | live (22,727 / 2,067) |
| Spontaneous v5, clips | 7,305 | live |
| Spontaneous v5, hours | 35.41 | live |
| Spontaneous v5, speakers | 110 | live |
| Spontaneous v5, clips with a transcription | 1,222 (5.81 h) | live |
| Spontaneous v5, clips without one | 6,083 | live |
| Distinct prompts (questions) | 150 | live |
| Silver clips reused from the baseline release | 5,782 | live |
| Training set (our `sharedtask`) | 25,960 clips, 49.84 h | live |
| └ scripted gold / spont gold / pending / silver (h) | 26.61 / 1.79 / 0.51 / 20.93 | live |

## Vocabulary — state the denominator

| Text set | Tokens | Types | Hapax |
|---|---:|---:|---:|
| Curated scripted sentences (what the organisers' README cites) | 8,273 | 4,594 | **77%** |
| Spontaneous training text | 76,838 | 20,584 | 71% |
| All training clips, with repetitions | 156,131 | 23,595 | 60% |

**Use 77% and say "of the scripted sentence corpus".** The 60% figure counts
repeated readings and is not a vocabulary statistic.

## The curated reference

| Fact | Value | Source |
|---|---|---|
| Curated sentences released | 2,067 | live |
| Differ in spelling beyond case/punctuation | **629** | live |
| Hyphens in the curated text | 0 | live |
| Accented vowels in the curated text | 0 | live |
| Straight apostrophes | 884 | live |
| Examples | `allqu`→`alqu`, `iskay`→`ishkay`, `misk'i`→`mishk'i`, `aswan`→`ashwan` | live |
| Baseline hypotheses vs raw reference | 8.13 WER / 2.76 CER | S07 |
| Baseline hypotheses vs curated reference | 16.08 WER / 3.77 CER | S07 |
| **Cost of the mismatch** | **7.95 WER points** | derived |

## Text normalisation

| Fact | Value | Source |
|---|---|---|
| Spontaneous rows with off-convention characters | 243 | live |
| Distinct off-convention characters | 28 | live |
| Control character U+0002, at file lines | 515, 759, 784 | live |
| Transcriptions wrapped in quotes if read with `quoting=3` | 21 | live |
| Final training vocabulary | 29 symbols (a–z, ñ, apostrophe, space) | live |
| Released checkpoint's CTC head | 50 outputs | live |

## The test set does not reuse training sentences

Normalised edit distance to the nearest sentence in the 2,067-sentence pool,
computed from our own greedy hypotheses, no labels used.

| Condition | Median | Within 0.05 | Source |
|---|---:|---:|---|
| Dev, its sentence in the pool | 0.000 | 91% | S07 |
| Dev, its sentence removed from the pool | 0.516 | 0% | S07 |
| **Test, scripted** | **0.480** | **3.7%** | S07 |

Test sentences also run 8–12 words against a pool mean of 4. (S07)

## Splits

| Split | Scripted dev clips | Speakers disjoint | Sentences disjoint | Source |
|---|---:|---|---|---|
| `sharedtask` | 1,493 | yes | **no** (1,188 of 1,188 shared) | live |
| `sharedtask_both` | 339 (1,347 reference words, 4 speakers) | yes | yes | S09, H finding G |
| The baseline's own split | 948 dev / 909 test | yes | **no** (805/805, 757/757) | live |

Spontaneous uses the organisers' own `split` column (train 478 / dev 300 /
test 296); their `test` clips are held out of training and dev entirely. (live)

Held-out sizes in `sharedtask`: 1,328 scripted clips (1.67 h), 279 spontaneous
(1.42 h). (live)

## The test set

| Fact | Value | Source |
|---|---|---|
| Clips | 659 (438 scripted, 221 spontaneous) | live |
| Duration | 59.3 min (25.7 scripted, 33.5 spontaneous) | live |
| Speakers | 7 | live |
| Scripted speaker ages | twenties, thirties | live |
| **Spontaneous speaker ages** | **all sixties (60+)** | live |
| Clips over 30 s | 0 | live |
| Audio format | 16 kHz mono PCM | live |
| Our 60+ spontaneous training audio | 2.90 h of 35.41 h (8.2%) | live |

## Main result — `sharedtask` development, decoding v3

| System | scr WER | scr CER | spo WER | spo CER | **mean WER** | **mean CER** |
|---|---:|---:|---:|---:|---:|---:|
| Published baseline, converted | 16.08 | 3.77 | 10.23 | 1.39 | **13.15** | **2.58** |
| This work (`final_n500_lr2e-5`) | 9.23 | 3.01 | 9.06 | 1.30 | **9.14** | **2.16** |

Source: S09 §10. **The headline claim is 13.15 → 9.14 mean WER.** Do not write
"13.15 → 9.23": 9.23 is the scripted column, not the mean. (9.38 appears in
earlier notes and is the *greedy* mean, without the spontaneous LM.)

## Conversion fidelity

| Fact | Value | Source |
|---|---|---|
| Converted model on the baseline's out-of-domain set | 27.26 WER / 4.46 CER | live (2026-09-20) |
| Published figure for the same model | 27.4 WER / 4.55 CER | paper, Table 3 |
| Tensors mapped / missing / unexpected | 424 / 0 / 0 | live |

## Training: the plateau and seed noise

`sharedtask_both` development, mean WER. Every run ends its own LR schedule.

| Recipe | Mean WER | Source |
|---|---:|---|
| 100 steps | 7.74 | H finding H |
| 200 steps | 7.56 | H |
| 250 steps, seeds 42/43 | 7.24 / 7.22 | H |
| 300 steps | 7.30 | H |
| 500 steps | 7.35 | H |
| **500 steps, LR 2e-5, seeds 42/43** | **6.96 / 7.26** | H |
| 1000 steps, LR 2e-5 | 7.23 | H |
| 1000 steps, LR 1e-5 | 7.16 | H |

Seed variation of the final recipe, three seeds (S09 §10):

| Split | Mean WER | Std. dev. | Range |
|---|---:|---:|---:|
| `sharedtask_both` (339 scripted clips) | 7.02 | 0.19 | 0.37 |
| `sharedtask` (1,493 scripted clips) | 9.23 | 0.08 | 0.13 |

Individual seeds, `sharedtask_both` dev, v3: 6.81 / 7.17 / 7.09 (s42/43/44).
Individual seeds, `sharedtask` dev, v3: 9.14 / 9.27 / 9.28.

## Sentence overlap, controlled experiment

Same dev (`sharedtask_both`), same recipe (250 updates); only training differs.

| Training set | Scripted WER | Scripted CER | Source |
|---|---|---|---|
| A: dev sentences never in training (39.94 h) | 5.20 / 4.97 | 0.65 / 0.62 | H finding I |
| B_eq: heard, identical hours (39.94 h) | **3.79 / 4.23** | **0.48 / 0.55** | H |
| B: heard, 43.58 h (one seed) | 4.68 | 0.60 | H |

Means: A = 5.085, B_eq = 4.01. **Effect = 1.1 WER points, 21% relative; CER 19%.**
Seed ranges do not overlap (worst B_eq 4.23 < best A 4.97).

## Held-out, measured once after freezing decisions

`sharedtask_both`, mean of three seeds (S09 §10):

| | scr WER | scr CER | spo WER | spo CER | mean WER | mean CER |
|---|---:|---:|---:|---:|---:|---:|
| Development | 4.85 | 0.62 | 9.20 | 1.31 | 7.02 | 0.96 |
| Held-out | 10.79 | 2.33 | 3.38 | 0.39 | 7.08 | 1.36 |

Seed spread on held-out: 0.05 mean WER.

`sharedtask` held-out, mean of three seeds: 9.70 / 1.77 scripted,
3.56 / 0.41 spontaneous, **6.63 / 1.09 mean**.

Composition, not overfitting:
- One speaker (`e990fbdf`, 59 clips) scores 98.67 WER / 54.06 CER alone;
  removing them moves the `sharedtask` dev scripted figure from 9.23 to 5.71. (S09 Annex B)
- 46% of the held-out spontaneous portion comes from speakers already heard
  through silver transcriptions in training. (S09)

## Negative results

**Ensembling three seeds, `sharedtask_both` dev** (S09 §10):

| System | Mean WER | Mean CER |
|---|---:|---:|
| Mean of three individual seeds | 7.02 | 0.96 |
| Averaged logits | 8.38 | 1.14 |
| Averaged probabilities | 7.30 | 1.02 |

Mechanism: character deletions rise from 21 to 50 when averaging logits. (S09)

**SpecAugment masking**, `sharedtask_both` dev, mean WER (S09 §10):
0.05 → 6.81, 0.2 → 7.05, 0.5 → 7.68. Worse on all four numbers.

**Language model**, spontaneous WER gain over greedy:

| Seed | Dev | Held-out |
|---|---:|---:|
| 42 | +0.31 | 0.00 |
| 43 | +0.17 | −0.04 |
| 44 | +0.12 | −0.08 |

Derived from S09 §10 main table and Annex A.

Earlier trap (S07): tuned on the speaker-only dev the grid selects alpha 2.0
and reports 4.54 scripted WER; under the new-sentence condition that setting
gives 9.83, worse than greedy.

## System configuration

| Fact | Value | Source |
|---|---|---|
| Acoustic model | XLS-R-300M, 315M parameters | live |
| Starting checkpoint | `QuechuaBase/xls-r-cpt-qxp-silver` (CPT + CTC on V+S) | H |
| Fine-tuning | 500 updates, LR 2e-5, seed 42 | S09 |
| Batching | by total audio duration, not clip count | live |
| Clip length filter | 0.4–30 s | live |
| Decoding, scripted | greedy | S09 |
| Decoding, spontaneous | `spont_all` order 8, alpha 0.5, beta 0, beam 64 | S09 |
| LM grid searched | 3 LM sources x 6 alphas x 4 betas = 72 per domain | S07 |
| LM smoothing | modified Kneser-Ney, ARPA, verified against KenLM (max error ~1e-6) | S07 |
| Beam search | flashlight-text, lexicon-free | S07 |

## Derived figures used in the paper

| Claim | Arithmetic | Where |
|---|---|---|
| Orthography mismatch | 16.08 − 8.13 = **7.95** | abstract, §2.2, conclusion |
| Scripted improvement | 16.08 − 9.23 = **6.85** | conclusion |
| Overlap effect | mean(5.20, 4.97) − mean(3.79, 4.23) = **1.08 ≈ 1.1** | abstract, §4.2 |
| Relative overlap effect | 1.08 / 5.085 = **21%** | §4.2 |
| Mean WER, baseline | (16.08 + 10.23) / 2 = **13.15** | Table 2 |
| Mean WER, this work | (9.23 + 9.06) / 2 = **9.14** | Table 2 |

## Figures NOT to use

- **77% hapax without naming the set** — it is the scripted sentence corpus.
- **13.15 → 9.23** — wrong, mixes a mean with a scripted column. Use 9.14.
- **13.15 → 9.38** — that is the greedy mean, not the submitted system's.
- **"The LM gains 0.8 WER per domain"** (S07 finding E) — superseded. On the
  final recipe it gains nothing measurable on scripted, and on held-out data
  the spontaneous gain disappears too.
- **The paper's 2.11 / 3.15 compared with our figures** — different test set,
  different reference text, and their Mean averages three domains including
  out-of-domain. Only compare on our own split with our own reference.
