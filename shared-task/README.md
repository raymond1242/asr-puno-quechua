# SIMBig 2026 — Task 1: Puno Quechua ASR

Pipeline for the shared task, built on top of this repository's baseline.

Everything runs on **transformers**, not fairseq: the published XLS-R checkpoints
are fairseq `.pt` files and fairseq 0.12.2 has no wheel for modern Python, so
`eval/hf/convert_fairseq_w2v2.py` ports the weights across. The converted model
reproduces the paper to within 0.14 WER points on its out-of-domain set
(27.26% vs 27.4%), so the architecture and weights are faithful.

**Continuing this work?** Read `HANDOFF.md` first: state, findings and next steps.

## Quick start on the GPU machine

```bash
git clone -b shared-task <your fork> && cd asr-puno-quechua
python -m venv .venv && .venv/bin/pip install -r shared-task/requirements.txt
cp shared-task/.env.example .env     # paste your MDC_API_KEY

bash shared-task/run_all.sh
```

Each stage skips work already done, so re-running after a crash or a reboot is
safe. Run part of it with `--from train` or `--only predict`.

The **test audio is not downloadable** — copy `data/test/qxp_test_dataset/`
across by hand.

## The scripts

| | What it does |
|---|---|
| `00_download.py` | Fetches Scripted 27.0 + Spontaneous 5.0. Resumable. Replaces `data/download_data.py`, whose dataset IDs now 404. |
| `01_build_manifests.py` | Builds train/dev/heldout. Joins the organisers' curated reference, folds text to their orthography, enforces the split rules. |
| `03_train.py` | CTC fine-tuning. Resumable, batches by audio duration, evaluates both domains. |
| `04_evaluate.py` | WER and CER for scripted and spontaneous, plus the mean the ranking uses. |
| `05_predict.py` | Transcribes the test audio into `qxp.tsv` + `qxp.zip`. |
| `IMPROVEMENTS.md` | Ideas not yet tried, each with what to measure. |

## What the ranking actually rewards

The organisers compute a **Mean Ranking over four numbers** — WER and CER for
scripted *and* for spontaneous. A model that wins one domain and loses the other
does not win, which rules out the paper's best scripted model:

| Model | Scripted WER | Spontaneous WER |
|---|---:|---:|
| xls-r + CPT, **V** | **1.19** | 13.6 |
| xls-r + CPT, **V+S** | 2.11 | **3.15** |

So V+S is the target, and `04_evaluate.py` refuses to print a single headline
figure.

## Measured baseline

`QuechuaBase/xls-r-cpt-qxp-silver`, converted, scored on our dev split:

| Domain | WER | CER |
|---|---:|---:|
| scripted | 16.08 | 3.77 |
| spontaneous | 10.23 | 1.39 |
| **mean** | **13.15** | **2.58** |

This is the number to beat. Two things explain why it is far above the paper's
2.11%:

**The curated reference costs 7.95 WER points.** Scoring the same hypotheses
against the raw `sentence` instead of `sentence_curated` gives 8.13% rather
than 16.08%. The baseline was trained on the uncurated spelling, and 627 of
2,067 sentences differ. Retraining on the curated text is the single largest
improvement available, and it is the default run.

**Our dev is harder than the paper's test split.** One speaker (`e990fbdf`,
59 clips) scores 106% WER — the audio is technically fine, the model simply
fails on them. They appear only in dev, which is correct for a speaker-disjoint
split but makes dev pessimistic relative to the real test, whose scripted
speakers are in their twenties and thirties.

## Data decisions

Recorded in full, with the reasoning, at the top of `01_build_manifests.py`.
In brief:

- **Reference text** is `sentence_curated`, never the raw `sentence`.
- **Clips are capped at 30 s.** Spontaneous speech holds a 28.6-minute outlier.
  A 20 s cap would cost 9 of 25.8 usable hours; no test clip exceeds 30 s.
- **The 296 spontaneous clips the organisers marked `split == "test"`** are
  held out of training and of dev. The rules say the test must not appear in
  training at any point.
- **The scripted split is speaker-disjoint**, as the paper's was. Sentences
  necessarily overlap across splits — 77 speakers read the same 2,067
  sentences — so no speaker-disjoint split can also separate text.
  `--split_by sentence` builds the text-disjoint variant.
- **`prompt` is not a label.** It is the question read to the speaker; only
  `transcription` is what they said.

### Two file-format traps

`ss-corpus-qxp.tsv` is standard-CSV quoted and must be read with pandas'
default quoting; reading it with `quoting=3` leaves 21 transcriptions wrapped
in quotes. The Common Voice files are the opposite — the organisers' own
instructions say to use `quoting=3`.

The spontaneous transcriptions carry 28 off-convention characters across
243 rows: digits, braces, brackets, Spanish accents, three Unicode spellings of
the apostrophe, and a control character at lines 515, 759 and 784. Left in,
they become CTC outputs the model can never usefully predict, and the three
apostrophes split one ejective phoneme into three symbols. `normalize()` folds
all of it; verified against the curated reference, which has zero hyphens and
zero accents in 2,067 sentences.

## Submission

```bash
python shared-task/05_predict.py \
    --model checkpoints/hf/ft_curated \
    --audio_dir data/test/qxp_test_dataset/audios \
    --template data/test/qxp_test_dataset/qxp.tsv \
    --out submission/qxp
```

`--template` makes the organisers' own `qxp.tsv` drive the file list and row
order, and fails loudly if any listed file is missing rather than dropping it.

The TSV is written by hand rather than through `csv`: every dialect either
quotes the apostrophes that are part of Quechua spelling or escapes something
their parser will not unescape.

Before sending, check `head -3 submission/qxp.tsv` and that the row count
matches the template.

## Dates

| | |
|---|---|
| Test released | 30 Sep 2026 |
| **Predictions + system paper due** | **18 Oct 2026** |
| Results | 28-30 Oct 2026, Arequipa |

The paper is 4-8 pages, Springer LNCS template, submitted through CMT.
