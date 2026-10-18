# Video script — English

**SIMBig 2026 · Task 1 · Team Innova**
Required video · up to 15 minutes · English · MP4, 1080p or better, max 100 MB
Deck: `simbig2026_task1-en.pptx` (11 slides)

> This is the **required** English narration. The Spanish `GUION.md` is the optional second
> video and is not a translation of this one — it is the version you wrote and corrected.
> Changes made to one do not propagate to the other.

---

## How to use this script

- Normal paragraphs are **what you say**. Lines in *italics inside brackets* are **cues**: don't read them.
- Each slide carries a time window. They are cumulative: at the end of slide 6 you should be at 5:45.
- Planned length: **11:15** at **130 words per minute**, plus two seconds per slide change. The rate
  is set below Spanish pace on purpose: reading technical English aloud is slower, and rushing costs
  more clarity than the extra minute buys you.
- **Check your own pace before recording.** Read slide 7 aloud and time it. About 65 seconds means you
  are on 130 wpm and the talk lands at 11:15; 75 seconds means 115 wpm and roughly 12:40. Both sit well
  inside the limit, so there is no reason to hurry.
- The organisers say *up to* 15 minutes, and the deck lost two slides when the paper did, so there is
  now close to four minutes of headroom. Spend it on pauses after the numbers, not on extra material.

**Numbers are read the English way.** Decimal **point**, not comma: `9.14` is "nine point one four",
`465,000` is "four hundred sixty-five thousand". `1e-5` is "one e minus five". `WER` is said as a word,
"wer", or spelled out — pick one on slide 2 and keep it.

**Dataset names.** `scripted` and `spontaneous`, as printed.

---

## Before you record

**Recording.** No camera needed: capture the screen in presenter mode and narrate. Record in one pass —
if you stumble, repeat the whole sentence rather than stopping, which makes it trivial to cut.

**Export, to stay under 100 MB.** Slides compress extremely well, so a 1080p screen recording of this
length is usually well under the cap. If yours is not:

```
ffmpeg -i raw.mp4 -c:v libx264 -preset slow -crf 27 -c:a aac -b:a 96k innova_task1_en.mp4
```

Raise `-crf` to shrink further; check with `ls -lh`. Keep the output at 1920×1080.

---

## Time budget

| # | Slide | Window | Length |
|---|---|---|---|
| 1 | Title | 0:00 – 0:55 | 55 s |
| 2 | Puno Quechua, and why ASR | 0:55 – 1:50 | 55 s |
| 3 | Finding 1: spelling conventions | 1:50 – 2:45 | 55 s |
| 4 | The system | 2:45 – 3:40 | 55 s |
| 5 | Main result | 3:40 – 4:25 | 45 s |
| 6 | How many updates, and what is noise | 4:25 – 5:45 | 80 s |
| 7 | Did not work: averaging models | 5:45 – 6:50 | 65 s |
| 8 | Did not work: masking and the language model | 6:50 – 7:55 | 65 s |
| 9 | The held-out set | 7:55 – 8:50 | 55 s |
| 10 | Limitations | 8:50 – 10:00 | 70 s |
| 11 | What we take away | 10:00 – 11:15 | 75 s |

---

# Slide 1 · Title

**0:00 – 0:55**

*[On screen: title, authors, and the three cards — 13.15 → 9.14 · 8 points · Measured once]*

Good morning. My name is Raymond Negron, from the Department of Computing at Universidad Católica San
Pablo, in Arequipa. Today I will present team Innova's system for Task 1 of the SIMBig 2026 shared
task: automatic speech recognition for Puno Quechua. This is joint work with Rosa Paccotacya-Yanque.

Three things frame this talk. We improve the published baseline from 13.15 to 9.14 mean WER. The
curated reference's spelling conventions are worth nearly eight points of scripted WER on their own.
And because the established split lets a model hear every development sentence in training, we
selected on a split that does not, and kept a held-out set we scored only once.

*[Point at the three cards with the cursor as you say them. Advance.]*

---

# Slide 2 · Puno Quechua, and why ASR

**0:55 – 1:50**

*[On screen: four cards — 465.000 · 66 h · 4 · 659 — and the table of the paper's two configurations]*

Puno Quechua is spoken by around 465,000 people, but almost nobody writes it, because schooling is in
Spanish. That is why speech recognition here is not a convenience: a text interface shuts out exactly
the people we want to help.

Two things about the task. We start from the 66 hours of speech the organisers released. And the
ranking averages four numbers: WER and CER — word error rate and character error rate — on scripted
and on spontaneous.

Look at the table at the bottom. The paper's best configuration on scripted has 1.19 WER, but 13.6 on
spontaneous. So averaging the four numbers discards that configuration and rewards the balanced one.

*[Advance.]*

---

# Slide 3 · Finding 1: spelling conventions are worth nearly eight points

**1:50 – 2:45**

*[On screen: three spelling examples, the two-reference table, and the conclusion in terracotta]*

Let me start with the first finding.

Puno Quechua has no standardised spelling, so the Common Voice contributors wrote it as they saw fit.
The organisers released a curated reference for the 2,067 sentences, and 629 of them change. These are
single-letter changes — *allqu* becomes *alqu*, *iskay* becomes *ishkay* — but under WER each one costs
a whole word.

This experiment isolates the effect. We took the paper's model, had it transcribe the development set
once, and scored those transcriptions against both references. Against the raw text it scores 8.13.
Against the curated one, 16.08.

The model did not change between the two measurements: it is the same set of hypotheses. Those 7.95
points of difference are spelling, not acoustics.

*[Pause after "not acoustics". This is the sentence you want them to remember. Advance.]*

---

# Slide 4 · The system

**2:45 – 3:40**

*[On screen: the pipeline diagram, full page]*

This is the system. We start from the published XLS-R-300M checkpoint with continued pre-training, and
fine-tune for 500 updates. Both corpora and the curated reference go into normalisation, which folds
everything down to an alphabet of 29 symbols.

From there we get two splits, and this is the important part. The one on the left is speaker-disjoint,
like the baseline, but it shares sentences between training and development. The one on the right is
disjoint by speaker **and** by text.

The model we submit is trained on the first. The second is what gives us a clean measurement.

Bottom left, decoding: scripted with greedy argmax, spontaneous with beam search and an order-eight
character language model.

*[Trace the diagram with the cursor in the order you narrate it. Advance.]*

---

# Slide 5 · Main result

**3:40 – 4:25**

*[On screen: the seven-column table and the three takeaways below]*

This is the result. We improve the baseline from 13.15 to 9.14 mean WER, on the same split and against
the same curated reference. It is a symmetric comparison: same base model, same scoring rule.

And to be honest about where the gain comes from. On scripted we gain 6.85 points, which is the same
order as the 7.95 points of spelling mismatch. Most of what we gain is spelling, not acoustic ability.

What we do achieve is improving on all four numbers at once, as the table shows, which is what the
ranking demands.

*[Advance.]*

---

# Slide 6 · How many updates to train, and what is noise

**4:25 – 5:45**

*[On screen: three cards — 7,02 · 9,23 · 0,3 — and the correction paragraph]*

On how much to train. Starting from a checkpoint already adapted to Quechua, you need far fewer updates
than the 8,000 of the original recipe. Between 250 and 1,000 updates, with learning rates from one e
minus five to five e minus five, everything lands between 7.1 and 7.3: a wide surface, not a single
optimum.

And here is a correction worth making. One run of 500 updates at two e minus five gave 6.96 mean WER,
the best of all configurations on all four numbers, and we recorded it as the winner. But a second seed
of the identical recipe gave 7.26. It was not a reliable measurement.

That is why we measure noise with three seeds and set a threshold of 0.3: below that, a difference
cannot be told apart from chance. And notice the contrast between the two development sets: the small
one has more than twice the spread. It does not only measure less precisely — it manufactures
differences that are not there.

*[Advance.]*

---

# Slide 7 · What did not work: averaging models

**5:45 – 6:50**

*[On screen: the ensemble table on the left, the mechanism explained on the right]*

Three things we expected to help, and they did not. The first: averaging models across seeds.

A seed-to-seed noise of 0.3 is exactly what an ensemble should remove. But we gained nothing with
either way of combining.

And the failure has a mechanism we could measure. CTC does not emit letters continuously: it emits
sharp spikes at specific frames. Two separately trained models place those spikes at neighbouring
frames, but not identical ones. So averaging logits is a geometric mean, and a confident spike averaged
against a near-zero value collapses: the letter disappears. Character deletions go from 21 to 50.

Averaging probabilities, which is an arithmetic mean, no longer loses letters, but it does not gain
anything either. So what we learned is that with CTC alignments, the fusion rule matters more than the
ensemble.

*[Advance.]*

---

# Slide 8 · What did not work: masking and the language model

**6:50 – 7:55**

*[On screen: two halves — SpecAugment on the left, the language model on the right]*

The other two things that did not work.

On the left, more regularisation. Overfitting past a thousand updates was the failure we observed, and
SpecAugment attacks it directly. But raising the masking probability makes things monotonically worse —
6.81, 7.05, 7.68 — and worse on all four numbers. With 500 updates there is little time to overfit.

On the right, the language model. On scripted no configuration wins measurably, so the final system
decodes scripted without it. On spontaneous it does win on the development set, between 0.12 and 0.31
depending on the seed. But on the held-out set it gains zero, and loses 0.04 and 0.08.

A gain tuned on development data did not survive a clean set. We keep it because the effect is within
the noise, but we no longer claim that it helps.

*[Advance.]*

---

# Slide 9 · The held-out set, measured once

**7:55 – 8:50**

*[On screen: development versus held-out table, and the three takeaways]*

This slide is the one that matters most to me methodologically.

A development set that has guided every decision stops being an unbiased estimate. Ours set the number
of updates, the learning rates, 72 language model configurations, and which decoder to use in each
domain.

So we set a held-out set aside on day one, never looked at it, and measured it once, after freezing
every decision.

The means agree: 7.02 against 7.08. But that agreement hides two large and opposite movements: scripted
doubles its error and spontaneous divides it by three. We think that is composition rather than
generalisation, because the held-out set contains one speaker who alone scores 98.67 WER over 59
scripted clips.

*[Advance.]*

---

# Slide 10 · Limitations

**8:50 – 10:00**

*[On screen: three cards — 1 · 2 de 9 · 3 — and the terracotta block below]*

We identified four limitations.

First, and most seriously, the system we submit contains data leakage. It is trained on the established
split, whose training portion contains the development sentences. We submit it for comparability with
the baseline, and because it sees 15 percent more distinct sentences, but there is leakage across
splits and we say so.

Second: the search was narrow, two hyperparameters out of nine. Third: three seeds are enough to expose
apparent wins, but not to separate similar configurations.

And fourth, half of the ranking cannot be measured faithfully. The spontaneous test consists entirely
of speakers aged sixty or over, and speakers of that age are a small and almost unlabelled part of the
corpus: about eight percent of the spontaneous audio, across which exactly one clip carries a human
transcription. The domain that decides half of our rank is the one we can least evaluate.

*[Advance.]*

---

# Slide 11 · What we take away

**10:00 – 11:15**

*[On screen: three closing cards, the future directions, and the repository link]*

And finally, to close, three things.

We improve the published baseline from 13.15 to 9.14 mean WER, on the same split and with the same
reference, and we are explicit that most of that gain is spelling, not acoustics.

Second, the evaluation protocol. Because the established split shares every development sentence with
training, we selected on a speaker- and text-disjoint split and kept a held-out set we scored once,
after freezing every decision. The two means agree, 7.02 against 7.08, so the clean estimate did not
collapse.

And third, three approaches that did not help, with the mechanism explained in the ensemble case.

Two directions for future work. A text-disjoint development set built on the official Common Voice
split would give 679 honest clips instead of 339. And above all, correcting the silver labels of
elderly speakers would make half of this benchmark measurable.

All the code and the splits are in the repository. Thank you very much.

*[Leave the slide on screen for two or three seconds before you stop recording.]*

---

## If you are running long

Unlikely at this length, but if you do run over, cut in this order. Each cut is written so the
following paragraph still fits:

1. **Slide 6**, the correction anecdote (6.96 against 7.26) — about **20 s**. The 0.3 threshold stays,
   which is what slides 7 and 8 need.
2. **Slide 8**, the SpecAugment half — about **25 s**. Say only "raising the masking probability makes
   things monotonically worse" and move to the language model.
3. **Slide 10**, the second and third limitations — about **15 s**. Do not sacrifice the first or the
   fourth: those are the two that actually bound the result.

## If you are running short

Do not stretch the closing. Add on **slide 9** why a development set that drove every decision stops
being an unbiased estimate: it chose the number of updates, the learning rates, 72 language model
configurations and the decoder for each domain. None of that is training, but all of it is selection,
which is the reason the held-out set exists.

---

*Every figure here matches the deck and the paper; their provenance is documented in
`deliverables/paper/FACTS.md`. Figures are written English-style (decimal point) while the deck prints
them Spanish-style (decimal comma).*
