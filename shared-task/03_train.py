#!/usr/bin/env python3
"""
CTC fine-tuning for Puno Quechua ASR (SIMBig 2026, Task 1).

Reimplements the paper's XLS-R recipe on top of transformers so the run is
resumable, runs on a single modern CUDA box with plain pip, and leaves room
for the improvements the baseline does not have.

Reads the manifests written by 01_build_manifests.py:
    data/manifests/sharedtask/{train,dev_scripted,dev_spontaneous}.tsv
with columns: path, text, duration_s, speaker, source, label_type

Start points:
  --init checkpoints/hf/cpt            a converted CPT checkpoint (no CTC head)
  --init checkpoints/hf/ft_cpt_silver  an already fine-tuned model (warm start)
  --init facebook/wav2vec2-xls-r-300m  the public XLS-R, no Quechua adaptation

Typical run:
  python shared-task/03_train.py \
      --init checkpoints/hf/ft_cpt_silver \
      --out  checkpoints/hf/ft_curated \
      --max_steps 8000

Resume after a crash or a reboot: re-run the identical command. The Trainer
picks up the last checkpoint in --out automatically.
"""

import argparse
import json
import os
import sys
from dataclasses import dataclass
from math import gcd
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from scipy.signal import resample_poly

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

PUNCT = ["?", "!", "¿", "¡", ".", ","]


def normalize(text):
    """Same normalization the organisers' curated reference already applies."""
    return " ".join("".join(c.lower() for c in str(text) if c not in PUNCT).split())


def load_audio(path, target_sr=16000):
    audio, sr = sf.read(str(path), always_2d=True)
    audio = audio.mean(axis=1)
    if sr != target_sr:
        g = gcd(sr, target_sr)
        audio = resample_poly(audio, target_sr // g, sr // g)
    return audio.astype(np.float32)


# --------------------------------------------------------------------------
# Dataset
# --------------------------------------------------------------------------
class ManifestDataset(torch.utils.data.Dataset):
    def __init__(self, tsv, processor, min_s=0.4, max_s=20.0, upsample=None):
        df = pd.read_csv(tsv, sep="\t", quoting=3)
        for col in ("path", "text"):
            if col not in df.columns:
                sys.exit(f"{tsv} has no '{col}' column")

        df["text"] = df["text"].map(normalize)
        before = len(df)
        df = df[df["text"].str.strip() != ""]
        if "duration_s" in df.columns:
            df = df[(df.duration_s >= min_s) & (df.duration_s <= max_s)]
        df = df.reset_index(drop=True)
        hours = f"  ({df.duration_s.sum()/3600:.2f} h)" if "duration_s" in df else ""
        print(f"  {Path(tsv).name}: {len(df)} clips kept of {before}{hours}")

        # Upsample a source (the paper repeats spontaneous x3 to offset the
        # scripted/spontaneous imbalance).
        if upsample and "source" in df.columns:
            name, factor = upsample
            extra = df[df["source"] == name]
            if len(extra) and factor > 1:
                df = pd.concat([df] + [extra] * (factor - 1), ignore_index=True)
                print(f"    upsampled '{name}' x{factor} -> {len(df)} clips")

        self.df = df
        self.processor = processor
        # Length-grouped batching needs this; duration is a fine proxy.
        self.lengths = (df["duration_s"] * 16000).astype(int).tolist() \
            if "duration_s" in df else [16000] * len(df)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        audio = load_audio(row["path"])
        values = self.processor(audio, sampling_rate=16000).input_values[0]
        labels = self.processor.tokenizer(row["text"]).input_ids
        return {"input_values": values, "labels": labels}


@dataclass
class CTCCollator:
    processor: object

    def __call__(self, features):
        inputs = self.processor.pad(
            [{"input_values": f["input_values"]} for f in features],
            padding=True, return_tensors="pt",
        )
        labels = self.processor.tokenizer.pad(
            [{"input_ids": f["labels"]} for f in features],
            padding=True, return_tensors="pt",
        )
        # -100 is ignored by the CTC loss.
        inputs["labels"] = labels["input_ids"].masked_fill(
            labels.attention_mask.ne(1), -100
        )
        return inputs


# --------------------------------------------------------------------------
# Vocabulary
# --------------------------------------------------------------------------
def build_vocab(texts, out_dir):
    """Character vocabulary over the training text, in fairseq-compatible order."""
    chars = sorted({c for t in texts for c in t if c != " "})
    vocab = {"<pad>": 0, "<s>": 1, "</s>": 2, "<unk>": 3, "|": 4}
    for c in chars:
        vocab.setdefault(c, len(vocab))
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "vocab.json").write_text(
        json.dumps(vocab, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"  vocab: {len(vocab)} symbols -> {out_dir/'vocab.json'}")
    return vocab


class DurationBatchSampler(torch.utils.data.Sampler):
    """Batch by total audio duration, the way fairseq's `max_tokens` does.

    A fixed clip count is the wrong unit here: spontaneous answers run to 30 s
    while scripted clips average 4.8 s, so any count that fits the long clips
    wastes most of the GPU on the short ones — and any count tuned for the short
    ones dies on a batch of long ones. Budgeting a batch by padded duration
    keeps memory flat and throughput high.

    Clips are bucketed by length first, so a batch holds similar durations and
    padding stays cheap.
    """

    def __init__(self, lengths_s, max_batch_s, seed=0, shuffle=True, window=256):
        self.lengths_s = list(lengths_s)
        self.max_batch_s = max_batch_s
        self.seed = seed
        self.shuffle = shuffle
        self.window = window
        self.epoch = 0
        self._batches = self._build(0)

    def _build(self, epoch):
        g = torch.Generator().manual_seed(self.seed + epoch)
        order = (torch.randperm(len(self.lengths_s), generator=g).tolist()
                 if self.shuffle else list(range(len(self.lengths_s))))

        # Sort within a sliding window: near-length grouping that still varies
        # between epochs, rather than one fixed global ordering.
        grouped = []
        for i in range(0, len(order), self.window):
            chunk = order[i:i + self.window]
            grouped += sorted(chunk, key=lambda j: self.lengths_s[j])

        batches, cur, longest = [], [], 0.0
        for idx in grouped:
            d = self.lengths_s[idx]
            nxt = max(longest, d)
            if cur and nxt * (len(cur) + 1) > self.max_batch_s:
                batches.append(cur)
                cur, longest = [idx], d
            else:
                cur.append(idx)
                longest = nxt
        if cur:
            batches.append(cur)

        if self.shuffle:
            perm = torch.randperm(len(batches), generator=g).tolist()
            batches = [batches[i] for i in perm]
        return batches

    def set_epoch(self, epoch):
        self.epoch = epoch
        self._batches = self._build(epoch)

    def __iter__(self):
        return iter(self._batches)

    def __len__(self):
        return len(self._batches)


def auto_batch_seconds(explicit):
    """Audio seconds per batch, from available VRAM.

    Reference point: the paper's fairseq run used max_tokens 5.6M samples =
    350 s per GPU on a 48 GB L40S, without gradient checkpointing. We train
    with checkpointing on, so these are deliberately conservative — raise
    --max_batch_s if nvidia-smi shows headroom.
    """
    if explicit:
        return explicit
    if not torch.cuda.is_available():
        return 20.0
    gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    for threshold, secs in ((70, 400.0), (44, 240.0), (30, 160.0), (20, 110.0),
                            (14, 70.0), (0, 40.0)):
        if gb >= threshold:
            print(f"  detected {gb:.0f} GB VRAM -> {secs:.0f} s of audio per batch")
            return secs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--init", required=True, help="Start point: HF dir or hub id")
    p.add_argument("--out", required=True)
    p.add_argument("--manifest_dir", default=str(ROOT / "data/manifests/sharedtask"))
    p.add_argument("--max_steps", type=int, default=8000)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--warmup_ratio", type=float, default=0.1)
    p.add_argument("--freeze_steps", type=int, default=0,
                   help="Freeze the encoder for N steps. Use ~half of --max_steps "
                        "when --init has no CTC head yet (the paper's schedule).")
    p.add_argument("--max_batch_s", type=float, default=None,
                   help="Audio seconds per batch (padded). Default: from VRAM")
    p.add_argument("--eval_batch_size", type=int, default=4)
    p.add_argument("--grad_accum", type=int, default=4)
    p.add_argument("--max_s", type=float, default=30.0)
    p.add_argument("--upsample_spontaneous", type=int, default=1,
                   help="Repeat spontaneous clips N times (paper uses 3 for V-only)")
    p.add_argument("--eval_steps", type=int, default=500)
    p.add_argument("--eval_after_steps", type=int, default=0,
                   help="Skip evaluation before this step. Mirrors fairseq's "
                        "validate_after_updates: while the encoder is frozen "
                        "the model is barely learning, so early evals are "
                        "wasted minutes. Set it near --freeze_steps.")
    p.add_argument("--mask_time_prob", type=float, default=0.05)
    p.add_argument("--layerdrop", type=float, default=0.0)
    p.add_argument("--fp16", action="store_true", default=torch.cuda.is_available())
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    from transformers import (
        Trainer, TrainingArguments, Wav2Vec2CTCTokenizer,
        Wav2Vec2FeatureExtractor, Wav2Vec2ForCTC, Wav2Vec2Processor,
    )
    import jiwer

    man = Path(args.manifest_dir)
    train_tsv = man / "train.tsv"
    if not train_tsv.exists():
        sys.exit(f"No manifests at {man}. Run 01_build_manifests.py first.")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # ---- processor: reuse the start point's vocabulary when it has one, so a
    # warm start keeps its CTC head; otherwise build one from the training text.
    init_vocab = Path(args.init) / "vocab.json"
    if init_vocab.exists():
        print(f"Vocabulary: reusing {init_vocab}")
        processor = Wav2Vec2Processor.from_pretrained(args.init)
        processor.save_pretrained(out)
    else:
        print("Vocabulary: building from training text")
        texts = pd.read_csv(train_tsv, sep="\t", quoting=3)["text"].map(normalize)
        build_vocab(texts.tolist(), out)
        tokenizer = Wav2Vec2CTCTokenizer(
            str(out / "vocab.json"), unk_token="<unk>", pad_token="<pad>",
            bos_token="<s>", eos_token="</s>", word_delimiter_token="|",
        )
        fe = Wav2Vec2FeatureExtractor(
            feature_size=1, sampling_rate=16000, padding_value=0.0,
            do_normalize=True, return_attention_mask=True,
        )
        processor = Wav2Vec2Processor(feature_extractor=fe, tokenizer=tokenizer)
        processor.save_pretrained(out)

    vocab_size = len(processor.tokenizer)

    print("\nDatasets:")
    upsample = ("spontaneous", args.upsample_spontaneous) if args.upsample_spontaneous > 1 else None
    train_ds = ManifestDataset(train_tsv, processor, max_s=args.max_s, upsample=upsample)

    dev_sets = {}
    for name in ("dev_scripted", "dev_spontaneous"):
        tsv = man / f"{name}.tsv"
        if tsv.exists():
            dev_sets[name.replace("dev_", "")] = ManifestDataset(
                tsv, processor, max_s=args.max_s
            )
    if not dev_sets:
        sys.exit(f"No dev manifests in {man}")

    print(f"\nModel: {args.init}")
    model = Wav2Vec2ForCTC.from_pretrained(
        args.init,
        vocab_size=vocab_size,
        pad_token_id=processor.tokenizer.pad_token_id,
        ctc_loss_reduction="mean",
        ctc_zero_infinity=True,
        attention_dropout=0.0,
        hidden_dropout=0.0,
        feat_proj_dropout=0.0,
        mask_time_prob=args.mask_time_prob,
        layerdrop=args.layerdrop,
        ignore_mismatched_sizes=True,
    )
    model.freeze_feature_encoder()  # the conv frontend stays frozen throughout
    print(f"  {sum(q.numel() for q in model.parameters())/1e6:.0f}M params, "
          f"vocab {vocab_size}")

    def preprocess_logits(logits, labels):
        """Collapse logits to token ids on the GPU.

        Without this the Trainer accumulates a full (frames x vocab) tensor
        for every dev clip before metrics run — tens of GB on a real dev set.
        """
        return logits.argmax(dim=-1)

    def compute_metrics(pred):
        # Already argmaxed by preprocess_logits. The Trainer pads predictions
        # across batches with -100, which decodes as a literal "<unk>" glued to
        # the last word; map it to the blank so CTC decoding drops it.
        ids = np.where(pred.predictions != -100, pred.predictions,
                       processor.tokenizer.pad_token_id)
        labels = np.where(pred.label_ids != -100, pred.label_ids,
                          processor.tokenizer.pad_token_id)
        hyp = processor.batch_decode(ids)
        ref = processor.batch_decode(labels, group_tokens=False)
        pairs = [(r, h) for r, h in zip(ref, hyp) if r.strip()]
        if not pairs:
            return {"wer": 1.0, "cer": 1.0}
        ref, hyp = [r for r, _ in pairs], [h for _, h in pairs]
        return {"wer": jiwer.wer(ref, hyp), "cer": jiwer.cer(ref, hyp)}

    batch_s = auto_batch_seconds(args.max_batch_s)
    targs = TrainingArguments(
        output_dir=str(out),
        max_steps=args.max_steps,
        per_device_train_batch_size=1,   # unused: DurationBatchSampler sizes batches
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        # An integer step count: transformers 5 dropped `warmup_ratio`, and 4.x
        # truncates a fractional `warmup_steps` to no warmup at all.
        warmup_steps=int(args.max_steps * args.warmup_ratio),
        lr_scheduler_type="linear",
        fp16=args.fp16,
        gradient_checkpointing=True,
        # Length grouping is DurationBatchSampler's job (see the Trainer below).
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_steps=args.eval_steps,
        save_total_limit=2,
        logging_steps=50,
        load_best_model_at_end=True,
        # Both domains count equally in the ranking, so checkpoints are picked
        # on their mean WER (added by DurationBatchTrainer.evaluate), never on
        # scripted alone.
        metric_for_best_model=("eval_mean_wer" if len(dev_sets) > 1
                               else f"eval_{next(iter(dev_sets))}_wer"),
        greater_is_better=False,
        dataloader_num_workers=4,
        seed=args.seed,
        report_to=["wandb"] if os.getenv("WANDB_API_KEY") else [],
        remove_unused_columns=False,
    )

    class DurationBatchTrainer(Trainer):
        """Feed training batches sized by audio duration.

        The stock `group_by_length` reads precomputed lengths only from a
        `datasets.Dataset`; with a plain torch Dataset it falls back to
        iterating every example to measure it, which would decode the whole
        corpus into memory. We know every duration from the manifest already.
        """

        def get_train_dataloader(self):
            from torch.utils.data import DataLoader
            sampler = DurationBatchSampler(
                [l / 16000 for l in self.train_dataset.lengths],
                max_batch_s=batch_s,
                seed=self.args.seed,
            )
            self._duration_sampler = sampler
            print(f"  {len(sampler)} batches/epoch at <={batch_s:.0f}s each")
            return DataLoader(
                self.train_dataset,
                batch_sampler=sampler,
                collate_fn=self.data_collator,
                num_workers=self.args.dataloader_num_workers,
                pin_memory=self.args.dataloader_pin_memory,
            )

        def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
            """Force each clip's padded frames to the blank before decoding.

            The model still emits symbols over the zero padding, and they land
            at the end of the shorter clips' transcripts (04_evaluate.py trims
            for the same reason). Left in, the in-training WER read 24.65 for a
            model that 04_evaluate.py scored at 10.95 -- and checkpoint
            selection runs on that number.
            """
            loss, logits, labels = super().prediction_step(
                model, inputs, prediction_loss_only, ignore_keys)
            mask = inputs.get("attention_mask")
            if isinstance(logits, torch.Tensor) and mask is not None:
                frames = self.model._get_feat_extract_output_lengths(mask.sum(-1))
                pad = (torch.arange(logits.shape[1], device=logits.device)[None, :]
                       >= frames[:, None])
                blank = torch.full_like(logits[0, 0], torch.finfo(logits.dtype).min)
                blank[processor.tokenizer.pad_token_id] = 0
                logits = torch.where(pad[..., None], blank, logits)
            return loss, logits, labels

        def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
            """Add the mean WER over both dev domains, for checkpoint selection."""
            metrics = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)
            keys = [f"{metric_key_prefix}_{d}_wer" for d in ("scripted", "spontaneous")]
            if all(k in metrics for k in keys):
                metrics[f"{metric_key_prefix}_mean_wer"] = sum(metrics[k] for k in keys) / 2
                self.log({f"{metric_key_prefix}_mean_wer": metrics[f"{metric_key_prefix}_mean_wer"]})
            return metrics

    trainer = DurationBatchTrainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset={k: v for k, v in dev_sets.items()},
        data_collator=CTCCollator(processor),
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits,
        processing_class=processor,
    )

    if args.eval_after_steps > 0:
        from transformers import TrainerCallback

        class DelayEval(TrainerCallback):
            """Suppress evaluation and checkpoint saving before a given step."""

            def __init__(self, after):
                self.after = after

            def on_step_end(self, a, state, control, **kw):
                if state.global_step < self.after:
                    control.should_evaluate = False
                    control.should_save = False
                return control

        trainer.add_callback(DelayEval(args.eval_after_steps))
        print(f"  evaluation starts at step {args.eval_after_steps}")

    if args.freeze_steps > 0:
        from transformers import TrainerCallback

        class ThawEncoder(TrainerCallback):
            """Keep the encoder frozen until the fresh CTC head has settled,
            so random gradients cannot wreck the pretrained representations."""
            def __init__(self, n):
                self.n, self.done = n, False

            def on_train_begin(self, a, state, c, model=None, **kw):
                if state.global_step < self.n:
                    model.wav2vec2.encoder.requires_grad_(False)
                    print(f"  encoder frozen for the first {self.n} steps")
                else:
                    self.done = True

            def on_step_end(self, a, state, c, model=None, **kw):
                if not self.done and state.global_step >= self.n:
                    model.wav2vec2.encoder.requires_grad_(True)
                    self.done = True
                    print(f"  step {state.global_step}: encoder unfrozen")

        trainer.add_callback(ThawEncoder(args.freeze_steps))

    resume = any(out.glob("checkpoint-*"))
    if resume:
        print(f"\nResuming from the last checkpoint in {out}")
    print(f"\nEffective batch: ~{batch_s:.0f}s x {args.grad_accum} = "
          f"~{batch_s*args.grad_accum:.0f}s of audio per update "
          f"(the paper's fairseq run used ~700s)")

    trainer.train(resume_from_checkpoint=resume)
    trainer.save_model(str(out))
    processor.save_pretrained(out)

    print("\nFinal dev metrics:")
    summary = {}
    for name, ds in dev_sets.items():
        m = trainer.evaluate(eval_dataset=ds, metric_key_prefix=name)
        summary[name] = {"wer": m[f"{name}_wer"] * 100, "cer": m[f"{name}_cer"] * 100}
        print(f"  {name:12s} WER {summary[name]['wer']:6.2f}%   CER {summary[name]['cer']:6.2f}%")
    if len(summary) > 1:
        mw = sum(v["wer"] for v in summary.values()) / len(summary)
        mc = sum(v["cer"] for v in summary.values()) / len(summary)
        print(f"  {'mean':12s} WER {mw:6.2f}%   CER {mc:6.2f}%   <- what the ranking tracks")
    (out / "dev_metrics.json").write_text(json.dumps(summary, indent=2))
    print(f"\nModel saved -> {out}")


if __name__ == "__main__":
    main()
