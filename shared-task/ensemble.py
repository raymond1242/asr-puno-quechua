"""
Ensembles of CTC models: fuse the per-frame posteriors of N models, decode once.

Why. The final recipe (500 steps, LR 2e-5) has ~0.30 mean-WER of noise between
seeds (HANDOFF finding H). Averaging the frame scores of models trained with
different seeds cancels part of that noise, at the cost of N forward passes.

How. `CTCEnsemble` looks like one CTC model to everything downstream: its
`forward` returns the fused frame scores, and it answers
`_get_feat_extract_output_lengths` (used to trim padding) from its first
member. So greedy decoding in 04_evaluate.py, `ctc_lm.ctc_logprobs` and the
beam-search decoder in ctc_lm.py all work on it unchanged.

Two fusions, measured on 2026-10-09 (HANDOFF finding L):

  "logit"  mean of the logits = normalised GEOMETRIC mean of the frame
           posteriors. Loses badly: scripted 4.60 -> 6.53 WER. CTC posteriors
           are spikes, and 14% of the time two seeds put the same character's
           spike one frame apart. The geometric mean needs every member to
           agree on the frame, so the blank wins both frames and the character
           is lost: character deletions went from 21 to 50.
  "prob"   ARITHMETIC mean of the posteriors (returned as its log). One member
           seeing the character is enough; deletions stay at 23. The default.

The members must agree on vocabulary (the same symbol at the same index),
feature extraction and conv frontend (the same frame rate), or the frames they
average are not the same frames. `load_ctc` checks all three and refuses
otherwise.
"""

import math
import sys
from types import SimpleNamespace

import torch


FUSIONS = ("prob", "logit")


class CTCEnsemble(torch.nn.Module):
    def __init__(self, members, fusion="prob"):
        super().__init__()
        if fusion not in FUSIONS:
            raise ValueError(f"fusion must be one of {FUSIONS}")
        self.members = torch.nn.ModuleList(members)
        self.fusion = fusion

    def forward(self, **inputs):
        logits = torch.stack([m(**inputs).logits.float() for m in self.members])
        if self.fusion == "logit":
            fused = logits.mean(0)
        else:  # log of the mean probability, computed stably
            fused = torch.logsumexp(logits.log_softmax(-1), dim=0) - math.log(len(self.members))
        return SimpleNamespace(logits=fused)

    def _get_feat_extract_output_lengths(self, lengths):
        return self.members[0]._get_feat_extract_output_lengths(lengths)


def _frontend(config):
    keys = ("conv_dim", "conv_stride", "conv_kernel", "feat_extract_norm", "vocab_size")
    return {k: getattr(config, k, None) for k in keys}


def load_ctc(paths, device, fusion="prob"):
    """One path -> that model; several -> a CTCEnsemble of them.

    Returns (model, processor). The processor is the first member's; every
    other member's must be identical in vocabulary and feature extraction.
    """
    from transformers import AutoModelForCTC, AutoProcessor

    paths = [str(p) for p in paths]
    processor = AutoProcessor.from_pretrained(paths[0])
    ref_vocab = processor.tokenizer.get_vocab()
    ref_fe = processor.feature_extractor.to_dict()

    members, ref_frontend = [], None
    for p in paths:
        proc = AutoProcessor.from_pretrained(p)
        if proc.tokenizer.get_vocab() != ref_vocab:
            sys.exit(f"{p}: vocabulary differs from {paths[0]}; cannot average its logits")
        if proc.feature_extractor.to_dict() != ref_fe:
            sys.exit(f"{p}: feature extraction differs from {paths[0]}")
        model = AutoModelForCTC.from_pretrained(p).to(device).eval()
        frontend = _frontend(model.config)
        if ref_frontend is None:
            ref_frontend = frontend
        elif frontend != ref_frontend:
            sys.exit(f"{p}: conv frontend differs from {paths[0]} ({frontend} vs "
                     f"{ref_frontend}); its frames would not line up")
        members.append(model)

    model = members[0] if len(members) == 1 else CTCEnsemble(members, fusion).eval()
    return model, processor
