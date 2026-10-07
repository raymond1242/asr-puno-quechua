#!/usr/bin/env python3
"""
Convert a fairseq wav2vec2/XLS-R CTC checkpoint into a HuggingFace
Wav2Vec2ForCTC model — without installing fairseq.

fairseq 0.12.2 has no wheel for Python 3.11 / arm64, so the released
`checkpoint_best.pt` files cannot be loaded on Apple Silicon the normal way.
This script unpickles the checkpoint with a stub-importer (fairseq config
objects become inert placeholders; the tensors load normally) and remaps the
state-dict keys onto the HF architecture.

Usage:
  python eval/hf/convert_fairseq_w2v2.py \
      --ckpt checkpoints/ft_cpt_silver/checkpoint_best.pt \
      --dict data/manifests/finetune/qxp_v2/dict.ltr.txt \
      --out checkpoints/hf/ft_cpt_silver
"""

import argparse
import io
import json
import pickle
import sys
import types
from pathlib import Path

import torch


# --------------------------------------------------------------------------
# Unpickling without fairseq
# --------------------------------------------------------------------------
class _Stub:
    """Stands in for any fairseq/omegaconf class we cannot import."""

    def __init__(self, *a, **kw):
        self._args, self._kwargs = a, kw

    def __setstate__(self, state):
        if isinstance(state, dict):
            self.__dict__.update(state)

    def __reduce__(self):
        return (_Stub, ())


def _stub_factory(module, name):
    return type(name, (_Stub,), {"__module__": module})


class _StubUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        try:
            return super().find_class(module, name)
        except (ImportError, AttributeError):
            return _stub_factory(module, name)


_shim = types.ModuleType("stub_pickle")
_shim.Unpickler = _StubUnpickler
_shim.load = lambda f, **kw: _StubUnpickler(f, **kw).load()
_shim.loads = lambda b, **kw: _StubUnpickler(io.BytesIO(b), **kw).load()
_shim.dump, _shim.dumps, _shim.Pickler = pickle.dump, pickle.dumps, pickle.Pickler
_shim.HIGHEST_PROTOCOL = pickle.HIGHEST_PROTOCOL


def load_fairseq_ckpt(path):
    return torch.load(path, map_location="cpu", pickle_module=_shim, weights_only=False)


# --------------------------------------------------------------------------
# Key remapping: fairseq -> HF
# --------------------------------------------------------------------------
PREFIX = "w2v_encoder.w2v_model."

ATTN = {
    "self_attn.k_proj": "attention.k_proj",
    "self_attn.v_proj": "attention.v_proj",
    "self_attn.q_proj": "attention.q_proj",
    "self_attn.out_proj": "attention.out_proj",
    "self_attn_layer_norm": "layer_norm",
    "fc1": "feed_forward.intermediate_dense",
    "fc2": "feed_forward.output_dense",
    "final_layer_norm": "final_layer_norm",
}


def convert_state_dict(sd):
    out, skipped = {}, []
    for k, v in sd.items():
        if k.startswith("w2v_encoder.proj."):
            out[k.replace("w2v_encoder.proj.", "lm_head.")] = v
            continue
        if not k.startswith(PREFIX):
            skipped.append(k)
            continue
        r = k[len(PREFIX):]

        if r.startswith("feature_extractor.conv_layers."):
            rest = r[len("feature_extractor.conv_layers."):]
            idx, sub = rest.split(".", 1)
            base = f"wav2vec2.feature_extractor.conv_layers.{idx}."
            if sub.startswith("0."):          # conv
                out[base + "conv." + sub[2:]] = v
            elif sub.startswith("2."):
                # layer_norm mode: conv_layers.{i}.2 is Sequential(
                #   TransposeLast(), Fp32LayerNorm(), TransposeLast())
                # so the affine params live at .2.1.{weight,bias}.
                out[base + "layer_norm." + sub.rsplit(".", 1)[1]] = v
            else:
                skipped.append(k)
        elif r.startswith("post_extract_proj."):
            out["wav2vec2.feature_projection.projection." + r.split(".", 1)[1]] = v
        elif r.startswith("layer_norm."):
            out["wav2vec2.feature_projection.layer_norm." + r.split(".", 1)[1]] = v
        elif r.startswith("encoder.pos_conv.0."):
            out["wav2vec2.encoder.pos_conv_embed.conv." + r.split("encoder.pos_conv.0.")[1]] = v
        elif r.startswith("encoder.layers."):
            rest = r[len("encoder.layers."):]
            idx, sub = rest.split(".", 1)
            for fq, hf in ATTN.items():
                if sub.startswith(fq + "."):
                    out[f"wav2vec2.encoder.layers.{idx}.{hf}." + sub[len(fq) + 1:]] = v
                    break
            else:
                skipped.append(k)
        elif r.startswith("encoder.layer_norm."):
            out["wav2vec2.encoder.layer_norm." + r.split(".", 2)[2]] = v
        elif r in ("mask_emb",):
            out["wav2vec2.masked_spec_embed"] = v
        else:
            skipped.append(k)
    return out, skipped


def read_fairseq_dict(path):
    """fairseq Dictionary: index 0..3 = <s>, <pad>, </s>, <unk>, then file order."""
    syms = ["<s>", "<pad>", "</s>", "<unk>"]
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            syms.append(line.rsplit(" ", 1)[0])
    return syms


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--dict", required=True, help="dict.ltr.txt used at training time")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    from transformers import (
        Wav2Vec2CTCTokenizer,
        Wav2Vec2FeatureExtractor,
        Wav2Vec2ForCTC,
        Wav2Vec2Config,
        Wav2Vec2Processor,
    )

    print(f"Loading {args.ckpt} ...")
    ckpt = load_fairseq_ckpt(args.ckpt)
    sd = ckpt["model"]
    print(f"  {len(sd)} tensors")

    lm_w = sd.get("w2v_encoder.proj.weight")
    if lm_w is None:
        sys.exit("No w2v_encoder.proj.weight — this is not a CTC fine-tuned checkpoint.")
    vocab_size, hidden = lm_w.shape
    n_layers = 1 + max(
        int(k.split("encoder.layers.")[1].split(".")[0])
        for k in sd if "encoder.layers." in k
    )
    print(f"  vocab_size={vocab_size}  hidden={hidden}  layers={n_layers}")

    syms = read_fairseq_dict(args.dict)
    print(f"  dict file has {len(syms)} symbols (incl. 4 fairseq specials)")
    if len(syms) != vocab_size:
        print(f"  !! MISMATCH: checkpoint expects {vocab_size}, dict gives {len(syms)}")
        if len(syms) == vocab_size + 1 and "|" in syms:
            syms.remove("|")
            print("     -> dropped '|' (checkpoint was trained without it); "
                  "<unk> acts as the word boundary")
        else:
            sys.exit("Cannot reconcile dictionary with checkpoint.")

    config = Wav2Vec2Config(
        vocab_size=vocab_size,
        hidden_size=hidden,
        num_hidden_layers=n_layers,
        num_attention_heads=16 if hidden == 1024 else 12,
        intermediate_size=4 * hidden,
        feat_extract_norm="layer",
        do_stable_layer_norm=True,
        conv_bias=True,
        final_dropout=0.0,
        ctc_loss_reduction="mean",
        pad_token_id=0,          # fairseq blank = index 0 (<s>), see inference/transcribe.py
        bos_token_id=None,
        eos_token_id=None,
    )

    model = Wav2Vec2ForCTC(config)
    new_sd, skipped = convert_state_dict(sd)
    missing, unexpected = model.load_state_dict(new_sd, strict=False)

    ignorable = {"wav2vec2.masked_spec_embed"}
    real_missing = [k for k in missing if k not in ignorable]
    print(f"  mapped {len(new_sd)} tensors | missing {len(real_missing)} | unexpected {len(unexpected)}")
    if real_missing:
        print("  MISSING:", real_missing[:10])
    if unexpected:
        print("  UNEXPECTED:", unexpected[:10])
    if skipped:
        print(f"  skipped {len(skipped)} fairseq-only keys, e.g. {skipped[:5]}")
    if real_missing or unexpected:
        sys.exit("Conversion incomplete — refusing to write a broken model.")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # Tokenizer: fairseq specials -> HF names; '|' is the word delimiter.
    vocab = {}
    for i, s in enumerate(syms):
        vocab[{"<s>": "<pad>", "<pad>": "<s>", "</s>": "</s>", "<unk>": "<unk>"}.get(s, s)
              if i < 4 else s] = i
    delim = "|" if "|" in vocab else "<unk>"
    (out / "vocab.json").write_text(json.dumps(vocab, ensure_ascii=False, indent=2))

    tokenizer = Wav2Vec2CTCTokenizer(
        str(out / "vocab.json"),
        unk_token="<unk>", pad_token="<pad>", bos_token="<s>", eos_token="</s>",
        word_delimiter_token=delim,
    )
    fe = Wav2Vec2FeatureExtractor(
        feature_size=1, sampling_rate=16000, padding_value=0.0,
        do_normalize=True, return_attention_mask=True,
    )
    Wav2Vec2Processor(feature_extractor=fe, tokenizer=tokenizer).save_pretrained(out)
    model.save_pretrained(out)
    print(f"\nSaved HF model -> {out}  (word delimiter: {delim!r})")


if __name__ == "__main__":
    main()
